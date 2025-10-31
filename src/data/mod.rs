use std::{path::PathBuf, slice::Iter};

use eyre::Result;
use itertools::Itertools;
use rand::{rng, seq::SliceRandom};

use crate::utils::matrix::Vector;

#[derive(Clone)]
pub struct Entry<const F: usize, const L: usize> {
    pub features: Vector<F, f64>,
    pub labels: Vector<L, f64>,
}

#[derive(Clone)]
pub struct Sequence<const F: usize, const L: usize> {
    entries: Vec<Entry<F, L>>,
}

impl<const F: usize, const L: usize> Sequence<F, L> {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> Iter<'_, Entry<F, L>> {
        self.entries.iter()
    }

    pub fn labels(&self) -> Vec<Vector<L, f64>> {
        self.entries.iter().map(|e| e.labels.clone()).collect()
    }
}

pub struct Dataset<const F: usize, const L: usize> {
    pub sequences: Vec<Sequence<F, L>>,
    pub label_min: Option<[f64; L]>,
    pub label_max: Option<[f64; L]>,
}

impl<const F: usize, const L: usize> Dataset<F, L> {
    /// reads from a csv
    /// csv must have the following format:
    /// "name","t","x1","x2",...,"xF","y1","y2",...,"yL"
    /// "t" is used to group sequences, "name" is used to differentiate sequences with the same "t"
    /// for example:
    /// "seq1",1,0.5,0.3,0.2,1.0,0.0
    /// "seq1",2,0.6,0.4,0.3,0.9,0.1
    /// "seq2",1,0.1,0.2,0.3,0.4,0.5
    /// "seq2",2,0.2,0.3,0.4,0.5,0.6
    /// the above csv contains two sequences: "seq1" and "seq2"
    pub fn from_csv(path: PathBuf) -> Result<Self> {
        let string = std::fs::read_to_string(path)?;
        let mut lines = string.lines();

        let header = lines
            .next()
            .ok_or_else(|| eyre::eyre!("could not find CSV header"))?;

        let expected_columns = 2 + F + L; // name, t, features, labels
        let header_columns: Vec<&str> = header.split(',').collect();

        if header_columns.len() != expected_columns {
            return Err(eyre::eyre!(
                "expected {} total columns (\"name\", \"t\", {} feature columns and {} label columns), found {}",
                expected_columns,
                F,
                L,
                header_columns.len()
            ));
        }

        let x_indexes: Vec<usize> = header_columns
            .iter()
            .enumerate()
            .filter_map(|(i, header)| match header.starts_with('x') {
                true => Some(i),
                false => None,
            })
            .collect();

        let y_indexes: Vec<usize> = header_columns
            .iter()
            .enumerate()
            .filter_map(|(i, header)| match header.starts_with('y') {
                true => Some(i),
                false => None,
            })
            .collect();

        let t_index = header_columns
            .iter()
            .position(|&h| h == "t")
            .ok_or_else(|| eyre::eyre!("could not find 't' column in header"))?;

        let name_index = header_columns
            .iter()
            .position(|&h| h == "name")
            .ok_or_else(|| eyre::eyre!("could not find 'name' column in header"))?;

        let mut data = lines
            .enumerate()
            .map(|(i, line)| {
                let columns: Vec<&str> = line.split(',').collect();

                if columns.len() != expected_columns {
                    return Err(eyre::eyre!(
                        "expected {} columns on line {}, found {}",
                        expected_columns,
                        i + 2, // +2 to account for header and 0-indexing
                        columns.len()
                    ));
                }

                let features: [f64; F] = columns
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| match x_indexes.contains(&i) {
                        true => Some(s.parse::<f64>()),
                        false => None,
                    })
                    .collect::<Result<Vec<f64>, _>>()?
                    .try_into()
                    .map_err(|_| eyre::eyre!("failed to convert features to array"))?;

                let labels: [f64; L] = columns
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| match y_indexes.contains(&i) {
                        true => Some(s.parse::<f64>()),
                        false => None,
                    })
                    .collect::<Result<Vec<f64>, _>>()?
                    .try_into()
                    .map_err(|_| eyre::eyre!("failed to convert features to array"))?;

                let name = columns[name_index].to_string();
                let t = columns[t_index].parse::<usize>()?;

                Ok((
                    name,
                    t,
                    Entry {
                        features: Vector::new(features),
                        labels: Vector::new(labels),
                    },
                ))
            })
            .collect::<Result<Vec<(String, usize, Entry<F, L>)>>>()?
            .into_iter()
            .chunk_by(|(name, _, _)| name.clone())
            .into_iter()
            .map(|(_, group)| {
                let entries = group
                    .sorted_by(|(_, t1, _), (_, t2, _)| t1.cmp(t2))
                    .map(|(_, _, entry)| entry)
                    .collect();

                Sequence { entries }
            })
            .collect::<Vec<Sequence<F, L>>>();

        let mut rng = rng();
        data.shuffle(&mut rng);

        Ok(Dataset {
            sequences: data,
            label_min: None,
            label_max: None,
        })
    }

    pub fn normalize(&mut self) {
        let mut feature_mins = [f64::INFINITY; F];
        let mut feature_maxs = [f64::NEG_INFINITY; F];
        let mut label_mins = [f64::INFINITY; L];
        let mut label_maxs = [f64::NEG_INFINITY; L];

        for sequence in &self.sequences {
            for entry in &sequence.entries {
                for (i, &value) in entry.features.iter().enumerate() {
                    if value < feature_mins[i] {
                        feature_mins[i] = value;
                    }
                    if value > feature_maxs[i] {
                        feature_maxs[i] = value;
                    }
                }

                for (i, &value) in entry.labels.iter().enumerate() {
                    if value < label_mins[i] {
                        label_mins[i] = value;
                    }
                    if value > label_maxs[i] {
                        label_maxs[i] = value;
                    }
                }
            }
        }

        for sequence in &mut self.sequences {
            for entry in &mut sequence.entries {
                for i in 0..F {
                    let min = feature_mins[i];
                    let max = feature_maxs[i];
                    if max - min != 0.0 {
                        entry.features.data[i] = (entry.features.data[i] - min) / (max - min);
                    } else {
                        entry.features.data[i] = 0.0;
                    }
                }

                for i in 0..L {
                    let min = label_mins[i];
                    let max = label_maxs[i];
                    if max - min != 0.0 {
                        entry.labels.data[i] = (entry.labels.data[i] - min) / (max - min);
                    } else {
                        entry.labels.data[i] = 0.0;
                    }
                }
            }
        }

        self.label_min = Some(label_mins);
        self.label_max = Some(label_maxs);
    }

    pub fn split(&self, train_ratio: f64) -> (Dataset<F, L>, Dataset<F, L>) {
        let split_idx = (self.sequences.len() as f64 * train_ratio) as usize;

        let train_sequences = self.sequences[..split_idx].to_vec();
        let test_sequences = self.sequences[split_idx..].to_vec();

        (
            Dataset {
                sequences: train_sequences,
                label_max: self.label_max,
                label_min: self.label_min,
            },
            Dataset {
                sequences: test_sequences,
                label_max: self.label_max,
                label_min: self.label_min,
            },
        )
    }

    #[allow(dead_code)]
    pub fn denormalize_label(&self, normalized: &Vector<L, f64>) -> Vector<L, f64> {
        if let (Some(mins), Some(maxs)) = (&self.label_min, &self.label_max) {
            let mut denormalized = [0.0; L];
            for i in 0..L {
                denormalized[i] = normalized.data[i] * (maxs[i] - mins[i]) + mins[i];
            }
            Vector::new(denormalized)
        } else {
            normalized.clone()
        }
    }
}
