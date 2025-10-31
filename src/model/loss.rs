use std::{array::from_fn, ops::Div};

use crate::utils::matrix::Vector;

pub enum LossFunction {
    MSE,
}

impl LossFunction {
    pub fn loss<const SIZE: usize>(
        &self,
        pred: &Vector<SIZE, f64>,
        actual: &Vector<SIZE, f64>,
    ) -> f64 {
        let loss = match self {
            Self::MSE => pred
                .iter()
                .zip(actual.iter())
                .map(|(p, a)| (a - p).powi(2))
                .sum::<f64>()
                .div(SIZE as f64),
        };

        loss
    }

    pub fn sequence_loss<const SIZE: usize>(
        &self,
        predictions: &Vec<Vector<SIZE, f64>>,
        labels: &Vec<Vector<SIZE, f64>>,
    ) -> f64 {
        predictions
            .iter()
            .zip(labels)
            .map(|(pred, actual)| self.loss(pred, actual))
            .sum::<f64>()
            .div(predictions.len() as f64) // assume pred.len == actual.len
    }

    pub fn derivative<const SIZE: usize>(
        &self,
        pred: &Vector<SIZE, f64>,
        actual: &Vector<SIZE, f64>,
    ) -> Vector<SIZE, f64> {
        match self {
            Self::MSE => {
                let constant = 2.0 / SIZE as f64;
                Vector::new(from_fn(|i| {
                    let pred = pred.data[i];
                    let actual = actual.data[i];

                    constant * (pred - actual)
                }))

                // optimized from
                // pred.iter()
                //     .zip(actual.iter())
                //     .map(|(p, a)| (2.0 / SIZE as f64) * (p - a))
                //     .collect::<Vec<f64>>()
                //     .try_into()
                //     .unwrap(),
            }
        }
    }
}
