use crate::{data::Dataset, model::RNN};

pub trait Evaluator {
    fn evaluate<const INPUT: usize, const OUTPUT: usize, const HIDDEN: usize>(
        &self,
        model: &RNN<INPUT, HIDDEN, OUTPUT>,
        dataset: &Dataset<INPUT, OUTPUT>,
    ) -> EvaluationResult;
}

pub struct EvaluationResult {
    pub primary_metric: f64,
    pub details: String,
}

#[allow(dead_code)]
pub struct BinaryClassificationEvaluator {
    pub threshold: f64,
}

impl Evaluator for BinaryClassificationEvaluator {
    fn evaluate<const INPUT: usize, const OUTPUT: usize, const HIDDEN: usize>(
        &self,
        model: &RNN<INPUT, HIDDEN, OUTPUT>,
        dataset: &Dataset<INPUT, OUTPUT>,
    ) -> EvaluationResult {
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_count = 0;

        for sequence in &dataset.sequences {
            let predictions = model.predict(sequence);
            let labels = sequence.labels();

            for (pred, label) in predictions.iter().zip(labels.iter()) {
                let predicted_class = if pred.data[0] > self.threshold {
                    1.0
                } else {
                    0.0
                };
                let actual_class = label.data[0];

                match (predicted_class as i32, actual_class as i32) {
                    (1, 1) => tp += 1,
                    (1, 0) => fp += 1,
                    (0, 0) => tn += 1,
                    (0, 1) => fn_count += 1,
                    _ => {}
                }
            }
        }

        let precision = tp as f64 / (tp + fp) as f64;
        let recall = tp as f64 / (tp + fn_count) as f64;
        let f1 = 2.0 * (precision * recall) / (precision + recall);
        let accuracy = (tp + tn) as f64 / (tp + tn + fp + fn_count) as f64;

        EvaluationResult {
            primary_metric: accuracy,
            details: format!(
                "Accuracy: {:.2}% | Precision: {:.2}% | Recall: {:.2}% | F1: {:.2}%\n\
                TP: {}, FP: {}, TN: {}, FN: {}\n\
                Positive Accuracy: {:.2}% | Negative Accuracy: {:.2}%",
                accuracy * 100.0,
                precision * 100.0,
                recall * 100.0,
                f1 * 100.0,
                tp,
                fp,
                tn,
                fn_count,
                (tp as f64 / (tp + fp) as f64) * 100.0,
                (tn as f64 / (tn + fn_count) as f64) * 100.0
            ),
        }
    }
}

#[allow(dead_code)]
pub struct RegressionEvaluator;

impl Evaluator for RegressionEvaluator {
    fn evaluate<const INPUT: usize, const OUTPUT: usize, const HIDDEN: usize>(
        &self,
        model: &RNN<INPUT, HIDDEN, OUTPUT>,
        dataset: &Dataset<INPUT, OUTPUT>,
    ) -> EvaluationResult {
        let mut total_error = 0.0;
        let mut total_squared_error = 0.0;
        let mut total_percentage_error = 0.0;
        let mut count = 0;
        let mut sum_actual = 0.0;
        let mut within_5_percent = 0;
        let mut within_10_percent = 0;

        for sequence in &dataset.sequences {
            let predictions = model.predict(sequence);
            let labels = sequence.labels();

            for (pred, label) in predictions.iter().zip(labels.iter()) {
                let actual_denorm = dataset.denormalize_label(label);
                let pred_denorm = dataset.denormalize_label(pred);

                let actual = actual_denorm.data[0];
                let predicted = pred_denorm.data[0];

                let error = (predicted - actual).abs();
                let squared_error = (predicted - actual).powi(2);

                // MAPE calculation (skip if actual is near zero to avoid division issues)
                if actual.abs() > 1e-6 {
                    let percentage_error = (error / actual.abs()) * 100.0;
                    total_percentage_error += percentage_error;

                    // Accuracy within thresholds
                    if percentage_error <= 5.0 {
                        within_5_percent += 1;
                    }
                    if percentage_error <= 10.0 {
                        within_10_percent += 1;
                    }
                }

                total_error += error;
                total_squared_error += squared_error;
                sum_actual += actual;
                count += 1;
            }
        }

        let mae = total_error / count as f64;
        let rmse = (total_squared_error / count as f64).sqrt();
        let mape = total_percentage_error / count as f64;
        let mean_actual = sum_actual / count as f64;

        let accuracy_5 = (within_5_percent as f64 / count as f64) * 100.0;
        let accuracy_10 = (within_10_percent as f64 / count as f64) * 100.0;

        // Calculate R²
        let mut ss_tot = 0.0;
        for sequence in &dataset.sequences {
            let labels = sequence.labels();
            for label in labels.iter() {
                ss_tot += (label.data[0] - mean_actual).powi(2);
            }
        }
        let r_squared = 1.0 - (total_squared_error / ss_tot);

        EvaluationResult {
            primary_metric: (100.0 - mape) / 100.0,
            details: format!(
                "MAE: {:.4} | RMSE: {:.4} | MAPE: {:.2}% | R²: {:.4}\n\
                 Accuracy (±5%): {:.2}% | Accuracy (±10%): {:.2}%",
                mae, rmse, mape, r_squared, accuracy_5, accuracy_10
            ),
        }
    }
}
