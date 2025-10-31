use log::info;

use crate::{
    data::{Dataset, Sequence},
    model::{
        activation::ActivationFunction, evaluator::Evaluator, gradient::Gradients,
        loss::LossFunction, optimizer::Optimizer, workspace::Workspace,
    },
    utils::matrix::{Matrix, Vector},
};
pub mod activation;
pub mod evaluator;
mod gradient;
pub mod loss;
pub mod optimizer;
mod workspace;

pub struct Weights<const INPUT: usize, const HIDDEN: usize, const OUTPUT: usize> {
    pub input: Matrix<HIDDEN, INPUT>,   // input to hidden
    pub hidden: Matrix<HIDDEN, HIDDEN>, // hidden to hidden
    pub output: Matrix<OUTPUT, HIDDEN>, // hidden to output

    pub bias_hidden: Vector<HIDDEN, f64>,
    pub bias_output: Vector<OUTPUT, f64>,
}

impl<const INPUT: usize, const HIDDEN: usize, const OUTPUT: usize> Weights<INPUT, HIDDEN, OUTPUT> {
    pub fn new() -> Self {
        Self {
            input: Matrix::rand(),
            hidden: Matrix::rand(),
            output: Matrix::rand(),
            bias_hidden: Vector::rand(),
            bias_output: Vector::rand(),
        }
    }
}

pub struct RNN<const INPUT: usize, const HIDDEN: usize, const OUTPUT: usize> {
    weights: Weights<INPUT, HIDDEN, OUTPUT>,
    activation: ActivationFunction,
    loss: LossFunction,
    optimizer: Box<dyn Optimizer<INPUT, HIDDEN, OUTPUT>>,
}

impl<const INPUT: usize, const HIDDEN: usize, const OUTPUT: usize> RNN<INPUT, HIDDEN, OUTPUT> {
    pub fn new(
        activation: ActivationFunction,
        loss: LossFunction,
        optimizer: Box<dyn Optimizer<INPUT, HIDDEN, OUTPUT>>,
    ) -> Self {
        Self {
            weights: Weights::new(),
            activation,
            loss,
            optimizer,
        }
    }

    pub fn predict(&self, sequence: &Sequence<INPUT, OUTPUT>) -> Vec<Vector<OUTPUT, f64>> {
        let mut workspace = Workspace::new();
        let (_, outputs) = self.feedforward(sequence, &mut workspace);
        outputs
    }

    pub fn train<E: Evaluator>(
        &mut self,
        dataset: &Dataset<INPUT, OUTPUT>,
        epochs: usize,
        train_ratio: f64,
        print_every: usize,
        evaluator: E,
    ) -> (f64, f64) {
        let (train_data, test_data) = dataset.split(train_ratio);

        (0..epochs).for_each(|epoch| {
            let loss = self.train_epoch(&train_data);

            if (epoch + 1) % print_every == 0 || epoch == 0 || epoch == epochs - 1 {
                let train_eval = evaluator.evaluate(self, dataset);
                let test_eval = evaluator.evaluate(self, dataset);
                info!(
                    "Epoch {}: Loss = {} | Train Accuracy: {:.2}% | Test Accuracy: {:.2}%",
                    epoch + 1,
                    loss,
                    train_eval.primary_metric * 100.0,
                    test_eval.primary_metric * 100.0
                );
            };
        });

        let train_eval = evaluator.evaluate(self, &train_data);
        let test_eval = evaluator.evaluate(self, &test_data);

        info!("Train Eval:\n{}", train_eval.details);
        info!("Test Eval:\n{}", test_eval.details);

        (train_eval.primary_metric, test_eval.primary_metric)
    }

    fn step<'a>(
        &self,
        input: &Vector<INPUT, f64>,
        prev_hidden: &Vector<HIDDEN, f64>,
        workspace: &'a mut Workspace<INPUT, HIDDEN, OUTPUT>,
    ) -> (&'a Vector<HIDDEN, f64>, &'a Vector<OUTPUT, f64>) {
        // Wx * xt
        self.weights
            .input
            .dot_vec_into(input, &mut workspace.input_contrib);

        // Wh * ht−1
        self.weights
            .hidden
            .dot_vec_into(prev_hidden, &mut workspace.hidden_contrib);

        // Wh * ht−1 + Wx * xt + bh
        workspace.new_hidden = workspace.input_contrib.clone();

        workspace.new_hidden.add_inplace(&workspace.hidden_contrib);
        workspace.new_hidden.add_inplace(&self.weights.bias_hidden);

        // tanh(Wh * ht−1 + Wx * xt + bh)
        self.activation.activate_vec(&mut workspace.new_hidden); // "ht"

        // Wy * ht + by
        self.weights
            .output
            .dot_vec_into(&workspace.new_hidden, &mut workspace.output);
        workspace.output.add_inplace(&self.weights.bias_output);

        return (&workspace.new_hidden, &workspace.output); // "ht" and "yt"
    }

    fn feedforward(
        &self,
        sequence: &Sequence<INPUT, OUTPUT>,
        workspace: &mut Workspace<INPUT, HIDDEN, OUTPUT>,
    ) -> (Vec<Vector<HIDDEN, f64>>, Vec<Vector<OUTPUT, f64>>) {
        let mut prev_hidden = Vector::<HIDDEN, f64>::zero();
        let mut hiddens = Vec::with_capacity(sequence.len());
        let mut outputs = Vec::with_capacity(sequence.len());

        for entry in sequence.iter() {
            let (h, o) = self.step(&entry.features, &prev_hidden, workspace);
            hiddens.push(h.clone());
            outputs.push(o.clone());
            prev_hidden = h.clone();
        }

        (hiddens, outputs)
    }

    fn backpropagate(
        &self,
        sequence: &Sequence<INPUT, OUTPUT>,
        hiddens: Vec<Vector<HIDDEN, f64>>,
        outputs: Vec<Vector<OUTPUT, f64>>,
        workspace: &mut Workspace<INPUT, HIDDEN, OUTPUT>,
    ) -> Gradients<INPUT, HIDDEN, OUTPUT> {
        let output_weights_t = self.weights.output.transpose();
        let hidden_weights_t = self.weights.hidden.transpose();

        let mut grads = Gradients::<INPUT, HIDDEN, OUTPUT>::zero();

        sequence
            .iter()
            .zip(outputs.iter())
            .zip(hiddens.iter())
            .enumerate()
            .rev()
            .for_each(|(i, ((entry, output), hidden))| {
                let loss_grad = self.loss.derivative(output, &entry.labels);

                grads
                    .output_weights_grad
                    .add_outer_product(&loss_grad, hidden);
                grads.bias_output_grad.add_inplace(&loss_grad);

                output_weights_t.dot_vec_into(&loss_grad, &mut workspace.dl_dh);
                workspace.dl_dh.add_inplace(&workspace.dh_next_temp);

                self.activation
                    .derivative_vec_into(hidden, &mut workspace.activation_grad);

                workspace.dl_dh.mul_inplace(&workspace.activation_grad);

                let prev_hidden = match i {
                    0 => &Vector::<HIDDEN, f64>::zero(),
                    _ => &hiddens[i - 1],
                };

                grads
                    .input_weights_grad
                    .add_outer_product(&workspace.dl_dh, &entry.features);
                grads
                    .hidden_weights_grad
                    .add_outer_product(&workspace.dl_dh, prev_hidden);
                grads.bias_hidden_grad.add_inplace(&workspace.dl_dh);

                hidden_weights_t.dot_vec_into(&workspace.dl_dh, &mut workspace.dh_next_temp);
            });

        grads
    }

    fn train_sequence(&mut self, sequence: &Sequence<INPUT, OUTPUT>) -> f64 {
        let mut workspace = Workspace::new();

        // 1. Forward pass
        let (hiddens, outputs) = self.feedforward(sequence, &mut workspace);

        // 2. Compute loss (for monitoring)
        let loss = self.loss.sequence_loss(&outputs, &sequence.labels());

        // 3. Backward pass
        let grads = self.backpropagate(sequence, hiddens, outputs, &mut workspace);

        // 4. Update weights
        self.optimizer.update(&mut self.weights, &grads);

        loss // return loss for monitoring
    }

    fn train_epoch(&mut self, dataset: &Dataset<INPUT, OUTPUT>) -> f64 {
        let total_loss: f64 = dataset
            .sequences
            .iter()
            .map(|seq| self.train_sequence(seq))
            .sum();

        total_loss / dataset.sequences.len() as f64
    }
}
