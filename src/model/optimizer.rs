use crate::model::{Weights, gradient::Gradients};

pub trait Optimizer<const INPUT: usize, const HIDDEN: usize, const OUTPUT: usize> {
    fn update(
        &mut self,
        weights: &mut Weights<INPUT, HIDDEN, OUTPUT>,
        grads: &Gradients<INPUT, HIDDEN, OUTPUT>,
    );

    fn boxed(self) -> Box<dyn Optimizer<INPUT, HIDDEN, OUTPUT>>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl<const INPUT: usize, const HIDDEN: usize, const OUTPUT: usize> Optimizer<INPUT, HIDDEN, OUTPUT>
    for SGD
{
    fn update(
        &mut self,
        weights: &mut Weights<INPUT, HIDDEN, OUTPUT>,
        grads: &Gradients<INPUT, HIDDEN, OUTPUT>,
    ) {
        weights
            .input
            .sub_scaled(&grads.input_weights_grad, self.learning_rate);
        weights
            .hidden
            .sub_scaled(&grads.hidden_weights_grad, self.learning_rate);
        weights
            .output
            .sub_scaled(&grads.output_weights_grad, self.learning_rate);
        weights
            .bias_hidden
            .sub_scaled(&grads.bias_hidden_grad, self.learning_rate);
        weights
            .bias_output
            .sub_scaled(&grads.bias_output_grad, self.learning_rate);
    }
}
