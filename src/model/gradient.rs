use crate::utils::matrix::{Matrix, Vector};

pub struct Gradients<const INPUT: usize, const HIDDEN: usize, const OUTPUT: usize> {
    pub input_weights_grad: Matrix<HIDDEN, INPUT>,
    pub hidden_weights_grad: Matrix<HIDDEN, HIDDEN>,
    pub output_weights_grad: Matrix<OUTPUT, HIDDEN>,
    pub bias_hidden_grad: Vector<HIDDEN, f64>,
    pub bias_output_grad: Vector<OUTPUT, f64>,
}

impl<const INPUT: usize, const HIDDEN: usize, const OUTPUT: usize>
    Gradients<INPUT, HIDDEN, OUTPUT>
{
    pub fn zero() -> Self {
        Self {
            bias_hidden_grad: Vector::zero(),
            bias_output_grad: Vector::zero(),
            hidden_weights_grad: Matrix::zero(),
            input_weights_grad: Matrix::zero(),
            output_weights_grad: Matrix::zero(),
        }
    }
}
