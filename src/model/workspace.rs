use crate::utils::matrix::Vector;

pub struct Workspace<const INPUT: usize, const HIDDEN: usize, const OUTPUT: usize> {
    pub input_contrib: Vector<HIDDEN, f64>,
    pub hidden_contrib: Vector<HIDDEN, f64>,
    pub new_hidden: Vector<HIDDEN, f64>,
    pub output: Vector<OUTPUT, f64>,

    pub activation_grad: Vector<HIDDEN, f64>,
    pub dl_dh: Vector<HIDDEN, f64>,
    pub dh_next_temp: Vector<HIDDEN, f64>,
}

impl<const INPUT: usize, const HIDDEN: usize, const OUTPUT: usize>
    Workspace<INPUT, HIDDEN, OUTPUT>
{
    pub fn new() -> Self {
        Self {
            input_contrib: Vector::zero(),
            hidden_contrib: Vector::zero(),
            new_hidden: Vector::zero(),
            output: Vector::zero(),
            dl_dh: Vector::zero(),
            dh_next_temp: Vector::zero(),
            activation_grad: Vector::zero(),
        }
    }
}
