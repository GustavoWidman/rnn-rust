use std::ops::Neg;

use crate::utils::matrix::Vector;

#[allow(dead_code)]
pub enum ActivationFunction {
    Tanh,
    ReLU,
    Sigmoid,
}

impl ActivationFunction {
    pub fn activate(&self, z: f64) -> f64 {
        match self {
            Self::ReLU => z.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + z.neg().exp()),

            // Self::Tanh => (z.exp() - z.neg().exp()) / (z.exp() + z.neg().exp()),
            Self::Tanh => z.tanh(),
        }
    }

    pub fn derivative(&self, activated: f64) -> f64 {
        match self {
            Self::ReLU => match activated > 0.0 {
                true => 1.0,
                false => 0.0,
            },
            Self::Sigmoid => activated * (1.0 - activated),
            Self::Tanh => 1.0 - activated.powi(2),
        }
    }

    pub fn activate_vec<const SIZE: usize>(&self, vec: &mut Vector<SIZE, f64>) {
        vec.iter_mut().for_each(|z| *z = self.activate(*z));
    }

    pub fn derivative_vec_into<const SIZE: usize>(
        &self,
        activated: &Vector<SIZE, f64>,
        target: &mut Vector<SIZE, f64>,
    ) {
        activated
            .iter()
            .zip(target.iter_mut())
            .for_each(|(a, t)| *t = self.derivative(*a));
    }
}
