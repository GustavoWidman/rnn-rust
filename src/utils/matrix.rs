use std::{
    array::from_fn,
    fmt::Debug,
    iter::Sum,
    ops::{AddAssign, Mul, MulAssign, SubAssign},
    slice::{Iter, IterMut},
};

use rand::random_range;

#[derive(Clone, Debug)]
pub struct Vector<const SIZE: usize, T> {
    pub data: [T; SIZE],
}

impl<const SIZE: usize, T> Vector<SIZE, T> {
    pub fn new(data: [T; SIZE]) -> Self {
        Self { data }
    }

    pub fn zero() -> Vector<SIZE, T>
    where
        T: From<u8> + Copy + Debug,
    {
        Vector::new([T::from(0u8); SIZE])
    }

    pub fn rand() -> Vector<SIZE, T>
    where
        T: From<f64> + Copy + Debug,
    {
        Vector::new(from_fn(|_| T::from(random_range(-1.0..1.0))))
    }

    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    pub fn add_inplace(&mut self, other: &Vector<SIZE, T>)
    where
        T: Copy + AddAssign,
    {
        self.iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| *x += *y);
    }

    // fused operation
    pub fn sub_scaled(&mut self, other: &Vector<SIZE, T>, scalar: T)
    where
        T: Copy + SubAssign + Mul<Output = T>,
    {
        self.iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| *x -= *y * scalar);
    }

    pub fn mul_sum<'a>(&'a self, other: &'a Vector<SIZE, T>) -> T
    where
        &'a T: Mul<&'a T>,
        T: Sum<<&'a T as Mul>::Output>,
    {
        self.iter().zip(other.iter()).map(|(x, y)| x * y).sum::<T>()
    }

    /// [a, b, c] .* [x, y, z] = [a*x, b*y, c*z]
    pub fn mul_inplace(&mut self, other: &Vector<SIZE, T>)
    where
        T: Copy + MulAssign,
    {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| *x *= *y);
    }
}

pub struct Matrix<const ROWS: usize, const COLS: usize> {
    pub data: Vector<ROWS, Vector<COLS, f64>>,
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    pub fn zero() -> Self {
        Matrix {
            data: Vector::new(from_fn(|_| Vector::zero())),
        }
    }

    pub fn rand() -> Self {
        Matrix {
            data: Vector::new(from_fn(|_| Vector::rand())),
        }
    }

    pub fn transpose(&self) -> Matrix<COLS, ROWS> {
        let mut result = Matrix::<COLS, ROWS>::zero();

        for i in 0..ROWS {
            for j in 0..COLS {
                result.data.data[j].data[i] = self.data.data[i].data[j];
            }
        }

        result
    }

    pub fn sub_scaled(&mut self, other: &Matrix<ROWS, COLS>, scalar: f64) {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(row_a, row_b)| row_a.sub_scaled(row_b, scalar));
    }

    pub fn add_outer_product(&mut self, vec_a: &Vector<ROWS, f64>, vec_b: &Vector<COLS, f64>) {
        for i in 0..ROWS {
            let a_val = vec_a.data[i];
            for j in 0..COLS {
                self.data.data[i].data[j] += a_val * vec_b.data[j];
            }
        }
    }

    pub fn iter(&self) -> Iter<'_, Vector<COLS, f64>> {
        self.data.iter()
    }

    pub fn dot_vec_into(&self, vector: &Vector<COLS, f64>, out: &mut Vector<ROWS, f64>) {
        self.iter()
            .zip(out.iter_mut())
            .for_each(|(row, out_elem)| *out_elem = row.mul_sum(vector));
    }
}
