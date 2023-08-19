use std::{
    f32,
    fmt::Debug,
    borrow::Borrow,
    ops::{Mul, Add, Sub, Div, AddAssign, SubAssign, DivAssign, Neg}
};

use serde::{Serialize, Deserialize};

use nalgebra::DMatrix;

use super::{SoftmaxedLayer, LEAKY_SLOPE, leaky_relu_d};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixWrapper(DMatrix<f32>);

impl Add<f32> for MatrixWrapper
{
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output
    {
        Self(self.0.add_scalar(rhs))
    }
}

impl Add<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn add(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper(self.0.add_scalar(rhs))
    }
}

impl<T> Add<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output
    {
        Self(self.0 + &rhs.borrow().0)
    }
}

impl<T> Add<T> for &MatrixWrapper
where
    T: Borrow<MatrixWrapper>
{
    type Output = MatrixWrapper;

    fn add(self, rhs: T) -> Self::Output
    {
        MatrixWrapper(&self.0 + &rhs.borrow().0)
    }
}

impl<T> Sub<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output
    {
        Self(self.0 - &rhs.borrow().0)
    }
}

impl<T> Sub<T> for &MatrixWrapper
where
    T: Borrow<MatrixWrapper>
{
    type Output = MatrixWrapper;

    fn sub(self, rhs: T) -> Self::Output
    {
        MatrixWrapper(&self.0 - &rhs.borrow().0)
    }
}

impl Sub<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn sub(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper(self.0.add_scalar(-rhs))
    }
}

impl<T> Mul<T> for &MatrixWrapper
where
    T: Borrow<MatrixWrapper>
{
    type Output = MatrixWrapper;

    fn mul(self, rhs: T) -> Self::Output
    {
        MatrixWrapper(self.0.component_mul(&rhs.borrow().0))
    }
}

impl Mul<f32> for MatrixWrapper
{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output
    {
        Self(self.0 * rhs)
    }
}

impl Mul<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn mul(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper(&self.0 * rhs)
    }
}

impl<T> Mul<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output
    {
        Self(self.0.component_mul(&rhs.borrow().0))
    }
}

impl Div<f32> for MatrixWrapper
{
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output
    {
        Self(self.0 / rhs)
    }
}

impl Div<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn div(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper(&self.0 / rhs)
    }
}

impl<T> Div<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output
    {
        Self(self.0.component_div(&rhs.borrow().0))
    }
}

impl<T> Div<T> for &MatrixWrapper
where
    T: Borrow<MatrixWrapper>
{
    type Output = MatrixWrapper;

    fn div(self, rhs: T) -> Self::Output
    {
        MatrixWrapper(self.0.component_div(&rhs.borrow().0))
    }
}

impl<T> SubAssign<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    fn sub_assign(&mut self, rhs: T)
    {
        self.0 -= &rhs.borrow().0;
    }
}

impl<T> AddAssign<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    fn add_assign(&mut self, rhs: T)
    {
        self.0 += &rhs.borrow().0;
    }
}

impl DivAssign<f32> for MatrixWrapper
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.0 /= rhs;
    }
}

impl Neg for MatrixWrapper
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        Self(-self.0)
    }
}

impl Neg for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn neg(self) -> Self::Output
    {
        MatrixWrapper(-&self.0)
    }
}

impl MatrixWrapper
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self(DMatrix::zeros(previous_size, this_size))
    }

    pub fn new_with<F: FnMut() -> f32>(
        previous_size: usize,
        this_size: usize,
        mut f: F
    )-> Self
    {
        Self(DMatrix::from_fn(previous_size, this_size, |_, _| f()))
    }

    pub fn from_raw<V: Into<Vec<f32>>>(values: V, previous_size: usize, this_size: usize) -> Self
    {
        Self(DMatrix::from_vec(previous_size, this_size, values.into()))
    }

    pub fn swap_raw_values<V: Into<Vec<f32>>>(&mut self, values: V)
    {
        self.0.copy_from_slice(&values.into());
    }

    pub fn fill(&mut self, value: f32)
    {
        self.0.fill(value);
    }

    pub fn matmul(&self, rhs: impl Borrow<Self>) -> Self
    {
        Self(&self.0 * &rhs.borrow().0)
    }

    pub fn matmul_transposed(&self, rhs: impl Borrow<Self>) -> Self
    {
        Self(self.0.tr_mul(&rhs.borrow().0))
    }

    pub fn matmul_by_transposed(&self, rhs: impl Borrow<Self>) -> Self
    {
        Self(&self.0 * rhs.borrow().0.transpose())
    }

    pub fn dot(self, rhs: &Self) -> f32
    {
        self.0.dot(&rhs.0)
    }

    pub fn sqrt(&mut self)
    {
        self.0.apply(|v| *v = v.sqrt());
    }

    pub fn clone_sqrt(&self) -> Self
    {
        let mut out = self.clone();
        out.sqrt();

        out
    }

    pub fn exp(&mut self)
    {
        self.0.apply(|v| *v = v.exp());
    }

    pub fn ln(&mut self)
    {
        self.0.apply(|v| *v = v.ln());
    }

    pub fn reciprocal(&mut self)
    {
        self.0.apply(|v| *v = 1.0 / *v);
    }

    pub fn sigmoid(&mut self)
    {
        self.0.apply(|v| *v = 1.0 / (1.0 + (-*v).exp()));
    }

    pub fn tanh(&mut self)
    {
        self.0.apply(|v| *v = v.tanh());
    }

    pub fn leaky_relu(&mut self)
    {
        self.0.apply(|v| *v = v.max(LEAKY_SLOPE * *v));
    }

    pub fn leaky_relu_d(&mut self)
    {
        self.0.apply(|v| *v = leaky_relu_d(*v));
    }

    pub fn sum(&self) -> f32
    {
        self.0.sum()
    }

    pub fn total_len(&self) -> usize
    {
        self.0.as_slice().len()
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.0.as_slice().to_vec()
    }

    pub fn iter(&self) -> impl Iterator<Item=&f32> + ExactSizeIterator
    {
        self.0.as_slice().iter()
    }

    pub fn pick_weighed(&self, temperature: f32) -> usize
    {
        SoftmaxedLayer::pick_weighed_associated(self, temperature)
    }

    pub fn highest_index(&self) -> usize
    {
        SoftmaxedLayer::highest_index(self.iter())
    }
}

