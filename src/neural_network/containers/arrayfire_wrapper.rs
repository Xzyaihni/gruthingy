use std::{
    fmt::{self, Debug},
    borrow::Borrow,
    ops::{Mul, Add, Sub, Div, AddAssign, SubAssign, DivAssign, Neg}
};

use serde::{Serialize, Deserialize};

use arrayfire::{dim4, Seq, MatProp, NormType, Array};

use super::{Softmaxer, Softmaxable, Joinable, JoinableSelector, LEAKY_SLOPE};

#[derive(Clone, Serialize, Deserialize)]
pub struct ArrayfireWrapper(Array<f32>);

macro_rules! op_impl_scalar
{
    (
        $op_trait:ident,
        $op_fn_name:ident
    ) =>
    {
        impl $op_trait<f32> for ArrayfireWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: f32) -> Self::Output
            {
                Self(self.0.$op_fn_name(rhs))
            }
        }

        impl $op_trait<f32> for &ArrayfireWrapper
        {
            type Output = ArrayfireWrapper;

            fn $op_fn_name(self, rhs: f32) -> Self::Output
            {
                ArrayfireWrapper((&self.0).$op_fn_name(rhs))
            }
        }

        impl $op_trait<&f32> for ArrayfireWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: &f32) -> Self::Output
            {
                Self(self.0.$op_fn_name(*rhs))
            }
        }

        impl $op_trait<&f32> for &ArrayfireWrapper
        {
            type Output = ArrayfireWrapper;

            fn $op_fn_name(self, rhs: &f32) -> Self::Output
            {
                ArrayfireWrapper((&self.0).$op_fn_name(*rhs))
            }
        }
    }
}

macro_rules! op_impl
{
    (
        $op_trait:ident,
        $op_fn_name:ident
    ) =>
    {
        impl $op_trait for ArrayfireWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: Self) -> Self::Output
            {
                Self(self.0.$op_fn_name(rhs.0))
            }
        }

        impl $op_trait<ArrayfireWrapper> for &ArrayfireWrapper
        {
            type Output = ArrayfireWrapper;

            fn $op_fn_name(self, rhs: ArrayfireWrapper) -> Self::Output
            {
                ArrayfireWrapper((&self.0).$op_fn_name(rhs.0))
            }
        }

        impl $op_trait<&ArrayfireWrapper> for ArrayfireWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: &Self) -> Self::Output
            {
                Self(self.0.$op_fn_name(&rhs.0))
            }
        }

        impl $op_trait<&ArrayfireWrapper> for &ArrayfireWrapper
        {
            type Output = ArrayfireWrapper;

            fn $op_fn_name(self, rhs: &ArrayfireWrapper) -> Self::Output
            {
                ArrayfireWrapper((&self.0).$op_fn_name(&rhs.0))
            }
        }
    }
}

macro_rules! op_assign_impl
{
    (
        $op_trait:ident,
        $op_fn_name:ident
    ) =>
    {
        impl $op_trait for ArrayfireWrapper
        {
            fn $op_fn_name(&mut self, rhs: Self)
            {
                self.0.$op_fn_name(rhs.0);
            }
        }

        impl $op_trait<&ArrayfireWrapper> for ArrayfireWrapper
        {
            fn $op_fn_name(&mut self, rhs: &Self)
            {
                self.0.$op_fn_name(rhs.0.clone());
            }
        }
    }
}

op_impl_scalar!{Add, add}
op_impl_scalar!{Sub, sub}
op_impl_scalar!{Mul, mul}
op_impl_scalar!{Div, div}

op_impl!{Add, add}
op_impl!{Sub, sub}
op_impl!{Mul, mul}
op_impl!{Div, div}

op_assign_impl!{SubAssign, sub_assign}
op_assign_impl!{AddAssign, add_assign}

impl DivAssign<f32> for ArrayfireWrapper
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.0 = &self.0 / rhs;
    }
}

impl Neg for ArrayfireWrapper
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        Self(-self.0)
    }
}

impl Neg for &ArrayfireWrapper
{
    type Output = ArrayfireWrapper;

    fn neg(self) -> Self::Output
    {
        ArrayfireWrapper(-self.0.clone())
    }
}

impl Softmaxable for ArrayfireWrapper
{
    fn exp(&mut self)
    {
        self.exp();
    }

    fn sum(&self) -> f32
    {
        self.sum()
    }
}

pub struct JoinableWrapper
{
    data: ArrayfireWrapper,
    len: i64,
    index: i64
}

impl FromIterator<(ArrayfireWrapper, ArrayfireWrapper)> for JoinableWrapper
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item=(ArrayfireWrapper, ArrayfireWrapper)>
    {
        let mut len = 0;
        let mut data: Option<ArrayfireWrapper> = None;

        for (input, output) in iter
        {
            let this = arrayfire::join(1, &input.0, &output.0);

            if let Some(data) = data.as_mut()
            {
                data.0 = arrayfire::join(2, &data.0, &this);
            } else
            {
                data = Some(ArrayfireWrapper(this));
            }

            len += 1;
        }

        Self{
            data: data.expect("cant have an empty joinable"),
            len,
            index: 0
        }
    }
}

pub struct JoinableDeepWrapper
{
    data: ArrayfireWrapper,
    len: i64,
    index: i64
}

impl FromIterator<JoinableWrapper> for JoinableDeepWrapper
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item=JoinableWrapper>
    {
        let mut len = 0;
        let mut data: Option<ArrayfireWrapper> = None;

        for input in iter
        {
            if let Some(data) = data.as_mut()
            {
                data.0 = arrayfire::join(3, &data.0, &input.data.0);
            } else
            {
                data = Some(ArrayfireWrapper(input.data.0));
            }

            len += 1;
        }

        Self{
            data: data.expect("cant have an empty joinable"),
            len,
            index: 0
        }
    }
}

impl JoinableSelector<(ArrayfireWrapper, ArrayfireWrapper)> for ArrayfireWrapper
{
    type This = JoinableWrapper;
    type Deep = JoinableDeepWrapper;
}

impl Joinable<(ArrayfireWrapper, ArrayfireWrapper)> for JoinableWrapper {}

impl Iterator for JoinableWrapper
{
    type Item = (ArrayfireWrapper, ArrayfireWrapper);

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.index == self.len
        {
            None
        } else
        {
            let index_f64 = self.index as f64;
            let seqs = [
                Seq::default(),
                Seq::default(),
                Seq::new(index_f64, index_f64, 1.0)
            ];

            let a_output = arrayfire::index(&self.data.0, &seqs);

            let inputs = arrayfire::col(&a_output, 0);
            let outputs = arrayfire::col(&a_output, 1);

            self.index += 1;

            Some((ArrayfireWrapper(inputs), ArrayfireWrapper(outputs)))
        }
    }
}

impl Joinable<JoinableWrapper> for JoinableDeepWrapper {}

impl Iterator for JoinableDeepWrapper
{
    type Item = JoinableWrapper;

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.index == self.len
        {
            None
        } else
        {
            let index_f64 = self.index as f64;
            let seqs = [
                Seq::default(),
                Seq::default(),
                Seq::default(),
                Seq::new(index_f64, index_f64, 1.0)
            ];

            let a_output = arrayfire::index(&self.data.0, &seqs);

            self.index += 1;

            Some(JoinableWrapper{data: ArrayfireWrapper(a_output), len: 0, index: 0})
        }
    }
}

#[allow(dead_code)]
impl ArrayfireWrapper
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self(arrayfire::constant(0.0, dim4!(previous_size as u64, this_size as u64)))
    }

    pub fn new_with<F: FnMut() -> f32>(
        previous_size: usize,
        this_size: usize,
        mut f: F
    )-> Self
    {
        let values = (0..(previous_size * this_size)).map(|_| f()).collect::<Vec<_>>();
        Self::from_raw(values, previous_size, this_size)
    }

    pub fn from_raw<V: Into<Vec<f32>>>(values: V, previous_size: usize, this_size: usize) -> Self
    {
        let values = values.into();
        Self(Array::new(&values, dim4!(previous_size as u64, this_size as u64)))
    }

    pub fn swap_raw_values<V: Into<Vec<f32>>>(&mut self, values: V)
    {
        let values = values.into();
        self.0 = Array::new(&values, self.0.dims());
    }

    pub fn fill(&mut self, value: f32)
    {
        self.0 = arrayfire::constant(value, self.0.dims());
    }

    pub fn softmax_cross_entropy(mut self, targets: &Self) -> (Self, f32)
    {
        Softmaxer::softmax(&mut self);
        let softmaxed = self.clone();

        // assumes that targets r either 0 or 1
        self.ln();

        let s = self.dot(targets);

        (softmaxed, -s)
    }

    pub fn matmulv(&self, rhs: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();
        Self(arrayfire::matmul(&self.0, &rhs.0, MatProp::NONE, MatProp::NONE))
    }

    pub fn matmulv_transposed(&self, rhs: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();
        Self(arrayfire::matmul(&self.0, &rhs.0, MatProp::TRANS, MatProp::NONE))
    }

    pub fn outer_product(&self, rhs: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();
        Self(arrayfire::matmul(&self.0, &rhs.0, MatProp::NONE, MatProp::TRANS))
    }

    pub fn matmulv_add(&self, rhs: impl Borrow<Self>, added: impl Borrow<Self>) -> Self
    {
        // arrayfire doesnt let me give it the C matrix in gemm >_<
        
        self.matmulv(rhs) + added.borrow()
    }

    pub fn max(&mut self, rhs: &Self)
    {
        self.0 = arrayfire::maxof(&self.0, &rhs.0, false);
    }

    pub fn dot(self, rhs: &Self) -> f32
    {
        let out = arrayfire::dot(&self.0, &rhs.0, MatProp::NONE, MatProp::NONE);

        Self::as_vec_assoc(&out)[0]
    }

    pub fn sqrt(&mut self)
    {
        self.0 = arrayfire::sqrt(&self.0);
    }

    pub fn clone_sqrt(&self) -> Self
    {
        Self(arrayfire::sqrt(&self.0))
    }

    pub fn exp(&mut self)
    {
        self.0 = arrayfire::exp(&self.0);
    }

    pub fn ln(&mut self)
    {
        self.0 = arrayfire::log(&self.0);
    }

    pub fn reciprocal(&mut self)
    {
        self.0 = arrayfire::div(&1.0_f32, &self.0, true);
    }

    pub fn sigmoid(&mut self)
    {
        self.0 = arrayfire::sigmoid(&self.0);
    }

    pub fn tanh(&mut self)
    {
        self.0 = arrayfire::tanh(&self.0);
    }

    pub fn leaky_relu(&mut self)
    {
        let sloped = &self.0 * LEAKY_SLOPE;
        let sloped = Self(sloped);

        self.max(&sloped);
    }

    pub fn leaky_relu_d(&mut self)
    {
        let gz = arrayfire::gt(&self.0, &0.0_f32, true);

        let ones = arrayfire::constant(1.0, self.0.dims());
        let slopes = arrayfire::constant(LEAKY_SLOPE, self.0.dims());

        self.0 = arrayfire::select(&ones, &gz, &slopes);
    }

    pub fn sum(&self) -> f32
    {
        arrayfire::sum_all(&self.0).0
    }

    pub fn signum(&self) -> Self
    {
        Self(arrayfire::sign(&self.0))
    }

    pub fn cap_magnitude(&self, cap: f32) -> Self
    {
        let magnitude = arrayfire::norm(&self.0, NormType::VECTOR_2, 0.0, 0.0) as f32;

        let s = cap / magnitude;
        
        Self(&self.0 * s)
    }

    pub fn total_len(&self) -> usize
    {
        self.0.elements()
    }

    fn as_vec_assoc(a: &Array<f32>) -> Vec<f32>
    {
        let mut out = vec![0.0_f32; a.elements()];
        a.host(&mut out);

        out
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        Self::as_vec_assoc(&self.0)
    }

    pub fn iter(&self) -> impl Iterator<Item=f32> + ExactSizeIterator
    {
        self.as_vec().into_iter()
    }

    pub fn pick_weighed(&self) -> usize
    {
        Softmaxer::pick_weighed_inner(self.iter())
    }

    pub fn highest_index(&self) -> usize
    {
        arrayfire::imax_all(&self.0).2 as usize
    }

    pub const fn is_arrayfire() -> bool
    {
        true
    }
}

impl Debug for ArrayfireWrapper
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        f.debug_struct("ArrayfireWrapper")
            .field("array", &self.as_vec())
            .finish()
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    use std::iter;

    #[test]
    fn joinable_works()
    {
        let a = ArrayfireWrapper::new(5, 1);
        let b = ArrayfireWrapper::new(5, 1);

        let j = iter::once((a, b)).collect::<JoinableWrapper>();

        assert_eq!(j.data.0.dims(), dim4!{5, 2, 1, 1});
    }
}
