use std::{
    ffi::c_char,
    borrow::Borrow,
    ops::{Neg, Add, Sub, Mul, Div, AddAssign, DivAssign}
};

use super::{DefaultJoinableWrapper, DefaultJoinableDeepWrapper, JoinableSelector, Softmaxable};

use serde::{Serialize, Deserialize};

extern crate blas_sys;
extern crate openblas_src;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlasWrapper
{
    data: Vec<f32>,
    previous_size: i32,
    this_size: i32
}

macro_rules! op_impl_scalar
{
    (
        $op_trait:ident,
        $op_fn_name:ident,
        $operation:expr
    ) =>
    {
        impl $op_trait<f32> for BlasWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: f32) -> Self::Output
            {
                todo!();
            }
        }

        impl $op_trait<f32> for &BlasWrapper
        {
            type Output = BlasWrapper;

            fn $op_fn_name(self, rhs: f32) -> Self::Output
            {
                todo!();
            }
        }

        impl $op_trait<&f32> for BlasWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: &f32) -> Self::Output
            {
                todo!();
            }
        }

        impl $op_trait<&f32> for &BlasWrapper
        {
            type Output = BlasWrapper;

            fn $op_fn_name(self, rhs: &f32) -> Self::Output
            {
                todo!();
            }
        }
    }
}

macro_rules! op_impl
{
    (
        $op_trait:ident,
        $op_fn_name:ident,
        $operation:expr
    ) =>
    {
        impl $op_trait for BlasWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: Self) -> Self::Output
            {
                todo!();
            }
        }

        impl $op_trait<BlasWrapper> for &BlasWrapper
        {
            type Output = BlasWrapper;

            fn $op_fn_name(self, rhs: BlasWrapper) -> Self::Output
            {
                todo!();
            }
        }

        impl $op_trait<&BlasWrapper> for BlasWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: &Self) -> Self::Output
            {
                todo!();
            }
        }

        impl $op_trait<&BlasWrapper> for &BlasWrapper
        {
            type Output = BlasWrapper;

            fn $op_fn_name(self, rhs: &BlasWrapper) -> Self::Output
            {
                todo!();
            }
        }
    }
}

op_impl_scalar!{Add, add,
    {}
}

op_impl_scalar!{Sub, sub,
    {}
}

op_impl_scalar!{Mul, mul,
    {}
}

op_impl_scalar!{Div, div,
    {}
}

op_impl!{Add, add,
    {}
}

op_impl!{Sub, sub,
    {}
}

op_impl!{Div, div,
    {}
}

op_impl!{Mul, mul,
    {}
}

impl AddAssign for BlasWrapper
{
    fn add_assign(&mut self, rhs: Self)
    {
        todo!();
    }
}

impl DivAssign<f32> for BlasWrapper
{
    fn div_assign(&mut self, rhs: f32)
    {
        todo!();
    }
}

impl Neg for &BlasWrapper
{
    type Output = BlasWrapper;

    fn neg(self) -> Self::Output
    {
        -self.clone()
    }
}

impl Neg for BlasWrapper
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        todo!();
    }
}

impl JoinableSelector<(BlasWrapper, BlasWrapper)> for BlasWrapper
{
    type This = DefaultJoinableWrapper<BlasWrapper>;
    type Deep = DefaultJoinableDeepWrapper<BlasWrapper>;
}

impl Softmaxable for BlasWrapper
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

#[allow(dead_code)]
impl BlasWrapper
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            data: vec![0.0; previous_size * this_size],
            previous_size: previous_size as i32,
            this_size: this_size as i32
        }
    }

    pub fn new_with<F: FnMut() -> f32>(
        previous_size: usize,
        this_size: usize,
        mut f: F
    )-> Self
    {
        Self{
            data: (0..(previous_size * this_size)).map(|_| f()).collect(),
            previous_size: previous_size as i32,
            this_size: this_size as i32
        }
    }

    pub fn from_raw<V: Into<Vec<f32>>>(values: V, previous_size: usize, this_size: usize) -> Self
    {
        Self{
            data: values.into(),
            previous_size: previous_size as i32,
            this_size: this_size as i32
        }
    }

    pub fn swap_raw_values<V: Into<Vec<f32>>>(&mut self, values: V)
    {
        self.data = values.into();
    }

    pub fn fill(&mut self, value: f32)
    {
        self.data.iter_mut().for_each(|v| *v = value);
    }

    pub fn softmax_cross_entropy(mut self, targets: &Self) -> (Self, f32)
    {
        todo!();
    }

    pub fn matmulv(&self, rhs: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();

        let output_len = self.previous_size as usize;
        let mut output = Vec::with_capacity(output_len);

        unsafe{
            blas_sys::sgemv_(
                &(b'N' as c_char),
                &self.previous_size,
                &self.this_size,
                &1.0,
                self.data.as_ptr(),
                &self.previous_size,
                rhs.data.as_ptr(),
                &1,
                &0.0,
                output.as_mut_ptr(),
                &1
            )
        }

        unsafe{ output.set_len(output_len); }

        Self{
            data: output,
            previous_size: self.previous_size,
            this_size: self.this_size
        }
    }

    pub fn matmulv_transposed(&self, rhs: impl Borrow<Self>) -> Self
    {
        todo!();
    }

    pub fn outer_product(&self, rhs: impl Borrow<Self>) -> Self
    {
        todo!();
    }

    pub fn matmulv_add(&self, rhs: impl Borrow<Self>, added: impl Borrow<Self>) -> Self
    {
        todo!();
    }

    pub fn max(&mut self, rhs: &Self)
    {
        todo!();
    }

    pub fn dot(self, rhs: &Self) -> f32
    {
        todo!();
    }

    pub fn sqrt(&mut self)
    {
        todo!();
    }

    pub fn clone_sqrt(&self) -> Self
    {
        todo!();
    }

    pub fn exp(&mut self)
    {
        todo!();
    }

    pub fn ln(&mut self)
    {
        todo!();
    }

    pub fn reciprocal(&mut self)
    {
        todo!();
    }

    pub fn sigmoid(&mut self)
    {
        todo!();
    }

    pub fn tanh(&mut self)
    {
        todo!();
    }

    pub fn leaky_relu(&mut self)
    {
        todo!();
    }

    pub fn leaky_relu_d(&mut self)
    {
        todo!();
    }

    pub fn sum(&self) -> f32
    {
        todo!();
    }

    pub fn signum(&self) -> Self
    {
        todo!();
    }

    pub fn cap_magnitude(&self, cap: f32) -> Self
    {
        todo!();
    }

    pub fn total_len(&self) -> usize
    {
        self.data.len()
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.data.to_vec()
    }

    pub fn iter(&self) -> impl Iterator<Item=&f32> + ExactSizeIterator
    {
        self.data.iter()
    }

    pub fn pick_weighed(&self) -> usize
    {
        todo!();
    }

    pub fn highest_index(&self) -> usize
    {
        todo!();
    }

    pub const fn is_arrayfire() -> bool
    {
        false
    }
}
