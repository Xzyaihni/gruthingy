use std::{
    ffi::c_char,
    borrow::Borrow,
    ops::{Neg, Add, Sub, Mul, Div, AddAssign, DivAssign}
};

use super::{
    DefaultJoinableWrapper,
    DefaultJoinableDeepWrapper,
    JoinableSelector,
    Softmaxable,
    Softmaxer,
    leaky_relu_d,
    LEAKY_SLOPE
};

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

macro_rules! sizes_match
{
    ($this:ident, $rhs:ident) =>
    {
        debug_assert!(
            ($this.previous_size == $rhs.previous_size) && ($this.this_size == $rhs.this_size),
            "[{}, {}] != [{}, {}]",
            $this.previous_size, $this.this_size,
            $rhs.previous_size, $rhs.this_size
        );
    }
}

macro_rules! op_impl_scalar
{
    (
        $op_trait:ident,
        $op_fn_name:ident,
        $operation:ident
    ) =>
    {
        unsafe fn $operation(lhs: &[f32], rhs: f32, output: *mut f32)
        {
            for (i, lhs) in lhs.iter().enumerate()
            {
                unsafe{ output.add(i).write(lhs.$op_fn_name(rhs)); }
            }
        }

        impl $op_trait<f32> for BlasWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: f32) -> Self::Output
            {
                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, rhs, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                BlasWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<f32> for &BlasWrapper
        {
            type Output = BlasWrapper;

            fn $op_fn_name(self, rhs: f32) -> Self::Output
            {
                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, rhs, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                BlasWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<&f32> for BlasWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: &f32) -> Self::Output
            {
                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, *rhs, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                BlasWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<&f32> for &BlasWrapper
        {
            type Output = BlasWrapper;

            fn $op_fn_name(self, rhs: &f32) -> Self::Output
            {
                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, *rhs, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                BlasWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }
    }
}

macro_rules! op_impl
{
    (
        $op_trait:ident,
        $op_fn_name:ident,
        $operation:ident
    ) =>
    {
        unsafe fn $operation(lhs: &[f32], rhs: &[f32], output: *mut f32)
        {
            // i feel like all of this is useless but wutever who cares
            for i in 0..lhs.len()
            {
                let lhs = unsafe{ lhs.get_unchecked(i) };
                let rhs = unsafe{ rhs.get_unchecked(i) };

                unsafe{ output.add(i).write(lhs.$op_fn_name(rhs)); }
            }
        }

        impl $op_trait for BlasWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: Self) -> Self::Output
            {
                sizes_match!(self, rhs);

                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, &rhs.data, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                BlasWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<BlasWrapper> for &BlasWrapper
        {
            type Output = BlasWrapper;

            fn $op_fn_name(self, rhs: BlasWrapper) -> Self::Output
            {
                sizes_match!(self, rhs);

                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, &rhs.data, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                BlasWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<&BlasWrapper> for BlasWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: &Self) -> Self::Output
            {
                sizes_match!(self, rhs);

                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, &rhs.data, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                BlasWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<&BlasWrapper> for &BlasWrapper
        {
            type Output = BlasWrapper;

            fn $op_fn_name(self, rhs: &BlasWrapper) -> Self::Output
            {
                sizes_match!(self, rhs);

                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, &rhs.data, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                BlasWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }
    }
}

op_impl_scalar!{Add, add, add_scalar}
op_impl_scalar!{Sub, sub, sub_scalar}
op_impl_scalar!{Mul, mul, mul_scalar}
op_impl_scalar!{Div, div, div_scalar}

op_impl!{Add, add, add_tensor}
op_impl!{Sub, sub, sub_tensor}
op_impl!{Mul, mul, mul_tensor}
op_impl!{Div, div, div_tensor}

impl AddAssign for BlasWrapper
{
    fn add_assign(&mut self, rhs: Self)
    {
        self.data.iter_mut().zip(rhs.data.iter()).for_each(|(v, rhs)| *v += rhs);
    }
}

impl DivAssign<f32> for BlasWrapper
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.data.iter_mut().for_each(|v| *v /= rhs);
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

    fn neg(mut self) -> Self::Output
    {
        self.data.iter_mut().for_each(|v| *v = -*v);

        self
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

    pub fn matmulv(&self, rhs: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();

        debug_assert!(
            self.this_size == rhs.previous_size,
            "{} != {}",
            self.this_size, rhs.previous_size
        );

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
            );
        }

        unsafe{ output.set_len(output_len); }

        Self{
            data: output,
            previous_size: self.previous_size,
            this_size: 1
        }
    }

    pub fn matmulv_transposed(&self, rhs: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();

        debug_assert!(
            self.previous_size == rhs.previous_size,
            "{} != {}",
            self.previous_size, rhs.previous_size
        );

        let output_len = self.this_size as usize;
        let mut output = Vec::with_capacity(output_len);

        unsafe{
            blas_sys::sgemv_(
                &(b'T' as c_char),
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
            );
        }

        unsafe{ output.set_len(output_len); }

        Self{
            data: output,
            previous_size: self.this_size,
            this_size: 1
        }
    }

    pub fn outer_product(&self, rhs: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();

        let mut output = vec![0.0; (rhs.previous_size * self.previous_size) as usize];

        unsafe{
            blas_sys::sger_(
                &self.previous_size,
                &rhs.previous_size,
                &1.0,
                self.data.as_ptr(),
                &1,
                rhs.data.as_ptr(),
                &1,
                output.as_mut_ptr(),
                &self.previous_size
            );
        }

        Self{
            data: output,
            previous_size: self.previous_size,
            this_size: rhs.previous_size
        }
    }

    pub fn matmulv_add(&self, rhs: impl Borrow<Self>, added: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();
        let mut added = added.borrow().clone();

        debug_assert!(
            self.this_size == rhs.previous_size,
            "{} != {}",
            self.this_size, rhs.previous_size
        );

        debug_assert!(
            added.previous_size == self.previous_size,
            "{} != {}",
            added.previous_size, self.previous_size
        );

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
                &1.0,
                added.data.as_mut_ptr(),
                &1
            );
        }

        added
    }

    pub fn max(&mut self, rhs: &Self)
    {
        self.data.iter_mut().zip(rhs.data.iter()).for_each(|(v, rhs)| *v = v.max(*rhs));
    }

    pub fn dot(self, rhs: &Self) -> f32
    {
        debug_assert!(
            self.data.len() == rhs.data.len(),
            "{} != {}",
            self.data.len(), rhs.data.len()
        );

        unsafe{
            blas_sys::sdot_(
                &(self.data.len() as i32),
                self.data.as_ptr(),
                &1,
                rhs.data.as_ptr(),
                &1,
            )
        }
    }

    pub fn sqrt(&mut self)
    {
        self.data.iter_mut().for_each(|v| *v = v.sqrt());
    }

    pub fn clone_sqrt(&self) -> Self
    {
        let mut this = self.clone();

        this.sqrt();

        this
    }

    pub fn exp(&mut self)
    {
        self.data.iter_mut().for_each(|v| *v = v.exp());
    }

    pub fn ln(&mut self)
    {
        self.data.iter_mut().for_each(|v| *v = v.ln());
    }

    pub fn reciprocal(&mut self)
    {
        self.data.iter_mut().for_each(|v| *v = v.recip());
    }

    pub fn sigmoid(&mut self)
    {
        self.data.iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp()));
    }

    pub fn tanh(&mut self)
    {
        self.data.iter_mut().for_each(|v| *v = v.tanh());
    }

    pub fn leaky_relu(&mut self)
    {
        self.data.iter_mut().for_each(|v| *v = v.max(LEAKY_SLOPE * *v));
    }

    pub fn leaky_relu_d(&mut self)
    {
        self.data.iter_mut().for_each(|v| *v = leaky_relu_d(*v));
    }

    pub fn sum(&self) -> f32
    {
        self.data.iter().sum()
    }

    pub fn signum(&self) -> Self
    {
        let mut this = self.clone();

        this.data.iter_mut().for_each(|v| *v = v.signum());

        this
    }

    pub fn cap_magnitude(&self, cap: f32) -> Self
    {
        let magnitude = unsafe{
            blas_sys::snrm2_(
                &(self.data.len() as i32),
                self.data.as_ptr(),
                &1
            )
        };

        if magnitude > cap
        {
            let scale = cap / magnitude;

            self * scale
        } else
        {
            self.clone()
        }
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
        Softmaxer::pick_weighed_inner(self.iter())
    }

    pub fn highest_index(&self) -> usize
    {
        Softmaxer::highest_index(self.iter())
    }

    pub const fn is_arrayfire() -> bool
    {
        false
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    fn tensors_equal(a: BlasWrapper, b: BlasWrapper)
    {
        for (a, b) in a.data.into_iter().zip(b.data.into_iter())
        {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn add_scalar_works()
    {
        let (previous, this) = (10, 15);

        let a = BlasWrapper::new_with(previous, this, || fastrand::f32());
        let b = fastrand::f32();

        let manual_added = BlasWrapper{
            data: a.data.iter().map(|a| a + b).collect(),
            previous_size: previous as i32,
            this_size: this as i32
        };

        tensors_equal(a + b, manual_added);
    }

    #[test]
    fn add_works()
    {
        let (previous, this) = (10, 15);

        let a = BlasWrapper::new_with(previous, this, || fastrand::f32());
        let b = BlasWrapper::new_with(previous, this, || fastrand::f32());

        let manual_added = BlasWrapper{
            data: a.data.iter().zip(b.data.iter()).map(|(a, b)| a + b).collect(),
            previous_size: previous as i32,
            this_size: this as i32
        };

        tensors_equal(a + b, manual_added);
    }
}
