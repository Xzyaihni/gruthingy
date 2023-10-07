use std::{
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


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NyanWrapper
{
    data: Vec<f32>,
    previous_size: isize,
    this_size: isize
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

        impl $op_trait<f32> for NyanWrapper
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

                NyanWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<f32> for &NyanWrapper
        {
            type Output = NyanWrapper;

            fn $op_fn_name(self, rhs: f32) -> Self::Output
            {
                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, rhs, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                NyanWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<&f32> for NyanWrapper
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

                NyanWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<&f32> for &NyanWrapper
        {
            type Output = NyanWrapper;

            fn $op_fn_name(self, rhs: &f32) -> Self::Output
            {
                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, *rhs, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                NyanWrapper{
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

        impl $op_trait for NyanWrapper
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

                NyanWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<NyanWrapper> for &NyanWrapper
        {
            type Output = NyanWrapper;

            fn $op_fn_name(self, rhs: NyanWrapper) -> Self::Output
            {
                sizes_match!(self, rhs);

                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, &rhs.data, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                NyanWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<&NyanWrapper> for NyanWrapper
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

                NyanWrapper{
                    data: output,
                    previous_size: self.previous_size,
                    this_size: self.this_size
                }
            }
        }

        impl $op_trait<&NyanWrapper> for &NyanWrapper
        {
            type Output = NyanWrapper;

            fn $op_fn_name(self, rhs: &NyanWrapper) -> Self::Output
            {
                sizes_match!(self, rhs);

                let output_len = self.data.len();
                let mut output = Vec::with_capacity(output_len);

                unsafe{
                    $operation(&self.data, &rhs.data, output.as_mut_ptr());
                }

                unsafe{ output.set_len(output_len); }

                NyanWrapper{
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

impl AddAssign for NyanWrapper
{
    fn add_assign(&mut self, rhs: Self)
    {
        self.data.iter_mut().zip(rhs.data.iter()).for_each(|(v, rhs)| *v += rhs);
    }
}

impl DivAssign<f32> for NyanWrapper
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.data.iter_mut().for_each(|v| *v /= rhs);
    }
}

impl Neg for &NyanWrapper
{
    type Output = NyanWrapper;

    fn neg(self) -> Self::Output
    {
        -self.clone()
    }
}

impl Neg for NyanWrapper
{
    type Output = Self;

    fn neg(mut self) -> Self::Output
    {
        self.data.iter_mut().for_each(|v| *v = -*v);

        self
    }
}

impl JoinableSelector<(NyanWrapper, NyanWrapper)> for NyanWrapper
{
    type This = DefaultJoinableWrapper<NyanWrapper>;
    type Deep = DefaultJoinableDeepWrapper<NyanWrapper>;
}

impl Softmaxable for NyanWrapper
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
impl NyanWrapper
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            data: vec![0.0; previous_size * this_size],
            previous_size: previous_size as isize,
            this_size: this_size as isize
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
            previous_size: previous_size as isize,
            this_size: this_size as isize
        }
    }

    pub fn from_raw<V: Into<Vec<f32>>>(values: V, previous_size: usize, this_size: usize) -> Self
    {
        Self{
            data: values.into(),
            previous_size: previous_size as isize,
            this_size: this_size as isize
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

        let total_len = self.previous_size as usize;
        let mut data = Vec::with_capacity(total_len);

        let lhs_ptr: *const f32 = self.data.as_ptr();
        let rhs_ptr: *const f32 = rhs.data.as_ptr();

        let data_ptr: *mut f32 = data.as_mut_ptr();
        for y in 0..self.previous_size
        {
            let mut s = 0.0;
            for x in 0..self.this_size
            {
                unsafe{
                    s += *(lhs_ptr.offset(x * self.previous_size + y))
                        * *(rhs_ptr.offset(x));
                }
            }

            unsafe{
                data_ptr.offset(y).write(s);
            }
        }

        unsafe{ data.set_len(total_len) }

        Self{
            data,
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

        let total_len = self.this_size as usize;
        let mut data = Vec::with_capacity(total_len);

        let lhs_ptr: *const f32 = self.data.as_ptr();
        let rhs_ptr: *const f32 = rhs.data.as_ptr();

        let data_ptr: *mut f32 = data.as_mut_ptr();
        for y in 0..self.this_size
        {
            let y_column = y * self.previous_size;

            let mut s = 0.0;
            for x in 0..self.previous_size
            {
                unsafe{
                    s += *(lhs_ptr.offset(x + y_column))
                        * *(rhs_ptr.offset(x));
                }
            }

            unsafe{
                data_ptr.offset(y).write(s);
            }
        }

        unsafe{ data.set_len(total_len) }

        Self{
            data,
            previous_size: self.this_size,
            this_size: 1
        }
    }

    pub fn outer_product(&self, rhs: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();

        let total_len = (self.previous_size * rhs.previous_size) as usize;
        let mut data = Vec::with_capacity(total_len);
        
        let mut i = 0;

        let lhs_ptr: *const f32 = self.data.as_ptr();
        let rhs_ptr: *const f32 = rhs.data.as_ptr();

        let data_ptr: *mut f32 = data.as_mut_ptr();
        for x in 0..rhs.previous_size
        {
            for y in 0..self.previous_size
            {
                unsafe{
                    let value: f32 = *(lhs_ptr.offset(y)) * *(rhs_ptr.offset(x));

                    data_ptr.offset(i).write(value);
                }

                i += 1;
            }
        }

        unsafe{ data.set_len(total_len) }

        Self{
            data,
            previous_size: self.previous_size,
            this_size: rhs.previous_size
        }
    }

    pub fn matmulv_add(&self, rhs: impl Borrow<Self>, added: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();
        let added = added.borrow();

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

        let total_len = self.previous_size as usize;
        let mut data = Vec::with_capacity(total_len);

        let added_ptr: *const f32 = added.data.as_ptr();
        
        let lhs_ptr: *const f32 = self.data.as_ptr();
        let rhs_ptr: *const f32 = rhs.data.as_ptr();

        let data_ptr: *mut f32 = data.as_mut_ptr();
        for y in 0..self.previous_size
        {
            let mut s = 0.0;
            for x in 0..self.this_size
            {
                unsafe{
                    s += *(lhs_ptr.offset(x * self.previous_size + y))
                        * *(rhs_ptr.offset(x));
                }
            }

            unsafe{
                s += *(added_ptr.offset(y));

                data_ptr.offset(y).write(s);
            }
        }

        unsafe{ data.set_len(total_len) }

        Self{
            data,
            previous_size: self.previous_size,
            this_size: 1
        }
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

        self.data.into_iter().zip(rhs.data.iter()).map(|(a, b)| a * b).sum()
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

    fn magnitude(&self) -> f32
    {
        let s: f32 = self.data.iter().map(|x| x * x).sum();

        s.sqrt()
    }

    pub fn cap_magnitude(&self, cap: f32) -> Self
    {
        let magnitude = self.magnitude();

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

    #[test]
    fn matmulv_works()
    {
        let a = vec![
            7.0, 2.0, 9.0,
            8.0, 4.0, 6.0
        ];

        let b = vec![5.0, 4.0];

        let correct = vec![67.0, 26.0, 69.0];

        let a = NyanWrapper::from_raw(a, 3, 2);
        let b = NyanWrapper::from_raw(b, 2, 1);

        let r = a.matmulv(b);

        r.data.into_iter().zip(correct.into_iter()).for_each(|(nyan, correct)|
        {
            assert_eq!(correct, nyan)
        });
    }
}
