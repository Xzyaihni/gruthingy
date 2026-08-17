use std::{
    f32,
    fmt::{self, Debug},
    borrow::Borrow,
    ops::{Mul, Add, Sub, Div, AddAssign, SubAssign, DivAssign, Neg}
};

use serde::{Serialize, Deserialize};

use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut};

use super::{
    Softmaxer,
    Softmaxable,
    OneHotLayer,
    TensorRawDataPointer,
    TensorIndexRaw,
    LEAKY_SLOPE,
    leaky_relu_d
};

#[derive(Clone, Serialize, Deserialize)]
pub struct MatrixWrapper(DMatrix<f32>);

impl Debug for MatrixWrapper
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        write!(f, "{{rows: {}, columns: {}, data: {:?}}}", self.0.nrows(), self.0.ncols(), self.0.data.as_vec())
    }
}

macro_rules! op_impl_scalar
{
    (
        $op_trait:ident,
        $op_fn_name:ident,
        $op_real_fn:ident
    ) =>
    {
        impl $op_trait<f32> for MatrixWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: f32) -> Self::Output
            {
                Self(self.0.$op_real_fn(rhs))
            }
        }

        impl $op_trait<f32> for &MatrixWrapper
        {
            type Output = MatrixWrapper;

            fn $op_fn_name(self, rhs: f32) -> Self::Output
            {
                MatrixWrapper((&self.0).$op_real_fn(rhs))
            }
        }

        impl $op_trait<&f32> for MatrixWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: &f32) -> Self::Output
            {
                Self(self.0.$op_real_fn(*rhs))
            }
        }

        impl $op_trait<&f32> for &MatrixWrapper
        {
            type Output = MatrixWrapper;

            fn $op_fn_name(self, rhs: &f32) -> Self::Output
            {
                MatrixWrapper((&self.0).$op_real_fn(*rhs))
            }
        }
    }
}

macro_rules! op_impl
{
    (
        $op_trait:ident,
        $op_fn_name:ident,
        $op_real_fn:ident
    ) =>
    {
        op_impl_owned!{$op_trait, $op_fn_name, $op_real_fn}
        op_impl_borrowed!{$op_trait, $op_fn_name, $op_real_fn}
    }
}

macro_rules! op_impl_rhs_ref
{
    (
        $op_trait:ident,
        $op_fn_name:ident,
        $op_real_fn:ident
    ) =>
    {
        op_impl_owned_ref!{$op_trait, $op_fn_name, $op_real_fn}
        op_impl_borrowed!{$op_trait, $op_fn_name, $op_real_fn}
    }
}

macro_rules! op_impl_owned_ref
{
    (
        $op_trait:ident,
        $op_fn_name:ident,
        $op_real_fn:ident
    ) =>
    {
        impl $op_trait for MatrixWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: Self) -> Self::Output
            {
                Self(self.0.$op_real_fn(&rhs.0))
            }
        }

        impl $op_trait<MatrixWrapper> for &MatrixWrapper
        {
            type Output = MatrixWrapper;

            fn $op_fn_name(self, rhs: MatrixWrapper) -> Self::Output
            {
                MatrixWrapper((&self.0).$op_real_fn(&rhs.0))
            }
        }

    }
}

macro_rules! op_impl_owned
{
    (
        $op_trait:ident,
        $op_fn_name:ident,
        $op_real_fn:ident
    ) =>
    {
        impl $op_trait for MatrixWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: Self) -> Self::Output
            {
                Self(self.0.$op_real_fn(rhs.0))
            }
        }

        impl $op_trait<MatrixWrapper> for &MatrixWrapper
        {
            type Output = MatrixWrapper;

            fn $op_fn_name(self, rhs: MatrixWrapper) -> Self::Output
            {
                MatrixWrapper((&self.0).$op_real_fn(rhs.0))
            }
        }

    }
}

macro_rules! op_impl_borrowed
{
    (
        $op_trait:ident,
        $op_fn_name:ident,
        $op_real_fn:ident
    ) =>
    {
        impl $op_trait<&MatrixWrapper> for MatrixWrapper
        {
            type Output = Self;

            fn $op_fn_name(self, rhs: &Self) -> Self::Output
            {
                Self(self.0.$op_real_fn(&rhs.0))
            }
        }

        impl $op_trait<&MatrixWrapper> for &MatrixWrapper
        {
            type Output = MatrixWrapper;

            fn $op_fn_name(self, rhs: &MatrixWrapper) -> Self::Output
            {
                MatrixWrapper((&self.0).$op_real_fn(&rhs.0))
            }
        }
    }
}

op_impl_scalar!{Add, add, add_scalar}
op_impl_scalar!{Mul, mul, mul}
op_impl_scalar!{Div, div, div}

op_impl!{Add, add, add}
op_impl!{Sub, sub, sub}
op_impl_rhs_ref!{Mul, mul, component_mul}
op_impl_rhs_ref!{Div, div, component_div}

impl Sub<f32> for MatrixWrapper
{
    type Output = MatrixWrapper;

    fn sub(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper((-self.0).add_scalar(rhs))
    }
}

impl Sub<&f32> for MatrixWrapper
{
    type Output = MatrixWrapper;

    fn sub(self, rhs: &f32) -> Self::Output
    {
        MatrixWrapper((-self.0).add_scalar(*rhs))
    }
}

impl Sub<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn sub(self, rhs: f32) -> Self::Output
    {
        self.clone().sub(rhs)
    }
}

impl Sub<&f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn sub(self, rhs: &f32) -> Self::Output
    {
        self.clone().sub(rhs)
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

impl Softmaxable for MatrixWrapper
{
    fn exp(&mut self)
    {
        self.exp_inplace();
    }

    fn sum(&self) -> f32
    {
        self.sum()
    }
}

#[allow(dead_code)]
impl MatrixWrapper
{
    pub fn new(rows: usize, columns: usize) -> Self
    {
        Self(DMatrix::zeros(rows, columns))
    }

    pub fn new_with<F: FnMut() -> f32>(
        rows: usize,
        columns: usize,
        mut f: F
    )-> Self
    {
        Self(DMatrix::from_fn(rows, columns, |_, _| f()))
    }

    pub fn repeat(rows: usize, columns: usize, value: f32) -> Self
    {
        Self(DMatrix::repeat(rows, columns, value))
    }

    pub fn from_raw<V: Into<Vec<f32>>>(values: V, rows: usize, columns: usize) -> Self
    {
        Self(DMatrix::from_vec(rows, columns, values.into()))
    }

    pub fn rows(&self) -> usize
    {
        self.0.nrows()
    }

    pub fn columns(&self) -> usize
    {
        self.0.ncols()
    }

    pub fn shape(&self) -> (usize, usize)
    {
        self.0.shape()
    }

    pub fn swap_raw_values<V: Into<Vec<f32>>>(&mut self, values: V)
    {
        self.0.copy_from_slice(&values.into());
    }

    pub fn fill(&mut self, value: f32)
    {
        self.0.fill(value);
    }

    pub fn fill_with(&mut self, f: impl Fn() -> f32)
    {
        self.0.fill_with(f);
    }

    pub fn max(&mut self, rhs: &Self)
    {
        self.0.zip_apply(&rhs.0, |lhs, rhs|
        {
            *lhs = lhs.max(rhs);
        });
    }

    pub fn dot(&self, rhs: &Self) -> f32
    {
        self.0.dot(&rhs.0)
    }

    #[must_use]
    pub fn sqrt(&self) -> Self
    {
        Self(self.0.map(|x| x.sqrt()))
    }

    #[must_use]
    pub fn pow(&self, power: u32) -> Self
    {
        let power = power as i32;
        Self(self.0.map(|x| x.powi(power)))
    }

    pub fn tanh_mul(&self, rhs: &Self) -> Self
    {
        Self(self.0.zip_map(&rhs.0, |x, rhs| x.tanh() * rhs))
    }

    pub fn leaky_relu_mul(&self, rhs: &Self) -> Self
    {
        Self(self.0.zip_map(&rhs.0, |x, rhs| x.max(LEAKY_SLOPE * x) * rhs))
    }

    pub fn exp_inplace(&mut self)
    {
        self.0.apply(|x| *x = x.exp());
    }

    pub fn sqrt_plus(&self, added: f32) -> Self
    {
        Self(self.0.map(|x| x.sqrt() + added))
    }

    pub fn sum(&self) -> f32
    {
        self.0.sum()
    }

    pub fn signum(&self) -> Self
    {
        Self(self.0.map(|v| v.signum()))
    }

    pub fn cosine_similarity(&self, other: &Self) -> f32
    {
        let top = self.dot(other);

        let bottom = self.magnitude() * other.magnitude();

        top / bottom
    }

    pub fn cap_magnitude(&self, cap: f32) -> Self
    {
        Self(self.0.simd_cap_magnitude(cap))
    }

    pub fn magnitude(&self) -> f32
    {
        self.0.magnitude()
    }

    pub fn as_slice(&self) -> &[f32]
    {
        self.0.as_slice()
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

    pub fn pick_weighed(&self) -> usize
    {
        Softmaxer::pick_weighed_inner(self.iter())
    }

    pub fn highest_index(&self) -> usize
    {
        Softmaxer::highest_index(self.iter())
    }
}

#[derive(Clone, Copy)]
pub struct MatrixWrapperRef<'a>(DMatrixView<'a, f32>);

impl Debug for MatrixWrapperRef<'_>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        write!(f, "{{rows: {}, columns: {}, data: {:?}}}", self.0.nrows(), self.0.ncols(), self.as_vec())
    }
}

impl<'a> From<&'a MatrixWrapper> for MatrixWrapperRef<'a>
{
    fn from(value: &'a MatrixWrapper) -> Self
    {
        Self(DMatrixView::from(&value.0))
    }
}

impl<'a> MatrixWrapperRef<'a>
{
    pub fn from_data(data: &'a [f32], info: TensorRawDataPointer) -> Self
    {
        Self::from_data_with_start(data, TensorRawDataPointer{raw_index: TensorIndexRaw(0), ..info})
    }

    pub fn from_data_with_start(data: &'a [f32], info: TensorRawDataPointer) -> Self
    {
        Self(DMatrixView::from_slice(&data[info.raw_index.0..], info.rows, info.columns))
    }

    pub fn dot(self, rhs: Self) -> f32
    {
        self.0.dot(&rhs.0)
    }

    pub fn dot_onehot(self, rhs: &OneHotLayer) -> f32
    {
        debug_assert!(self.0.shape().1 == 1);

        let this = self.0.column(0);

        rhs.positions.iter().map(|position| this.index(*position)).sum()
    }

    pub fn softmax_cross_entropy(self, targets: &OneHotLayer) -> f32
    {
        let mut cloned = self.0.clone_owned();

        MatrixWrapperMut(DMatrixViewMut::from(&mut cloned)).softmax_cross_entropy_inplace(targets)
    }

    pub fn clone_owned(self) -> MatrixWrapper
    {
        MatrixWrapper(self.0.clone_owned())
    }

    pub fn as_vec(self) -> Vec<f32>
    {
        self.clone_owned().as_vec()
    }
}

pub struct MatrixWrapperMut<'a>(DMatrixViewMut<'a, f32>);

impl Debug for MatrixWrapperMut<'_>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        write!(f, "{{rows: {}, columns: {}, data: {:?}}}", self.0.nrows(), self.0.ncols(), self.as_vec())
    }
}

impl<'a> MatrixWrapperMut<'a>
{
    pub fn from_data(data: &'a mut [f32], info: TensorRawDataPointer) -> Self
    {
        Self::from_data_with_start(data, TensorRawDataPointer{raw_index: TensorIndexRaw(0), ..info})
    }

    pub fn from_data_with_start(data: &'a mut [f32], info: TensorRawDataPointer) -> Self
    {
        Self(DMatrixViewMut::from_slice(&mut data[info.raw_index.0..], info.rows, info.columns))
    }

    pub fn add_to(mut self, lhs: MatrixWrapperRef, rhs: MatrixWrapperRef)
    {
        lhs.0.add_to(&rhs.0, &mut self.0);
    }

    pub fn sub_to(&mut self, lhs: MatrixWrapperRef, rhs: MatrixWrapperRef)
    {
        lhs.0.sub_to(&rhs.0, &mut self.0);
    }

    pub fn sub_from_scalar(mut self, lhs: f32, rhs: MatrixWrapperRef)
    {
        self.0.zip_apply(&rhs.0, |out, rhs| *out = lhs - rhs);
    }

    pub fn add_scalar(mut self, rhs: f32)
    {
        self.0.add_scalar_mut(rhs);
    }

    pub fn scale(mut self, rhs: f32)
    {
        self.0.scale_mut(rhs);
    }

    pub fn pow_inplace(mut self, power: u32)
    {
        let power = power as i32;

        self.0.apply(|x| *x = x.powi(power));
    }

    pub fn tanh_inplace(mut self)
    {
        self.0.apply(|x| *x = x.tanh());
    }

    pub fn tanh_gradient_inplace(mut self, value: MatrixWrapperRef, gradient: MatrixWrapperRef)
    {
        self.0.zip_zip_apply(&value.0, &gradient.0, |output, a, b| *output = (1.0 - a * a) * b);
    }

    pub fn sigmoid_inplace(mut self)
    {
        self.0.apply(|x| *x = 1.0 / (1.0 + (-*x).exp()));
    }

    pub fn sigmoid_gradient_inplace(mut self, value: MatrixWrapperRef, gradient: MatrixWrapperRef)
    {
        self.0.zip_zip_apply(&value.0, &gradient.0, |output, a, b| *output = (1.0 - a) * a * b);
    }

    pub fn leaky_relu_inplace(mut self)
    {
        self.0.apply(|x| *x = x.max(LEAKY_SLOPE * *x));
    }

    pub fn leaky_relu_gradient_inplace(mut self, value: MatrixWrapperRef, gradient: MatrixWrapperRef)
    {
        self.0.zip_zip_apply(&value.0, &gradient.0, |output, a, b| *output = leaky_relu_d(a) * b);
    }

    pub fn component_mul_into(mut self, lhs: MatrixWrapperRef, rhs: MatrixWrapperRef)
    {
        self.0.cmpy(1.0, &lhs.0, &rhs.0, 0.0);
    }

    pub fn matmulv_into(mut self, lhs: MatrixWrapperRef, rhs: MatrixWrapperRef)
    {
        self.0.column_mut(0).gemv(1.0, &lhs.0, &rhs.0.column(0), 0.0);
    }

    pub fn matmulv_add_into(mut self, lhs: MatrixWrapperRef, rhs: MatrixWrapperRef, added: MatrixWrapperRef)
    {
        self.0.copy_from(&added.0);
        self.0.column_mut(0).gemv(1.0, &lhs.0, &rhs.0.column(0), 1.0);
    }

    pub fn matmul_onehotv_add_into(mut self, lhs: MatrixWrapperRef, rhs: &OneHotLayer, added: MatrixWrapperRef)
    {
        debug_assert!(added.0.shape().1 == 1);

        self.0.copy_from(&added.0);

        for position in rhs.positions.iter()
        {
            self.0 += lhs.0.column(*position);
        }
    }

    pub fn matmulv_transposed_into(mut self, lhs: MatrixWrapperRef, rhs: MatrixWrapperRef)
    {
        self.0.column_mut(0).gemv_tr(1.0, &lhs.0, &rhs.0.column(0), 0.0);
    }

    pub fn outer_product_into(mut self, lhs: MatrixWrapperRef, rhs: MatrixWrapperRef)
    {
        self.0.ger(1.0, &lhs.0.column(0), &rhs.0.column(0), 0.0);
    }

    pub fn outer_product_one_hot_into(mut self, lhs: MatrixWrapperRef, rhs: &OneHotLayer)
    {
        let a = &lhs.0;

        debug_assert!(a.shape().1 == 1);

        self.0.fill(0.0);

        let a = &a.column(0);

        for position in rhs.positions.iter().copied()
        {
            self.0.set_column(position, a);
        }
    }

    fn ln_onehot(&mut self, onehot: &OneHotLayer)
    {
        debug_assert!(self.0.shape().1 == 1);

        let mut this = self.0.column_mut(0);

        onehot.positions.iter().for_each(|position|
        {
            let value = this.index_mut(*position);

            *value = value.ln();
        });
    }

    pub fn softmax_cross_entropy_inplace(mut self, targets: &OneHotLayer) -> f32
    {
        self.0.apply(|x| *x = x.exp());
        let s = self.0.sum();

        self.0 /= s;

        let mut new = self.0.clone_owned();
        MatrixWrapperMut(DMatrixViewMut::from(&mut new)).ln_onehot(targets);

        -MatrixWrapperRef(DMatrixView::from(&new)).dot_onehot(targets)
    }

    pub fn clone_owned(&self) -> MatrixWrapper
    {
        MatrixWrapper(self.0.clone_owned())
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.clone_owned().as_vec()
    }
}
