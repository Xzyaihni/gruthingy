use std::{
    f32,
    fmt::Debug,
    borrow::Borrow,
    ops::{Mul, Add, Sub, Div, AddAssign, SubAssign, DivAssign, Neg}
};

use serde::{Serialize, Deserialize};

use nalgebra::{DMatrix, Dyn};

use super::{
    Softmaxer,
    Softmaxable,
    JoinableSelector,
    DefaultJoinableWrapper,
    DefaultJoinableDeepWrapper,
    LEAKY_SLOPE,
    leaky_relu_d
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixWrapper(DMatrix<f32>);

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

impl Sub<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn sub(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper(self.0.add_scalar(-rhs))
    }
}

impl Sub<&f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn sub(self, rhs: &f32) -> Self::Output
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

impl Softmaxable for MatrixWrapper
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

impl JoinableSelector<(MatrixWrapper, MatrixWrapper)> for MatrixWrapper
{
    type This = DefaultJoinableWrapper<MatrixWrapper>;
    type Deep = DefaultJoinableDeepWrapper<MatrixWrapper>;
}

#[allow(dead_code)]
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

    pub fn matmulv(&self, rhs: impl Borrow<Self>) -> Self
    {
        let this = (&self.0).mul(&rhs.borrow().0.column(0));

        let rows = this.shape_generic().0;
        Self(this.reshape_generic(rows, Dyn(1)))
    }

    pub fn matmulv_transposed(&self, rhs: impl Borrow<Self>) -> Self
    {
        // nope no gemv_tr cuz FUCK me
        Self(self.0.tr_mul(&rhs.borrow().0))
    }

    pub fn outer_product(&self, rhs: impl Borrow<Self>) -> Self
    {
        // cant use ger here cuz all linear algebra libraries hate me and r slow
        // if only i could call ger on a matrix that has MaybeUninit<T> >_<
        Self(&self.0 * &rhs.borrow().0.transpose())
    }

    pub fn matmulv_add(&self, rhs: impl Borrow<Self>, added: impl Borrow<Self>) -> Self
    {
        let mut this = added.borrow().0.clone();
        this.column_mut(0).gemv(1.0, &self.0, &rhs.borrow().0.column(0), 1.0);

        Self(this)
    }

    pub fn max(&mut self, rhs: &Self)
    {
        self.0.zip_apply(&rhs.0, |lhs, rhs|
        {
            *lhs = lhs.max(rhs);
        });
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

    pub fn pow(&mut self, power: u32)
    {
        let power = power as i32;
        self.0.apply(|v| *v = v.powi(power));
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

    pub fn signum(&self) -> Self
    {
        let mut this = self.0.clone();
        this.apply(|v| *v = v.signum());

        Self(this)
    }

    pub fn cap_magnitude(&self, cap: f32) -> Self
    {
        Self(self.0.simd_cap_magnitude(cap))
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

    pub const fn is_arrayfire() -> bool
    {
        false
    }
}
