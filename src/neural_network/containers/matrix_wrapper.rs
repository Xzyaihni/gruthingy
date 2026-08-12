use std::{
    f32,
    fmt::{self, Debug},
    borrow::Borrow,
    ops::{Mul, Add, Sub, Div, AddAssign, SubAssign, DivAssign, Neg}
};

use serde::{Serialize, Deserialize};

use nalgebra::{DMatrix, Matrix, UninitMatrix, Dyn};

use super::{
    Softmaxer,
    Softmaxable,
    OneHotLayer,
    LayerType,
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

const UNCOMMENT_MEA: () = {let a = (); ()};
/*impl<'a> DiffOperation for MulGTensorOperation<'a>
{
    fn inplace_tensor(self) -> impl FnOnce(&LayerType, &mut Option<LayerType>)
    {
        |_, output| *output = Some(self.compute_tensor())
    }

    fn add_tensor(self) -> impl FnOnce(&LayerType, &mut LayerType)
    {
        move |_, output|
        {
            match &*self.a
            {
                AnyDiffType::Tensor(_) => unimplemented!(),
                AnyDiffType::Scalar(x) => output.0 += &self.b.0 * x.value
            }
        }
    }

    fn scalar_sum(self) -> f32
    {
        match &*self.a
        {
            AnyDiffType::Tensor(x) => x.value.0.dot(&self.b.0),
            AnyDiffType::Scalar(_) => unimplemented!()
        }
    }

    fn compute_tensor(self) -> LayerType
    {
        match &*self.a
        {
            AnyDiffType::Tensor(x) => &x.value * self.b,
            AnyDiffType::Scalar(x) => self.b * x.value
        }
    }
}

impl<'a> DiffOperation for MulGF32Operation<'a>
{
    fn inplace_tensor(self) -> impl FnOnce(&LayerType, &mut Option<LayerType>)
    {
        |_, output| *output = Some(self.compute_tensor())
    }

    fn add_tensor(self) -> impl FnOnce(&LayerType, &mut LayerType)
    {
        move |_, output|
        {
            match &*self.a
            {
                AnyDiffType::Tensor(_) => unimplemented!(),
                AnyDiffType::Scalar(x) => output.0.add_scalar_mut(self.b * x.value)
            }
        }
    }

    fn scalar_sum(self) -> f32
    {
        unimplemented!()
    }

    fn compute_tensor(self) -> LayerType
    {
        match &*self.a
        {
            AnyDiffType::Tensor(x) => &x.value * self.b,
            AnyDiffType::Scalar(_) => unimplemented!()
        }
    }
}

impl<'a> DiffOperation for MatMulVTransposed<'a>
{
    fn inplace_tensor(self) -> impl FnOnce(&LayerType, &mut Option<LayerType>)
    {
        |_, output| *output = Some(self.compute_tensor())
    }

    fn add_tensor(self) -> impl FnOnce(&LayerType, &mut LayerType)
    {
        move |_, output|
        {
            debug_assert!(self.b.0.shape().1 == 1);

            output.0.column_mut(0).gemv_tr(1.0, &self.a.0, &self.b.0.column(0), 1.0);
        }
    }

    fn scalar_sum(self) -> f32
    {
        unimplemented!()
    }

    fn compute_tensor(self) -> LayerType
    {
        debug_assert!(self.b.0.shape().1 == 1);

        MatrixWrapper(self.a.0.tr_mul(&self.b.0))
    }
}

impl<'a> DiffOperation for OuterProduct<'a>
{
    fn inplace_tensor(self) -> impl FnOnce(&LayerType, &mut Option<LayerType>)
    {
        |_, output| *output = Some(self.compute_tensor())
    }

    fn add_tensor(self) -> impl FnOnce(&LayerType, &mut LayerType)
    {
        move |_, output|
        {
            // let compare_result = output.clone() + self.clone().compute_tensor();

            let a = &self.a.0;
            let b = &self.b.0;

            debug_assert!(a.shape().1 == 1);
            debug_assert!(b.shape().1 == 1);

            output.0.ger(1.0, &a.column(0), &b.column(0), 1.0);

            // assert_eq!(output.0, compare_result.0); let im_debug = ();
        }
    }

    fn scalar_sum(self) -> f32
    {
        unimplemented!()
    }

    fn compute_tensor(self) -> LayerType
    {
        let a = &self.a.0;
        let b = &self.b.0;

        let rows = a.nrows();
        let columns = b.nrows();

        debug_assert!(a.shape().1 == 1);
        debug_assert!(b.shape().1 == 1);

        let mut output_uninit: UninitMatrix<f32, Dyn, Dyn> = UninitMatrix::uninit(Dyn(rows), Dyn(columns));

        {
            let this = &a.column(0);
            let rhs = &b.column(0);

            for column in 0..columns
            {
                let rhs_value = unsafe{ *rhs.vget_unchecked(column) };

                for row in 0..rows
                {
                    let this_value = unsafe{ *this.vget_unchecked(row) };

                    unsafe{ output_uninit.get_unchecked_mut((row, column)).write(this_value * rhs_value); }
                }
            }
        }

        let output: Matrix<f32, Dyn, Dyn, _> = unsafe{ UninitMatrix::assume_init(output_uninit) };

        // assert_eq!(output.clone(), &self.0 * &rhs.borrow().0.transpose()); let im_debug = ();

        MatrixWrapper(output)
    }
}

impl<'a> DiffOperation for OuterProductOneHot<'a>
{
    fn inplace_tensor(self) -> impl FnOnce(&LayerType, &mut Option<LayerType>)
    {
        |_, output| *output = Some(self.compute_tensor())
    }

    fn add_tensor(self) -> impl FnOnce(&LayerType, &mut LayerType)
    {
        move |_, output|
        {
            let a = &self.a.0;

            let output = &mut output.0;

            debug_assert!(a.shape().1 == 1);

            let a = &a.column(0);

            for position in self.b.positions.iter().copied()
            {
                output.column_mut(position).axpy(1.0, a, 1.0)
            }
        }
    }

    fn scalar_sum(self) -> f32
    {
        unimplemented!()
    }

    fn compute_tensor(self) -> LayerType
    {
        let a = &self.a.0;

        debug_assert!(a.shape().1 == 1);

        let mut output = DMatrix::zeros(a.nrows(), self.b.size);

        let a = &a.column(0);

        for position in self.b.positions.iter().copied()
        {
            output.set_column(position, a);
        }

        MatrixWrapper(output)
    }
}*/

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
        debug_assert!(rhs.borrow().0.shape().1 == 1);

        let this = (&self.0).mul(&rhs.borrow().0.column(0));

        let rows = this.shape_generic().0;
        Self(this.reshape_generic(rows, Dyn(1)))
    }

    pub fn matmulv_add(&self, rhs: impl Borrow<Self>, added: impl Borrow<Self>) -> Self
    {
        debug_assert!(rhs.borrow().0.shape().1 == 1);
        debug_assert!(added.borrow().0.shape().1 == 1);

        let mut this = added.borrow().0.clone();
        this.column_mut(0).gemv(1.0, &self.0, &rhs.borrow().0.column(0), 1.0);

        Self(this)
    }

    pub fn matmul_onehotv_add(&self, rhs: &OneHotLayer, added: impl Borrow<Self>) -> Self
    {
        debug_assert!(added.borrow().0.shape().1 == 1);

        let mut this = added.borrow().0.clone();

        for position in rhs.positions.iter()
        {
            this += self.0.column(*position);
        }

        Self(this)
    }

    pub fn max(&mut self, rhs: &Self)
    {
        self.0.zip_apply(&rhs.0, |lhs, rhs|
        {
            *lhs = lhs.max(rhs);
        });
    }

    pub fn dot_onehot(self, rhs: &OneHotLayer) -> f32
    {
        debug_assert!(self.0.shape().1 == 1);

        let this = self.0.column(0);

        rhs.positions.iter().map(|position| this.index(*position)).sum()
    }

    pub fn dot(&self, rhs: &Self) -> f32
    {
        self.0.dot(&rhs.0)
    }

    pub fn ln_onehot(&mut self, onehot: &OneHotLayer)
    {
        debug_assert!(self.0.shape().1 == 1);

        let mut this = self.0.column_mut(0);

        onehot.positions.iter().for_each(|position|
        {
            let value = this.index_mut(*position);

            *value = value.ln();
        });
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

    #[must_use]
    pub fn sigmoid(&self) -> Self
    {
        Self(self.0.map(|x| 1.0 / (1.0 + (-x).exp())))
    }

    pub fn sigmoid_gradient_inplace(&mut self, value: &Self, gradient: &Self)
    {
        self.0.zip_zip_apply(&value.0, &gradient.0, |output, a, b| *output = (1.0 - a) * a * b);
    }

    #[must_use]
    pub fn tanh(&self) -> Self
    {
        Self(self.0.map(|x| x.tanh()))
    }

    pub fn tanh_gradient_inplace(&mut self, value: &Self, gradient: &Self)
    {
        self.0.zip_zip_apply(&value.0, &gradient.0, |output, a, b| *output = (1.0 - a * a) * b);
    }

    #[must_use]
    pub fn leaky_relu(&self) -> Self
    {
        Self(self.0.map(|x| x.max(LEAKY_SLOPE * x)))
    }

    pub fn leaky_relu_gradient_inplace(&mut self, value: &Self, gradient: &Self)
    {
        self.0.zip_zip_apply(&value.0, &gradient.0, |output, a, b| *output = leaky_relu_d(a) * b);
    }

    pub fn exp_inplace(&mut self)
    {
        self.0.apply(|x| *x = x.exp());
    }

    pub fn sum(&self) -> f32
    {
        self.0.sum()
    }

    pub fn signum(&self) -> Self
    {
        Self(self.0.map(|v| v.signum()))
    }

    pub fn cap_magnitude(&self, cap: f32) -> Self
    {
        Self(self.0.simd_cap_magnitude(cap))
    }

    pub fn magnitude(&self) -> f32
    {
        self.0.magnitude()
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
