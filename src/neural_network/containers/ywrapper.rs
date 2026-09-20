use std::{
    f32,
    iter,
    num::FpCategory
};

use serde::{Serialize, Deserialize};

use oxiblas_matrix::{MatRef, MatMut};

use super::{
    TensorShape,
    Softmaxer,
    Softmaxable,
    OneHotLayer,
    TensorRawDataPointer,
    LEAKY_SLOPE,
    leaky_relu_d
};


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YWrapper
{
    shape: TensorShape,
    values: Box<[f32]>
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YWrapperRef<'a>
{
    shape: TensorShape,
    values: &'a [f32]
}

impl<'a> From<&'a YWrapper> for YWrapperRef<'a>
{
    fn from(value: &'a YWrapper) -> Self
    {
        value.as_ref()
    }
}

#[derive(Debug, PartialEq)]
pub struct YWrapperMut<'a>
{
    shape: TensorShape,
    values: &'a mut [f32]
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YVectorWrapperRef<'a>
{
    rows: usize,
    batch_size: usize,
    values: &'a [f32]
}

#[derive(Debug, PartialEq)]
pub struct YVectorWrapperMut<'a>
{
    rows: usize,
    batch_size: usize,
    values: &'a mut [f32]
}

impl Softmaxable for YWrapper
{
    fn exp_inplace(&mut self)
    {
        self.exp_inplace();
    }

    fn sum(&self) -> f32
    {
        self.sum()
    }

    fn mul_scalar_inplace(&mut self, value: f32)
    {
        self.mul_scalar_inplace(value);
    }
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32
{
    oxiblas_blas::level1::dot_f32(lhs, rhs)
}

#[allow(dead_code)]
impl YWrapper
{
    pub fn new(rows: usize, columns: usize) -> Self
    {
        Self::repeat(rows, columns, 0.0)
    }

    pub fn repeat(rows: usize, columns: usize, value: f32) -> Self
    {
        Self::from_boxed(vec![value; rows * columns].into_boxed_slice(), rows, columns)
    }

    pub fn repeat_full(shape: TensorShape, value: f32) -> Self
    {
        Self::from_raw(vec![value; shape.size()].into_boxed_slice(), shape)
    }

    pub fn new_with(rows: usize, columns: usize, f: impl Fn() -> f32) -> Self
    {
        Self::from_boxed(iter::repeat_with(f).take(rows * columns).collect(), rows, columns)
    }

    pub fn from_boxed(values: Box<[f32]>, rows: usize, columns: usize) -> Self
    {
        Self::from_raw(values, TensorShape{rows, columns, batch_size: 1})
    }

    pub fn from_raw(values: Box<[f32]>, shape: TensorShape) -> Self
    {
        debug_assert_eq!(values.len(), shape.size());

        Self{
            shape,
            values
        }
    }

    pub fn as_ref(&self) -> YWrapperRef<'_>
    {
        YWrapperRef{
            shape: self.shape,
            values: &self.values
        }
    }

    pub fn as_mut(&mut self) -> YWrapperMut<'_>
    {
        YWrapperMut{
            shape: self.shape,
            values: &mut self.values
        }
    }

    pub fn mul_scalar(&self, value: f32) -> Self
    {
        self.clone().map(|x| x * value)
    }

    pub fn mul_scalar_inplace(&mut self, value: f32)
    {
        self.as_mut().mul_scalar_inplace(value)
    }

    pub fn sqrt_plus(&self, added: f32) -> Self
    {
        self.clone().map(|x| x.sqrt() + added)
    }

    pub fn mul_componentwise(&self, other: YWrapperRef) -> Self
    {
        self.clone().zip_map(other, |a, b| a * b)
    }

    pub fn div_componentwise(&self, other: YWrapperRef) -> Self
    {
        self.clone().zip_map(other, |a, b| a / b)
    }

    pub fn add_inplace(&mut self, other: YWrapperRef)
    {
        self.as_mut().add_inplace(other)
    }

    pub fn add(&self, other: YWrapperRef) -> Self
    {
        self.clone().zip_map(other, |a, b| a + b)
    }

    pub fn pow(&self, power: u32) -> Self
    {
        self.clone().map(|x| x.powi(power as i32))
    }

    pub fn signum(&self) -> Self
    {
        self.clone().map(|x| x.signum())
    }

    pub fn exp_inplace(&mut self)
    {
        self.as_mut().apply(|x| x.exp())
    }

    pub fn sum(&self) -> f32
    {
        debug_assert_eq!(self.shape.batch_size, 1);

        self.values.iter().copied().sum::<f32>()
    }

    pub fn max(&self, other: YWrapperRef) -> Self
    {
        self.clone().zip_map(other, |a, b| a.max(b))
    }

    fn zip_map(self, b: YWrapperRef, f: impl Fn(f32, f32) -> f32) -> Self
    {
        debug_assert_eq!(self.shape(), b.shape());

        Self{
            shape: self.shape,
            values: self.values.into_iter().zip(b.values).map(|(a, b)| f(a, *b)).collect()
        }
    }

    fn map(self, f: impl Fn(f32) -> f32) -> Self
    {
        Self{
            shape: self.shape,
            values: self.values.into_iter().map(f).collect()
        }
    }

    pub fn cosine_similarity(&self, other: YWrapperRef) -> f32
    {
        debug_assert_eq!(self.shape.batch_size, 1);

        let top = self.as_ref().dot(other);

        let bottom = self.magnitude() * other.magnitude();

        top / bottom
    }

    pub fn cap_magnitude_inplace(&mut self, cap: f32)
    {
        debug_assert_eq!(self.shape.batch_size, 1);

        let m = self.magnitude();

        if m > cap
        {
            self.as_mut().mul_scalar_inplace(cap / m);
        }
    }

    pub fn magnitude(&self) -> f32
    {
        self.as_ref().magnitude()
    }

    pub fn rows(&self) -> usize
    {
        self.shape.rows
    }

    pub fn columns(&self) -> usize
    {
        self.shape.columns
    }

    pub fn shape(&self) -> TensorShape
    {
        self.shape
    }

    pub fn total_len(&self) -> usize
    {
        self.values.len()
    }

    pub fn as_slice(&self) -> &[f32]
    {
        &self.values
    }

    pub fn swap_raw_values(&mut self, values: Box<[f32]>)
    {
        self.values = values;
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.values.to_vec()
    }

    pub fn iter(&self) -> impl Iterator<Item=&f32> + ExactSizeIterator
    {
        self.values.iter()
    }

    pub fn pick_weighed(&self) -> usize
    {
        debug_assert_eq!(self.shape.batch_size, 1);

        Softmaxer::pick_weighed_inner(self.iter())
    }

    pub fn highest_index(&self) -> usize
    {
        debug_assert_eq!(self.shape.batch_size, 1);

        Softmaxer::highest_index(self.iter())
    }
}

#[allow(dead_code)]
impl<'a> YWrapperRef<'a>
{
    pub fn from_data(values: &'a [f32], shape: TensorShape) -> Self
    {
        debug_assert_eq!(values.len(), shape.size());

        Self{shape, values}
    }

    pub fn from_data_with_start(data: &'a [f32], info: TensorRawDataPointer) -> Self
    {
        Self::from_data(&data[info.raw_index.0..(info.raw_index.0 + info.size())], info.shape)
    }

    pub fn matmul_onehotv_add(self, rhs: &OneHotLayer, added: YVectorWrapperRef) -> YWrapper
    {
        debug_assert_eq!(self.shape.batch_size, 1);

        let mut output = YWrapper::new(self.shape.rows, 1);

        output.as_mut().as_vector_mut().matmul_onehotv_add_into(self, rhs, added);

        output
    }

    pub fn dot(self, rhs: Self) -> f32
    {
        debug_assert_eq!(self.shape.batch_size, 1);

        dot(self.values, rhs.values)
    }

    pub fn magnitude(&self) -> f32
    {
        debug_assert_eq!(self.shape.batch_size, 1);

        oxiblas_blas::level1::nrm2_f32(self.values)
    }

    pub fn average(&self) -> f32
    {
        debug_assert!(self.shape.is_batched_scalar());

        self.values.iter().copied().sum::<f32>() / self.shape.batch_size as f32
    }

    pub fn as_vector_ref(&self) -> YVectorWrapperRef<'_>
    {
        debug_assert_eq!(self.shape.columns, 1);

        YVectorWrapperRef::from_data(&self.values, self.shape)
    }

    pub fn batch_slice_ref(&self, batch_index: usize) -> YWrapperRef<'_>
    {
        YWrapperRef{
            values: &self.values[self.shape.batch_range(batch_index)],
            shape: TensorShape{batch_size: 1, ..self.shape}
        }
    }

    fn as_mat_ref(&self) -> MatRef<'_, f32>
    {
        MatRef::from_column_major(self.values, self.shape.rows, self.shape.columns).expect("dimensions must match")
    }

    pub fn rows(&self) -> usize
    {
        self.shape.rows
    }

    pub fn columns(&self) -> usize
    {
        self.shape.columns
    }

    pub fn shape(&self) -> TensorShape
    {
        self.shape
    }

    pub fn clone_owned(&self) -> YWrapper
    {
        YWrapper{
            shape: self.shape,
            values: self.values.to_vec().into_boxed_slice()
        }
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.values.to_vec()
    }
}

#[allow(dead_code)]
impl<'a> YWrapperMut<'a>
{
    pub fn from_data(values: &'a mut [f32], shape: TensorShape) -> Self
    {
        debug_assert_eq!(values.len(), shape.size());

        Self{shape, values}
    }

    pub fn from_data_with_start(data: &'a mut [f32], info: TensorRawDataPointer) -> Self
    {
        Self::from_data(&mut data[info.raw_index.0..(info.raw_index.0 + info.shape.size())], info.shape)
    }

    pub fn copy_from(self, value: YWrapperRef)
    {
        debug_assert_eq!(self.values.len(), value.values.len());
        debug_assert_eq!(self.shape, value.shape);

        self.values.copy_from_slice(value.values)
    }

    pub fn fill(self, value: f32)
    {
        self.values.fill(value);
    }

    pub fn fill_with(self, f: impl Fn() -> f32)
    {
        self.values.fill_with(f);
    }

    pub fn add_to(self, lhs: YWrapperRef, rhs: YWrapperRef)
    {
        debug_assert_eq!(self.shape, lhs.shape);

        if rhs.shape.is_batched_scalar()
        {
            if self.shape.batch_size != rhs.shape.batch_size
            {
                debug_assert_eq!(self.shape.batch_size, 1);

                for batch_index in 0..rhs.shape.batch_size
                {
                    let rhs = rhs.values[batch_index];

                    for i in 0..self.shape.single_size()
                    {
                        let s = lhs.values[i] + rhs;

                        if batch_index == 0
                        {
                            self.values[i] = s;
                        } else
                        {
                            self.values[i] += s;
                        }
                    }
                }
            } else
            {
                for batch_index in 0..self.shape.batch_size
                {
                    let rhs = rhs.values[batch_index];

                    let batch_start = self.shape.batch_range(batch_index).start;

                    for i in 0..self.shape.single_size()
                    {
                        let index = batch_start + i;

                        self.values[index] = lhs.values[index] + rhs;
                    }
                }
            }

            return;
        }

        debug_assert_eq!(lhs.shape.rows, rhs.shape.rows);
        debug_assert_eq!(lhs.shape.columns, rhs.shape.columns);

        debug_assert_eq!(self.shape.batch_size, 1);
        debug_assert_eq!(lhs.shape.batch_size, 1);
        debug_assert_eq!(rhs.shape.batch_size, 1);

        for i in 0..self.values.len()
        {
            unsafe{
                *self.values.get_unchecked_mut(i) = *lhs.values.get_unchecked(i) + *rhs.values.get_unchecked(i);
            }
        }
    }

    pub fn sub_to(&mut self, lhs: YWrapperRef, rhs: YWrapperRef)
    {
        debug_assert_eq!(self.shape.batch_size, lhs.shape.batch_size);
        debug_assert_eq!(lhs.shape.batch_size, rhs.shape.batch_size);

        debug_assert_eq!(self.shape, rhs.shape);

        if lhs.shape.is_batched_scalar()
        {
            for batch_index in 0..lhs.shape.batch_size
            {
                self.batch_slice_mut(batch_index).sub_from_scalar(lhs.values[batch_index], rhs.batch_slice_ref(batch_index));
            }

            return;
        }

        debug_assert_eq!(lhs.shape, rhs.shape);

        let mut out = nalgebra::DVectorViewMut::from(&mut *self.values);
        let lhs = nalgebra::DVectorView::from(lhs.values);
        let rhs = nalgebra::DVectorView::from(rhs.values);

        lhs.sub_to(&rhs, &mut out);
    }

    pub fn sub_from_scalar(self, lhs: f32, rhs: YWrapperRef)
    {
        debug_assert_eq!(self.shape, rhs.shape);

        for i in 0..self.values.len()
        {
            unsafe{
                *self.values.get_unchecked_mut(i) = lhs - rhs.values.get_unchecked(i);
            }
        }
    }

    pub fn sub_inplace(self, rhs: YWrapperRef)
    {
        debug_assert_eq!(self.shape, rhs.shape);

        oxiblas_blas::level1::axpy_f32(-1.0, rhs.values, self.values)
    }

    pub fn add_inplace(self, rhs: YWrapperRef)
    {
        debug_assert_eq!(self.shape, rhs.shape);

        oxiblas_blas::level1::axpy_f32(1.0, rhs.values, self.values)
    }

    pub fn add_scalar_inplace(mut self, other: f32)
    {
        self.apply(|x| x + other)
    }

    pub fn mul_scalar_inplace(&mut self, value: f32)
    {
        self.apply(|x| x * value)
    }

    pub fn mul_batched_scalar_inplace(&mut self, value: YWrapperRef)
    {
        debug_assert!(value.shape.is_batched_scalar());

        debug_assert_eq!(self.shape.batch_size, value.shape.batch_size);

        for batch_index in 0..value.shape.batch_size
        {
            self.batch_slice_mut(batch_index).mul_scalar_inplace(value.values[batch_index]);
        }
    }

    pub fn sum_tensor_into(self, value: YWrapperRef)
    {
        debug_assert!(self.shape.is_batched_scalar());

        debug_assert_eq!(self.shape.batch_size, value.shape.batch_size);

        for batch_index in 0..self.shape.batch_size
        {
            self.values[batch_index] = value.values[value.shape.batch_range(batch_index)].iter().copied().sum();
        }
    }

    pub fn fill_into(self, value: YWrapperRef)
    {
        debug_assert!(value.shape.is_batched_scalar());

        debug_assert_eq!(self.shape.batch_size, value.shape.batch_size);

        for batch_index in 0..self.shape.batch_size
        {
            self.values[self.shape.batch_range(batch_index)].fill(value.values[batch_index]);
        }
    }

    pub fn dot_into(self, lhs: YWrapperRef, rhs: YWrapperRef)
    {
        debug_assert!(self.shape.is_batched_scalar());

        debug_assert_eq!(self.shape.batch_size, lhs.shape.batch_size);
        debug_assert_eq!(lhs.shape, rhs.shape);

        for batch_index in 0..self.shape.batch_size
        {
            let batch_range = lhs.shape.batch_range(batch_index);

            self.values[batch_index] = dot(&lhs.values[batch_range.clone()], &rhs.values[batch_range]);
        }
    }

    pub fn pow_inplace(mut self, power: u32)
    {
        self.apply(|x| x.powi(power as i32))
    }

    pub fn tanh_inplace(mut self)
    {
        self.apply(|x| x.tanh())
    }

    pub fn tanh_gradient_inplace(self, value: YWrapperRef, gradient: YWrapperRef)
    {
        debug_assert_eq!(self.shape, value.shape);
        debug_assert_eq!(value.shape, gradient.shape);

        (0..self.values.len()).for_each(|i|
        {
            let a = value.values[i];

            self.values[i] = (1.0 - a * a) * gradient.values[i]
        })
    }

    pub fn sigmoid_inplace(mut self)
    {
        self.apply(|x| 1.0 / (1.0 + (-x).exp()))
    }

    pub fn sigmoid_gradient_inplace(self, value: YWrapperRef, gradient: YWrapperRef)
    {
        debug_assert_eq!(self.shape, value.shape);
        debug_assert_eq!(value.shape, gradient.shape);

        (0..self.values.len()).for_each(|i|
        {
            let a = value.values[i];

            self.values[i] = (1.0 - a) * a * gradient.values[i]
        })
    }

    pub fn leaky_relu_inplace(mut self)
    {
        self.apply(|x| x.max(LEAKY_SLOPE * x))
    }

    pub fn leaky_relu_gradient_inplace(self, value: YWrapperRef, gradient: YWrapperRef)
    {
        debug_assert_eq!(self.shape, value.shape);
        debug_assert_eq!(value.shape, gradient.shape);

        (0..self.values.len()).for_each(|i| self.values[i] = leaky_relu_d(value.values[i]) * gradient.values[i])
    }

    pub fn component_mul_into(mut self, lhs: YWrapperRef, rhs: YWrapperRef)
    {
        let (lhs, rhs) = if lhs.shape.is_batched_scalar()
        {
            (rhs, lhs)
        } else
        {
            (lhs, rhs)
        };

        if rhs.shape.is_batched_scalar()
        {
            let (values, scalar) = (lhs, rhs);

            debug_assert_eq!(values.shape.batch_size, scalar.shape.batch_size);

            if self.shape != values.shape
            {
                debug_assert_eq!(values.shape.batch_size, 1);

                let single_size = self.shape.single_size();
                self.values[..single_size].copy_from_slice(values.values);

                self.batch_slice_mut(0).mul_scalar_inplace(scalar.values[0]);

                for batch_index in 1..scalar.shape.batch_size
                {
                    self.values[self.shape.batch_range(batch_index)].copy_within(0..single_size, batch_index * single_size);
                }
            } else
            {
                self.values.copy_from_slice(values.values);

                for batch_index in 0..scalar.shape.batch_size
                {
                    self.batch_slice_mut(batch_index).mul_scalar_inplace(scalar.values[batch_index]);
                }
            }

            return;
        }

        debug_assert_eq!(self.shape, lhs.shape);
        debug_assert_eq!(lhs.shape, rhs.shape);

        (0..self.values.len()).for_each(|i| self.values[i] = lhs.values[i] * rhs.values[i]);
    }

    pub fn component_mul_add_into(self, lhs: YWrapperRef, rhs: YWrapperRef, added: YWrapperRef)
    {
        debug_assert_eq!(self.shape, lhs.shape);
        debug_assert_eq!(lhs.shape, rhs.shape);
        debug_assert_eq!(rhs.shape, added.shape);

        (0..self.values.len()).for_each(|i| self.values[i] = lhs.values[i] * rhs.values[i] + added.values[i]);
    }

    pub fn outer_product_into(self, lhs: YVectorWrapperRef, rhs: YVectorWrapperRef)
    {
        debug_assert_eq!(self.shape.rows, lhs.rows);
        debug_assert_eq!(self.shape.columns, rhs.rows);

        debug_assert_eq!(self.shape.batch_size, lhs.batch_size);
        debug_assert_eq!(self.shape.batch_size, rhs.batch_size);

        debug_assert_eq!(self.shape.batch_size, 1);

        let mut out = nalgebra::DMatrixViewMut::from_slice(self.values, self.shape.rows, self.shape.columns);
        let lhs = nalgebra::DVectorView::from(lhs.values);
        let rhs = nalgebra::DVectorView::from(rhs.values);

        out.ger(1.0, &lhs, &rhs, 0.0);
    }

    pub fn outer_product_add_inplace(mut self, lhs: YVectorWrapperRef, rhs: YVectorWrapperRef)
    {
        fn inner_single_batch(output: &mut YWrapperMut, lhs: YVectorWrapperRef, rhs: YVectorWrapperRef)
        {
            debug_assert_eq!(output.shape.batch_size, 1);
            debug_assert_eq!(lhs.batch_size, 1);
            debug_assert_eq!(rhs.batch_size, 1);

            let mut out = nalgebra::DMatrixViewMut::from_slice(output.values, output.shape.rows, output.shape.columns);
            let lhs = nalgebra::DVectorView::from(lhs.values);
            let rhs = nalgebra::DVectorView::from(rhs.values);

            out.ger(1.0, &lhs, &rhs, 1.0);
        }

        debug_assert_eq!(self.shape.rows, lhs.rows);
        debug_assert_eq!(self.shape.columns, rhs.rows);

        if self.shape.batch_size != lhs.batch_size
        {
            debug_assert_eq!(self.shape.batch_size, 1);
            debug_assert_eq!(lhs.batch_size, rhs.batch_size);

            for batch_index in 0..lhs.batch_size
            {
                inner_single_batch(&mut self, lhs.batch_slice_ref(batch_index), rhs.batch_slice_ref(batch_index));
            }

            return;
        }

        if self.shape.batch_size != rhs.batch_size
        {
            debug_assert_eq!(rhs.batch_size, 1);
            debug_assert_eq!(self.shape.batch_size, lhs.batch_size);

            for batch_index in 0..lhs.batch_size
            {
                inner_single_batch(&mut self.batch_slice_mut(batch_index), lhs.batch_slice_ref(batch_index), rhs);
            }

            return;
        }

        for batch_index in 0..self.shape.batch_size
        {
            inner_single_batch(&mut self.batch_slice_mut(batch_index), lhs.batch_slice_ref(batch_index), rhs.batch_slice_ref(batch_index));
        }
    }

    pub fn outer_product_one_hot_into(self, lhs: YVectorWrapperRef, rhs: &OneHotLayer)
    {
        debug_assert_eq!(self.shape.rows, lhs.rows);
        debug_assert_eq!(self.shape.columns, rhs.size);

        debug_assert_eq!(self.shape.batch_size, lhs.batch_size);
        debug_assert_eq!(self.shape.batch_size, rhs.batch_size());

        debug_assert_eq!(self.shape.batch_size, 1);

        let rows = self.shape.rows;

        self.values.fill(0.0);

        rhs.positions[0].iter().for_each(|column|
        {
            (0..rows).for_each(|row|
            {
                self.values[column * rows + row] = lhs.values[row];
            })
        })
    }

    pub fn outer_product_one_hot_add_inplace(mut self, lhs: YVectorWrapperRef, rhs: &OneHotLayer)
    {
        fn inner_single_batch(output: &mut YWrapperMut, lhs: YVectorWrapperRef, rhs: &[usize])
        {
            debug_assert_eq!(output.shape.batch_size, 1);
            debug_assert_eq!(lhs.batch_size, 1);

            let rows = output.shape.rows;

            rhs.iter().for_each(|column|
            {
                let start = column * rows;

                oxiblas_blas::level1::axpy_f32(1.0, lhs.values, &mut output.values[start..(start + rows)]);
            });
        }

        debug_assert_eq!(self.shape.rows, lhs.rows);
        debug_assert_eq!(self.shape.columns, rhs.size);

        if self.shape.batch_size != lhs.batch_size
        {
            debug_assert_eq!(self.shape.batch_size, 1);
            debug_assert_eq!(lhs.batch_size, rhs.batch_size());

            for batch_index in 0..lhs.batch_size
            {
                inner_single_batch(&mut self, lhs.batch_slice_ref(batch_index), &rhs.positions[batch_index]);
            }

            return;
        }

        debug_assert_eq!(self.shape.batch_size, rhs.batch_size());

        for batch_index in 0..self.shape.batch_size
        {
            inner_single_batch(&mut self.batch_slice_mut(batch_index), lhs.batch_slice_ref(batch_index), &rhs.positions[batch_index]);
        }
    }

    pub fn softmax_cross_entropy_into(self, mut values: YWrapperMut, targets: &OneHotLayer)
    {
        debug_assert_eq!(values.shape.rows, targets.size);

        debug_assert_eq!(values.shape.batch_size, targets.batch_size());

        values.apply(|x| x.exp());

        for batch_index in 0..self.shape.batch_size
        {
            let batch_range = values.shape.batch_range(batch_index);

            let s = values.values[batch_range.clone()].iter().copied().sum::<f32>();

            debug_assert!(s.classify() != FpCategory::Zero);
            debug_assert!(s.classify() != FpCategory::Infinite);

            values.batch_slice_mut(batch_index).mul_scalar_inplace(s.recip());

            let batch_start = batch_range.start;

            let entropy = -targets.positions[batch_index].iter().map(|position| values.values[batch_start + *position].ln()).sum::<f32>();

            self.values[batch_index] = entropy;
        }
    }

    fn apply(&mut self, f: impl Fn(f32) -> f32)
    {
        self.values.iter_mut().for_each(|x| *x = f(*x));
    }

    pub fn as_vector_mut(&mut self) -> YVectorWrapperMut<'_>
    {
        debug_assert_eq!(self.shape.columns, 1);

        YVectorWrapperMut::from_data(&mut self.values, self.shape)
    }

    fn batch_slice_mut(&mut self, batch_index: usize) -> YWrapperMut<'_>
    {
        YWrapperMut{
            values: &mut self.values[self.shape.batch_range(batch_index)],
            shape: TensorShape{batch_size: 1, ..self.shape}
        }
    }

    fn as_mat_mut(&mut self) -> MatMut<'_, f32>
    {
        MatMut::from_column_major(self.values, self.shape.rows, self.shape.columns).expect("dimensions must match")
    }

    pub fn rows(&self) -> usize
    {
        self.shape.rows
    }

    pub fn columns(&self) -> usize
    {
        self.shape.columns
    }

    pub fn shape(&self) -> TensorShape
    {
        self.shape
    }

    pub fn clone_owned(&self) -> YWrapper
    {
        YWrapper{
            shape: self.shape,
            values: self.values.to_vec().into_boxed_slice()
        }
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.values.to_vec()
    }
}

impl<'a> YVectorWrapperRef<'a>
{
    pub fn from_data(values: &'a [f32], shape: TensorShape) -> Self
    {
        debug_assert_eq!(shape.columns, 1);
        debug_assert_eq!(values.len(), shape.rows * shape.batch_size);

        Self{
            rows: shape.rows,
            batch_size: shape.batch_size,
            values
        }
    }

    pub fn from_data_with_start(data: &'a [f32], info: TensorRawDataPointer) -> Self
    {
        Self::from_data(&data[info.raw_index.0..(info.raw_index.0 + info.shape.size())], info.shape)
    }

    fn batch_slice_ref(&self, batch_index: usize) -> YVectorWrapperRef<'_>
    {
        let shape = TensorShape{rows: self.rows, columns: 1, batch_size: self.batch_size};

        YVectorWrapperRef{
            values: &self.values[shape.batch_range(batch_index)],
            rows: self.rows,
            batch_size: 1
        }
    }

    pub fn len(&self) -> usize
    {
        self.values.len()
    }
}

impl<'a> YVectorWrapperMut<'a>
{
    pub fn from_data(values: &'a mut [f32], shape: TensorShape) -> Self
    {
        debug_assert_eq!(shape.columns, 1);
        debug_assert_eq!(values.len(), shape.rows * shape.batch_size);

        Self{
            rows: shape.rows,
            batch_size: shape.batch_size,
            values
        }
    }

    pub fn from_data_with_start(data: &'a mut [f32], info: TensorRawDataPointer) -> Self
    {
        Self::from_data(&mut data[info.raw_index.0..(info.raw_index.0 + info.shape.size())], info.shape)
    }

    pub fn matmulv_transposed_into(mut self, lhs: YWrapperRef, rhs: YVectorWrapperRef)
    {
        fn inner_single_batch(output: YVectorWrapperMut, lhs: YWrapperRef, rhs: YVectorWrapperRef)
        {
            debug_assert_eq!(output.batch_size, 1);
            debug_assert_eq!(lhs.shape.batch_size, 1);
            debug_assert_eq!(rhs.batch_size, 1);

            let rows = lhs.shape.rows;
            let columns = lhs.shape.columns;

            for i in 0..columns
            {
                let lhs_column_start = i * rows;
                let lhs_column = unsafe{ lhs.values.get_unchecked(lhs_column_start..(lhs_column_start + rows)) };

                *(unsafe{ output.values.get_unchecked_mut(i) }) = dot(lhs_column, rhs.values);
            }
        }

        debug_assert_eq!(self.rows, lhs.shape.columns);
        debug_assert_eq!(lhs.shape.rows, rhs.rows);

        if self.batch_size != lhs.shape.batch_size
        {
            debug_assert_eq!(lhs.shape.batch_size, 1);
            debug_assert_eq!(self.batch_size, rhs.batch_size);

            for batch_index in 0..self.batch_size
            {
                inner_single_batch(self.batch_slice_mut(batch_index), lhs, rhs.batch_slice_ref(batch_index));
            }

            return;
        }

        debug_assert_eq!(self.batch_size, rhs.batch_size);

        inner_single_batch(self, lhs, rhs);
    }

    pub fn matmulv_transposed_add_inplace(mut self, lhs: YWrapperRef, rhs: YVectorWrapperRef)
    {
        fn inner_single_batch(output: YVectorWrapperMut, lhs: YWrapperRef, rhs: YVectorWrapperRef)
        {
            let rows = lhs.shape.rows;
            let columns = lhs.shape.columns;

            for i in 0..columns
            {
                let lhs_column_start = i * rows;
                let lhs_column = unsafe{ lhs.values.get_unchecked(lhs_column_start..(lhs_column_start + rows)) };

                *(unsafe{ output.values.get_unchecked_mut(i) }) += dot(lhs_column, rhs.values);
            }
        }

        debug_assert_eq!(self.rows, lhs.shape.columns);
        debug_assert_eq!(lhs.shape.rows, rhs.rows);

        if self.batch_size != lhs.shape.batch_size
        {
            debug_assert_eq!(lhs.shape.batch_size, 1);
            debug_assert_eq!(self.batch_size, rhs.batch_size);

            for batch_index in 0..self.batch_size
            {
                inner_single_batch(self.batch_slice_mut(batch_index), lhs, rhs.batch_slice_ref(batch_index));
            }

            return;
        }

        debug_assert_eq!(self.batch_size, rhs.batch_size);

        inner_single_batch(self, lhs, rhs);
    }

    pub fn matmulv_into(mut self, lhs: YWrapperRef, rhs: YVectorWrapperRef)
    {
        fn inner_single_batch(output: YVectorWrapperMut, lhs: YWrapperRef, rhs: YVectorWrapperRef)
        {
            debug_assert_eq!(output.batch_size, 1);
            debug_assert_eq!(lhs.shape.batch_size, 1);
            debug_assert_eq!(rhs.batch_size, 1);

            let rows = lhs.shape.rows;
            let columns = lhs.shape.columns;

            output.values.fill(0.0);

            for i in 0..columns
            {
                let lhs_column_start = i * rows;
                let lhs_column = unsafe{ lhs.values.get_unchecked(lhs_column_start..(lhs_column_start + rows)) };

                oxiblas_blas::level1::axpy_f32(unsafe{ *rhs.values.get_unchecked(i) }, lhs_column, output.values);
            }
        }

        debug_assert_eq!(self.rows, lhs.shape.rows);
        debug_assert_eq!(lhs.shape.columns, rhs.rows);

        debug_assert_eq!(self.batch_size, rhs.batch_size);

        if self.batch_size != lhs.shape.batch_size
        {
            debug_assert_eq!(lhs.shape.batch_size, 1);
            debug_assert_eq!(self.batch_size, rhs.batch_size);

            for batch_index in 0..self.batch_size
            {
                inner_single_batch(self.batch_slice_mut(batch_index), lhs, rhs.batch_slice_ref(batch_index));
            }

            return;
        }

        debug_assert_eq!(self.batch_size, 1);

        inner_single_batch(self, lhs, rhs);
    }

    pub fn matmulv_add_into(mut self, lhs: YWrapperRef, rhs: YVectorWrapperRef, added: YVectorWrapperRef)
    {
        fn inner_single_batch(output: YVectorWrapperMut, lhs: YWrapperRef, rhs: YVectorWrapperRef, added: YVectorWrapperRef)
        {
            debug_assert_eq!(output.batch_size, 1);
            debug_assert_eq!(lhs.shape.batch_size, 1);
            debug_assert_eq!(rhs.batch_size, 1);
            debug_assert_eq!(added.batch_size, 1);

            let rows = lhs.shape.rows;
            let columns = lhs.shape.columns;

            output.values.copy_from_slice(added.values);

            for i in 0..columns
            {
                let lhs_column_start = i * rows;
                let lhs_column = unsafe{ lhs.values.get_unchecked(lhs_column_start..(lhs_column_start + rows)) };

                oxiblas_blas::level1::axpy_f32(unsafe{ *rhs.values.get_unchecked(i) }, lhs_column, output.values);
            }
        }

        debug_assert_eq!(self.rows, lhs.shape.rows);
        debug_assert_eq!(lhs.shape.columns, rhs.rows);
        debug_assert_eq!(self.rows, added.rows);

        if self.batch_size != lhs.shape.batch_size
        {
            debug_assert_eq!(self.batch_size, rhs.batch_size);
            debug_assert_eq!(self.batch_size, added.batch_size);

            debug_assert_eq!(lhs.shape.batch_size, 1);

            for batch_index in 0..self.batch_size
            {
                inner_single_batch(self.batch_slice_mut(batch_index), lhs, rhs.batch_slice_ref(batch_index), added.batch_slice_ref(batch_index));
            }

            return;
        }

        debug_assert_eq!(self.batch_size, rhs.batch_size);

        debug_assert_eq!(self.batch_size, 1);

        inner_single_batch(self, lhs, rhs, added);
    }

    pub fn matmul_onehotv_add_into(mut self, lhs: YWrapperRef, rhs: &OneHotLayer, added: YVectorWrapperRef)
    {
        fn inner_single_batch(output: YVectorWrapperMut, lhs: YWrapperRef, rhs: &[usize], added: YVectorWrapperRef)
        {
            debug_assert_eq!(output.batch_size, 1);
            debug_assert_eq!(lhs.shape.batch_size, 1);
            debug_assert_eq!(added.batch_size, 1);

            let o_size = output.len();

            (0..o_size).for_each(|r|
            {
                output.values[r] = added.values[r];

                rhs.iter().for_each(|m|
                {
                    output.values[r] += lhs.values[m * o_size + r];
                });
            });
        }

        debug_assert_eq!(self.rows, lhs.shape.rows);
        debug_assert_eq!(lhs.shape.columns, rhs.size);
        debug_assert_eq!(self.rows, added.rows);

        if self.batch_size != lhs.shape.batch_size
        {
            debug_assert_eq!(lhs.shape.batch_size, 1);
            debug_assert_eq!(added.batch_size, 1);

            debug_assert_eq!(self.batch_size, rhs.batch_size());

            for batch_index in 0..self.batch_size
            {
                inner_single_batch(self.batch_slice_mut(batch_index), lhs, &rhs.positions[batch_index], added);
            }

            return;
        }

        debug_assert_eq!(rhs.batch_size(), 1);

        inner_single_batch(self, lhs, &rhs.positions[0], added);
    }

    fn batch_slice_mut(&mut self, batch_index: usize) -> YVectorWrapperMut<'_>
    {
        let shape = TensorShape{rows: self.rows, columns: 1, batch_size: self.batch_size};

        YVectorWrapperMut{
            values: &mut self.values[shape.batch_range(batch_index)],
            rows: self.rows,
            batch_size: 1
        }
    }

    pub fn len(&self) -> usize
    {
        self.values.len()
    }
}
