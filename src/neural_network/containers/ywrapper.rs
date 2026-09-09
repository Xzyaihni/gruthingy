use std::{
    f32,
    iter,
    num::FpCategory
};

use serde::{Serialize, Deserialize};

use super::{
    Softmaxer,
    Softmaxable,
    OneHotLayer,
    TensorRawDataPointer,
    TensorIndexRaw,
    LEAKY_SLOPE,
    leaky_relu_d
};


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YWrapper
{
    rows: usize,
    columns: usize,
    values: Vec<f32>
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YWrapperRef<'a>
{
    rows: usize,
    columns: usize,
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
    rows: usize,
    columns: usize,
    values: &'a mut [f32]
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YVectorWrapperRef<'a>(&'a [f32]);

#[derive(Debug, PartialEq)]
pub struct YVectorWrapperMut<'a>(&'a mut [f32]);

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

#[allow(dead_code)]
impl YWrapper
{
    pub fn new(rows: usize, columns: usize) -> Self
    {
        Self::repeat(rows, columns, 0.0)
    }

    pub fn repeat(rows: usize, columns: usize, value: f32) -> Self
    {
        Self::from_raw(vec![value; rows * columns], rows, columns)
    }

    pub fn new_with(rows: usize, columns: usize, f: impl Fn() -> f32) -> Self
    {
        Self::from_raw(iter::repeat_with(f).take(rows * columns).collect::<Vec<f32>>(), rows, columns)
    }

    pub fn from_raw(values: impl Into<Vec<f32>>, rows: usize, columns: usize) -> Self
    {
        let values = values.into();

        debug_assert_eq!(values.len(), rows * columns);

        Self{
            rows,
            columns,
            values
        }
    }

    pub fn as_ref(&self) -> YWrapperRef<'_>
    {
        YWrapperRef{
            rows: self.rows,
            columns: self.columns,
            values: &self.values
        }
    }

    pub fn as_mut(&mut self) -> YWrapperMut<'_>
    {
        YWrapperMut{
            rows: self.rows,
            columns: self.columns,
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

    pub fn mul_componentwise(&self, other: YWrapperRef) -> Self
    {
        todo!()
    }

    pub fn add_inplace(&mut self, other: YWrapperRef)
    {
        todo!()
    }

    pub fn add(&self, other: YWrapperRef) -> Self
    {
        self.clone().zip_map(other, |a, b| a + b)
    }

    pub fn pow(&self, power: u32) -> Self
    {
        self.clone().map(|x| x.powi(power as i32))
    }

    pub fn exp_inplace(&mut self)
    {
        todo!()
    }

    pub fn sum(&self) -> f32
    {
        self.values.iter().copied().sum::<f32>()
    }

    fn zip_map(self, b: YWrapperRef, f: impl Fn(f32, f32) -> f32) -> Self
    {
        debug_assert_eq!(self.shape(), b.shape());

        Self{
            rows: self.rows,
            columns: self.columns,
            values: self.values.into_iter().zip(b.values).map(|(a, b)| f(a, *b)).collect()
        }
    }

    fn map(self, f: impl Fn(f32) -> f32) -> Self
    {
        Self{
            rows: self.rows,
            columns: self.columns,
            values: self.values.into_iter().map(f).collect()
        }
    }

    pub fn rows(&self) -> usize
    {
        self.rows
    }

    pub fn columns(&self) -> usize
    {
        self.columns
    }

    pub fn shape(&self) -> (usize, usize)
    {
        (self.rows, self.columns)
    }

    pub fn total_len(&self) -> usize
    {
        self.values.len()
    }

    pub fn as_slice(&self) -> &[f32]
    {
        &self.values
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.values.clone()
    }

    pub fn swap_raw_values<V: Into<Vec<f32>>>(&mut self, values: V)
    {
        self.values = values.into();
    }
}

impl<'a> YWrapperRef<'a>
{
    pub fn from_data(data: &'a [f32], info: TensorRawDataPointer) -> Self
    {
        Self::from_data_with_start(data, TensorRawDataPointer{raw_index: TensorIndexRaw(0), ..info})
    }

    pub fn from_data_with_start(data: &'a [f32], info: TensorRawDataPointer) -> Self
    {
        let len = info.rows * info.columns;

        Self{
            rows: info.rows,
            columns: info.columns,
            values: &data[info.raw_index.0..(info.raw_index.0 + len)]
        }
    }

    pub fn dot(self, rhs: Self) -> f32
    {
        todo!()
    }

    pub fn softmax_cross_entropy(self, targets: &OneHotLayer) -> f32
    {
        let mut cloned = self.clone_owned();

        cloned.as_mut().softmax_cross_entropy_inplace(targets)
    }

    pub fn shape(&self) -> (usize, usize)
    {
        (self.rows, self.columns)
    }

    pub fn clone_owned(&self) -> YWrapper
    {
        YWrapper{
            rows: self.rows,
            columns: self.columns,
            values: self.values.to_vec()
        }
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.values.to_vec()
    }
}

impl<'a> YWrapperMut<'a>
{
    pub fn from_data(data: &'a mut [f32], info: TensorRawDataPointer) -> Self
    {
        Self::from_data_with_start(data, TensorRawDataPointer{raw_index: TensorIndexRaw(0), ..info})
    }

    pub fn from_data_with_start(data: &'a mut [f32], info: TensorRawDataPointer) -> Self
    {
        let len = info.rows * info.columns;

        Self{
            rows: info.rows,
            columns: info.columns,
            values: &mut data[info.raw_index.0..(info.raw_index.0 + len)]
        }
    }

    pub fn copy_from(self, value: YWrapperRef)
    {
        debug_assert_eq!(self.values.len(), value.values.len());
        debug_assert_eq!(self.shape(), value.shape());

        self.values.copy_from_slice(value.values)
    }

    pub fn add_to(self, lhs: YWrapperRef, rhs: YWrapperRef)
    {
        (0..self.values.len()).for_each(|i| self.values[i] = lhs.values[i] + rhs.values[i]);
    }

    pub fn sub_to(&mut self, lhs: YWrapperRef, rhs: YWrapperRef)
    {
        (0..self.values.len()).for_each(|i| self.values[i] = lhs.values[i] - rhs.values[i]);
    }

    pub fn sub_from_scalar(self, lhs: f32, rhs: YWrapperRef)
    {
        (0..self.values.len()).for_each(|i| self.values[i] = lhs - rhs.values[i]);
    }

    pub fn add_scalar(mut self, other: f32) -> Self
    {
        todo!()
    }

    pub fn mul_scalar_inplace(mut self, value: f32)
    {
        todo!()
    }

    pub fn pow_inplace(mut self, power: u32)
    {
        todo!()
    }

    pub fn tanh_inplace(mut self)
    {
        todo!()
    }

    pub fn tanh_gradient_inplace(mut self, value: YWrapperRef, gradient: YWrapperRef)
    {
        todo!()
    }

    pub fn sigmoid_inplace(mut self)
    {
        todo!()
    }

    pub fn sigmoid_gradient_inplace(mut self, value: YWrapperRef, gradient: YWrapperRef)
    {
        todo!()
    }

    pub fn leaky_relu_inplace(mut self)
    {
        todo!()
    }

    pub fn leaky_relu_gradient_inplace(mut self, value: YWrapperRef, gradient: YWrapperRef)
    {
        todo!()
    }

    pub fn component_mul_into(mut self, lhs: YWrapperRef, rhs: YWrapperRef)
    {
        todo!()
    }

    pub fn component_mul_add_into(mut self, lhs: YWrapperRef, rhs: YWrapperRef, added: YWrapperRef)
    {
        todo!()
    }

    pub fn matmulv_add_into(mut self, lhs: YWrapperRef, rhs: YWrapperRef, added: YWrapperRef)
    {
        todo!()
    }

    pub fn matmul_onehotv_add_into(mut self, lhs: YWrapperRef, rhs: &OneHotLayer, added: YWrapperRef)
    {
        todo!()
    }

    pub fn matmulv_transposed_into(mut self, lhs: YWrapperRef, rhs: YWrapperRef)
    {
        todo!()
    }

    pub fn outer_product_into(mut self, lhs: YWrapperRef, rhs: YWrapperRef)
    {
        todo!()
    }

    pub fn outer_product_one_hot_into(mut self, lhs: YWrapperRef, rhs: &OneHotLayer)
    {
        todo!()
    }

    pub fn softmax_cross_entropy_inplace(mut self, targets: &OneHotLayer) -> f32
    {
        todo!()
    }

    pub fn shape(&self) -> (usize, usize)
    {
        (self.rows, self.columns)
    }

    pub fn clone_owned(&self) -> YWrapper
    {
        YWrapper{
            rows: self.rows,
            columns: self.columns,
            values: self.values.to_vec()
        }
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.values.to_vec()
    }
}

impl<'a> YVectorWrapperRef<'a>
{
    pub fn from_data(data: &'a [f32], info: TensorRawDataPointer) -> Self
    {
        Self::from_data_with_start(data, TensorRawDataPointer{raw_index: TensorIndexRaw(0), ..info})
    }

    pub fn from_data_with_start(data: &'a [f32], info: TensorRawDataPointer) -> Self
    {
        let len = info.rows * info.columns;

        Self(&data[info.raw_index.0..(info.raw_index.0 + len)])
    }
}

impl<'a> YVectorWrapperMut<'a>
{
    pub fn from_data(data: &'a mut [f32], info: TensorRawDataPointer) -> Self
    {
        Self::from_data_with_start(data, TensorRawDataPointer{raw_index: TensorIndexRaw(0), ..info})
    }

    pub fn from_data_with_start(data: &'a mut [f32], info: TensorRawDataPointer) -> Self
    {
        let len = info.rows * info.columns;

        Self(&mut data[info.raw_index.0..(info.raw_index.0 + len)])
    }

    pub fn matmulv_into(mut self, lhs: YWrapperRef, rhs: YVectorWrapperRef)
    {
        todo!()
    }
}
