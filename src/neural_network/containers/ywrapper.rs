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

    pub fn cosine_similarity(&self, other: YWrapperRef) -> f32
    {
        let top = self.as_ref().dot(other);

        let bottom = self.magnitude() * other.magnitude();

        top / bottom
    }

    pub fn cap_magnitude_inplace(&mut self, cap: f32)
    {
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

    pub fn swap_raw_values<V: Into<Vec<f32>>>(&mut self, values: V)
    {
        self.values = values.into();
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.values.clone()
    }

    pub fn iter(&self) -> impl Iterator<Item=&f32> + ExactSizeIterator
    {
        self.values.iter()
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

#[allow(dead_code)]
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

    pub fn matmul_onehotv_add(self, rhs: &OneHotLayer, added: YVectorWrapperRef) -> YWrapper
    {
        let mut output = YWrapper::new(self.rows, 1);

        output.as_mut().as_vector_mut().matmul_onehotv_add_into(self, rhs, added);

        output
    }

    pub fn dot(self, rhs: Self) -> f32
    {
        self.values.iter().zip(rhs.values).map(|(a, b)| *a * *b).sum::<f32>()
    }

    pub fn softmax_cross_entropy(self, targets: &OneHotLayer) -> f32
    {
        let mut cloned = self.clone_owned();

        cloned.as_mut().softmax_cross_entropy_inplace(targets)
    }

    pub fn magnitude(&self) -> f32
    {
        self.values.iter().map(|x| *x * *x).sum::<f32>().sqrt()
    }

    pub fn as_vector_ref(&self) -> YVectorWrapperRef<'_>
    {
        debug_assert_eq!(self.columns, 1);

        YVectorWrapperRef::from_data(&self.values, TensorRawDataPointer{
            raw_index: TensorIndexRaw(0),
            rows: self.rows,
            columns: 1
        })
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

#[allow(dead_code)]
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

    pub fn sub_inplace(self, rhs: YWrapperRef)
    {
        (0..self.values.len()).for_each(|i| self.values[i] = self.values[i] - rhs.values[i]);
    }

    pub fn add_inplace(self, rhs: YWrapperRef)
    {
        (0..self.values.len()).for_each(|i| self.values[i] = self.values[i] + rhs.values[i]);
    }

    pub fn add_scalar_inplace(mut self, other: f32)
    {
        self.apply(|x| x + other)
    }

    pub fn mul_scalar_inplace(&mut self, value: f32)
    {
        self.apply(|x| x * value)
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
        (0..self.values.len()).for_each(|i| self.values[i] = leaky_relu_d(value.values[i]) * gradient.values[i])
    }

    pub fn component_mul_into(self, lhs: YWrapperRef, rhs: YWrapperRef)
    {
        (0..self.values.len()).for_each(|i| self.values[i] = lhs.values[i] * rhs.values[i]);
    }

    pub fn component_mul_add_into(self, lhs: YWrapperRef, rhs: YWrapperRef, added: YWrapperRef)
    {
        (0..self.values.len()).for_each(|i| self.values[i] = lhs.values[i] * rhs.values[i] + added.values[i]);
    }

    pub fn outer_product_into(self, lhs: YVectorWrapperRef, rhs: YVectorWrapperRef)
    {
        debug_assert_eq!(self.rows(), lhs.len());
        debug_assert_eq!(self.columns(), rhs.len());

        let (rows, columns) = self.shape();

        (0..columns).for_each(|column|
        {
            (0..rows).for_each(|row|
            {
                self.values[column * rows + row] = lhs.0[row] * rhs.0[column];
            })
        })
    }

    pub fn outer_product_one_hot_into(self, lhs: YVectorWrapperRef, rhs: &OneHotLayer)
    {
        debug_assert_eq!(self.rows(), lhs.len());
        debug_assert_eq!(self.columns(), rhs.size);

        let rows = self.rows();

        self.values.fill(0.0);

        rhs.positions.iter().for_each(|column|
        {
            (0..rows).for_each(|row|
            {
                self.values[column * rows + row] = lhs.0[row];
            })
        })
    }

    pub fn softmax_cross_entropy_inplace(&mut self, targets: &OneHotLayer) -> f32
    {
        debug_assert_eq!(self.rows(), targets.size);

        self.apply(|x| x.exp());
        let s = self.values.iter().copied().sum::<f32>();

        debug_assert!(s.classify() != FpCategory::Zero);
        debug_assert!(s.classify() != FpCategory::Infinite);

        self.mul_scalar_inplace(s.recip());

        -targets.positions.iter().map(|position| self.values[*position].ln()).sum::<f32>()
    }

    fn apply(&mut self, f: impl Fn(f32) -> f32)
    {
        self.values.iter_mut().for_each(|x| *x = f(*x));
    }

    pub fn as_vector_mut(&mut self) -> YVectorWrapperMut<'_>
    {
        debug_assert_eq!(self.columns, 1);

        YVectorWrapperMut::from_data(&mut self.values, TensorRawDataPointer{
            raw_index: TensorIndexRaw(0),
            rows: self.rows,
            columns: 1
        })
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

    pub fn len(&self) -> usize
    {
        self.0.len()
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

    pub fn matmulv_transposed_into(self, lhs: YWrapperRef, rhs: YVectorWrapperRef)
    {
        debug_assert_eq!(self.len(), lhs.columns());
        debug_assert_eq!(lhs.rows(), rhs.len());

        let (rows, columns) = lhs.shape();

        (0..columns).for_each(|column|
        {
            self.0[column] = 0.0;

            (0..rows).for_each(|row|
            {
                self.0[column] += lhs.values[column * rows + row] * rhs.0[row];
            })
        })
    }

    pub fn matmulv_into(self, lhs: YWrapperRef, rhs: YVectorWrapperRef)
    {
        debug_assert_eq!(self.len(), lhs.rows());
        debug_assert_eq!(lhs.columns(), rhs.len());

        let o_size = self.len();
        let m_size = rhs.len();

        (0..o_size).for_each(|r|
        {
            self.0[r] = 0.0;

            (0..m_size).for_each(|m|
            {
                self.0[r] += lhs.values[m * o_size + r] * rhs.0[m];
            });
        })
    }

    pub fn matmulv_add_into(self, lhs: YWrapperRef, rhs: YVectorWrapperRef, added: YVectorWrapperRef)
    {
        debug_assert_eq!(self.len(), lhs.rows());
        debug_assert_eq!(lhs.columns(), rhs.len());
        debug_assert_eq!(self.len(), added.len());

        let o_size = self.len();
        let m_size = rhs.len();

        (0..o_size).for_each(|r|
        {
            self.0[r] = added.0[r];

            (0..m_size).for_each(|m|
            {
                self.0[r] += lhs.values[m * o_size + r] * rhs.0[m];
            });
        })
    }

    pub fn matmul_onehotv_add_into(self, lhs: YWrapperRef, rhs: &OneHotLayer, added: YVectorWrapperRef)
    {
        debug_assert_eq!(self.len(), lhs.rows());
        debug_assert_eq!(lhs.columns(), rhs.size);
        debug_assert_eq!(self.len(), added.len());

        let o_size = self.len();

        (0..o_size).for_each(|r|
        {
            self.0[r] = added.0[r];

            rhs.positions.iter().for_each(|m|
            {
                self.0[r] += lhs.values[m * o_size + r];
            });
        });
    }

    pub fn len(&self) -> usize
    {
        self.0.len()
    }
}
