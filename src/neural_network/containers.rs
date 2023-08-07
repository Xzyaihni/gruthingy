use std::{
    f32,
    vec,
    borrow::Borrow,
    ops::{Add, Sub, Mul, Div, AddAssign, DivAssign}
};

use nalgebra::DMatrix;

use arrayfire::{Array, MatProp, dim4};

use serde::{Serialize, Deserialize};


#[derive(Debug, Serialize, Deserialize)]
pub struct SoftmaxedLayer<T>(pub T);

impl<T> SoftmaxedLayer<T>
where
    T: NetworkType,
    for<'a> &'a T: Mul<f32, Output=T> + Mul<&'a T, Output=T> + Mul<T, Output=T>,
    for<'a> &'a T: Div<f32, Output=T>,
    for<'a> &'a T: Sub<&'a T, Output=T>
{
    #[allow(dead_code)]
    pub fn new(mut layer: T) -> Self
    {
        Self::softmax(&mut layer);
        Self(layer)
    }

    pub fn softmax(layer: &mut T) 
    {
        layer.exp();
        let s = layer.sum();

        *layer /= s;
    }

    #[allow(dead_code)]
    pub fn from_raw(layer: T) -> Self
    {
        Self(layer)
    }

    #[allow(dead_code)]
    pub fn new_empty(size: usize) -> Self
    {
        Self(T::new(size, 1))
    }

    #[allow(dead_code)]
    pub fn pick_weighed(&self, temperature: f32) -> usize
    {
        Self::pick_weighed_associated(&self.0, temperature)
    }

    pub fn pick_weighed_associated(values: &T, temperature: f32) -> usize
    {
        let values = values / temperature;

        values.pick_weighed()
    }

    pub fn pick_weighed_inner<'b, I>(mut iter: I) -> usize
    where
        I: Iterator<Item=&'b f32> + ExactSizeIterator
    {
        let mut c = fastrand::f32();

        let max_index = iter.len() - 1;

        iter.position(|v|
        {
            c -= v;

            c <= 0.0
        }).unwrap_or(max_index)
    }

    pub fn highest_index<'b, I>(iter: I) -> usize
    where
        I: Iterator<Item=&'b f32>
    {
        iter.enumerate().max_by(|a, b|
        {
            a.1.partial_cmp(b.1).unwrap()
        }).unwrap().0
    }
}

#[derive(Clone, Copy)]
pub struct WeightsIterValue<T>
{
    pub previous: usize,
    pub this: usize,
    pub value: T
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenericContainer<T=f32>
{
    values: Box<[T]>,
    previous_size: usize,
    this_size: usize
}

impl<T> GenericContainer<T>
{
    pub fn new_from(&self, values: &Array<f32>) -> GenericContainer<f32>
    {
        debug_assert!(self.previous_size == values.dims()[0] as usize);
        debug_assert!(self.this_size == values.dims()[1] as usize);
        debug_assert!(self.values.len() == values.elements());

        let mut values_host = vec![0.0_f32; values.elements()];
        values.host(&mut values_host);

        GenericContainer{
            values: values_host.into_boxed_slice(),
            previous_size: self.previous_size,
            this_size: self.this_size
        }
    }

    #[inline(always)]
    pub fn dims(&self) -> arrayfire::Dim4
    {
        dim4!(self.previous_size as u64, self.this_size as u64)
    }
}

impl<T> GenericContainer<T>
where
    T: Default
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        let values = (0..(previous_size * this_size)).map(|_| T::default()).collect();

        Self{values, previous_size, this_size}
    }
}

impl<T> GenericContainer<T>
where
    T: Copy
{
    #[allow(dead_code)]
    pub fn iter_pos(&self) -> impl Iterator<Item=WeightsIterValue<T>> + DoubleEndedIterator + '_
    {
        self.values.iter().enumerate().map(|(index, &value)|
        {
            WeightsIterValue{
                previous: index % self.previous_size,
                this: index / self.previous_size,
                value
            }
        })
    }

    #[allow(dead_code)]
    pub fn map<F>(&mut self, mut f: F)
    where
        F: FnMut(T) -> T
    {
        self.values.iter_mut().for_each(|v| *v = f(*v));
    }
}

impl<T> GenericContainer<T>
{
    pub fn new_with<F>(previous_size: usize, this_size: usize, mut f: F) -> Self
    where
        F: FnMut() -> T
    {
        let values = (0..(previous_size * this_size)).map(|_|
        {
            f()
        }).collect();

        Self{values, previous_size, this_size}
    }

    #[allow(dead_code)]
    pub fn from_raw<V>(values: V, previous_size: usize, this_size: usize) -> Self
    where
        V: Into<Box<[T]>>
    {
        Self{values: values.into(), previous_size, this_size}
    }
    
    #[allow(dead_code)]
    pub fn this(&self, previous: usize) -> impl Iterator<Item=&T>
    {
        (0..self.this_size).map(move |i|
        {
            self.weight(previous, i)
        })
    }

    #[allow(dead_code)]
    pub unsafe fn this_unchecked(&self, previous: usize) -> impl Iterator<Item=&T>
    {
        (0..self.this_size).map(move |i|
        {
            self.weight_unchecked(previous, i)
        })
    }

    #[allow(dead_code)]
    pub fn previous_size(&self) -> usize
    {
        self.previous_size
    }

    #[allow(dead_code)]
    pub fn this_size(&self) -> usize
    {
        self.this_size
    }

    pub fn total_len(&self) -> usize
    {
        self.values.len()
    }

    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item=&T> + ExactSizeIterator
    {
        self.values.iter()
    }

    #[allow(dead_code)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut T> + ExactSizeIterator
    {
        self.values.iter_mut()
    }

    #[inline(always)]
    fn index_of(&self, previous: usize, this: usize) -> usize
    {
        this * self.previous_size + previous
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn weight(&self, previous: usize, this: usize) -> &T
    {
        &self.values[self.index_of(previous, this)]
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn weight_unchecked(&self, previous: usize, this: usize) -> &T
    {
        debug_assert!(
            (0..self.previous_size).contains(&previous),
            "{} >= {}",
            previous,
            self.previous_size
        );

        debug_assert!((0..self.this_size).contains(&this), "{} >= {}", this, self.this_size);

        self.values.get_unchecked(self.index_of(previous, this))
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn weight_mut(&mut self, previous: usize, this: usize) -> &mut T
    {
        &mut self.values[self.index_of(previous, this)]
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn weight_unchecked_mut(&mut self, previous: usize, this: usize) -> &mut T
    {
        debug_assert!(
            (0..self.previous_size).contains(&previous),
            "{} >= {}",
            previous,
            self.previous_size
        );
        debug_assert!((0..self.this_size).contains(&this), "{} >= {}", this, self.this_size);
        self.values.get_unchecked_mut(self.index_of(previous, this))
    }
}

impl GenericContainer<f32>
{
    #[inline(always)]
    pub fn dot(self, rhs: GenericContainer) -> f32
    {
        self.values.into_iter().zip(rhs.values.into_iter()).map(|(v, rhs)| v * rhs).sum()
    }

    #[inline(always)]
    pub fn sum(&self) -> f32
    {
        self.values.into_iter().sum()
    }

    #[inline(always)]
    pub fn one_minus_this(mut self) -> Self
    {
        self.iter_mut().for_each(|v| *v = 1.0 - *v);

        self
    }

    #[inline(always)]
    pub fn matmul(&self, rhs: impl Borrow<GenericContainer>) -> GenericContainer
    {
        let rhs = rhs.borrow();

        debug_assert!(
            rhs.total_len() == self.previous_size,
            "{} != {}",
            rhs.total_len(),
            self.previous_size
        );

        Self{
            values: (0..self.this_size).map(|i|
            {
                (0..self.previous_size).map(|p|
                {
                    // no bounds checking, if i messed something up let it burn
                    unsafe{ self.weight_unchecked(p, i) * rhs.weight_unchecked(p, 0) }
                }).sum()
            }).collect(),
            previous_size: self.this_size,
            this_size: 1
        }
    }

    #[inline(always)]
    pub fn matmul_transposed(&self, rhs: impl Borrow<GenericContainer>) -> GenericContainer
    {
        let rhs = rhs.borrow();

        debug_assert!(
            rhs.total_len() == self.this_size,
            "{} != {}",
            rhs.total_len(),
            self.this_size
        );

        Self{
            values: (0..self.previous_size).map(|i|
            {
                (0..self.this_size).map(|p|
                {
                    // no bounds checking, if i messed something up let it burn
                    unsafe{ self.weight_unchecked(i, p) * rhs.weight_unchecked(p, 0) }
                }).sum()
            }).collect(),
            previous_size: self.previous_size,
            this_size: 1
        }
    }

    #[inline(always)]
    pub fn apply<F: FnMut(&f32) -> f32>(&mut self, mut f: F)
    {
        self.iter_mut().for_each(|v| *v = f(v));
    }

    #[inline(always)]
    pub fn applied<F: FnMut(&f32) -> f32>(&self, mut f: F) -> Self
    {
        Self{
            values: self.iter().map(|v| f(v)).collect(),
            ..*self
        }
    }

    // ballin
    #[inline(always)]
    pub fn add_outer_product(
        &mut self,
        lhs: impl Borrow<GenericContainer>,
        rhs: impl Borrow<GenericContainer>
    )
    {
        let lhs = lhs.borrow();
        let rhs = rhs.borrow();
        
        debug_assert!(
            (lhs.total_len() == self.this_size)
            &&
            (rhs.total_len() == self.previous_size)
        );

        let this_size = self.this_size;
        let previous_size = self.previous_size;

        let mut weights = self.iter_mut();
        for y in 0..this_size
        {
            for x in 0..previous_size
            {
                unsafe{
                    *weights.next().unwrap_unchecked() +=
                        lhs.weight_unchecked(y, 0)
                        * rhs.weight_unchecked(x, 0);
                }
            }
        }
    }
}

impl Mul<f32> for GenericContainer
{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output
    {
        let values = self.values.into_iter().map(|v|
        {
            v * rhs
        }).collect();

        GenericContainer{
            values,
            ..self
        }
    }
}

impl Mul<f32> for &GenericContainer
{
    type Output = GenericContainer;

    fn mul(self, rhs: f32) -> Self::Output
    {
        let values = self.iter().map(|v|
        {
            *v * rhs
        }).collect();

        GenericContainer{
            values,
            ..*self
        }
    }
}

impl<T> Mul<T> for GenericContainer
where
    T: Borrow<Self>
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        debug_assert!(
            self.previous_size == rhs.previous_size,
            "self.previous_size: {}, rhs.previous_size: {}",
            self.previous_size, rhs.previous_size
        );

        debug_assert!(
            self.this_size == rhs.this_size,
            "self.this_size: {}, rhs.this_size: {}",
            self.this_size, rhs.this_size
        );

        let values = self.values.into_iter().zip(rhs.iter()).map(|(v, rhs)|
        {
            v * rhs
        }).collect();

        Self{
            values,
            ..self
        }
    }
}

impl<T> Mul<T> for &GenericContainer
where
    T: Borrow<GenericContainer>
{
    type Output = GenericContainer;

    fn mul(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        debug_assert!(self.previous_size == rhs.previous_size);
        debug_assert!(self.this_size == rhs.this_size);

        let values = self.iter().zip(rhs.iter()).map(|(v, rhs)|
        {
            *v * *rhs
        }).collect();

        GenericContainer{
            values,
            ..*self
        }
    }
}

impl Div<f32> for GenericContainer
{
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output
    {
        let values = self.values.into_iter().map(|v|
        {
            v / rhs
        }).collect();

        Self{
            values,
            ..self
        }
    }
}

impl Div<f32> for &GenericContainer
{
    type Output = GenericContainer;

    fn div(self, rhs: f32) -> Self::Output
    {
        let values = self.iter().map(|v|
        {
            v / rhs
        }).collect();

        GenericContainer{
            values,
            ..*self
        }
    }
}

impl Div for GenericContainer
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output
    {
        debug_assert!(self.previous_size == rhs.previous_size);
        debug_assert!(self.this_size == rhs.this_size);

        let values = self.values.into_iter().zip(rhs.values.into_iter()).map(|(v, rhs)|
        {
            v / rhs
        }).collect();

        Self{
            values,
            ..self
        }
    }
}

impl DivAssign<f32> for GenericContainer
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.values.iter_mut().for_each(|v|
        {
            *v /= rhs;
        });
    }
}

impl<V> Sub<V> for GenericContainer
where
    V: Borrow<Self>
{
    type Output = Self;

    fn sub(self, rhs: V) -> Self::Output
    {
        let rhs = rhs.borrow();

        debug_assert!(self.previous_size == rhs.previous_size);
        debug_assert!(self.this_size == rhs.this_size);

        let values = self.values.into_iter().zip(rhs.iter()).map(|(v, rhs)|
        {
            v - rhs
        }).collect();

        Self{
            values,
            ..self
        }
    }
}

impl<V> Sub<V> for &GenericContainer
where
    V: Borrow<GenericContainer>
{
    type Output = GenericContainer;

    fn sub(self, rhs: V) -> Self::Output
    {
        let rhs = rhs.borrow();

        debug_assert!(self.previous_size == rhs.previous_size);
        debug_assert!(self.this_size == rhs.this_size);

        let values = self.values.into_iter().zip(rhs.iter()).map(|(v, rhs)|
        {
            v - rhs
        }).collect();

        GenericContainer{
            values,
            ..*self
        }
    }
}

impl<V> Add<V> for GenericContainer
where
    V: Borrow<Self>
{
    type Output = Self;

    fn add(self, rhs: V) -> Self::Output
    {
        let rhs = rhs.borrow();

        debug_assert!(
            self.previous_size == rhs.previous_size,
            "self.previous_size: {}, rhs.previous_size: {}",
            self.previous_size, rhs.previous_size
        );

        debug_assert!(
            self.this_size == rhs.this_size,
            "self.this_size: {}, rhs.this_size: {}",
            self.this_size, rhs.this_size
        );

        let values = self.values.into_iter().zip(rhs.iter()).map(|(v, rhs)|
        {
            v + rhs
        }).collect();

        Self{
            values,
            ..self
        }
    }
}

impl Add<f32> for GenericContainer
{
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output
    {
        let values = self.values.into_iter().map(|v|
        {
            v + rhs
        }).collect();

        Self{
            values,
            ..self
        }
    }
}

impl<T, R> AddAssign<R> for GenericContainer<T>
where
    T: for<'a> AddAssign<&'a T>,
    R: Borrow<Self>
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: R)
    {
        let rhs = rhs.borrow();

        debug_assert!(
            (self.previous_size == rhs.previous_size)
            &&
            (self.this_size == rhs.this_size)
        );

        self.iter_mut().zip(rhs.iter()).for_each(|(this, other)|
        {
            *this += other;
        });
    }
}

pub const LEAKY_SLOPE: f32 = 0.01;

pub trait NetworkType
where
    for<'a> Self: Sized + Serialize + Deserialize<'a> + Clone + Send + Sync,
    for<'a> Self: Mul<f32, Output=Self> + Mul<Self, Output=Self> + Mul<&'a Self, Output=Self>,
    for<'a> Self: Add<Output=Self> + Add<f32, Output=Self> + Add<&'a Self, Output=Self>,
    Self: AddAssign<Self>,
    Self: Div<Output=Self> + Div<f32, Output=Self> + DivAssign<f32>,
    for<'a> Self: Sub<&'a Self, Output=Self>,
    for<'a> &'a Self: Sub<&'a Self, Output=Self>,
    for<'a> &'a Self: Mul<f32, Output=Self> + Mul<&'a Self, Output=Self> + Mul<Self, Output=Self>,
    for<'a> &'a Self: Div<f32, Output=Self>
{
    fn new(previous_size: usize, this_size: usize) -> Self;
    fn new_with<F: FnMut() -> f32>(
        previous_size: usize,
        this_size: usize,
        f: F
    )-> Self;
    fn from_raw<V: Into<Box<[f32]>>>(values: V, previous_size: usize, this_size: usize) -> Self;
    
    fn matmul(&self, rhs: impl Borrow<Self>) -> Self;
    fn matmul_transposed(&self, rhs: impl Borrow<Self>) -> Self;
    fn add_outer_product(&mut self, lhs: impl Borrow<Self>, rhs: impl Borrow<Self>);
    fn dot(self, rhs: Self) -> f32;
    
    fn sqrt(&mut self);
    fn exp(&mut self);
    fn ln(&mut self);
    fn sigmoid(&mut self);
    fn tanh(&mut self);

    fn leaky_relu(&mut self);
    fn leaky_relu_d(&mut self);

    fn sum(&self) -> f32;
    
    fn one_minus_this(self) -> Self;

    fn total_len(&self) -> usize;

    fn as_vec(&self) -> Vec<f32>;

    fn pick_weighed(&self) -> usize;
    fn highest_index(&self) -> usize;

    fn clone_sqrt(&self) -> Self
    {
        let mut out = self.clone();
        out.sqrt();

        out
    }
}

impl NetworkType for GenericContainer
{
    fn new(previous_size: usize, this_size: usize) -> Self
    {
        GenericContainer::new(previous_size, this_size)
    }

    fn new_with<F: FnMut() -> f32>(
        previous_size: usize,
        this_size: usize,
        f: F
    )-> Self
    {
        GenericContainer::new_with(previous_size, this_size, f)
    }

    fn from_raw<V: Into<Box<[f32]>>>(values: V, previous_size: usize, this_size: usize) -> Self
    {
        GenericContainer::from_raw(values, previous_size, this_size)
    }

    fn matmul(&self, rhs: impl Borrow<Self>) -> Self
    {
        GenericContainer::matmul(self, rhs)
    }

    fn matmul_transposed(&self, rhs: impl Borrow<Self>) -> Self
    {
        GenericContainer::matmul_transposed(self, rhs)
    }

    fn add_outer_product(&mut self, lhs: impl Borrow<Self>, rhs: impl Borrow<Self>)
    {
        GenericContainer::add_outer_product(self, lhs, rhs);
    }

    fn dot(self, rhs: Self) -> f32
    {
        GenericContainer::dot(self, rhs)
    }

    fn sqrt(&mut self)
    {
        GenericContainer::apply(self, |x| x.sqrt())
    }

    fn exp(&mut self)
    {
        GenericContainer::apply(self, |x| x.exp())
    }

    fn ln(&mut self)
    {
        GenericContainer::apply(self, |x| x.ln())
    }

    fn sigmoid(&mut self)
    {
        GenericContainer::apply(self, |x| 1.0 / (1.0 + (-x).exp()))
    }

    fn tanh(&mut self)
    {
        GenericContainer::apply(self, |x| x.tanh())
    }

    fn leaky_relu(&mut self)
    {
        GenericContainer::apply(self, |x| x.max(LEAKY_SLOPE * x))
    }

    fn leaky_relu_d(&mut self)
    {
        GenericContainer::apply(self, |x|
        {
            if *x > 0.0
            {
                1.0
            } else
            {
                LEAKY_SLOPE
            }
        })
    }

    fn sum(&self) -> f32
    {
        GenericContainer::sum(self)
    }

    fn one_minus_this(self) -> Self
    {
        GenericContainer::one_minus_this(self)
    }

    fn total_len(&self) -> usize
    {
        GenericContainer::total_len(self)
    }

    fn as_vec(&self) -> Vec<f32>
    {
        self.values.clone().to_vec()
    }

    fn pick_weighed(&self) -> usize
    {
        SoftmaxedLayer::<Self>::pick_weighed_inner(self.values.iter())
    }

    fn highest_index(&self) -> usize
    {
        SoftmaxedLayer::<Self>::highest_index(self.values.iter())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixWrapper(pub DMatrix<f32>);

impl<T> Add<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output
    {
        Self(self.0 + &rhs.borrow().0)
    }
}

impl Add<f32> for MatrixWrapper
{
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output
    {
        Self(self.0.add_scalar(rhs))
    }
}

impl<T> Sub<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output
    {
        Self(self.0 - &rhs.borrow().0)
    }
}

impl<T> Sub<T> for &MatrixWrapper
where
    T: Borrow<MatrixWrapper>
{
    type Output = MatrixWrapper;

    fn sub(self, rhs: T) -> Self::Output
    {
        MatrixWrapper(&self.0 - &rhs.borrow().0)
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

impl Mul<f32> for MatrixWrapper
{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output
    {
        Self(self.0 * rhs)
    }
}

impl Mul<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn mul(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper(&self.0 * rhs)
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

impl Div<f32> for MatrixWrapper
{
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output
    {
        Self(self.0 / rhs)
    }
}

impl Div<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn div(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper(&self.0 / rhs)
    }
}

impl AddAssign for MatrixWrapper
{
    fn add_assign(&mut self, rhs: Self)
    {
        self.0 += rhs.0;
    }
}

impl DivAssign<f32> for MatrixWrapper
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.0 /= rhs;
    }
}

impl NetworkType for MatrixWrapper
{
    fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self(DMatrix::zeros(previous_size, this_size))
    }

    fn new_with<F: FnMut() -> f32>(
        previous_size: usize,
        this_size: usize,
        mut f: F
    )-> Self
    {
        Self(DMatrix::from_fn(previous_size, this_size, |_, _| f()))
    }

    fn from_raw<V: Into<Box<[f32]>>>(values: V, previous_size: usize, this_size: usize) -> Self
    {
        Self(DMatrix::from_vec(previous_size, this_size, values.into().to_vec()))
    }

    fn matmul(&self, rhs: impl Borrow<Self>) -> Self
    {
        Self(self.0.tr_mul(&rhs.borrow().0))
    }

    fn matmul_transposed(&self, rhs: impl Borrow<Self>) -> Self
    {
        Self(&self.0 * &rhs.borrow().0)
    }

    fn add_outer_product(&mut self, lhs: impl Borrow<Self>, rhs: impl Borrow<Self>)
    {
        let transposed_lhs = lhs.borrow().0.transpose();

        self.0 += &rhs.borrow().0 * transposed_lhs;
    }

    fn dot(self, rhs: Self) -> f32
    {
        self.0.dot(&rhs.0)
    }

    fn sqrt(&mut self)
    {
        self.0.apply(|v| *v = v.sqrt());
    }

    fn exp(&mut self)
    {
        self.0.apply(|v| *v = v.exp());
    }

    fn ln(&mut self)
    {
        self.0.apply(|v| *v = v.ln());
    }

    fn sigmoid(&mut self)
    {
        self.0.apply(|v| *v = 1.0 / (1.0 + (-*v).exp()));
    }

    fn tanh(&mut self)
    {
        self.0.apply(|v| *v = v.tanh());
    }

    fn leaky_relu(&mut self)
    {
        self.0.apply(|v| *v = v.max(LEAKY_SLOPE * *v));
    }

    fn leaky_relu_d(&mut self)
    {
        self.0.apply(|v| 
        {
            *v = if *v > 0.0
            {
                1.0
            } else
            {
                LEAKY_SLOPE
            };
        })
    }

    fn sum(&self) -> f32
    {
        self.0.sum()
    }

    fn one_minus_this(self) -> Self
    {
        Self((-self.0).add_scalar(1.0))
    }

    fn total_len(&self) -> usize
    {
        self.0.as_slice().len()
    }

    fn as_vec(&self) -> Vec<f32>
    {
        self.0.as_slice().to_vec()
    }

    fn pick_weighed(&self) -> usize
    {
        SoftmaxedLayer::<Self>::pick_weighed_inner(self.0.as_slice().iter())
    }

    fn highest_index(&self) -> usize
    {
        SoftmaxedLayer::<Self>::highest_index(self.0.as_slice().iter())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ArrayWrapper(pub Array<f32>);

impl<T> Add<T> for ArrayWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output
    {
        Self(self.0 + &rhs.borrow().0)
    }
}

impl Add<f32> for ArrayWrapper
{
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output
    {
        Self(self.0 + rhs)
    }
}

impl<T> Sub<T> for ArrayWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output
    {
        Self(self.0 - &rhs.borrow().0)
    }
}

impl<T> Sub<T> for &ArrayWrapper
where
    T: Borrow<ArrayWrapper>
{
    type Output = ArrayWrapper;

    fn sub(self, rhs: T) -> Self::Output
    {
        ArrayWrapper(&self.0 - &rhs.borrow().0)
    }
}

impl<T> Mul<T> for ArrayWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output
    {
        Self(self.0 * &rhs.borrow().0)
    }
}

impl<T> Mul<T> for &ArrayWrapper
where
    T: Borrow<ArrayWrapper>
{
    type Output = ArrayWrapper;

    fn mul(self, rhs: T) -> Self::Output
    {
        ArrayWrapper(&self.0 * &rhs.borrow().0)
    }
}

impl Mul<f32> for ArrayWrapper
{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output
    {
        Self(self.0 * rhs)
    }
}

impl Mul<f32> for &ArrayWrapper
{
    type Output = ArrayWrapper;

    fn mul(self, rhs: f32) -> Self::Output
    {
        ArrayWrapper(&self.0 * rhs)
    }
}

impl<T> Div<T> for ArrayWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output
    {
        Self(self.0 / &rhs.borrow().0)
    }
}

impl Div<f32> for ArrayWrapper
{
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output
    {
        Self(self.0 / rhs)
    }
}

impl Div<f32> for &ArrayWrapper
{
    type Output = ArrayWrapper;

    fn div(self, rhs: f32) -> Self::Output
    {
        ArrayWrapper(&self.0 / rhs)
    }
}

impl AddAssign for ArrayWrapper
{
    fn add_assign(&mut self, rhs: Self)
    {
        self.0 += rhs.0;
    }
}

impl DivAssign<f32> for ArrayWrapper
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.0 = &self.0 / rhs;
    }
}

impl NetworkType for ArrayWrapper
{
    fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self(arrayfire::constant(0.0, dim4!(previous_size as u64, this_size as u64)))
    }

    fn new_with<F: FnMut() -> f32>(
        previous_size: usize,
        this_size: usize,
        mut f: F
    )-> Self
    {
        let s = (0..(previous_size * this_size)).map(|_| f()).collect::<Vec<_>>();
        Self(Array::new(&s, dim4!(previous_size as u64, this_size as u64)))
    }

    fn from_raw<V: Into<Box<[f32]>>>(values: V, previous_size: usize, this_size: usize) -> Self
    {
        Self(Array::new(&values.into(), dim4!(previous_size as u64, this_size as u64)))
    }

    fn matmul(&self, rhs: impl Borrow<Self>) -> Self
    {
        Self(arrayfire::matmul(
            &self.0,
            &rhs.borrow().0,
            MatProp::TRANS,
            MatProp::NONE
        ))
    }

    fn matmul_transposed(&self, rhs: impl Borrow<Self>) -> Self
    {
        Self(arrayfire::matmul(
            &self.0,
            &rhs.borrow().0,
            MatProp::NONE,
            MatProp::NONE
        ))
    }

    fn add_outer_product(&mut self, lhs: impl Borrow<Self>, rhs: impl Borrow<Self>)
    {
        self.0 += arrayfire::matmul(
            &rhs.borrow().0,
            &lhs.borrow().0,
            MatProp::NONE,
            MatProp::TRANS
        );
    }

    fn dot(self, rhs: Self) -> f32
    {
        let d = arrayfire::dot(&self.0, &rhs.0, MatProp::NONE, MatProp::NONE);

        let mut out = [0.0_f32];
        d.host(&mut out);

        out[0]
    }

    fn sqrt(&mut self)
    {
        self.0 = arrayfire::sqrt(&self.0);
    }

    fn exp(&mut self)
    {
        self.0 = arrayfire::exp(&self.0);
    }

    fn ln(&mut self)
    {
        self.0 = arrayfire::log(&self.0);
    }

    fn sigmoid(&mut self)
    {
        self.0 = arrayfire::sigmoid(&self.0);
    }

    fn tanh(&mut self)
    {
        self.0 = arrayfire::tanh(&self.0);
    }

    fn leaky_relu(&mut self)
    {
        self.0 = arrayfire::maxof(&self.0, &(&self.0 * LEAKY_SLOPE), false);
    }

    fn leaky_relu_d(&mut self)
    {
        let ones = arrayfire::constant(1.0_f32, self.0.dims());
        let cond = arrayfire::gt(&self.0, &0.0_f32, true);
        let leakys = arrayfire::constant(LEAKY_SLOPE, self.0.dims());

        self.0 = arrayfire::select(&ones, &cond, &leakys);
    }

    fn sum(&self) -> f32
    {
        arrayfire::sum_all(&self.0).0
    }

    fn one_minus_this(self) -> Self
    {
        Self(1.0_f32 - self.0)
    }

    fn total_len(&self) -> usize
    {
        self.0.elements()
    }

    fn as_vec(&self) -> Vec<f32>
    {
        let mut out = vec![0.0_f32; self.0.elements()];
        self.0.host(&mut out);

        out
    }

    fn pick_weighed(&self) -> usize
    {
        SoftmaxedLayer::<Self>::pick_weighed_inner(self.as_vec().iter())
    }

    fn highest_index(&self) -> usize
    {
        SoftmaxedLayer::<Self>::highest_index(self.as_vec().iter())
    }
}
