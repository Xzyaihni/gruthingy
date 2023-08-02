use std::{
    f32,
    vec,
    borrow::Borrow,
    ops::{Add, Sub, Mul, Div, AddAssign, DivAssign}
};

use arrayfire::{Array, dim4};

use serde::{Serialize, Deserialize};


#[derive(Debug, Serialize, Deserialize)]
pub struct SoftmaxedLayer<T>(pub T);

impl<T> SoftmaxedLayer<T>
where
    T: NetworkType,
    for<'a> &'a T: Mul<f32, Output=T> + Mul<&'a T, Output=T> + Mul<T, Output=T>,
    for<'a> &'a T: Div<f32, Output=T>
{
    #[allow(dead_code)]
    pub fn new(layer: T) -> Self
    {
        Self(Self::softmax(layer))
    }

    pub fn softmax(layer: T) -> T
    {
        let s = layer.exp().sum();

        layer.exp() / s
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
    pub fn iter_pos(&self) -> impl Iterator<Item=WeightsIterValue<T>> + '_
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

        debug_assert!(self.previous_size == rhs.previous_size);
        debug_assert!(self.this_size == rhs.this_size);

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

impl<V> Add<V> for GenericContainer
where
    V: Borrow<Self>
{
    type Output = Self;

    fn add(self, rhs: V) -> Self::Output
    {
        let rhs = rhs.borrow();

        debug_assert!(self.previous_size == rhs.previous_size);
        debug_assert!(self.this_size == rhs.this_size);

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

pub trait NetworkType
where
    for<'a> Self: Sized + Serialize + Deserialize<'a> + Clone,
    for<'a> Self: Mul<f32, Output=Self> + Mul<Self, Output=Self> + Mul<&'a Self, Output=Self>,
    for<'a> Self: Add<Output=Self> + Add<f32, Output=Self> + Add<&'a Self, Output=Self>,
    Self: AddAssign<Self>,
    Self: Div<Output=Self> + Div<f32, Output=Self> + DivAssign<f32>,
    for<'a> Self: Sub<&'a Self, Output=Self>,
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
    
    fn sqrt(&self) -> Self;
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn sigmoid(&self) -> Self;
    fn tanh(&self) -> Self;

    fn sum(&self) -> f32;
    
    fn one_minus_this(self) -> Self;

    fn total_len(&self) -> usize;
    fn previous_size(&self) -> usize;
    fn this_size(&self) -> usize;

    fn as_slice(&self) -> &[f32];

    fn pick_weighed(&self) -> usize
    {
        let mut c = fastrand::f32();

        let max_index = self.total_len() - 1;

        self.as_slice().iter().position(|v|
        {
            c -= v;

            c <= 0.0
        }).unwrap_or(max_index)
    }

    fn highest_index(&self) -> usize
    {
        self.as_slice().iter().enumerate().max_by(|a, b|
        {
            a.1.partial_cmp(b.1).unwrap()
        }).unwrap().0
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

    fn sqrt(&self) -> Self
    {
        GenericContainer::applied(self, |x| x.sqrt())
    }

    fn exp(&self) -> Self
    {
        GenericContainer::applied(self, |x| x.exp())
    }

    fn ln(&self) -> Self
    {
        GenericContainer::applied(self, |x| x.ln())
    }

    fn sigmoid(&self) -> Self
    {
        GenericContainer::applied(self, |x| 1.0 / (1.0 + (-x).exp()))
    }

    fn tanh(&self) -> Self
    {
        GenericContainer::applied(self, |x| x.tanh())
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

    fn previous_size(&self) -> usize
    {
        GenericContainer::previous_size(self)
    }

    fn this_size(&self) -> usize
    {
        GenericContainer::this_size(self)
    }

    fn as_slice(&self) -> &[f32]
    {
        &self.values
    }
}
