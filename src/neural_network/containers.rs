use std::{
    f32,
    vec,
    slice,
    borrow::Borrow,
    ops::{Index, IndexMut, Add, Sub, Mul, Div, AddAssign, DivAssign}
};

use arrayfire::{Array, dim4};

use serde::{Serialize, Deserialize};


#[derive(Debug, Serialize, Deserialize)]
pub struct SoftmaxedLayer(LayerContainer);

impl SoftmaxedLayer
{
    #[allow(dead_code)]
    pub fn new(layer: LayerContainer) -> Self
    {
        Self(Self::softmax(layer))
    }

    pub fn softmax(mut layer: LayerContainer) -> LayerContainer
    {
        let s: f32 = layer.iter().map(|v|
        {
            f32::consts::E.powf(*v)
        }).sum();

        layer.map(|v|
        {
            f32::consts::E.powf(v) / s
        });

        layer
    }

    #[allow(dead_code)]
    pub fn from_raw(layer: LayerContainer) -> Self
    {
        Self(layer)
    }

    #[allow(dead_code)]
    pub fn new_empty(size: usize) -> Self
    {
        Self(LayerContainer::new(size))
    }

    #[allow(dead_code)]
    pub fn pick_weighed(&self, temperature: f32) -> usize
    {
        Self::pick_weighed_associated(&self.0, temperature)
    }

    pub fn pick_weighed_associated(values: &LayerContainer, temperature: f32) -> usize
    {
        let values = values / temperature;

        let mut c = fastrand::f32();

        let index = values.iter().position(|v|
        {
            c -= v;

            c <= 0.0
        }).unwrap_or(values.len() - 1);

        index
    }
}

pub struct SoftmaxedArray(Array<f32>);

impl SoftmaxedArray
{
    #[allow(dead_code)]
    pub fn new(layer: &Array<f32>) -> Self
    {
        Self(Self::softmax(layer))
    }

    pub fn softmax(layer: &Array<f32>) -> Array<f32>
    {
        let exp_layer = arrayfire::exp(layer);
        let s = arrayfire::sum_all(&exp_layer).0;

        exp_layer / s
    }

    #[allow(dead_code)]
    pub fn pick_weighed(&self, temperature: f32) -> usize
    {
        Self::pick_weighed_associated(&self.0, temperature)
    }

    pub fn pick_weighed_associated(layer: &Array<f32>, temperature: f32) -> usize
    {
        let values = layer / temperature;

        let mut c = fastrand::f32();

        let mut host_values = vec![0.0_f32; values.elements()];
        values.host(&mut host_values);

        let last_index = host_values.len() - 1;
        let index = host_values.into_iter().position(|v|
        {
            c -= v;

            c <= 0.0
        }).unwrap_or(last_index);

        index
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerContainer<T=f32>
{
    values: Vec<T>
}

impl<T> LayerContainer<T>
where
    T: Default + Copy
{
    pub fn new(size: usize) -> Self
    {
        let values = vec![T::default(); size];

        Self{values}
    }

    pub fn new_with<F>(size: usize, mut f: F) -> Self
    where
        F: FnMut() -> T
    {
        let values = (0..size).map(|_|
        {
            f()
        }).collect();

        Self{values}
    }
}

impl<T> LayerContainer<T>
{
    pub fn dims(&self) -> arrayfire::Dim4
    {
        dim4!(self.values.len() as u64)
    }

    pub fn iter(&self) -> impl Iterator<Item=&T>
    {
        self.values.iter()
    }

    #[allow(dead_code)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut T>
    {
        self.values.iter_mut()
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize
    {
        self.values.len()
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T
    {
        debug_assert!(
            (0..self.values.len()).contains(&index),
            "{} >= {}",
            index,
            self.values.len()
        );

        self.values.get_unchecked(index)
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T
    {
        debug_assert!(
            (0..self.values.len()).contains(&index),
            "{} >= {}",
            index,
            self.values.len()
        );

        self.values.get_unchecked_mut(index)
    }
}

impl LayerContainer
{
    pub fn as_arrayfire(&self) -> Array<f32>
    {
        Array::new(&self.values, self.dims())
    }

    pub fn highest_index(&self) -> usize
    {
        let (highest_index, _highest_value) = self.values.iter().enumerate().max_by(|x, y|
        {
            x.1.partial_cmp(y.1).unwrap()
        }).unwrap();

        highest_index
    }

    #[inline(always)]
    pub fn outer_product(&self, other: impl Borrow<Self>) -> WeightsContainer
    {
        let other = other.borrow();

        let raw_weights = (0..self.len()).flat_map(|y|
        {
            (0..other.len()).map(move |x|
            {
                unsafe{ other.get_unchecked(x) * self.get_unchecked(y) }
            })
        }).collect();

        WeightsContainer::from_raw(raw_weights, other.len(), self.len())
    }

    pub fn dot(&self, other: impl Borrow<Self>) -> f32
    {
        self.values.iter().zip(other.borrow().values.iter())
            .map(|(this, other)| this * other).sum()
    }

    #[inline(always)]
    pub fn powi(self, pow: i32) -> Self
    {
        Self{
            values: self.values.into_iter().map(|v|
            {
                v.powi(pow)
            }).collect()
        }
    }

    #[inline(always)]
    pub fn one_minus_this(mut self) -> Self
    {
        self.values.iter_mut().for_each(|v|
        {
            *v = 1.0 - *v;
        });

        self
    }

    pub fn sqrt(&self) -> Self
    {
        Self{
            values: self.iter().map(|v| v.sqrt()).collect(),
            ..*self
        }
    }

    pub fn map<F>(&mut self, mut f: F)
    where
        F: FnMut(f32) -> f32
    {
        self.values.iter_mut().for_each(|v| *v = f(*v));
    }
}

impl FromIterator<f32> for LayerContainer
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item=f32>
    {
        Self{
            values: Vec::<f32>::from_iter(iter)
        }
    }
}

impl IntoIterator for LayerContainer
{
    type Item = f32;
    type IntoIter = vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.values.into_iter()
    }
}

impl<'a> IntoIterator for &'a LayerContainer
{
    type Item = &'a f32;
    type IntoIter = slice::Iter<'a, f32>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.values.iter()
    }
}

impl From<Vec<f32>> for LayerContainer
{
    fn from(values: Vec<f32>) -> Self
    {
        Self{values}
    }
}

impl From<Array<f32>> for LayerContainer
{
    fn from(values: Array<f32>) -> Self
    {
        let mut values_host = vec![0.0_f32; values.elements()];
        values.host(&mut values_host);

        Self{values: values_host}
    }
}

impl Index<usize> for LayerContainer
{
    type Output = f32;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output
    {
        &self.values[index]
    }
}

impl IndexMut<usize> for LayerContainer
{
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output
    {
        &mut self.values[index]
    }
}

impl DivAssign<f32> for LayerContainer
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.values.iter_mut().for_each(|v|
        {
            *v /= rhs;
        });
    }
}

impl Add<f32> for LayerContainer
{
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output
    {
        Self{
            values: self.values.into_iter().map(|v|
            {
                v + rhs
            }).collect()
        }
    }
}

impl<T> Add<T> for LayerContainer
where
    T: Borrow<Self>
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output
    {
        debug_assert!(
            rhs.borrow().len() == self.len(),
            "{} != {}",
            rhs.borrow().len(),
            self.len()
        );

        Self{
            values: self.values.into_iter().zip(rhs.borrow().values.iter()).map(|(v, rv)|
            {
                v + rv
            }).collect()
        }
    }
}

impl<R> AddAssign<R> for LayerContainer
where
    R: Borrow<Self>
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: R)
    {
        let rhs = rhs.borrow();

        debug_assert!(self.len() == rhs.len());

        self.iter_mut().zip(rhs.iter()).for_each(|(this, other)|
        {
            *this = *this + other;
        });
    }
}

impl<T> Sub<T> for LayerContainer
where
    T: Borrow<Self>
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output
    {
        debug_assert!(
            rhs.borrow().len() == self.len(),
            "{} != {}",
            rhs.borrow().len(),
            self.len()
        );

        Self{
            values: self.values.into_iter().zip(rhs.borrow().values.iter()).map(|(v, rv)|
            {
                v - rv
            }).collect()
        }
    }
}

impl Div<f32> for &LayerContainer
{
    type Output = LayerContainer;

    fn div(self, rhs: f32) -> Self::Output
    {
        let values = self.into_iter().map(|v|
        {
            v / rhs
        }).collect();

        LayerContainer{
            values,
            ..*self
        }
    }
}

impl Div for LayerContainer
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output
    {
        let values = self.into_iter().zip(rhs.into_iter()).map(|(v, rhs)|
        {
            v / rhs
        }).collect();

        Self{
            values,
            ..self
        }
    }
}

impl Mul<f32> for LayerContainer
{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output
    {
        Self{
            values: self.values.into_iter().map(|v|
            {
                v * rhs
            }).collect()
        }
    }
}

impl Mul<f32> for &LayerContainer
{
    type Output = LayerContainer;

    fn mul(self, rhs: f32) -> Self::Output
    {
        LayerContainer{
            values: self.into_iter().map(|v|
            {
                v * rhs
            }).collect()
        }
    }
}

impl<T> Mul<T> for LayerContainer
where
    T: Borrow<Self>
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output
    {
        debug_assert!(
            rhs.borrow().len() == self.len(),
            "{} != {}",
            rhs.borrow().len(),
            self.len()
        );

        Self{
            values: self.into_iter().zip(rhs.borrow().into_iter()).map(|(v, rhs)|
            {
                v * rhs
            }).collect()
        }
    }
}

impl<T> Mul<T> for &LayerContainer
where
    T: Borrow<LayerContainer>
{
    type Output = LayerContainer;

    fn mul(self, rhs: T) -> Self::Output
    {
        debug_assert!(
            rhs.borrow().len() == self.len(),
            "{} != {}",
            rhs.borrow().len(),
            self.len()
        );

        LayerContainer{
            values: self.into_iter().zip(rhs.borrow().into_iter()).map(|(v, rhs)|
            {
                v * rhs
            }).collect()
        }
    }
}

#[derive(Clone, Copy)]
pub struct WeightsIterValue<T>
{
    pub previous: usize,
    pub this: usize,
    pub value: T
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WeightsContainer<T=f32>
{
    values: Box<[T]>,
    previous_size: usize,
    this_size: usize
}

impl<T> WeightsContainer<T>
{
    pub fn new_from(&self, values: &Array<f32>) -> WeightsContainer<f32>
    {
        debug_assert!(self.values.len() == values.elements());

        let mut values_host = vec![0.0_f32; values.elements()];
        values.host(&mut values_host);

        WeightsContainer{
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

impl<T> WeightsContainer<T>
where
    T: Default
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        let values = (0..(previous_size * this_size)).map(|_| T::default()).collect();

        Self{values, previous_size, this_size}
    }
}

impl<T> WeightsContainer<T>
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

impl<T> WeightsContainer<T>
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
    pub fn from_raw(values: Box<[T]>, previous_size: usize, this_size: usize) -> Self
    {
        Self{values, previous_size, this_size}
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

impl WeightsContainer<f32>
{
    pub fn as_arrayfire(&self) -> Array<f32>
    {
        Array::new(&self.values, self.dims())
    }

    #[inline(always)]
    pub fn mul(&self, rhs: impl Borrow<LayerContainer>) -> LayerContainer
    {
        let rhs = rhs.borrow();

        debug_assert!(
            rhs.len() == self.previous_size,
            "{} != {}",
            rhs.len(),
            self.previous_size
        );

        (0..self.this_size).map(|i|
        {
            (0..rhs.len()).map(|p|
            {
                // no bounds checking, if i messed something up let it burn
                unsafe{ self.weight_unchecked(p, i) * rhs.get_unchecked(p) }
            }).sum()
        }).collect()
    }

    #[inline(always)]
    pub fn mul_transposed(&self, rhs: impl Borrow<LayerContainer>) -> LayerContainer
    {
        let rhs = rhs.borrow();

        debug_assert!(
            rhs.len() == self.this_size,
            "{} != {}",
            rhs.len(),
            self.this_size
        );

        (0..self.previous_size).map(|i|
        {
            (0..rhs.len()).map(|p|
            {
                // no bounds checking, if i messed something up let it burn
                unsafe{ self.weight_unchecked(i, p) * rhs.get_unchecked(p) }
            }).sum()
        }).collect()
    }

    pub fn sqrt(&self) -> Self
    {
        Self{
            values: self.iter().map(|v| v.sqrt()).collect(),
            ..*self
        }
    }

    // ballin
    #[inline(always)]
    pub fn add_outer_product(
        &mut self,
        lhs: impl Borrow<LayerContainer>,
        rhs: impl Borrow<LayerContainer>
    )
    {
        let lhs = lhs.borrow();
        let rhs = rhs.borrow();
        
        debug_assert!(
            (lhs.len() == self.this_size)
            &&
            (rhs.len() == self.previous_size)
        );

        let this_size = self.this_size;
        let previous_size = self.previous_size;

        let mut weights = self.iter_mut();
        for y in 0..this_size
        {
            for x in 0..previous_size
            {
                unsafe{
                    *weights.next().unwrap_unchecked() += lhs.get_unchecked(y) * rhs.get_unchecked(x);
                }
            }
        }
    }
}

impl Mul<f32> for WeightsContainer
{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output
    {
        let values = self.values.into_iter().map(|v|
        {
            v * rhs
        }).collect();

        WeightsContainer{
            values,
            ..self
        }
    }
}

impl Mul<f32> for &WeightsContainer
{
    type Output = WeightsContainer;

    fn mul(self, rhs: f32) -> Self::Output
    {
        let values = self.iter().map(|v|
        {
            *v * rhs
        }).collect();

        WeightsContainer{
            values,
            ..*self
        }
    }
}

impl<T> Mul<T> for &WeightsContainer
where
    T: Borrow<WeightsContainer>
{
    type Output = WeightsContainer;

    fn mul(self, rhs: T) -> Self::Output
    {
        let values = self.iter().zip(rhs.borrow().iter()).map(|(v, rhs)|
        {
            *v * *rhs
        }).collect();

        WeightsContainer{
            values,
            ..*self
        }
    }
}

impl Div for WeightsContainer
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output
    {
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

impl DivAssign<f32> for WeightsContainer
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.values.iter_mut().for_each(|v|
        {
            *v /= rhs;
        });
    }
}

impl Add for WeightsContainer
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output
    {
        let values = self.values.into_iter().zip(rhs.values.into_iter()).map(|(v, rhs)|
        {
            v + rhs
        }).collect();

        Self{
            values,
            ..self
        }
    }
}

impl Add<f32> for WeightsContainer
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

impl<T, R> AddAssign<R> for WeightsContainer<T>
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

