use std::{
    f32,
    vec,
    mem,
    slice,
    borrow::Borrow,
    io::{self, Read},
    fs::File,
    path::Path,
    ops::{Index, IndexMut, Add, Sub, Mul, AddAssign}
};

use arrayfire::{Array, dim4};

use serde::{Serialize, Deserialize, de::DeserializeOwned};

// #[allow(unused_imports)]
// use rnn::{RNN, RNNGradients};

#[allow(unused_imports)]
use gru::{GRU, GRUGradients, GPUGradientsInfo, GPUGradientInfo, GRUOutput, GPUGRU};

use super::word_vectorizer::{NetworkDictionary, WordVectorizer, VectorWord, WordDictionary};

// mod rnn;
mod gru;


pub const HIDDEN_AMOUNT: usize = 1000;

#[derive(Debug, Serialize, Deserialize)]
pub struct SoftmaxedLayer(LayerContainer);

impl SoftmaxedLayer
{
    pub fn new(mut layer: LayerContainer) -> Self
    {
        let s: f32 = layer.iter().map(|v|
        {
            f32::consts::E.powf(*v)
        }).sum();

        layer.map(|v|
        {
            f32::consts::E.powf(v) / s
        });

        Self(layer)
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

    pub fn pick_weighed(&self, temperature: f32) -> usize
    {
        let mut values = self.0.clone();
        values.map(|v| v / temperature);

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
    pub fn new(layer: &Array<f32>) -> Self
    {
        let exp_layer = arrayfire::exp(layer);
        let s = arrayfire::sum_all(&exp_layer).0;

        Self(exp_layer / s)
    }

    pub fn pick_weighed(&self, temperature: f32) -> usize
    {
        let values = &self.0 / temperature;

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
            values: self.values.into_iter().zip(rhs.borrow().values.iter()).map(|(v, rhs)|
            {
                v * rhs
            }).collect()
        }
    }
}

impl Into<GPUGradientInfo> for &LayerContainer<GradientInfo>
{
    fn into(self) -> GPUGradientInfo
    {
        let dimensions = self.dims();

        let (m_data, v_data): (Vec<_>, Vec<_>) = self.iter().map(|v|
        {
            (v.m, v.v)
        }).unzip();

        let m = Array::new(&m_data, dimensions);
        let v = Array::new(&v_data, dimensions);

        GPUGradientInfo{m, v}
    }
}

impl LayerContainer<GradientInfo>
{
    pub fn copy_gradients_from(&mut self, value: &GPUGradientInfo)
    {
        debug_assert!(self.values.len() == value.m.elements());
        debug_assert!(self.values.len() == value.v.elements());

        let mut m_values = vec![0.0_f32; self.values.len()];
        let mut v_values = vec![0.0_f32; self.values.len()];

        value.m.host(&mut m_values);
        value.v.host(&mut v_values);

        self.values = m_values.into_iter().zip(v_values.into_iter()).map(|(m, v)|
        {
            GradientInfo{m, v}
        }).collect();
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
    pub fn new_from(&self, values: Array<f32>) -> WeightsContainer<f32>
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

impl WeightsContainer<GradientInfo>
{
    pub fn copy_gradients_from(&mut self, value: &GPUGradientInfo)
    {
        debug_assert!(self.values.len() == value.m.elements());
        debug_assert!(self.values.len() == value.v.elements());

        let mut m_values = vec![0.0_f32; self.values.len()];
        let mut v_values = vec![0.0_f32; self.values.len()];

        value.m.host(&mut m_values);
        value.v.host(&mut v_values);

        self.values = m_values.into_iter().zip(v_values.into_iter()).map(|(m, v)|
        {
            GradientInfo{m, v}
        }).collect();
    }
}

impl<R> AddAssign<R> for WeightsContainer<f32>
where
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
            *this = *this + other;
        });
    }
}

impl Into<GPUGradientInfo> for &WeightsContainer<GradientInfo>
{
    fn into(self) -> GPUGradientInfo
    {
        let dimensions = self.dims();

        let (m_data, v_data): (Vec<_>, Vec<_>) = self.iter().map(|v|
        {
            (v.m, v.v)
        }).unzip();

        let m = Array::new(&m_data, dimensions);
        let v = Array::new(&v_data, dimensions);

        GPUGradientInfo{m, v}
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct GradientInfo
{
    m: f32,
    v: f32
}

impl Default for GradientInfo
{
    fn default() -> Self
    {
        Self{
            m: 0.0,
            v: 0.0
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct GradientsInfo
{
    pub input_update_gradients: WeightsContainer<GradientInfo>,
    pub input_reset_gradients: WeightsContainer<GradientInfo>,
    pub input_activation_gradients: WeightsContainer<GradientInfo>,
    pub hidden_update_gradients: WeightsContainer<GradientInfo>,
    pub hidden_reset_gradients: WeightsContainer<GradientInfo>,
    pub hidden_activation_gradients: WeightsContainer<GradientInfo>,
    pub update_bias_gradients: LayerContainer<GradientInfo>,
    pub reset_bias_gradients: LayerContainer<GradientInfo>,
    pub activation_bias_gradients: LayerContainer<GradientInfo>,
    pub output_gradients: WeightsContainer<GradientInfo>
}

impl GradientsInfo
{
    pub fn new(word_vector_size: usize) -> Self
    {
        Self{
        	input_update_gradients: WeightsContainer::new(word_vector_size, HIDDEN_AMOUNT),
        	input_reset_gradients: WeightsContainer::new(word_vector_size, HIDDEN_AMOUNT),
        	input_activation_gradients: WeightsContainer::new(word_vector_size, HIDDEN_AMOUNT),
        	hidden_update_gradients: WeightsContainer::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
        	hidden_reset_gradients: WeightsContainer::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
        	hidden_activation_gradients: WeightsContainer::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
            update_bias_gradients: LayerContainer::new(HIDDEN_AMOUNT),
            reset_bias_gradients: LayerContainer::new(HIDDEN_AMOUNT),
            activation_bias_gradients: LayerContainer::new(HIDDEN_AMOUNT),
            output_gradients: WeightsContainer::new(HIDDEN_AMOUNT, word_vector_size)
        }
    }

    pub fn as_arrayfire(&self) -> GPUGradientsInfo
    {
        GPUGradientsInfo{
			input_update_gradients: (&self.input_update_gradients).into(),
			input_reset_gradients: (&self.input_reset_gradients).into(),
			input_activation_gradients: (&self.input_activation_gradients).into(),
			hidden_update_gradients: (&self.hidden_update_gradients).into(),
			hidden_reset_gradients: (&self.hidden_reset_gradients).into(),
			hidden_activation_gradients: (&self.hidden_activation_gradients).into(),
			update_bias_gradients: (&self.update_bias_gradients).into(),
			reset_bias_gradients: (&self.reset_bias_gradients).into(),
			activation_bias_gradients: (&self.activation_bias_gradients).into(),
			output_gradients: (&self.output_gradients).into()
        }
    }
}

pub struct InputOutput<T>
{
    container: Vec<T>
}

impl<T> InputOutput<T>
{
    pub fn values_slice<V, F>(
        values: &[V],
        f: F,
        start: usize,
        size: usize
    ) -> Self
    where
        F: FnMut(&V) -> T
    {
        let slice_end = (start + size + 1).min(values.len());
        let this_slice = &values[start..slice_end];

        debug_assert!(this_slice.len() > 1);

        Self::new(this_slice.iter().map(f).collect())
    }

    pub fn new(container: Vec<T>) -> Self
    {
        Self{container}
    }

    pub fn iter<'a>(&'a self) -> InputOutputIter<slice::Iter<'a, T>, &T>
    {
        InputOutputIter::new(self.container.iter())
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize
    {
        self.container.len() - 1
    }
}

#[derive(Clone)]
pub struct InputOutputIter<I, T>
{
    previous: T,
    inputs: I
}

impl<I, T> InputOutputIter<I, T>
where
    I: Iterator<Item=T>
{
    pub fn new(mut inputs: I) -> Self
    {
        Self{
            previous: inputs.next().expect("input must not be empty"),
            inputs
        }
    }
}

impl<I, T> Iterator for InputOutputIter<I, T>
where
    T: Clone,
    I: Iterator<Item=T>
{
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item>
    {
        let input = self.inputs.next();

        match input
        {
            None => None,
            Some(input) =>
            {
                let out = (mem::replace(&mut self.previous, input.clone()), input);

                Some(out)
            }
        }
    }
}

struct Predictor<'a, D>
{
    dictionary: &'a mut D,
    words: Vec<LayerContainer>,
    predicted: Vec<u8>,
    temperature: f32,
    predict_amount: usize
}

impl<'a, D> Predictor<'a, D>
where
    D: NetworkDictionary
{
    pub fn new(
        dictionary: &'a mut D,
        words: Vec<LayerContainer>,
        temperature: f32,
        predict_amount: usize
    ) -> Self
    {
        Self{
            dictionary,
            words,
            predicted: Vec::with_capacity(predict_amount),
            temperature,
            predict_amount
        }
    }

    // uses the cpu
    pub fn predict_bytes(mut self, network: &GRU) -> Box<[u8]>
    {
        let input_amount = self.words.len();

        let mut previous_hidden = LayerContainer::new(self.dictionary.words_amount());
        for i in 0..(input_amount + self.predict_amount)
        {
            debug_assert!(self.words.len() < i);
            let this_input = unsafe{ self.words.get_unchecked(i) };

            let GRUOutput{
                output,
                hidden,
                ..
            } = network.feedforward_cpu_single(&previous_hidden, this_input);
            previous_hidden = hidden;

            if i >= (input_amount - 1)
            {
                let word = self.dictionary.layer_to_word(&output, self.temperature);
                self.words.push(self.dictionary.word_to_layer(word));

                self.predicted.extend(self.dictionary.word_to_bytes(word).into_iter());
            }
        }

        self.predicted.into_boxed_slice()
    }
}

pub struct TrainingInfo
{
    pub epochs: usize,
    pub batch_size: usize,
    pub steps_num: usize,
    pub calculate_accuracy: bool,
    pub ignore_loss: bool
}

#[derive(Serialize, Deserialize)]
pub struct AdamHyperparams
{
    pub a: f32,
    pub b1: f32,
    pub b2: f32,
    pub epsilon: f32,
    pub t: i32,
    pub b1_t: f32,
    pub b2_t: f32

}

impl AdamHyperparams
{
    pub fn new() -> Self
    {
        let mut this = Self{
            a: 0.001,
            b1: 0.9,
            b2: 0.999,
            epsilon: 10e-8,
            t: 1,
            b1_t: 0.0,
            b2_t: 0.0
        };

        this.update_t_vars();

        this
    }

    fn update_t_vars(&mut self)
    {
        self.b1_t = self.b1.powi(self.t);
        self.b2_t = self.b2.powi(self.t);
    }

    pub fn advance_time(&mut self)
    {
        self.t += 1;
    }
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork<D=WordDictionary>
{
    dictionary: D,
    network: GRU,
    gradients_info: GradientsInfo,
    hyper: AdamHyperparams
}

impl<D: NetworkDictionary> NeuralNetwork<D>
where
    D: Serialize + DeserializeOwned
{
    pub fn new(dictionary: D) -> Self
    {
        let words_vector_size = dictionary.words_amount();
        let network = GRU::new(words_vector_size);

        let gradients_info = GradientsInfo::new(words_vector_size);

        let hyper = AdamHyperparams::new();

        Self{dictionary, network, gradients_info, hyper}
    }

    pub fn save<P: AsRef<Path>>(&self, path: P)
    {
        let writer = File::create(path).unwrap();

        ciborium::into_writer(self, writer).unwrap();
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ciborium::de::Error<io::Error>>
    {
        let reader = File::open(path)?;

        ciborium::from_reader(reader)
    }

    pub fn input_expected_from_text(
        &mut self,
        text: impl Read
    ) -> Vec<VectorWord>
    {
        let word_vectorizer = WordVectorizer::new(&mut self.dictionary, text);

        word_vectorizer.collect()
    }

    pub fn test_loss(&mut self, file: impl Read, calculate_accuracy: bool)
    {
        let inputs = self.input_expected_from_text(file);

        let gpu_adapter = self.network.gpu_adapter(&self.gradients_info);

        self.test_loss_inner(&gpu_adapter, &inputs, calculate_accuracy);
    }

    fn test_loss_inner(
        &self,
        gpu_adapter: &GPUGRU,
        testing_inputs: &[VectorWord],
        calculate_accuracy: bool
    )
    {
        let input_outputs = InputOutputIter::new(
            testing_inputs.iter().map(|word|
            {
                self.dictionary.word_to_array(*word)
            })
        );

        if calculate_accuracy
        {
            let accuracy = gpu_adapter.accuracy(input_outputs);

            println!("accuracy: {}%", accuracy * 100.0);
        } else
        {
            let loss = gpu_adapter.loss(input_outputs);

            println!("loss: {loss}");
        }
    }

    pub fn train<R: Read>(
        &mut self,
        info: TrainingInfo,
        testing_data: Option<R>,
        text: impl Read
    )
    {
        let TrainingInfo{
            batch_size,
            steps_num,
            epochs,
            calculate_accuracy,
            ignore_loss
        } = info;

        let mut gpu_adapter = self.network.gpu_adapter(&self.gradients_info);

        let inputs = self.input_expected_from_text(text);
        let testing_inputs = if info.ignore_loss
        {
            Vec::new()
        } else
        {
            match testing_data
            {
                None => inputs.clone(),
                Some(testing_data) =>
                {
                    self.input_expected_from_text(testing_data)
                }
            }
        };

        println!("batch size: {batch_size}");
        println!("steps amount: {steps_num}");
        
        let epochs_per_input = (inputs.len() / steps_num).max(1);
        println!("calculate loss every {epochs_per_input} epochs");

        let output_loss = |network: &NeuralNetwork<D>, gpu_adapter: &GPUGRU|
        {
            if ignore_loss
            {
                return;
            }

            network.test_loss_inner(gpu_adapter, &testing_inputs, calculate_accuracy);
        };

        let mut batch_start: usize = 0;

        let empty_hidden = ||
        {
            arrayfire::constant(0.0_f32, dim4!(HIDDEN_AMOUNT as u64))
        };

        let mut previous_hidden: Array<f32> = empty_hidden();

        // whats an epoch? cool word is wut it is
        // at some point i found out wut it was (going over the whole training data once)
        // but i dont rly feel like changing a silly thing like that
        for epoch in 0..epochs
        {
            eprintln!("epoch: {epoch}");

            let input_vectorizer = |dictionary: &D, word: &VectorWord|
            {
                dictionary.word_to_array(*word)
            };

            let print_loss = (epoch % epochs_per_input) == epochs_per_input - 1;
            if print_loss
            {
                output_loss(self, &gpu_adapter);
            }

            let mut batch_gradients = None;
            for _ in 0..batch_size
            {
                let values = InputOutput::values_slice(
                    &inputs,
                    |word| input_vectorizer(&self.dictionary, word),
                    batch_start,
                    steps_num
                );

                let (final_hidden, gradients) =
                    gpu_adapter.gradients_with_hidden::<true, _>(&previous_hidden, values.iter());

                if batch_gradients.is_none()
                {
                    batch_gradients = Some(gradients);
                } else
                {
                    batch_gradients.as_mut().map(|batch_gradients| *batch_gradients += gradients);
                }

                previous_hidden = final_hidden;

                batch_start += steps_num;
                if batch_start >= (inputs.len() - 1)
                {
                    batch_start = 0;
                    previous_hidden = empty_hidden();
                }
            }

            let gradients = batch_gradients.unwrap() / batch_size as f32;

            gpu_adapter.apply_gradients(gradients, &mut self.hyper);
        }

        output_loss(self, &gpu_adapter);

        self.transfer_gradient_info(&gpu_adapter);
        self.network.transfer_weights(gpu_adapter);
    }

    fn transfer_gradient_info(&mut self, gpugru: &GPUGRU)
    {
        let gradients = gpugru.gradients_info();

		self.gradients_info.input_update_gradients.copy_gradients_from(
			&gradients.input_update_gradients
		);

		self.gradients_info.input_reset_gradients.copy_gradients_from(
			&gradients.input_reset_gradients
		);

		self.gradients_info.input_activation_gradients.copy_gradients_from(
			&gradients.input_activation_gradients
		);

		self.gradients_info.hidden_update_gradients.copy_gradients_from(
			&gradients.hidden_update_gradients
		);

		self.gradients_info.hidden_reset_gradients.copy_gradients_from(
			&gradients.hidden_reset_gradients
		);

		self.gradients_info.hidden_activation_gradients.copy_gradients_from(
			&gradients.hidden_activation_gradients
		);

		self.gradients_info.update_bias_gradients.copy_gradients_from(
			&gradients.update_bias_gradients
		);

		self.gradients_info.reset_bias_gradients.copy_gradients_from(
			&gradients.reset_bias_gradients
		);

		self.gradients_info.activation_bias_gradients.copy_gradients_from(
			&gradients.activation_bias_gradients
		);

		self.gradients_info.output_gradients.copy_gradients_from(
			&gradients.output_gradients
		);
    }

    #[allow(dead_code)]
    pub fn predict(&mut self, text: &str, amount: usize, temperature: f32) -> String
    {
        let word_vectorizer = WordVectorizer::new(&mut self.dictionary, text.as_bytes());

        let predictor = {
            let words = word_vectorizer.collect::<Vec<_>>();

            let words = words.into_iter().map(|v|
            {
                self.dictionary.word_to_layer(v)
            }).collect::<Vec<_>>();

            Predictor::new(&mut self.dictionary, words, temperature, amount)
        };

        let output = predictor.predict_bytes(&self.network);
        
        String::from_utf8_lossy(&output).to_string()
    }
}

#[cfg(test)]
mod tests
{
    use super::*;
    
    fn close_enough(a: f32, b: f32, epsilon: f32) -> bool
    {
        if (a == b) || ((a.min(b) == -0.0) && (a.max(b) == 0.0))
        {
            return true;
        }

        ((a - b).abs() / (a.abs() + b.abs())) < epsilon
    }

    fn test_dictionary() -> WordDictionary
    {
        let _words = "test.testing.tester.epic.cool.true";

        // WordDictionary::build(words.as_bytes())
        
        let words = "testing test - a b c 6 1 A o l d e f g h i j s o r s m";
        WordDictionary::no_defaults(words.as_bytes())
    }

    fn test_network() -> NeuralNetwork
    {
        NeuralNetwork::new(test_dictionary())
    }

    fn test_texts_many() -> Vec<&'static str>
    {
        vec![
            "testing tests or sm",
            "abcdefghij",
            "coolllllll",
            "AAAAAAAAAA"
        ]
    }

    fn test_texts_one() -> Vec<&'static str>
    {
        vec!["testing", "a", "6", "A"]
    }

    fn test_texts_two() -> Vec<&'static str>
    {
        vec!["testing-", "ab", "61", "AA"]
    }

    fn test_texts_three() -> Vec<&'static str>
    {
        vec!["testing test", "abc", "611", "AAA"]
    }

    fn test_input_outputs(
        test_texts: Vec<&'static str>,
        network: &mut NeuralNetwork
    ) -> Vec<InputOutput<LayerContainer>>
    {
        test_texts.into_iter().map(|text|
        {
            let inputs = network.input_expected_from_text(text.as_bytes());

            InputOutput::new(inputs.iter().map(|word: &VectorWord|
            {
                network.dictionary.word_to_layer(*word)
            }).collect())
        }).collect()
    }

    #[test]
    fn matrix_multiplication()
    {
        let v = LayerContainer::from(vec![5.0, 0.0, 5.0, 7.0]);

        let m = WeightsContainer::from_raw(vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        ].into_boxed_slice(), 4, 4);

        assert_eq!(
            m.mul(&v),
            LayerContainer::from(vec![48.0, 116.0, 184.0, 252.0])
        );

        assert_eq!(
            m.mul_transposed(&v),
            LayerContainer::from(vec![141.0, 158.0, 175.0, 192.0])
        );
    }

    #[test]
    fn softmax()
    {
        let test_layer = LayerContainer::from_iter([1.0, 2.0, 8.0].iter().cloned());

        let softmaxed = SoftmaxedLayer::new(test_layer);

        softmaxed.0.iter().zip([0.001, 0.002, 0.997].iter()).for_each(|(softmaxed, correct)|
        {
            assert!(
                close_enough(*softmaxed, *correct, 0.2),
                "softmaxed: {}, correct: {}",
                *softmaxed,
                *correct
            );
        });
    }

    #[test]
    fn loss()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_many(), &mut network);

        let l = inputs.len() as f32;
        let this_loss = inputs.into_iter().map(|input|
        {
            network.network.loss(input.iter().map(|(a, b)| (a.clone(), b.clone())))
        }).sum::<f32>() / l;

        let predicted_loss = (network.dictionary.words_amount() as f32).ln();

        assert!(
            close_enough(this_loss, predicted_loss, 0.1),
            "this_loss: {this_loss}, predicted_loss: {predicted_loss}"
        );
    }

    #[ignore]
    #[test]
    fn gradients_check_many()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_many(), &mut network);

        gradients_check(&mut network, inputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_one()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_one(), &mut network);

        gradients_check(&mut network, inputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_two()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_two(), &mut network);

        gradients_check(&mut network, inputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_three()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_three(), &mut network);

        gradients_check(&mut network, inputs);
    }

    fn print_status(description: &str, f: impl FnOnce())
    {
        print!("checking {description}: ⛔ ");
        f();
        println!("\rchecking {description}: ✔️");
    }

    fn gradients_check(
        network: &mut NeuralNetwork,
        inputs: Vec<InputOutput<LayerContainer>>
    )
    {
        let network = &mut network.network;
        inputs.into_iter().for_each(|input|
        {
            print_status("output gradients", ||
            {
                gru::tests::output_gradients_check(network, input.iter())
            });

            print_status("hidden update gradients", ||
            {
                gru::tests::hidden_update_gradients_check(network, input.iter())
            });

            print_status("hidden reset gradients", ||
            {
                gru::tests::hidden_reset_gradients_check(network, input.iter())
            });

            print_status("hidden activation gradients", ||
            {
                gru::tests::hidden_activation_gradients_check(network, input.iter())
            });

            print_status("update bias gradients", ||
            {
                gru::tests::update_bias_gradients_check(network, input.iter())
            });

            print_status("reset bias gradients", ||
            {
                gru::tests::reset_bias_gradients_check(network, input.iter())
            });

            print_status("activation bias gradients", ||
            {
                gru::tests::activation_bias_gradients_check(network, input.iter())
            });

            print_status("input update gradients", ||
            {
                gru::tests::input_update_gradients_check(network, input.iter())
            });

            print_status("input reset gradients", ||
            {
                gru::tests::input_reset_gradients_check(network, input.iter())
            });

            print_status("input activation gradients", ||
            {
                gru::tests::input_activation_gradients_check(network, input.iter())
            });
        });
    }
}
