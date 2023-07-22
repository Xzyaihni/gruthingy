use std::{
    f64,
    vec,
    slice,
    cmp::Ordering,
    borrow::Borrow,
    io::{self, Read},
    fs::File,
    path::Path,
    ops::{Index, Add, Sub, Mul, AddAssign}
};

use serde::{Serialize, Deserialize};

// #[allow(unused_imports)]
// use rnn::{RNN, RNNGradients};

#[allow(unused_imports)]
use gru::{GRU, GRUGradients};

use super::word_vectorizer::{WordVectorizer, VectorWord, WordDictionary};

// mod rnn;
mod gru;


pub const HIDDEN_AMOUNT: usize = 10;

#[derive(Debug, Serialize, Deserialize)]
pub struct SoftmaxedLayer(LayerContainer);

impl SoftmaxedLayer
{
    pub fn new(mut layer: LayerContainer) -> Self
    {
        let s: f64 = layer.iter().map(|v|
        {
            f64::consts::E.powf(*v)
        }).sum();

        layer.map(|v|
        {
            f64::consts::E.powf(v) / s
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

    pub fn pick_weighed(&self, temperature: f64) -> usize
    {
        let mut values = self.0.clone();
        values.map(|v| v / temperature);

        let mut c = fastrand::f64();

        let index = values.iter().position(|v|
        {
            c -= v;

            c <= 0.0
        }).unwrap();

        index
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerContainer
{
    values: Vec<f64>
}

impl LayerContainer
{
    pub fn new(size: usize) -> Self
    {
        let values = vec![0.0; size];

        Self{values}
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize
    {
        self.values.len()
    }

    #[allow(dead_code)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut f64>
    {
        self.values.iter_mut()
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> &f64
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
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut f64
    {
        debug_assert!(
            (0..self.values.len()).contains(&index),
            "{} >= {}",
            index,
            self.values.len()
        );

        self.values.get_unchecked_mut(index)
    }

    pub fn iter(&self) -> impl Iterator<Item=&f64>
    {
        self.values.iter()
    }

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

    pub fn dot(&self, other: impl Borrow<Self>) -> f64
    {
        self.values.iter().zip(other.borrow().values.iter())
            .map(|(this, other)| this * other).sum()
    }

    pub fn powi(self, pow: i32) -> Self
    {
        Self{
            values: self.values.into_iter().map(|v|
            {
                v.powi(pow)
            }).collect()
        }
    }

    pub fn map<F>(&mut self, mut f: F)
    where
        F: FnMut(f64) -> f64
    {
        self.values.iter_mut().for_each(|v| *v = f(*v));
    }
}

impl FromIterator<f64> for LayerContainer
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item=f64>
    {
        Self{
            values: Vec::<f64>::from_iter(iter)
        }
    }
}

impl IntoIterator for LayerContainer
{
    type Item = f64;
    type IntoIter = vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.values.into_iter()
    }
}

impl<'a> IntoIterator for &'a LayerContainer
{
    type Item = &'a f64;
    type IntoIter = slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.values.iter()
    }
}

impl From<Vec<f64>> for LayerContainer
{
    fn from(values: Vec<f64>) -> Self
    {
        Self{values}
    }
}

impl Index<usize> for LayerContainer
{
    type Output = f64;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output
    {
        &self.values[index]
    }
}

impl Add<f64> for LayerContainer
{
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output
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

impl Mul<f64> for LayerContainer
{
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output
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

#[derive(Clone, Copy)]
pub struct WeightsIterValue<T>
{
    pub previous: usize,
    pub this: usize,
    pub value: T
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WeightsContainer<T=f64>
{
    values: Box<[T]>,
    previous_size: usize,
    this_size: usize
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

impl WeightsContainer<f64>
{
    pub fn mul(&self, rhs: impl Borrow<LayerContainer>) -> LayerContainer
    {
        let rhs = rhs.borrow();

        debug_assert!(
            // checks if theyre both equal, or rhs is shorter by 1
            // (weights might have a bias unused in multiplication)
            (rhs.len() == self.previous_size) || (rhs.len() == (self.previous_size - 1)),
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

    pub fn mul_transposed_skip_last(&self, rhs: impl Borrow<LayerContainer>) -> LayerContainer
    {
        let rhs = rhs.borrow();

        debug_assert!(
            rhs.len() == self.this_size,
            "{} != {}",
            rhs.len(),
            self.this_size
        );

        (0..(self.previous_size - 1)).map(|i|
        {
            (0..rhs.len()).map(|p|
            {
                // no bounds checking, if i messed something up let it burn
                unsafe{ self.weight_unchecked(i, p) * rhs.get_unchecked(p) }
            }).sum()
        }).collect()
    }
}

impl<R> AddAssign<R> for WeightsContainer<f64>
where
    R: Borrow<Self>
{
    fn add_assign(&mut self, rhs: R)
    {
        let rhs = rhs.borrow();

        debug_assert!(
            ((self.previous_size == rhs.previous_size)
            || (self.previous_size - 1 == rhs.previous_size))
            &&
            ((self.this_size == rhs.this_size)
            || (self.this_size - 1 == rhs.this_size))
        );

        for y in 0..rhs.this_size
        {
            for x in 0..rhs.previous_size
            {
                let this = unsafe{ self.weight_unchecked_mut(x, y) };
                let other = unsafe{ rhs.weight_unchecked(x, y) };

                *this = *this + other;
            }
        }
    }
}

type Sign = i8;
fn new_sign(num: f64) -> Sign
{
    if num==0.0
    {
        0
    } else if num>0.0
    {
        1
    } else
    {
        -1
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct GradientInfo
{
    learning_rate: f64,
    previous_sign: Sign
}

impl GradientInfo
{
    pub fn update(&mut self, gradient: f64) -> f64
    {
        let current_sign = new_sign(gradient);
        
        let combination = current_sign * self.previous_sign;
        match combination.cmp(&0)
        {
            Ordering::Greater =>
            {
                self.learning_rate = (self.learning_rate * 1.2).min(0.01);

                self.previous_sign = current_sign;

                -self.learning_rate * current_sign as f64
            },
            Ordering::Less =>
            {
                self.learning_rate = (self.learning_rate * 0.5).max(f64::MIN_POSITIVE);

                self.previous_sign = 0;

                0.0
            },
            Ordering::Equal =>
            {
                self.previous_sign = current_sign;
                -self.learning_rate * current_sign as f64
            }
        }
    }
}

impl Default for GradientInfo
{
    fn default() -> Self
    {
        Self{
            learning_rate: 0.01,
            previous_sign: 1
        }
    }
}

#[derive(Serialize, Deserialize)]
struct GradientsInfo
{
    pub input_update_gradients: WeightsContainer<GradientInfo>,
    pub input_reset_gradients: WeightsContainer<GradientInfo>,
    pub input_activation_gradients: WeightsContainer<GradientInfo>,
    pub hidden_update_gradients: WeightsContainer<GradientInfo>,
    pub hidden_reset_gradients: WeightsContainer<GradientInfo>,
    pub hidden_activation_gradients: WeightsContainer<GradientInfo>,
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
        	hidden_update_gradients: WeightsContainer::new(HIDDEN_AMOUNT + 1, HIDDEN_AMOUNT),
        	hidden_reset_gradients: WeightsContainer::new(HIDDEN_AMOUNT + 1, HIDDEN_AMOUNT),
        	hidden_activation_gradients: WeightsContainer::new(HIDDEN_AMOUNT + 1, HIDDEN_AMOUNT),
            output_gradients: WeightsContainer::new(HIDDEN_AMOUNT, word_vector_size)
        }
    }
}

pub struct InputOutput
{
    container: Vec<LayerContainer>
}

impl InputOutput
{
    pub fn batch<V, F>(
        values: &[V],
        f: F,
        batch_start: usize,
        batch_size: usize
    ) -> Self
    where
        F: FnMut(&V) -> LayerContainer
    {
        let batch_end = (batch_start + batch_size).min(values.len());
        let batch = &values[batch_start..batch_end];

        Self::new(batch.iter().map(f).collect())
    }

    pub fn new(container: Vec<LayerContainer>) -> Self
    {
        Self{container}
    }

    pub fn iter<'a>(&'a self) -> InputOutputIter<'a>
    {
        InputOutputIter{
            container: &self.container,
            current_index: 1
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize
    {
        self.container.len() - 1
    }
}

#[derive(Clone)]
pub struct InputOutputIter<'a>
{
    container: &'a [LayerContainer],
    current_index: usize
}

impl InputOutputIter<'_>
{
    pub fn len(&self) -> usize
    {
        self.container.len() - 1
    }
}

impl<'a> Iterator for InputOutputIter<'a>
{
    type Item = (&'a LayerContainer, &'a LayerContainer);

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.container.len() == self.current_index
        {
            None
        } else
        {
            debug_assert!((0..self.container.len()).contains(&(self.current_index - 1)));
            let previous = unsafe{ self.container.get_unchecked(self.current_index - 1) };
            
            debug_assert!((0..self.container.len()).contains(&self.current_index));
            let this = unsafe{ self.container.get_unchecked(self.current_index) };
            
            self.current_index += 1;

            Some((previous, this))
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork
{
    dictionary: WordDictionary,
    network: GRU,
    gradients_info: GradientsInfo
}

impl NeuralNetwork
{
    pub fn new(dictionary: WordDictionary) -> Self
    {
        let words_vector_size = dictionary.words_amount();
        let network = GRU::new(words_vector_size);

        let gradients_info = GradientsInfo::new(words_vector_size);

        Self{dictionary, network, gradients_info}
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

    #[allow(dead_code)]
    pub fn dictionary(&self) -> &WordDictionary
    {
        &self.dictionary
    }

    pub fn input_expected_from_text(
        &self,
        text: impl Read
    ) -> Vec<VectorWord>
    {
        let word_vectorizer = WordVectorizer::new(&self.dictionary, text);

        word_vectorizer.collect()
    }

    pub fn train(&mut self, epochs: usize, batch_size: usize, text: impl Read)
    {
        let inputs = self.input_expected_from_text(text);
        println!("batch size: {batch_size}");
        
        let epochs_per_input = (inputs.len() / batch_size).max(1);
        println!("calculate loss every {epochs_per_input} epochs");

        let input_vectorizer = |dictionary: &WordDictionary, word: &VectorWord|
        {
            dictionary.word_to_layer(*word)
        };

        let output_loss = |network: &GRU, dictionary: &WordDictionary|
        {
            let mut total_batches = 0;
            let mut loss = 0.0;

            let mut batch_start = 0;

            loop
            {
                let batch = InputOutput::batch(
                    &inputs,
                    |word| input_vectorizer(dictionary, word),
                    batch_start,
                    batch_size
                );

                loss += network.average_loss(batch.iter());
                total_batches += 1;

                batch_start += batch_size;
                if batch_start >= inputs.len()
                {
                    break;
                }
            }

            loss /= total_batches as f64;

            println!("loss: {loss}");
        };
        
        let new_batch_start = ||
        {
            let max_length = inputs.len().saturating_sub(batch_size);

            if max_length != 0
            {
                fastrand::usize(0..max_length)
            } else
            {
                0
            }
        };

        let mut batch_start = new_batch_start();

        // whats an epoch? cool word is wut it is
        for epoch in 0..epochs
        {
            let batch = InputOutput::batch(
                &inputs,
                |word| input_vectorizer(&self.dictionary, word),
                batch_start,
                batch_size
            );

            let print_loss = (epoch % epochs_per_input) == epochs_per_input - 1;
            if print_loss
            {
                output_loss(&self.network, &self.dictionary);
            }

            let gradients = self.network.gradients(batch.iter());

            self.apply_gradients(gradients);

            batch_start = new_batch_start();
        }

        output_loss(&self.network, &self.dictionary);
    }

    fn apply_gradients(&mut self, mut gradients: GRUGradients)
    {
        gradients.input_update_gradients.iter_mut()
            .zip(self.gradients_info.input_update_gradients.iter_mut())
            .for_each(|(g, tg)|
            {
                *g = tg.update(*g);
            });

        gradients.input_reset_gradients.iter_mut()
            .zip(self.gradients_info.input_reset_gradients.iter_mut())
            .for_each(|(g, tg)|
            {
                *g = tg.update(*g);
            });

        gradients.input_activation_gradients.iter_mut()
            .zip(self.gradients_info.input_activation_gradients.iter_mut())
            .for_each(|(g, tg)|
            {
                *g = tg.update(*g);
            });

        gradients.hidden_update_gradients.iter_mut()
            .zip(self.gradients_info.hidden_update_gradients.iter_mut())
            .for_each(|(g, tg)|
            {
                *g = tg.update(*g);
            });

        gradients.hidden_reset_gradients.iter_mut()
            .zip(self.gradients_info.hidden_reset_gradients.iter_mut())
            .for_each(|(g, tg)|
            {
                *g = tg.update(*g);
            });

        gradients.hidden_activation_gradients.iter_mut()
            .zip(self.gradients_info.hidden_activation_gradients.iter_mut())
            .for_each(|(g, tg)|
            {
                *g = tg.update(*g);
            });

        gradients.output_gradients.iter_mut()
            .zip(self.gradients_info.output_gradients.iter_mut())
            .for_each(|(g, tg)|
            {
                *g = tg.update(*g);
            });
        
        self.network.adjust_weights(gradients);
    }

    #[allow(dead_code)]
    pub fn predict(&self, text: &str, amount: usize, temperature: f64) -> String
    {
        let word_vectorizer = WordVectorizer::new(&self.dictionary, text.as_bytes());

        let mut words = word_vectorizer.map(|v|
        {
            self.dictionary.word_to_layer(v)
        }).collect::<Vec<_>>();

        let mut total_bytes = Vec::new();

        for _ in 0..amount
        {
            let output = self.network.feedforward(&words);

            let output = match output.outputs.last()
            {
                Some(x) => x,
                None => return String::new()
            };

            let word = self.dictionary.layer_to_word(output, temperature);
            words.push(self.dictionary.word_to_layer(word));

            let bytes = self.dictionary.word_to_bytes(word).unwrap();
            total_bytes.extend(bytes.iter());
        }
        
        String::from_utf8_lossy(&total_bytes).to_string()
    }
}

#[cfg(test)]
mod tests
{
    use super::*;
    
    fn close_enough(a: f64, b: f64, epsilon: f64) -> bool
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
        network: &NeuralNetwork
    ) -> Vec<InputOutput>
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
        let network = test_network();
        let inputs = test_input_outputs(test_texts_many(), &network);

        let len = inputs.len();
        let this_loss = inputs.into_iter().map(|input|
        {
            network.network.average_loss(input.iter())
        }).sum::<f64>() / len as f64;

        let predicted_loss = (network.dictionary.words_amount() as f64).ln();

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
        let inputs = test_input_outputs(test_texts_many(), &network);

        gradients_check(&mut network, inputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_one()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_one(), &network);

        gradients_check(&mut network, inputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_two()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_two(), &network);

        gradients_check(&mut network, inputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_three()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_three(), &network);

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
        inputs: Vec<InputOutput>
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
