use std::{
    f64,
    cmp::Ordering,
    slice,
    borrow::Borrow,
    io::{self, Read},
    fs::File,
    path::Path,
    ops::{Index, Add, Sub, Mul}
};

use serde::{Serialize, Deserialize};

#[allow(unused_imports)]
use rnn::{RNN, RNNGradients};

#[allow(unused_imports)]
use gru::GRU;

use super::word_vectorizer::{WordVectorizer, WordDictionary};

mod rnn;
mod gru;


pub const HIDDEN_AMOUNT: usize = 100;

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
        let mut c = fastrand::f64() / temperature;

        let index = self.0.iter().position(|v|
        {
            c -= v;

            c <= 0.0
        }).unwrap_or_else(||
        {
            // value above 1, just pick the highest
            self.0.iter().cloned().enumerate().reduce(|acc, (i, v)|
            {
                if v > acc.1
                {
                    (i, v)
                } else
                {
                    acc
                }
            }).unwrap().0
        });

        index
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn dot<T>(&self, other: impl Iterator<Item=T>) -> f64
    where
        T: Borrow<f64>
    {
        self.values.iter().zip(other).map(|(this, other)| this * other.borrow()).sum()
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
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.values.into_iter()
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

impl<I, T> Add<I> for LayerContainer
where
    I: IntoIterator<Item=T>,
    T: Borrow<f64>
{
    type Output = Self;

    fn add(self, rhs: I) -> Self::Output
    {
        Self{
            values: self.values.into_iter().zip(rhs.into_iter()).map(|(v, rv)|
            {
                v + rv.borrow()
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
    pub fn mul(&self, rhs: &LayerContainer) -> LayerContainer
    {
        let layer = (0..self.this_size).map(|i|
        {
            (0..rhs.len()).map(|p|
            {
                // no bounds checking, if i messed something up let it burn
                unsafe{ self.weight_unchecked(p, i) * rhs.get_unchecked(p) }
            }).sum()
        }).collect();

        layer
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
                self.learning_rate = (self.learning_rate * 0.5).max(0.0000000001);

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
    pub input_gradients: WeightsContainer<GradientInfo>,
    pub hidden_gradients: WeightsContainer<GradientInfo>,
    pub output_gradients: WeightsContainer<GradientInfo>
}

impl GradientsInfo
{
    pub fn new(word_vector_size: usize) -> Self
    {
        Self{
            input_gradients: WeightsContainer::new(word_vector_size, HIDDEN_AMOUNT),
            hidden_gradients: WeightsContainer::new(HIDDEN_AMOUNT + 1, HIDDEN_AMOUNT),
            output_gradients: WeightsContainer::new(HIDDEN_AMOUNT, word_vector_size)
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork
{
    dictionary: WordDictionary,
    network: RNN,
    gradients_info: GradientsInfo
}

impl NeuralNetwork
{
    pub fn new(dictionary: WordDictionary) -> Self
    {
        let words_vector_size = dictionary.words_amount();
        let network = RNN::new(words_vector_size);

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
    ) -> (Vec<LayerContainer>, Vec<LayerContainer>)
    {
        let word_vectorizer = WordVectorizer::new(&self.dictionary, text);

        let mut words = word_vectorizer.map(|v|
        {
            self.dictionary.word_to_layer(v)
        });

        let mut previous = words.next().expect("text file must not be empty");
        let words = words.skip(1).map(|word|
        {
            let mapped = (previous.clone(), word.clone());

            previous = word;

            mapped
        });

        words.unzip()
    }

    pub fn train(&mut self, epochs: usize, text: impl Read)
    {
        let (inputs, outputs) = self.input_expected_from_text(text);
        
        // whats an epoch? cool word is wut it is
        for epoch in 0..epochs
        {
            let loss =
                self.network.average_loss(slice::from_ref(&inputs), slice::from_ref(&outputs));

            println!("epoch {}: loss: {loss}", epoch + 1);

            let gradients = self.network.gradients(&inputs, &outputs);

            self.apply_gradients(gradients);
        }
    }

    fn apply_gradients(&mut self, mut gradients: RNNGradients)
    {
        gradients.input_gradients.iter_mut().zip(self.gradients_info.input_gradients.iter_mut())
            .for_each(|(g, tg)|
            {
                *g = tg.update(*g);
            });

        gradients.hidden_gradients.iter_mut().zip(self.gradients_info.hidden_gradients.iter_mut())
            .for_each(|(g, tg)|
            {
                *g = tg.update(*g);
            });

        gradients.output_gradients.iter_mut().zip(self.gradients_info.output_gradients.iter_mut())
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
    ) -> (Vec<Vec<LayerContainer>>, Vec<Vec<LayerContainer>>)
    {
        test_texts.into_iter().map(|text|
        {
            let (inputs, outputs) = network.input_expected_from_text(text.as_bytes());

            (inputs, outputs)
        }).unzip()
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
        let (inputs, outputs) = test_input_outputs(test_texts_many(), &network);

        let this_loss = network.network.average_loss(&inputs, &outputs);
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
        let (inputs, outputs) = test_input_outputs(test_texts_many(), &network);

        gradients_check(&mut network, inputs, outputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_one()
    {
        let mut network = test_network();
        let (inputs, outputs) = test_input_outputs(test_texts_one(), &network);

        gradients_check(&mut network, inputs, outputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_two()
    {
        let mut network = test_network();
        let (inputs, outputs) = test_input_outputs(test_texts_two(), &network);

        gradients_check(&mut network, inputs, outputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_three()
    {
        let mut network = test_network();
        let (inputs, outputs) = test_input_outputs(test_texts_three(), &network);

        gradients_check(&mut network, inputs, outputs);
    }

    fn gradients_check(
        network: &mut NeuralNetwork,
        inputs: Vec<Vec<LayerContainer>>,
        outputs: Vec<Vec<LayerContainer>>
    )
    {
        inputs.iter().zip(outputs.iter()).for_each(|(input, output)|
        {
            println!("checking output gradients");
            output_gradients_check(network, input, output);
            println!("checking hidden gradients");
            hidden_gradients_check(network, input, output);
            println!("checking input gradients");
            input_gradients_check(network, input, output);
        });
    }

    fn output_gradients_check(
        network: &mut NeuralNetwork,
        input: &Vec<LayerContainer>,
        output: &Vec<LayerContainer>
    )
    {
        let true_gradient = rnn::tests::output_gradient_check(&mut network.network, input, output);
        let calculated_gradient = network.network.gradients(input, output);

        true_gradient.iter_pos().zip(calculated_gradient.output_gradients.iter_pos())
            .for_each(|(true_gradient, calculated_gradient)|
            {
                let (previous, this) = (true_gradient.previous, true_gradient.this);

                let true_gradient = true_gradient.value;
                let calculated_gradient = calculated_gradient.value;

                assert!(
                    close_enough(true_gradient, calculated_gradient, 0.01),
                    "true_gradient: {true_gradient}, calculated_gradient: {calculated_gradient}, previous_index: {previous}, this_index: {this}"
                );
            });
    }

    fn hidden_gradients_check(
        network: &mut NeuralNetwork,
        input: &Vec<LayerContainer>,
        output: &Vec<LayerContainer>
    )
    {
        let f_true_gradient = rnn::tests::hidden_gradient_check(&mut network.network, input, output);
        let f_calculated_gradient = network.network.gradients(input, output);

        f_true_gradient.iter_pos().zip(f_calculated_gradient.hidden_gradients.iter_pos())
            .for_each(|(true_gradient, calculated_gradient)|
            {
                let (previous, this) = (true_gradient.previous, true_gradient.this);
                
                let true_gradient = true_gradient.value;
                let calculated_gradient = calculated_gradient.value;

                if !close_enough(true_gradient, calculated_gradient, 0.01)
                {
                    dbg!(&network.network);
                    dbg!(&f_calculated_gradient, &f_true_gradient);
                    dbg!(input, output);
                    dbg!(network.network.feedforward(input));
                }

                assert!(
                    close_enough(true_gradient, calculated_gradient, 0.01),
                    "true_gradient: {true_gradient}, calculated_gradient: {calculated_gradient}, previous_index: {previous}, this_index: {this}",
                    //rnn::tests::debug_biases(&network.network, input, output)
                );
            });
    }

    fn input_gradients_check(
        network: &mut NeuralNetwork,
        input: &Vec<LayerContainer>,
        output: &Vec<LayerContainer>
    )
    {
        let true_gradient = rnn::tests::input_gradient_check(&mut network.network, input, output);
        let calculated_gradient = network.network.gradients(input, output);

        true_gradient.iter_pos().zip(calculated_gradient.input_gradients.iter_pos())
            .for_each(|(true_gradient, calculated_gradient)|
            {
                let (previous, this) = (true_gradient.previous, true_gradient.this);

                let true_gradient = true_gradient.value;
                let calculated_gradient = calculated_gradient.value;

                assert!(
                    close_enough(true_gradient, calculated_gradient, 0.01),
                    "true_gradient: {true_gradient}, calculated_gradient: {calculated_gradient}, previous_index: {previous}, this_index: {this}"
                );
            });
    }
}
