use std::{
    f64,
    slice,
    borrow::Borrow,
    io::{self, Read},
    fs::File,
    path::Path,
    ops::{Index, Add, Sub, Mul}
};

use serde::{Serialize, Deserialize};

#[allow(unused_imports)]
use rnn::RNN;

#[allow(unused_imports)]
use gru::GRU;

use super::word_vectorizer::{WordVectorizer, WordDictionary};

mod rnn;
mod gru;


const DEFAULT_LEARNING_RATE: f64 = 0.001;

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

    pub fn new_empty(size: usize) -> Self
    {
        Self(LayerContainer::new(size))
    }

    pub fn pick_weighed(&self) -> usize
    {
        let mut c = fastrand::f64();

        let index = self.0.iter().position(|v|
        {
            c -= v;

            c <= 0.0
        }).unwrap();

        index
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerContainer
{
    values: Box<[f64]>
}

impl LayerContainer
{
    pub fn new(size: usize) -> Self
    {
        let values = vec![0.0; size].into_boxed_slice();

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
        self.values.get_unchecked(index)
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
            values: Box::<[f64]>::from_iter(iter)
        }
    }
}

impl IntoIterator for LayerContainer
{
    type Item = f64;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.values.into_vec().into_iter()
    }
}

impl From<Box<[f64]>> for LayerContainer
{
    fn from(values: Box<[f64]>) -> Self
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

impl Add for LayerContainer
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output
    {
        Self{
            values: self.values.into_iter().zip(rhs.values.into_iter()).map(|(v, rv)|
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
        Self{
            values: self.values.into_iter().zip(rhs.borrow().values.into_iter()).map(|(v, rv)|
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
pub struct WeightsIterValue
{
    pub previous: usize,
    pub this: usize,
    pub value: f64
}

#[derive(Serialize, Deserialize)]
pub struct WeightsContainer
{
    values: Box<[f64]>,
    previous_size: usize,
    this_size: usize
}

impl WeightsContainer
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        let values = (0..(previous_size * this_size)).map(|_| 0.0).collect();

        Self{values, previous_size, this_size}
    }

    pub fn new_with<F>(previous_size: usize, this_size: usize, mut f: F) -> Self
    where
        F: FnMut() -> f64
    {
        let values = (0..(previous_size * this_size)).map(|_|
        {
            f()
        }).collect();

        Self{values, previous_size, this_size}
    }

    #[allow(dead_code)]
    pub fn from_raw(values: Box<[f64]>, previous_size: usize, this_size: usize) -> Self
    {
        Self{values, previous_size, this_size}
    }
    
    #[allow(dead_code)]
    pub fn this(&self, previous: usize) -> impl Iterator<Item=&f64>
    {
        (0..self.this_size).map(move |i|
        {
            self.weight(previous, i)
        })
    }

    #[allow(dead_code)]
    pub unsafe fn this_unchecked(&self, previous: usize) -> impl Iterator<Item=&f64>
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

    pub fn iter(&self) -> impl Iterator<Item=&f64>
    {
        self.values.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut f64>
    {
        self.values.iter_mut()
    }

    #[allow(dead_code)]
    pub fn iter_pos(&self) -> impl Iterator<Item=WeightsIterValue> + '_
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

    pub fn map<F>(&mut self, mut f: F)
    where
        F: FnMut(f64) -> f64
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
    pub fn weight(&self, previous: usize, this: usize) -> &f64
    {
        &self.values[self.index_of(previous, this)]
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn weight_unchecked(&self, previous: usize, this: usize) -> &f64
    {
        self.values.get_unchecked(self.index_of(previous, this))
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn weight_mut(&mut self, previous: usize, this: usize) -> &mut f64
    {
        &mut self.values[self.index_of(previous, this)]
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn weight_unchecked_mut(&mut self, previous: usize, this: usize) -> &mut f64
    {
        self.values.get_unchecked_mut(self.index_of(previous, this))
    }

    pub fn mul(&self, rhs: &LayerContainer) -> LayerContainer
    {
        let layer = (0..self.this_size).map(|i|
        {
            (0..self.previous_size).map(|p|
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

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork
{
    dictionary: WordDictionary,
    network: RNN,
    learning_rate: f64
}

impl NeuralNetwork
{
    pub fn new(dictionary: WordDictionary) -> Self
    {
        let words_vector_size = dictionary.words_amount();
        let network = RNN::new(words_vector_size);

        let learning_rate = DEFAULT_LEARNING_RATE;

        Self{dictionary, network, learning_rate}
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
        
        let mut previous_loss: Option<f64> = None;

        // whats an epoch? cool word is wut it is
        for epoch in 0..epochs
        {
            let loss =
                self.network.average_loss(slice::from_ref(&inputs), slice::from_ref(&outputs));

            println!("epoch {}: loss: {loss}, learning_rate: {}", epoch + 1, self.learning_rate);

            if let Some(previous_loss) = previous_loss
            {
                if previous_loss < loss
                {
                    self.learning_rate *= 0.5;
                }
            }

            previous_loss = Some(loss);

            self.train_inner(self.learning_rate, &inputs, &outputs);
        }
    }

    fn train_inner(
        &mut self,
        learning_rate: f64,
        inputs: &Vec<LayerContainer>,
        outputs: &Vec<LayerContainer>
    )
    {
        let mut gradients = self.network.gradients(inputs, outputs);

        gradients.output_gradients.map(|g| -g * learning_rate);
        gradients.hidden_gradients.map(|g| -g * learning_rate);
        gradients.input_gradients.map(|g| -g * learning_rate);

        self.network.adjust_weights(gradients);
    }

    #[allow(dead_code)]
    pub fn predict(&self, text: &str, amount: usize) -> String
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

            let word = self.dictionary.layer_to_word(output);
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
        let words = "test.testing.tester.epic.cool.true";

        WordDictionary::build(words.as_bytes())
    }

    fn test_network() -> NeuralNetwork
    {
        NeuralNetwork::new(test_dictionary())
    }

    fn test_input_outputs(
        network: &NeuralNetwork
    ) -> (Vec<Vec<LayerContainer>>, Vec<Vec<LayerContainer>>)
    {
        let texts = vec![
            "testing tests or sm",
            "abcdefghij",
            "coolllllll",
            "AAAAAAAAAA"
        ];

        texts.into_iter().map(|text|
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
        let (inputs, outputs) = test_input_outputs(&network);

        let this_loss = network.network.average_loss(&inputs, &outputs);
        let predicted_loss = (network.dictionary.words_amount() as f64).ln();

        assert!(
            close_enough(this_loss, predicted_loss, 0.1),
            "this_loss: {this_loss}, predicted_loss: {predicted_loss}"
        );
    }

    #[test]
    fn gradients_check()
    {
        let mut network = test_network();
        let (inputs, outputs) = test_input_outputs(&network);

        inputs.iter().zip(outputs.iter()).for_each(|(input, output)|
        {
            println!("checking output gradients");
            output_gradients_check(&mut network, input, output);
            println!("checking hidden gradients");
            hidden_gradients_check(&mut network, input, output);
            println!("checking input gradients");
            input_gradients_check(&mut network, input, output);
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
        let true_gradient = rnn::tests::hidden_gradient_check(&mut network.network, input, output);
        let calculated_gradient = network.network.gradients(input, output);

        true_gradient.iter_pos().zip(calculated_gradient.hidden_gradients.iter_pos())
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
