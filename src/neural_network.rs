use std::{
    f32,
    mem,
    slice,
    io::{self, Read},
    fs::File,
    path::Path
};

use serde::{Serialize, Deserialize};

use network::{NetworkOutput, Network};
use network_unit::NetworkUnit;

#[allow(unused_imports)]
use crate::word_vectorizer::{
    ByteDictionary,
    CharDictionary,
    WordDictionary,
    NetworkDictionary,
    WordVectorizer,
    VectorWord
};

use optimizers::*;

pub use network::WeightsNamed;

pub use containers::{
    LayerType,
    ScalarType,
    LayerInnerType,
    DiffWrapper,
    Softmaxer,
    CloneableWrapper,
    Joinable,
    JoinableType,
    JoinableDeepType
};

#[allow(unused_imports)]
use gru::GRU;

#[allow(unused_imports)]
use lstm::LSTM;

mod optimizers;
mod network_unit;
mod network;
mod gru;
mod lstm;

pub mod containers;


pub const HIDDEN_AMOUNT: usize = 256;
pub const LAYERS_AMOUNT: usize = 3;

pub const DROPCONNECT_PROBABILITY: f32 = 0.5;
pub const DROPOUT_PROBABILITY: f32 = 0.5;

pub const GRADIENT_CLIP: f32 = 1.0;

// options: Power, Division
pub const DECAY_FUNCTION: DecayFunction = DecayFunction::Power;

// options: SDG, Adam, AdamX, PowerSign (garbage (maybe i did it wrong))
pub type CurrentOptimizer = Adam;

// options: Tanh, LeakyRelu
pub const LAYER_ACTIVATION: AFType = AFType::LeakyRelu;

// options: GRU, LSTM
pub type CurrentNetworkUnit = LSTM;

// these 2 r related, WordDictionary uses a dictionary and ByteDictionary doesnt
pub const USES_DICTIONARY: bool = true;
pub const DICTIONARY_TEXT: &'static str = include_str!("../ascii_dictionary.txt");

pub const INPUT_SIZE: usize = DictionaryType::words_amount();

// options: WordDictionary, ByteDictionary, CharDictionary
pub type DictionaryType = CharDictionary;

#[allow(dead_code)]
pub enum DecayFunction
{
    Power,
    Division
}

impl DecayFunction
{
    fn decay(&self, value: f32, t: i32) -> f32
    {
        match self
        {
            DecayFunction::Power => value.powi(t),
            DecayFunction::Division => value / t as f32
        }
    }
}

#[allow(dead_code)]
pub enum AFType
{
    Tanh,
    LeakyRelu
}

pub struct KahanSum
{
    value: f64,
    compensation: f64
}

impl KahanSum
{
    pub fn new() -> Self
    {
        Self{
            value: 0.0,
            compensation: 0.0
        }
    }

    pub fn add(&mut self, rhs: f64)
    {
        let temp_n = rhs - self.compensation;
        let temp_sum = self.value + temp_n;

        self.compensation = (temp_sum - self.value) - temp_n;
        self.value = temp_sum;
    }

    pub fn value(&self) -> f64
    {
        self.value
    }
}

pub type ThisWeightsContainer<T> = <CurrentNetworkUnit as NetworkUnit>::ThisWeightsContainer<T>;

pub struct InputOutput<T>
{
    container: Vec<T>
}

impl<T> InputOutput<T>
{
    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

impl<I, T> ExactSizeIterator for InputOutputIter<I, T>
where
    T: Clone,
    I: Iterator<Item=T> + ExactSizeIterator
{
    fn len(&self) -> usize
    {
        self.inputs.len()
    }
}

struct Predictor<'a>
{
    dictionary: &'a mut DictionaryType,
    words: Vec<LayerType>,
    predicted: Vec<u8>,
    temperature: f32,
    predict_amount: usize
}

impl<'a> Predictor<'a>
{
    pub fn new(
        dictionary: &'a mut DictionaryType,
        words: Vec<LayerInnerType>,
        temperature: f32,
        predict_amount: usize
    ) -> Self
    {
        let words = words.into_iter().map(|word| LayerType::new_undiff(word)).collect();

        Self{
            dictionary,
            words,
            predicted: Vec::with_capacity(predict_amount),
            temperature,
            predict_amount
        }
    }

    pub fn predict_bytes(mut self, network: &mut Network<CurrentNetworkUnit>) -> Box<[u8]>
    {
        let input_amount = self.words.len();

        let mut previous_state: Option<_> = None;

        let dropout_masks = network.create_dropout_masks(
            self.dictionary.words_amount_trait(),
            0.0
        );

        for i in 0..(input_amount + self.predict_amount)
        {
            debug_assert!(i < self.words.len());
            let this_input = unsafe{ self.words.get_unchecked(i) };

            let NetworkOutput{
                state,
                output
            } = network.predict_single_input(
                previous_state.take(),
                &dropout_masks,
                this_input,
                self.temperature
            );

            if i >= (input_amount - 1)
            {
                let word = output.pick_weighed();
                let word = VectorWord::from_raw(word);

                let layer = self.dictionary.word_to_layer(word);
                self.words.push(LayerType::new_undiff(layer));

                self.predicted.extend(self.dictionary.word_to_bytes(word).into_iter());
            }

            previous_state = Some(state);
        }

        self.predicted.into_boxed_slice()
    }
}

pub struct TrainingInfo
{
    pub epochs: usize,
    pub batch_size: usize,
    pub steps_num: usize,
    pub learning_rate: Option<f32>,
    pub calculate_loss: bool,
    pub calculate_accuracy: bool,
    pub ignore_loss: bool
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork
{
    dictionary: DictionaryType,
    network: Network<CurrentNetworkUnit>,
    optimizer: CurrentOptimizer
}

impl NeuralNetwork
{
    pub fn new(dictionary: DictionaryType) -> Self
    {
        let network = Network::new();
        let optimizer = CurrentOptimizer::new();

        Self{dictionary, network, optimizer}
    }

    pub fn save<P: AsRef<Path>>(&mut self, path: P)
    {
        // clear all derivatives info
        self.network.clear();

        let writer = File::create(path).unwrap();

        ciborium::into_writer(self, writer).unwrap();
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ciborium::de::Error<io::Error>>
    {
        let reader = File::open(path)?;

        ciborium::from_reader(reader)
    }

    #[allow(dead_code)]
    pub fn inner_network(&self) -> &Network<CurrentNetworkUnit>
    {
        &self.network
    }

    pub fn apply_gradients(&mut self, gradients: Vec<ThisWeightsContainer<LayerInnerType>>)
    {
        let (gradient_info, hyper) = self.optimizer.info_mut();

        let combined_iter = gradients.into_iter()
            .zip(self.network.layers_mut().iter_mut()
                 .zip(gradient_info.iter_mut()));

        combined_iter.for_each(|(gradients, (network_weights, optimizer_info))|
        {
            *network_weights -= optimizer_info.gradients_to_change(gradients, hyper);
        });

        self.optimizer.advance_time();
    }

    pub fn input_expected_from_text(
        &mut self,
        text: impl Read
    ) -> Vec<VectorWord>
    {
        let word_vectorizer = WordVectorizer::new(&mut self.dictionary, text);

        word_vectorizer.collect()
    }

    pub fn test_loss(&mut self, file: impl Read, calculate_loss: bool, calculate_accuracy: bool)
    {
        let inputs = self.input_expected_from_text(file);

        self.test_loss_inner(&inputs, calculate_loss, calculate_accuracy);
    }


    fn test_loss_inner(
        &mut self,
        inputs: &[VectorWord],
        calculate_loss: bool,
        calculate_accuracy: bool
    )
    {
        let input_outputs = InputOutputIter::new(
            inputs.iter().map(|word|
            {
                self.dictionary.word_to_layer(*word)
            })
        );

        self.network.disable_gradients();

        if calculate_accuracy
        {
            let accuracy = self.network.accuracy(input_outputs.clone());

            println!("accuracy: {}%", accuracy * 100.0);
        }

        if calculate_loss
        {
            let input_outputs = Self::combine_inputs(input_outputs);

            let loss = self.network.feedforward(input_outputs);

            Self::print_loss(true, loss.value_clone() / inputs.len() as f32);
        }

        self.network.enable_gradients();
    }

    fn print_loss(testing: bool, loss: f32)
    {
        let loss_type = if testing
        {
            "testing"
        } else
        {
            "training"
        };

        println!("{loss_type} loss: {loss}");
    }

    fn combine_inputs(input: impl Iterator<Item=(LayerInnerType, LayerInnerType)>) -> JoinableType
    {
        let mut joined: Option<JoinableType> = None;

        for inputs in input
        {
            if let Some(joined) = joined.as_mut()
            {
                joined.join(inputs);
            } else
            {
                joined = Some(JoinableType::new(inputs));
            }
        }

        joined.unwrap()
    }

    fn create_batch(
        &self,
        inputs: &[VectorWord],
        steps_num: usize,
        batch_size: usize
    ) -> JoinableDeepType
    {
        let max_batch_start = inputs.len().saturating_sub(steps_num);

        let mut batch: Option<JoinableDeepType> = None;
        for _ in 0..batch_size
        {
            let batch_start = if max_batch_start == 0
            {
                0
            } else
            {
                fastrand::usize(0..max_batch_start)
            };

            let values = InputOutput::values_slice(
                inputs,
                |word| self.dictionary.word_to_layer(*word),
                batch_start,
                steps_num
            );

            let inputs: JoinableType = Self::combine_inputs(
                values.iter().map(|(a, b)| (a.clone(), b.clone()))
            );

            if let Some(batch) = batch.as_mut()
            {
                batch.join(inputs);
            } else
            {
                batch = Some(JoinableDeepType::new(inputs));
            }
        }

        batch.unwrap()
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
            learning_rate,
            calculate_loss,
            calculate_accuracy,
            ignore_loss
        } = info;

        if let Some(learning_rate) = learning_rate
        {
            self.optimizer.set_learning_rate(learning_rate);
        }

        let batch_step = batch_size * steps_num;

        let inputs = self.input_expected_from_text(text);
        let testing_inputs = if ignore_loss
        {
            Vec::new()
        } else
        {
            match testing_data
            {
                None => Vec::new(),
                Some(testing_data) =>
                {
                    self.input_expected_from_text(testing_data)
                }
            }
        };

        println!("input vector size: {}", self.dictionary.words_amount_trait());

        println!("parameters amount: {}", self.network.parameters_amount());

        println!("batch size: {batch_size}");

        let steps_deviation = steps_num / 10;
        let steps_half_deviation = steps_deviation / 2;

        println!(
            "steps amount: {} to {}",
            steps_num - steps_half_deviation,
            steps_num + steps_half_deviation
        );
        
        let inputs_per_epoch = (inputs.len() / batch_step).max(1);
        println!("calculate loss every ~{inputs_per_epoch} inputs");

        let output_loss = |network: &mut NeuralNetwork|
        {
            if testing_inputs.is_empty()
            {
                return;
            }

            network.test_loss_inner(&testing_inputs, calculate_loss, calculate_accuracy);
        };

        for input_index in 0..epochs
        {
            eprintln!("iteration: {input_index}");

            let steps_num = {
                let this_dev = fastrand::i32(0..(steps_deviation as i32 + 1));
                let this_dev = this_dev - steps_half_deviation as i32;

                (steps_num as i32 + this_dev) as usize
            };

            let print_loss = (input_index % inputs_per_epoch) == inputs_per_epoch - 1;
            if print_loss
            {
                output_loss(self);
            }

            let mut kahan_sum = KahanSum::new();

            let mut joinable = self.create_batch(&inputs, steps_num, batch_size);

            let gradients = (0..batch_size).map(|_|
            {
                let values = joinable.pop();

                let (loss, mut gradients): (f32, Vec<_>) = self.network.gradients(values);

                kahan_sum.add(loss as f64 / batch_size as f64);
                gradients.iter_mut().for_each(|gradient| *gradient /= batch_size as f32);

                gradients
            }).reduce(|mut acc, this|
            {
                acc.iter_mut().zip(this.into_iter()).for_each(|(acc, this)|
                {
                    *acc += this;
                });

                acc
            }).expect("batch size must not be 0");

            let batch_loss = kahan_sum.value() / steps_num as f64;

            Self::print_loss(false, batch_loss as f32);

            self.apply_gradients(gradients);
        }

        output_loss(self);
    }

    #[allow(dead_code)]
    pub fn predict_text(&mut self, text: &str, amount: usize, temperature: f32) -> String
    {
        let output = self.predict_inner(text.as_bytes(), amount, temperature)
            .into_iter().copied()
            .filter(|&c| c != b'\0').collect::<Vec<_>>();
        
        String::from_utf8_lossy(&output).to_string()
    }

    pub fn predict_bytes(&mut self, text: &str, amount: usize, temperature: f32) -> Box<[u8]>
    {
        self.predict_inner(text.as_bytes(), amount, temperature)
    }

    fn predict_inner(&mut self, start: &[u8], amount: usize, temperature: f32) -> Box<[u8]>
    {
        let word_vectorizer = WordVectorizer::new(&mut self.dictionary, start);

        self.network.disable_gradients();

        let predictor = {
            let words = word_vectorizer.collect::<Vec<_>>();

            let words = words.into_iter().map(|v|
            {
                self.dictionary.word_to_layer(v)
            }).collect::<Vec<_>>();

            Predictor::new(&mut self.dictionary, words, temperature, amount)
        };

        let predicted = predictor.predict_bytes(&mut self.network);

        self.network.enable_gradients();

        predicted
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

    #[test]
    fn softmax()
    {
        let mut test_layer = LayerType::from_raw([1.0, 2.0, 8.0], 3, 1).value_clone();
        Softmaxer::softmax(&mut test_layer);

        let softmaxed = test_layer;

        softmaxed.as_vec().into_iter().zip([0.001, 0.002, 0.997].iter())
            .for_each(|(softmaxed, correct)|
            {
                assert!(
                    close_enough(softmaxed, *correct, 0.2),
                    "softmaxed: {}, correct: {}",
                    softmaxed,
                    *correct
                );
            });
    }
}
