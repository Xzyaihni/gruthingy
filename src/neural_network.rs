use std::{
    f32,
    mem,
    slice,
    io::{self, Read},
    fs::File,
    path::Path
};

use serde::{Serialize, Deserialize};

use network::{NewableLayer, NetworkOutput, Network};
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

pub use network::WeightsNamed;

pub use containers::{
    LayerType,
    ScalarType,
    LayerInnerType,
    Softmaxer,
    CloneableWrapper
};

#[allow(unused_imports)]
use gru::GRU;

// #[allow(unused_imports)]
// use lstm::LSTM;

mod network_unit;
mod network;
mod gru;
// mod lstm;

pub mod containers;


pub const HIDDEN_AMOUNT: usize = 256;
pub const LAYERS_AMOUNT: usize = 4;

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
pub type CurrentNetworkUnit = GRU;

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
        words: Vec<LayerType>,
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

                self.words.push(self.dictionary.word_to_layer(word));

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamGradientInfo
{
    m: LayerInnerType,
    v: LayerInnerType
}

impl NewableLayer for AdamGradientInfo
{
    fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            m: LayerInnerType::new(previous_size, this_size),
            v: LayerInnerType::new(previous_size, this_size)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamXGradientInfo
{
    m: LayerInnerType,
    v: LayerInnerType,
    v_hat: Option<LayerInnerType>
}

impl NewableLayer for AdamXGradientInfo
{
    fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            m: LayerInnerType::new(previous_size, this_size),
            v: LayerInnerType::new(previous_size, this_size),
            v_hat: None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSignGradientInfo
{
    m: LayerInnerType
}

impl NewableLayer for PowerSignGradientInfo
{
    fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            m: LayerInnerType::new(previous_size, this_size)
        }
    }
}

type OutputGradients = <CurrentNetworkUnit as NetworkUnit>::ThisWeightsContainer<LayerInnerType>;
type AdamGradientsContainer = <CurrentNetworkUnit as NetworkUnit>::ThisWeightsContainer<AdamGradientInfo>;
type AdamXGradientsContainer = <CurrentNetworkUnit as NetworkUnit>::ThisWeightsContainer<AdamXGradientInfo>;
type PowerSignGradientsContainer = <CurrentNetworkUnit as NetworkUnit>::ThisWeightsContainer<PowerSignGradientInfo>;

pub trait Optimizer
{
    type HyperParams;
    type WeightParam;

    fn new() -> Self;

    fn gradient_to_change_indexed(
        &mut self,
        layer_index: usize,
        weight_index: usize,
        gradient: LayerInnerType
    ) -> LayerInnerType;

    fn gradient_to_change(
        gradient_info: &mut Self::WeightParam,
        gradient: LayerInnerType,
        hyper: &Self::HyperParams
    ) -> LayerInnerType;

    fn advance_time(&mut self);
    fn set_learning_rate(&mut self, learning_rate: f32);
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SGD
{
    learning_rate: f32
}

impl Optimizer for SGD
{
    type HyperParams = f32;
    type WeightParam = ();

    fn new() -> Self
    {
        Self{learning_rate: 0.001}
    }

    fn gradient_to_change_indexed(
        &mut self,
        _layer_index: usize,
        _weight_index: usize,
        gradient: LayerInnerType
    ) -> LayerInnerType
    {
        Self::gradient_to_change(&mut (), gradient, &self.learning_rate)
    }

    fn gradient_to_change(
        _gradient_info: &mut Self::WeightParam,
        gradient: LayerInnerType,
        hyper: &Self::HyperParams
    ) -> LayerInnerType
    {
        gradient * *hyper
    }

    fn advance_time(&mut self) {}
    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.learning_rate = learning_rate;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PowerSignHyperparams
{
    pub b1: f32,
    pub learning_rate: f32,
    pub t: i32
}

impl PowerSignHyperparams
{
    pub fn new() -> Self
    {
        Self{
            b1: 0.9,
            learning_rate: 0.1,
            t: 1
        }
    }

    pub fn advance_time(&mut self)
    {
        self.t += 1;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PowerSign
{
    gradients_info: Vec<PowerSignGradientsContainer>,
    hyper: PowerSignHyperparams
}

impl Optimizer for PowerSign
{
    type HyperParams = PowerSignHyperparams;
    type WeightParam = PowerSignGradientInfo;

    fn new() -> Self
    {
        let gradients_info = (0..LAYERS_AMOUNT).map(|_|
        {
            PowerSignGradientsContainer::new_container()
        }).collect::<Vec<_>>();

        let hyper = PowerSignHyperparams::new();

        Self{gradients_info, hyper}
    }

    fn gradient_to_change_indexed(
        &mut self,
        layer_index: usize,
        weight_index: usize,
        gradient: LayerInnerType
    ) -> LayerInnerType
    {
        let gradient_info = self.gradients_info[layer_index]
            .raw_index_mut(weight_index);

        Self::gradient_to_change(gradient_info, gradient, &self.hyper)
    }

    fn gradient_to_change(
        gradient_info: &mut Self::WeightParam,
        gradient: LayerInnerType,
        hyper: &Self::HyperParams
    ) -> LayerInnerType
    {
        gradient_info.m = &gradient_info.m * hyper.b1 + &gradient * (1.0 - hyper.b1);

        let decay = DECAY_FUNCTION.decay(hyper.learning_rate, hyper.t);

        let mut this = gradient.signum() * gradient_info.m.signum() * decay;
        this.exp();

        this * gradient
    }

    fn advance_time(&mut self)
    {
        self.hyper.advance_time();
    }

    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.hyper.learning_rate = learning_rate;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AdamXHyperparams
{
    pub a: f32,
    pub b1: f32,
    pub b2: f32,
    pub epsilon: f32,
    pub t: i32
}

impl AdamXHyperparams
{
    pub fn new() -> Self
    {
        Self{
            a: 0.001,
            b1: 0.9,
            b2: 0.999,
            epsilon: 1e-8,
            t: 1
        }
    }

    pub fn advance_time(&mut self)
    {
        self.t += 1;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AdamX
{
    gradients_info: Vec<AdamXGradientsContainer>,
    hyper: AdamXHyperparams
}

impl Optimizer for AdamX
{
    type HyperParams = AdamXHyperparams;
    type WeightParam = AdamXGradientInfo;

    fn new() -> Self
    {
        let gradients_info = (0..LAYERS_AMOUNT).map(|_|
        {
            AdamXGradientsContainer::new_container()
        }).collect::<Vec<_>>();

        let hyper = AdamXHyperparams::new();

        Self{gradients_info, hyper}
    }

    fn gradient_to_change_indexed(
        &mut self,
        layer_index: usize,
        weight_index: usize,
        gradient: LayerInnerType
    ) -> LayerInnerType
    {
        let gradient_info = self.gradients_info[layer_index]
            .raw_index_mut(weight_index);

        Self::gradient_to_change(gradient_info, gradient, &self.hyper)
    }

    fn gradient_to_change(
        gradient_info: &mut Self::WeightParam,
        gradient: LayerInnerType,
        hyper: &Self::HyperParams
    ) -> LayerInnerType
    {
        let b1_t = DECAY_FUNCTION.decay(hyper.b1, hyper.t);
        let one_minus_b1_t = 1.0 - b1_t;

        gradient_info.m = &gradient_info.m * b1_t + &gradient * one_minus_b1_t;
        gradient_info.v = &gradient_info.v * hyper.b2 + (&gradient * &gradient) * (1.0 - hyper.b2);

        if let Some(v_hat) = gradient_info.v_hat.as_mut()
        {
            let one_minus_b1_tlast = 1.0 - DECAY_FUNCTION.decay(hyper.b1, hyper.t - 1);

            let lhs = (one_minus_b1_t).powi(2) / (one_minus_b1_tlast).powi(2);

            let mut new_v_hat = &*v_hat * lhs;
            new_v_hat.max(&gradient_info.v);

            *v_hat = new_v_hat;
        } else
        {
            gradient_info.v_hat = Some(gradient_info.v.clone());
        }

        // it can be a / t.sqrt() but this is fine
        let a_t = hyper.a;

        let rhs = gradient_info.v_hat.as_ref().unwrap().clone_sqrt() + hyper.epsilon;

        (&gradient_info.m * a_t) / rhs
    }

    fn advance_time(&mut self)
    {
        self.hyper.advance_time();
    }

    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.hyper.a = learning_rate;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AdamHyperparams
{
    pub a: f32,
    pub b1: f32,
    pub b2: f32,
    pub epsilon: f32,
    pub t: i32
}

impl AdamHyperparams
{
    pub fn new() -> Self
    {
        Self{
            a: 0.001,
            b1: 0.9,
            b2: 0.999,
            epsilon: 1e-8,
            t: 1
        }
    }

    pub fn advance_time(&mut self)
    {
        self.t += 1;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Adam
{
    gradients_info: Vec<AdamGradientsContainer>,
    hyper: AdamHyperparams
}

impl Optimizer for Adam
{
    type HyperParams = AdamHyperparams;
    type WeightParam = AdamGradientInfo;

    fn new() -> Self
    {
        let gradients_info = (0..LAYERS_AMOUNT).map(|_|
        {
            AdamGradientsContainer::new_container()
        }).collect::<Vec<_>>();

        let hyper = AdamHyperparams::new();

        Self{gradients_info, hyper}
    }

    fn gradient_to_change_indexed(
        &mut self,
        layer_index: usize,
        weight_index: usize,
        gradient: LayerInnerType
    ) -> LayerInnerType
    {
        let gradient_info = self.gradients_info[layer_index]
            .raw_index_mut(weight_index);

        Self::gradient_to_change(gradient_info, gradient, &self.hyper)
    }

    fn gradient_to_change(
        gradient_info: &mut Self::WeightParam,
        gradient: LayerInnerType,
        hyper: &Self::HyperParams
    ) -> LayerInnerType
    {
        let one_minus_b1_t = 1.0 - DECAY_FUNCTION.decay(hyper.b1, hyper.t);
        let one_minus_b2_t = 1.0 - DECAY_FUNCTION.decay(hyper.b2, hyper.t);

        gradient_info.m = &gradient_info.m * hyper.b1 + &gradient * (1.0 - hyper.b1);
        gradient_info.v = &gradient_info.v * hyper.b2 + (&gradient * &gradient) * (1.0 - hyper.b2);

        let a_t = hyper.a * one_minus_b2_t.sqrt() / one_minus_b1_t;

        (&gradient_info.m * a_t) / (gradient_info.v.clone_sqrt() + hyper.epsilon)
    }

    fn advance_time(&mut self)
    {
        self.hyper.advance_time();
    }

    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.hyper.a = learning_rate;
    }
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

    pub fn apply_gradients(&mut self, gradients: Vec<OutputGradients>)
    {
        let combined_iter = gradients.into_iter()
            .zip(self.network.layers_mut().iter_mut());

        combined_iter.enumerate().for_each(|(layer_index, (gradients, network_weights))|
        {
            network_weights.iter_mut().zip(gradients).enumerate()
                .for_each(|(weight_index, (weight, gradient))|
                {
                    let change = self.optimizer.gradient_to_change_indexed(
                        layer_index,
                        weight_index,
                        Self::gradient_clipped(gradient)
                    );

                    *weight.value_mut() -= change;
                });
        });

        self.optimizer.advance_time();
    }

    fn gradient_clipped(gradient: LayerInnerType) -> LayerInnerType
    {
        gradient.cap_magnitude(GRADIENT_CLIP)
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
                CloneableWrapper(self.dictionary.word_to_layer(*word))
            })
        ).map(|(a, b)| (a.0, b.0));

        self.network.disable_gradients();

        if calculate_accuracy
        {
            let accuracy = self.network.accuracy(input_outputs.clone());

            println!("accuracy: {}%", accuracy * 100.0);
        }

        if calculate_loss
        {
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

            let max_batch_start = inputs.len().saturating_sub(steps_num);

            let print_loss = (input_index % inputs_per_epoch) == inputs_per_epoch - 1;
            if print_loss
            {
                output_loss(self);
            }

            let mut batch_loss = 0.0;
            let gradients = (0..batch_size).map(|_|
            {
                let batch_start = if max_batch_start == 0
                {
                    0
                } else
                {
                    fastrand::usize(0..max_batch_start)
                };

                let values = InputOutput::values_slice(
                    &inputs,
                    |word| CloneableWrapper(self.dictionary.word_to_layer(*word)),
                    batch_start,
                    steps_num
                );

                let (loss, mut gradients) = {
                    let values = values.iter().map(|(a, b)| (a.clone().0, b.clone().0));

                    self.network.gradients(values)
                };

                batch_loss += loss / batch_size as f32;
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

            Self::print_loss(false, batch_loss / steps_num as f32);

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
    fn adam_correct()
    {
        let mut old_weight = vec![3.21, 7.65];

        let mut m = vec![0.0, 0.0];
        let mut v = vec![0.0, 0.0];
        
        let mut g = vec![3.1_f32, -0.8_f32];

        let mut t = 1;

        for _ in 0..2
        {
            let a = 0.001;
            let b1 = 0.9;
            let b2 = 0.999;

            let epsilon = 10e-8;

            let adam_g = {
                let mut gradient_info = AdamGradientInfo{
                    m: LayerInnerType::from_raw(m.clone().into_boxed_slice(), 2, 1),
                    v: LayerInnerType::from_raw(v.clone().into_boxed_slice(), 2, 1)
                };

                let gradient = LayerInnerType::from_raw(g.clone().into_boxed_slice(), 2, 1);

                let hyper = AdamHyperparams{
                    a,
                    b1,
                    b2,
                    epsilon,
                    t
                };

                let change = Adam::gradient_to_change(
                    &mut gradient_info,
                    gradient.clone(),
                    &hyper
                );

                LayerInnerType::from_raw(old_weight.clone().into_boxed_slice(), 2, 1) + change
            };

            m = vec![
                b1 * m[0] + (1.0 - b1) * g[0],
                b1 * m[1] + (1.0 - b1) * g[1]
            ];

            v = vec![
                b2 * v[0] + (1.0 - b2) * g[0].powi(2),
                b2 * v[1] + (1.0 - b2) * g[1].powi(2)
            ];

            let m_hat = vec![
                m[0] / (1.0 - b1.powi(t)),
                m[1] / (1.0 - b1.powi(t))
            ];

            let v_hat = vec![
                v[0] / (1.0 - b2.powi(t)),
                v[1] / (1.0 - b2.powi(t))
            ];

            let new_weight = vec![
                old_weight[0] + a * m_hat[0] / (v_hat[0].sqrt() + epsilon),
                old_weight[1] + a * m_hat[1] / (v_hat[1].sqrt() + epsilon)
            ];

            if t == 1
            {
                let mut adam_g = adam_g.as_vec().into_iter();
                assert_eq!(new_weight[0], adam_g.next().unwrap());
                assert_eq!(new_weight[1], adam_g.next().unwrap());
            } else
            {
                let mut adam_g = adam_g.as_vec().into_iter();
                assert_eq!(new_weight[0], adam_g.next().unwrap());
                assert_eq!(new_weight[1], adam_g.next().unwrap());
            }

            t += 1;

            old_weight = new_weight;
            g = vec![
                g[0] - 1.1,
                g[1] - 2.34
            ];
        }
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
