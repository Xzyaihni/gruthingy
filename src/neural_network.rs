use std::{
    f32,
    mem,
    slice,
    io::{self, Read},
    fs::File,
    path::Path
};

use serde::{Serialize, Deserialize};

#[allow(unused_imports)]
use gru::{GRU, GRUGradients, GRUOutput, GRUFullGradients};

#[allow(unused_imports)]
use crate::word_vectorizer::{
    CharDictionary,
    WordDictionary,
    NetworkDictionary,
    WordVectorizer,
    VectorWord
};

pub use containers::{
    LayerType,
    ScalarType,
    LayerInnerType,
    SoftmaxedLayer
};

mod gru;

pub mod containers;


pub const HIDDEN_AMOUNT: usize = 25;
pub const LAYERS_AMOUNT: usize = 4;

pub const LAYER_ACTIVATION: AFType = AFType::LeakyRelu;

// these 2 r related, WordDictionary uses a dictionary and CharDictionary doesnt
pub const USES_DICTIONARY: bool = true;
pub type DictionaryType = WordDictionary;

#[allow(dead_code)]
pub enum AFType
{
    Tanh,
    LeakyRelu
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientInfo
{
    m: LayerInnerType,
    v: LayerInnerType
}

impl GradientInfo
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            m: LayerInnerType::new(previous_size, this_size),
            v: LayerInnerType::new(previous_size, this_size)
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GradientsInfo
{
    pub input_update_gradients: GradientInfo,
    pub input_reset_gradients: GradientInfo,
    pub input_activation_gradients: GradientInfo,
    pub hidden_update_gradients: GradientInfo,
    pub hidden_reset_gradients: GradientInfo,
    pub hidden_activation_gradients: GradientInfo,
    pub update_bias_gradients: GradientInfo,
    pub reset_bias_gradients: GradientInfo,
    pub activation_bias_gradients: GradientInfo,
    pub output_gradients: GradientInfo
}

impl GradientsInfo
{
    pub fn new(word_vector_size: usize) -> Self
    {
        Self{
        	input_update_gradients: GradientInfo::new(HIDDEN_AMOUNT, word_vector_size),
        	input_reset_gradients: GradientInfo::new(HIDDEN_AMOUNT, word_vector_size),
        	input_activation_gradients: GradientInfo::new(HIDDEN_AMOUNT, word_vector_size),
        	hidden_update_gradients: GradientInfo::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
        	hidden_reset_gradients: GradientInfo::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
        	hidden_activation_gradients: GradientInfo::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
            update_bias_gradients: GradientInfo::new(HIDDEN_AMOUNT, 1),
            reset_bias_gradients: GradientInfo::new(HIDDEN_AMOUNT, 1),
            activation_bias_gradients: GradientInfo::new(HIDDEN_AMOUNT, 1),
            output_gradients: GradientInfo::new(word_vector_size, HIDDEN_AMOUNT)
        }
    }

    fn gradient_to_change(
        gradient_info: &mut GradientInfo,
        gradient: LayerInnerType,
        hyper: &AdamHyperparams
    ) -> LayerInnerType
    {
        gradient_info.m = &gradient_info.m * hyper.b1 + &gradient * (1.0 - hyper.b1);
        gradient_info.v = &gradient_info.v * hyper.b2 + (&gradient * &gradient) * (1.0 - hyper.b2);

        let a_t = hyper.a * hyper.one_minus_b2_t.sqrt() / hyper.one_minus_b1_t;

        (&gradient_info.m * a_t) / (gradient_info.v.clone_sqrt() + hyper.epsilon)
    }
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

    pub fn predict_bytes(mut self, network: &mut GRU) -> Box<[u8]>
    {
        let input_amount = self.words.len();

        let mut previous_hiddens = vec![LayerType::new(HIDDEN_AMOUNT, 1); LAYERS_AMOUNT];
        for i in 0..(input_amount + self.predict_amount)
        {
            debug_assert!(i < self.words.len());
            let this_input = unsafe{ self.words.get_unchecked(i) };

            let outputs = network.feedforward_single(
                Some(&previous_hiddens),
                this_input
            );

            let output = outputs.last_output_ref().clone();
            previous_hiddens = outputs.hiddens();

            if i >= (input_amount - 1)
            {
                let word = output.pick_weighed(self.temperature);
                let word = VectorWord::from_raw(word);

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
    pub learning_rate: f32,
    pub calculate_accuracy: bool,
    pub ignore_loss: bool
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AdamHyperparams
{
    pub a: f32,
    pub b1: f32,
    pub b2: f32,
    pub epsilon: f32,
    pub t: i32,
    pub one_minus_b1_t: f32,
    pub one_minus_b2_t: f32
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
            one_minus_b1_t: 0.0,
            one_minus_b2_t: 0.0
        };

        this.update_t_vars();

        this
    }

    fn update_t_vars(&mut self)
    {
        self.one_minus_b1_t = 1.0 - self.b1.powi(self.t);
        self.one_minus_b2_t = 1.0 - self.b2.powi(self.t);
    }

    pub fn advance_time(&mut self)
    {
        self.t += 1;

        self.update_t_vars();
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork
{
    dictionary: DictionaryType,
    network: GRU,
    gradients_info: Vec<GradientsInfo>,
    hyper: AdamHyperparams
}

impl NeuralNetwork
{
    pub fn new(dictionary: DictionaryType) -> Self
    {
        let words_vector_size = dictionary.words_amount();
        let network = GRU::new(words_vector_size);

        let gradients_info = (0..LAYERS_AMOUNT).map(|_| GradientsInfo::new(words_vector_size))
            .collect::<Vec<_>>();

        let hyper = AdamHyperparams::new();

        Self{dictionary, network, gradients_info, hyper}
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

    pub fn words_amount(&self) -> usize
    {
        self.dictionary.words_amount()
    }

    pub fn inner_network(&self) -> &GRU
    {
        &self.network
    }

    // suckines
    pub fn apply_gradients(&mut self, gradients: GRUFullGradients)
    {
        let combined_iter = gradients.0.into_iter()
            .zip(self.network.layers.iter_mut()
                 .zip(self.gradients_info.iter_mut())
            );

        combined_iter.for_each(|(gradients, (network_weights, gradients_info))|
        {
            let GRUGradients{
                input_update_gradients,
                input_reset_gradients,
                input_activation_gradients,
                hidden_update_gradients,
                hidden_reset_gradients,
                hidden_activation_gradients,
                update_bias_gradients,
                reset_bias_gradients,
                activation_bias_gradients,
                output_gradients
            } = gradients;

            let hyper = &mut self.hyper;

            *network_weights.input_update_weights.value_mut() -=
                GradientsInfo::gradient_to_change(
                    &mut gradients_info.input_update_gradients,
                    input_update_gradients,
                    hyper
                );

            *network_weights.input_reset_weights.value_mut() -= 
				GradientsInfo::gradient_to_change(
                    &mut gradients_info.input_reset_gradients,
                    input_reset_gradients,
                    hyper
                );
            
            *network_weights.input_activation_weights.value_mut() -= 
				GradientsInfo::gradient_to_change(
                    &mut gradients_info.input_activation_gradients,
                    input_activation_gradients,
                    hyper
                );

            *network_weights.hidden_update_weights.value_mut() -= 
				GradientsInfo::gradient_to_change(
                    &mut gradients_info.hidden_update_gradients,
                    hidden_update_gradients,
                    hyper
                );

            *network_weights.hidden_reset_weights.value_mut() -= 
				GradientsInfo::gradient_to_change(
                    &mut gradients_info.hidden_reset_gradients,
                    hidden_reset_gradients,
                    hyper
                );
            
            *network_weights.hidden_activation_weights.value_mut() -= 
				GradientsInfo::gradient_to_change(
                    &mut gradients_info.hidden_activation_gradients,
                    hidden_activation_gradients,
                    hyper
                );
            
            *network_weights.update_biases.value_mut() -= 
				GradientsInfo::gradient_to_change(
                    &mut gradients_info.update_bias_gradients,
                    update_bias_gradients,
                    hyper
                );
            
            *network_weights.reset_biases.value_mut() -= 
				GradientsInfo::gradient_to_change(
                    &mut gradients_info.reset_bias_gradients,
                    reset_bias_gradients,
                    hyper
                );
            
            *network_weights.activation_biases.value_mut() -= 
				GradientsInfo::gradient_to_change(
                    &mut gradients_info.activation_bias_gradients,
                    activation_bias_gradients,
                    hyper
                );
            
            *network_weights.output_weights.value_mut() -= 
				GradientsInfo::gradient_to_change(
                    &mut gradients_info.output_gradients,
                    output_gradients,
                    hyper
                );
        });

        self.hyper.advance_time();
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

        self.test_loss_inner(&inputs, calculate_accuracy);
    }


    fn test_loss_inner(
        &mut self,
        inputs: &[VectorWord],
        calculate_accuracy: bool
    )
    {
        let input_outputs = InputOutputIter::new(
            inputs.iter().map(|word|
            {
                self.dictionary.word_to_layer(*word)
            })
        );

        if calculate_accuracy
        {
            let accuracy = self.network.accuracy(input_outputs);

            println!("accuracy: {}%", accuracy * 100.0);
        } else
        {
            let loss = self.network.loss(input_outputs);

            Self::print_loss(true, loss.value_clone());
        }
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
            calculate_accuracy,
            ignore_loss
        } = info;

        self.hyper.a = learning_rate;

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

        println!("batch size: {batch_size}");
        println!("steps amount: {steps_num}");
        
        let epochs_per_input = (inputs.len() / batch_step).max(1);
        println!("calculate loss every {epochs_per_input} epochs");

        let output_loss = |network: &mut NeuralNetwork|
        {
            if testing_inputs.is_empty()
            {
                return;
            }

            network.test_loss_inner(&testing_inputs, calculate_accuracy);
        };

        let max_batch_start = inputs.len().saturating_sub(steps_num);

        // whats an epoch? cool word is wut it is
        // at some point i found out wut it was (going over the whole training data once)
        // but i dont rly feel like changing a silly thing like that
        for epoch in 0..epochs
        {
            eprintln!("epoch: {epoch}");

            let print_loss = (epoch % epochs_per_input) == epochs_per_input - 1;
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
                    |word| self.dictionary.word_to_layer(*word),
                    batch_start,
                    steps_num
                );

                let (loss, mut gradients) = self.network.gradients(values.iter());

                batch_loss += loss / batch_size as f32;
                gradients /= batch_size as f32;

                gradients
            }).reduce(|mut acc, this|
            {
                acc.0.iter_mut().zip(this.0.into_iter()).for_each(|(acc, this)|
                {
                    *acc += this;
                });

                acc
            }).expect("batch size must not be 0");

            Self::print_loss(false, batch_loss);

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

        let predictor = {
            let words = word_vectorizer.collect::<Vec<_>>();

            let words = words.into_iter().map(|v|
            {
                self.dictionary.word_to_layer(v)
            }).collect::<Vec<_>>();

            Predictor::new(&mut self.dictionary, words, temperature, amount)
        };

        predictor.predict_bytes(&mut self.network)
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
                let mut gradient_info = GradientInfo{
                    m: LayerInnerType::from_raw(m.clone().into_boxed_slice(), 2, 1),
                    v: LayerInnerType::from_raw(v.clone().into_boxed_slice(), 2, 1)
                };

                let gradient = LayerInnerType::from_raw(g.clone().into_boxed_slice(), 2, 1);

                let hyper = AdamHyperparams{
                    a,
                    b1,
                    b2,
                    epsilon,
                    t,
                    one_minus_b1_t: 1.0 - b1.powi(t),
                    one_minus_b2_t: 1.0 - b2.powi(t)
                };

                let change = GradientsInfo::gradient_to_change(
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
        let test_layer = LayerType::from_raw([1.0, 2.0, 8.0], 3, 1);

        let softmaxed = SoftmaxedLayer::new(test_layer);

        softmaxed.0.as_vec().into_iter().zip([0.001, 0.002, 0.997].iter())
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
