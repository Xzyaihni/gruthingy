use std::{
    f64,
    mem,
    slice,
    io::{self, Read},
    fs::File,
    path::Path,
    ops::{Mul, Div}
};

use serde::{Serialize, Deserialize, de::DeserializeOwned};

// #[allow(unused_imports)]
// use rnn::{RNN, RNNGradients};

#[allow(unused_imports)]
use gru::{GRU, GRUGradients, GRUOutput};

use super::word_vectorizer::{NetworkDictionary, WordVectorizer, VectorWord};

pub use containers::{
    NetworkType,
    MatrixWrapper,
    ArrayWrapper,
    GenericContainer,
    WeightsIterValue,
    SoftmaxedLayer
};

// mod rnn;
mod gru;

pub mod containers;


pub const HIDDEN_AMOUNT: usize = 1;
pub const LAYERS_AMOUNT: usize = 2;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct GradientInfo<T>
{
    m: T,
    v: T
}

impl<T> GradientInfo<T>
where
    T: NetworkType,
    for<'a> &'a T: Mul<f64, Output=T> + Mul<&'a T, Output=T> + Mul<T, Output=T>,
    for<'a> &'a T: Div<f64, Output=T>
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            m: T::new(previous_size, this_size),
            v: T::new(previous_size, this_size)
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct GradientsInfo<T>
{
    pub input_update_gradients: GradientInfo<T>,
    pub input_reset_gradients: GradientInfo<T>,
    pub input_activation_gradients: GradientInfo<T>,
    pub hidden_update_gradients: GradientInfo<T>,
    pub hidden_reset_gradients: GradientInfo<T>,
    pub hidden_activation_gradients: GradientInfo<T>,
    pub update_bias_gradients: GradientInfo<T>,
    pub reset_bias_gradients: GradientInfo<T>,
    pub activation_bias_gradients: GradientInfo<T>,
    pub output_gradients: GradientInfo<T>
}

impl<T> GradientsInfo<T>
where
    T: NetworkType,
    for<'a> &'a T: Mul<f64, Output=T> + Mul<&'a T, Output=T> + Mul<T, Output=T>,
    for<'a> &'a T: Div<f64, Output=T>
{
    pub fn new(word_vector_size: usize) -> Self
    {
        Self{
        	input_update_gradients: GradientInfo::new(word_vector_size, HIDDEN_AMOUNT),
        	input_reset_gradients: GradientInfo::new(word_vector_size, HIDDEN_AMOUNT),
        	input_activation_gradients: GradientInfo::new(word_vector_size, HIDDEN_AMOUNT),
        	hidden_update_gradients: GradientInfo::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
        	hidden_reset_gradients: GradientInfo::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
        	hidden_activation_gradients: GradientInfo::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
            update_bias_gradients: GradientInfo::new(HIDDEN_AMOUNT, 1),
            reset_bias_gradients: GradientInfo::new(HIDDEN_AMOUNT, 1),
            activation_bias_gradients: GradientInfo::new(HIDDEN_AMOUNT, 1),
            output_gradients: GradientInfo::new(HIDDEN_AMOUNT, word_vector_size)
        }
    }

    fn gradient_to_change(
        gradient_info: &mut GradientInfo<T>,
        gradient: T,
        hyper: &AdamHyperparams
    ) -> T
    {
        gradient_info.m = &gradient_info.m * hyper.b1 + &gradient * (1.0 - hyper.b1);
        gradient_info.v = &gradient_info.v * hyper.b2 + (&gradient * &gradient) * (1.0 - hyper.b2);

        let a_t = hyper.a * hyper.one_minus_b2_t.sqrt() / hyper.one_minus_b1_t;

        (&gradient_info.m * -a_t) / (gradient_info.v.clone_sqrt() + hyper.epsilon)
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

struct Predictor<'a, N, D>
{
    dictionary: &'a mut D,
    words: Vec<N>,
    predicted: Vec<u8>,
    temperature: f64,
    predict_amount: usize
}

impl<'a, N, D> Predictor<'a, N, D>
where
    N: NetworkType,
    for<'b> &'b N: Mul<f64, Output=N> + Mul<&'b N, Output=N> + Mul<N, Output=N>,
    for<'b> &'b N: Div<f64, Output=N>,
    D: NetworkDictionary
{
    pub fn new(
        dictionary: &'a mut D,
        words: Vec<N>,
        temperature: f64,
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

    pub fn predict_bytes(mut self, network: &GRU<N>) -> Box<[u8]>
    {
        let input_amount = self.words.len();

        let mut previous_hiddens: Vec<N> = vec![N::new(HIDDEN_AMOUNT, 1); LAYERS_AMOUNT];
        for i in 0..(input_amount + self.predict_amount)
        {
            debug_assert!(i < self.words.len());
            let this_input = unsafe{ self.words.get_unchecked(i) };

            let outputs = network.feedforward_single(
                &previous_hiddens.iter().collect::<Vec<_>>(),
                this_input
            );

            let output = outputs.last_output_ref().clone();
            previous_hiddens = outputs.hiddens();

            if i >= (input_amount - 1)
            {
                let word = SoftmaxedLayer::pick_weighed_associated(&output, self.temperature);
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
    pub batch_start: usize,
    pub batch_size: usize,
    pub steps_num: usize,
    pub learning_rate: f64,
    pub calculate_accuracy: bool,
    pub ignore_loss: bool
}

#[derive(Serialize, Deserialize)]
pub struct AdamHyperparams
{
    pub a: f64,
    pub b1: f64,
    pub b2: f64,
    pub epsilon: f64,
    pub t: i32,
    pub one_minus_b1_t: f64,
    pub one_minus_b2_t: f64

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

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork<T, D>
{
    dictionary: D,
    network: GRU<T>,
    gradients_info: Vec<GradientsInfo<T>>,
    hyper: AdamHyperparams
}

impl<T, D> NeuralNetwork<T, D>
where
    T: NetworkType,
    for<'a> &'a T: Mul<f64, Output=T> + Mul<&'a T, Output=T> + Mul<T, Output=T>,
    for<'a> &'a T: Div<f64, Output=T>,
    D: NetworkDictionary + Serialize + DeserializeOwned
{
    pub fn new(dictionary: D) -> Self
    {
        let words_vector_size = dictionary.words_amount();
        let network = GRU::new(words_vector_size);

        let gradients_info = (0..LAYERS_AMOUNT).map(|_| GradientsInfo::new(words_vector_size))
            .collect::<Vec<_>>();

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

    // suckines
    pub fn apply_gradients(&mut self, gradients: Vec<GRUGradients<T>>)
    {
        let combined_iter = gradients.into_iter()
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

            network_weights.input_update_weights += GradientsInfo::gradient_to_change(
                &mut gradients_info.input_update_gradients,
                input_update_gradients,
                hyper
            );

            network_weights.input_reset_weights += GradientsInfo::gradient_to_change(
                &mut gradients_info.input_reset_gradients,
                input_reset_gradients,
                hyper
            );
            
            network_weights.input_activation_weights += GradientsInfo::gradient_to_change(
                &mut gradients_info.input_activation_gradients,
                input_activation_gradients,
                hyper
            );

            network_weights.hidden_update_weights += GradientsInfo::gradient_to_change(
                &mut gradients_info.hidden_update_gradients,
                hidden_update_gradients,
                hyper
            );

            network_weights.hidden_reset_weights += GradientsInfo::gradient_to_change(
                &mut gradients_info.hidden_reset_gradients,
                hidden_reset_gradients,
                hyper
            );
            
            network_weights.hidden_activation_weights += GradientsInfo::gradient_to_change(
                &mut gradients_info.hidden_activation_gradients,
                hidden_activation_gradients,
                hyper
            );
            
            network_weights.update_biases += GradientsInfo::gradient_to_change(
                &mut gradients_info.update_bias_gradients,
                update_bias_gradients,
                hyper
            );
            
            network_weights.reset_biases += GradientsInfo::gradient_to_change(
                &mut gradients_info.reset_bias_gradients,
                reset_bias_gradients,
                hyper
            );
            
            network_weights.activation_biases += GradientsInfo::gradient_to_change(
                &mut gradients_info.activation_bias_gradients,
                activation_bias_gradients,
                hyper
            );
            
            network_weights.output_weights += GradientsInfo::gradient_to_change(
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
        &self,
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
            batch_start,
            batch_size,
            steps_num,
            epochs,
            learning_rate,
            calculate_accuracy,
            ignore_loss
        } = info;

        self.hyper.a = learning_rate;

        let batch_step = batch_size * steps_num;
        let mut batch_start = batch_start * batch_step;

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
        
        let epochs_per_input = (inputs.len() / batch_step).max(1);
        println!("calculate loss every {epochs_per_input} epochs");

        let output_loss = |network: &NeuralNetwork<T, D>|
        {
            if ignore_loss
            {
                return;
            }

            network.test_loss_inner(&testing_inputs, calculate_accuracy);
        };

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

            let mut batch_gradients = None;

            for _ in 0..batch_size
            {
                let values = InputOutput::values_slice(
                    &inputs,
                    |word| self.dictionary.word_to_layer(*word),
                    batch_start,
                    steps_num
                );

                let gradients = self.network.gradients::<true>(values.iter());

                if batch_gradients.is_none()
                {
                    batch_gradients = Some(gradients);
                } else
                {
                    batch_gradients.as_mut().map(|batch_gradients|
                    {
                        batch_gradients.iter_mut().zip(gradients.into_iter())
                            .for_each(|(batch_gradients, gradients)|
                            {
                                *batch_gradients += gradients
                            });
                    });
                }

                batch_start += steps_num;
                if batch_start >= (inputs.len() - 1)
                {
                    batch_start = 0;
                }
            }

            let mut gradients = batch_gradients.unwrap();
            gradients.iter_mut().for_each(|gradients| *gradients /= batch_size as f64);

            self.apply_gradients(gradients);
        }

        output_loss(self);
    }

    #[allow(dead_code)]
    pub fn predict(&mut self, text: &str, amount: usize, temperature: f64) -> String
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

        let output = predictor.predict_bytes(&self.network).into_iter().copied()
            .filter(|&c| c != b'\0').collect::<Vec<_>>();
        
        String::from_utf8_lossy(&output).to_string()
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    use crate::word_vectorizer::WordDictionary;
    
    #[allow(unused_imports)]
    use arrayfire::af_print;
    
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

    fn test_network() -> NeuralNetwork<GenericContainer, WordDictionary>
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
        network: &mut NeuralNetwork<GenericContainer, WordDictionary>
    ) -> Vec<InputOutput<GenericContainer>>
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
    fn adam_correct()
    {
        let mut old_weight = vec![3.21, 7.65];

        let mut m = vec![0.0, 0.0];
        let mut v = vec![0.0, 0.0];
        
        let mut g = vec![3.1_f64, -0.8_f64];

        let mut t = 1;

        for _ in 0..2
        {
            let a = 0.001;
            let b1 = 0.9;
            let b2 = 0.999;

            let epsilon = 10e-8;

            let adam_g = {
                let mut gradient_info = GradientInfo{
                    m: GenericContainer::from_raw(m.clone().into_boxed_slice(), 2, 1),
                    v: GenericContainer::from_raw(v.clone().into_boxed_slice(), 2, 1)
                };

                let gradient = GenericContainer::from_raw(g.clone().into_boxed_slice(), 2, 1);

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

                GenericContainer::from_raw(old_weight.clone().into_boxed_slice(), 2, 1) + change
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
                old_weight[0] - a * m_hat[0] / (v_hat[0].sqrt() + epsilon),
                old_weight[1] - a * m_hat[1] / (v_hat[1].sqrt() + epsilon)
            ];

            if t == 1
            {
                assert_eq!(new_weight[0], 3.2090000000322581);
                assert_eq!(new_weight[1], 7.6509999998750000);

                let mut adam_g = adam_g.iter();
                assert_eq!(new_weight[0], *adam_g.next().unwrap());
                assert_eq!(new_weight[1], *adam_g.next().unwrap());
            } else
            {
                assert_eq!(new_weight[0], 3.2080334761008376);
                assert_eq!(new_weight[1], 7.6518864757914067);

                let mut adam_g = adam_g.iter();
                assert_eq!(new_weight[0], *adam_g.next().unwrap());
                assert_eq!(new_weight[1], *adam_g.next().unwrap());
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
    fn matrix_multiplication()
    {
        let v = GenericContainer::from_raw(vec![5.0, 0.0, 5.0, 7.0], 4, 1);

        let m = GenericContainer::from_raw(vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        ], 4, 4);

        assert_eq!(
            m.matmul(&v),
            GenericContainer::from_raw(vec![48.0, 116.0, 184.0, 252.0], 4, 1)
        );

        assert_eq!(
            m.matmul_transposed(&v),
            GenericContainer::from_raw(vec![141.0, 158.0, 175.0, 192.0], 4, 1)
        );
    }

    #[test]
    fn softmax()
    {
        let test_layer = GenericContainer::from_raw([1.0, 2.0, 8.0], 3, 1);

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

        let l = inputs.len() as f64;
        let this_loss = inputs.into_iter().map(|input|
        {
            network.network.loss(input.iter().map(|(a, b)| (a.clone(), b.clone())))
        }).sum::<f64>() / l;

        let predicted_loss = (network.dictionary.words_amount() as f64).ln();

        assert!(
            close_enough(this_loss, predicted_loss, 0.1),
            "this_loss: {this_loss}, predicted_loss: {predicted_loss}"
        );
    }

    #[ignore]
    // #[test]
    fn gradients_check_many()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_many(), &mut network);

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
    // #[test]
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
        network: &mut NeuralNetwork<GenericContainer, WordDictionary>,
        inputs: Vec<InputOutput<GenericContainer>>
    )
    {
        let network = &mut network.network;

        for l_i in (0..LAYERS_AMOUNT).rev()
        {
            println!("layer {l_i}");

            inputs.iter().for_each(|input|
            {
                print_status("output gradients", ||
                {
                    gru::tests::output_gradients_check(network, l_i, input.iter())
                });

                print_status("hidden update gradients", ||
                {
                    gru::tests::hidden_update_gradients_check(network, l_i, input.iter())
                });

                print_status("hidden reset gradients", ||
                {
                    gru::tests::hidden_reset_gradients_check(network, l_i, input.iter())
                });

                print_status("hidden activation gradients", ||
                {
                    gru::tests::hidden_activation_gradients_check(network, l_i, input.iter())
                });

                print_status("update bias gradients", ||
                {
                    gru::tests::update_bias_gradients_check(network, l_i, input.iter())
                });

                print_status("reset bias gradients", ||
                {
                    gru::tests::reset_bias_gradients_check(network, l_i, input.iter())
                });

                print_status("activation bias gradients", ||
                {
                    gru::tests::activation_bias_gradients_check(network, l_i, input.iter())
                });

                print_status("input update gradients", ||
                {
                    gru::tests::input_update_gradients_check(network, l_i, input.iter())
                });

                print_status("input reset gradients", ||
                {
                    gru::tests::input_reset_gradients_check(network, l_i, input.iter())
                });

                print_status("input activation gradients", ||
                {
                    gru::tests::input_activation_gradients_check(network, l_i, input.iter())
                });
            });
        }
    }
}
