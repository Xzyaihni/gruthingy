use std::{
    f32,
    mem,
    vec,
    slice,
    io::{Read, Write, BufReader},
    fs::File,
    path::Path
};

use serde::{Serialize, Deserialize, de::DeserializeOwned};

use network::{NetworkOutput, Network};

#[allow(unused_imports)]
use crate::word_vectorizer::{
    ByteDictionary,
    CharDictionary,
    WordDictionary,
    NetworkDictionary,
    WordVectorizer,
    VectorWord,
    ReaderAdapter
};

use optimizers::*;

pub use network::LayerSizes;
pub use optimizers::Optimizer;
pub use network_unit::{NetworkUnit, NewableLayer};
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
use gru::Gru;

#[allow(unused_imports)]
use lstm::Lstm;

mod optimizers;
mod network_unit;
mod network;
mod gru;
mod lstm;

pub mod containers;

pub use neural_network_config::*;
mod neural_network_config;


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

macro_rules! time_debug
{
    ($($token:tt)*) =>
    {
        #[cfg(feature = "timedebug")]
        use std::time::Instant;

        #[cfg(feature = "timedebug")]
        let now_time = Instant::now();

        {
            $($token)*
        }

        #[cfg(feature = "timedebug")]
        {
            let duration = Instant::now() - now_time;
            eprintln!("took {} ms", duration.as_millis());
        }
    }
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
    pub fn iter(&self) -> InputOutputIter<slice::Iter<'_, T>, &T>
    {
        InputOutputIter::new(self.container.iter())
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize
    {
        self.container.len() - 1
    }
}

impl<T: Clone> IntoIterator for InputOutput<T>
{
    type Item = (T, T);
    type IntoIter = InputOutputIter<vec::IntoIter<T>, T>;

    fn into_iter(self) -> Self::IntoIter
    {
        InputOutputIter::new(self.container.into_iter())
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

struct Predictor<'a, D>
{
    dictionary: &'a mut D,
    words: Vec<LayerType>,
    temperature: f32,
    predict_amount: usize
}

impl<'a, D: NetworkDictionary> Predictor<'a, D>
{
    pub fn new(
        dictionary: &'a mut D,
        words: Vec<LayerInnerType>,
        temperature: f32,
        predict_amount: usize
    ) -> Self
    {
        let words = words.into_iter().map(LayerType::new_undiff).collect();

        Self{
            dictionary,
            words,
            temperature,
            predict_amount
        }
    }

    pub fn predict_into<O>(
        mut self,
        network: &mut Network<O>,
        mut out: impl Write
    )
    {
        let input_amount = self.words.len();

        let mut previous_state: Option<_> = None;

        let dropout_masks = network.create_dropout_masks(
            self.dictionary.words_amount(),
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

                let bytes = self.dictionary.word_to_bytes(word);

                out.write_all(&bytes).unwrap();
            }

            previous_state = Some(state);
        }

        out.flush().unwrap();
    }

    pub fn predict_bytes<O>(self, network: &mut Network<O>) -> Box<[u8]>
    {
        let mut predicted = Vec::with_capacity(self.predict_amount);
        self.predict_into(network, &mut predicted);

        predicted.into_boxed_slice()
    }
}

type VectorizerType<'a, R, D> = WordVectorizer<<D as NetworkDictionary>::Adapter<BufReader<R>>, &'a mut D>;

pub struct TrainingInfo
{
    pub iterations: usize,
    pub batch_size: usize,
    pub steps_num: usize,
    pub learning_rate: Option<f32>,
    pub calculate_loss: bool,
    pub calculate_accuracy: bool,
    pub less_info: bool
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork<O: Optimizer, D>
where
    for<'a> O::WeightParam: Serialize + Deserialize<'a>
{
    dictionary: D,
    network: Network<O::WeightParam>,
    optimizer: O
}

impl<O, D> NeuralNetwork<O, D>
where
    O: Optimizer,
    for<'a> O::WeightParam: Serialize + Deserialize<'a>,
    D: NetworkDictionary
{
    pub fn new(dictionary: D, sizes: LayerSizes) -> Self
    where
        O::WeightParam: NewableLayer
    {
        let network = Network::new(sizes);

        let optimizer = O::new();

        Self{dictionary, network, optimizer}
    }

    // these trait bounds feel wrong somehow
    pub fn save<P: AsRef<Path>>(&mut self, path: P)
    where
        O: Serialize,
        D: Serialize
    {
        // clear all derivatives info
        self.network.clear();

        let writer = File::create(path).unwrap();

        bincode::serialize_into(writer, self).unwrap();
    }

    pub fn load<P: AsRef<Path>>(path: P) -> bincode::Result<Self>
    where
        O: DeserializeOwned,
        D: DeserializeOwned
    {
        let reader = File::open(path)?;

        bincode::deserialize_from(reader)
    }

    #[allow(dead_code)]
    pub fn inner_network(&self) -> &Network<O::WeightParam>
    {
        &self.network
    }

    pub fn apply_gradients(&mut self, gradients: Vec<NUnit<LayerInnerType>>)
    {
        let combined_iter = gradients.into_iter()
            .zip(self.network.gradients_info());

        combined_iter.for_each(|(gradients, (network_weights, optimizer_info))|
        {
            *network_weights -= optimizer_info.gradients_to_change(gradients, &self.optimizer);
        });

        self.optimizer.advance_time();
    }

    pub fn test_loss<R>(
        &mut self,
        reader: R,
        calculate_loss: bool,
        calculate_accuracy: bool
    )
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        let inputs = self.vectorized(reader);

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
            let loss = self.network.feedforward(JoinableType::from_iter(input_outputs));

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

    fn vectorizer<'a, R: Read>(
        &'a mut self,
        reader: R
    ) -> impl Iterator<Item=VectorWord> + 'a
    where
        D::Adapter<BufReader<R>>: 'a,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        WordVectorizer::new(&mut self.dictionary, reader)
    }

    fn vectorized<R: Read>(&mut self, reader: R) -> Vec<VectorWord>
    where
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        self.vectorizer(reader).collect()
    }

    pub fn train<R, RT>(
        &mut self,
        info: TrainingInfo,
        testing_reader: Option<RT>,
        reader: R
    )
    where
        R: Read,
        RT: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        for<'b> VectorizerType<'b, RT, D>: Iterator<Item=VectorWord>
    {
        let TrainingInfo{
            iterations,
            batch_size,
            steps_num,
            learning_rate,
            calculate_loss,
            calculate_accuracy,
            less_info
        } = info;

        if let Some(learning_rate) = learning_rate
        {
            self.optimizer.set_learning_rate(learning_rate);
        }

        let batch_step = batch_size * steps_num;

        let inputs: Vec<_> = self.vectorized(reader);
        let testing_inputs: Vec<_> = if !calculate_loss && !calculate_accuracy
        {
            Vec::new()
        } else
        {
            match testing_reader.map(|reader| self.vectorized(reader))
            {
                None => Vec::new(),
                Some(words) => words
            }
        };

        let steps_num = steps_num.min(inputs.len());

        let steps_deviation = steps_num / 10;
        let steps_half_deviation = steps_deviation / 2;

        let inputs_per_epoch = (inputs.len() / batch_step).max(1);

        let display_header = !less_info;
        let display_inner = !less_info;

        if display_header
        {
            println!("input vector size: {}", self.dictionary.words_amount());
            println!("parameters amount: {}", self.network.parameters_amount());
            println!("batch size: {batch_size}");

            println!(
                "steps amount: {} to {}",
                steps_num - steps_half_deviation,
                steps_num + steps_half_deviation
            );
        
            println!("calculate loss every ~{inputs_per_epoch} inputs");
        }

        let output_loss = |network: &mut NeuralNetwork<_, _>|
        {
            if testing_inputs.is_empty()
            {
                return;
            }

            network.test_loss_inner(&testing_inputs, calculate_loss, calculate_accuracy);
        };

        for input_index in 0..iterations
        {
            if display_inner
            {
                eprintln!("iteration: {input_index}");
            }
            
            time_debug! {
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

                let max_batch_start = inputs.len().saturating_sub(steps_num);

                let mut network = self.network.dropconnected();

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

                    let (loss, mut gradients): (f32, Vec<_>) =
                        network.gradients(values.into_iter());

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

                if display_inner
                {
                    Self::print_loss(false, batch_loss as f32);
                }

                self.apply_gradients(gradients);
            }
        }

        output_loss(self);
    }

    pub fn predict_into<R>(
        &mut self,
        reader: R,
        amount: usize,
        temperature: f32,
        out: impl Write
    )
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        self.predict_inner(reader, amount, temperature, |predictor, network|
        {
            predictor.predict_into(network, out)
        })
    }

    #[allow(dead_code)]
    pub fn predict_text<R>(
        &mut self,
        reader: R,
        amount: usize,
        temperature: f32
    ) -> String
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        let output = self.predict_inner(reader, amount, temperature, |predictor, network|
        {
            predictor.predict_bytes(network)
        }).iter().copied().filter(|&c| c != b'\0').collect::<Vec<_>>();
        
        String::from_utf8_lossy(&output).to_string()
    }

    pub fn predict_bytes<R>(
        &mut self,
        reader: R,
        amount: usize,
        temperature: f32
    ) -> Box<[u8]>
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        self.predict_inner(reader, amount, temperature, |predictor, network|
        {
            predictor.predict_bytes(network)
        })
    }

    fn predict_inner<R, T, F>(
        &mut self,
        reader: R,
        amount: usize,
        temperature: f32,
        f: F
    ) -> T
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        F: FnOnce(Predictor<D>, &mut Network<O::WeightParam>) -> T
    {
        self.network.disable_gradients();

        let predictor = {
            // could do this without a collect but wheres the fun in that
            let words = self.vectorized(reader).into_iter().map(|v|
            {
                self.dictionary.word_to_layer(v)
            }).collect::<Vec<_>>();

            Predictor::new(&mut self.dictionary, words, temperature, amount)
        };

        let predicted = f(predictor, &mut self.network);

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
