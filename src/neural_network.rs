use std::{
    f32,
    fmt,
    slice,
    io::{Read, Write, BufReader},
    fs::File,
    path::Path,
    ops::{Range, DivAssign, AddAssign, SubAssign}
};

use serde::{Serialize, Deserialize, de::DeserializeOwned};

use network::{NetworkOutput, Network};

#[allow(unused_imports)]
use crate::{
    Config,
    word_vectorizer::{
        ByteDictionary,
        CharDictionary,
        WordDictionary,
        NetworkDictionary,
        WordVectorizer,
        VectorWord,
        ReaderAdapter
    }
};

use optimizers::*;

pub use network::LayerSizes;
pub use optimizers::Optimizer;
pub use network_unit::{NetworkUnit, NewableLayer, GenericUnit, UnitFactory, OptimizerUnit};
pub use network::{WeightsNamed, WeightsSize};
pub use containers::{
    LayerInnerType,
    DiffWrapper,
    InputType,
    OneHotLayer,
    Softmaxer
};

#[allow(unused_imports)]
use gru::Gru;

#[allow(unused_imports)]
use lstm::Lstm;

pub use embedding_unit::EmbeddingUnit;

mod optimizers;
mod network_unit;
mod network;
mod gru;
mod lstm;
mod embedding_unit;

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

pub struct InputOutput<'a, const SURROUND: bool, D>
{
    dictionary: &'a D,
    values: &'a [VectorWord]
}

impl<'a, const SURROUND: bool, D> InputOutput<'a, SURROUND, D>
{
    #[allow(dead_code)]
    pub fn values_slice(
        dictionary: &'a D,
        values: &'a [VectorWord],
        start: usize,
        size: usize
    ) -> Self
    {
        let min_slice_len = Self::min_len();

        let slice_end = (start + size + min_slice_len).min(values.len());
        let this_slice = &values[start..slice_end];

        debug_assert!(this_slice.len() > min_slice_len);

        Self::new(dictionary, this_slice)
    }

    pub fn new(dictionary: &'a D, values: &'a [VectorWord]) -> Self
    {
        Self{
            dictionary,
            values
        }
    }

    pub const fn min_len() -> usize
    {
        if SURROUND { 2 } else { 1 }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize
    {
        self.values.len() - 1
    }
}

// jumping through hoops cuz its not obvious to the compiler that bools
// cant be anything except true or false
pub trait InputOutputable
{
    type Iter<'a>: Iterator<Item=(OneHotLayer, OneHotLayer)>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_>;
}

impl<'a, D> InputOutputable for InputOutput<'a, false, D>
where
    D: NetworkDictionary
{
    type Iter<'b> = InputOutputIter<'b, D, slice::Iter<'b, VectorWord>>
    where
        Self: 'b,
        D: 'b;

    fn iter(&self) -> Self::Iter<'_>
    {
        InputOutputIter::new(self.dictionary, self.values.iter())
    }
}

impl<'a, D> InputOutputable for InputOutput<'a, true, D>
where
    D: NetworkDictionary
{
    type Iter<'b> = InputOutputSurroundIter<'b, D, slice::Iter<'b, VectorWord>>
    where
        Self: 'b,
        D: 'b;

    fn iter(&self) -> Self::Iter<'_>
    {
        InputOutputSurroundIter::new(self.dictionary, self.values.iter())
    }
}

pub struct InputOutputIter<'a, D, I>
{
    dictionary: &'a D,
    inputs: I,
    previous: &'a VectorWord
}

// why cant the macro figure this out :/
impl<'a, D, I> Clone for InputOutputIter<'a, D, I>
where
    I: Clone
{
    fn clone(&self) -> Self
    {
        Self{
            dictionary: self.dictionary,
            inputs: self.inputs.clone(),
            previous: self.previous
        }
    }
}

impl<'a, D, I> InputOutputIter<'a, D, I>
where
    I: Iterator<Item=&'a VectorWord>
{
    pub fn new(dictionary: &'a D, mut inputs: I) -> Self
    {
        Self{
            dictionary,
            previous: inputs.next().expect("input must not be empty"),
            inputs
        }
    }

    fn value_map(&self, value: &'a VectorWord) -> OneHotLayer
    where
        D: NetworkDictionary
    {
        self.dictionary.words_to_layer([*value])
    }
}

impl<'a, D, I> Iterator for InputOutputIter<'a, D, I>
where
    D: NetworkDictionary,
    I: Iterator<Item=&'a VectorWord>
{
    type Item = (OneHotLayer, OneHotLayer);

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.inputs.next()
        {
            None => None,
            Some(input) =>
            {
                let out = Some((self.value_map(self.previous), self.value_map(input)));

                self.previous = input;

                out
            }
        }
    }
}

impl<'a, D, I> ExactSizeIterator for InputOutputIter<'a, D, I>
where
    D: NetworkDictionary,
    I: Iterator<Item=&'a VectorWord> + ExactSizeIterator
{
    fn len(&self) -> usize
    {
        self.inputs.len()
    }
}

pub struct InputOutputSurroundIter<'a, D, I>
{
    dictionary: &'a D,
    inputs: I,
    double_previous: &'a VectorWord,
    previous: &'a VectorWord
}

// why cant the macro figure this out :/
impl<'a, D, I> Clone for InputOutputSurroundIter<'a, D, I>
where
    I: Clone
{
    fn clone(&self) -> Self
    {
        Self{
            dictionary: self.dictionary,
            inputs: self.inputs.clone(),
            double_previous: self.double_previous,
            previous: self.previous
        }
    }
}

impl<'a, D, I> InputOutputSurroundIter<'a, D, I>
where
    I: Iterator<Item=&'a VectorWord>
{
    pub fn new(dictionary: &'a D, mut inputs: I) -> Self
    {
        Self{
            dictionary,
            double_previous: inputs.next().expect("input must not be empty"),
            previous: inputs.next().expect("input must not be 1 len"),
            inputs
        }
    }
}

impl<'a, D, I> Iterator for InputOutputSurroundIter<'a, D, I>
where
    D: NetworkDictionary,
    I: Iterator<Item=&'a VectorWord>
{
    type Item = (OneHotLayer, OneHotLayer);

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.inputs.next()
        {
            None => None,
            Some(input) =>
            {
                let this_input = self.dictionary.words_to_layer([*self.double_previous, *input]);
                let this_output = self.dictionary.words_to_layer([*self.previous]);

                let out = Some((this_input, this_output));

                self.double_previous = self.previous;
                self.previous = input;

                out
            }
        }
    }
}

impl<'a, D, I> ExactSizeIterator for InputOutputSurroundIter<'a, D, I>
where
    D: NetworkDictionary,
    I: Iterator<Item=&'a VectorWord> + ExactSizeIterator
{
    fn len(&self) -> usize
    {
        self.inputs.len()
    }
}

struct Predictor<'a, D>
{
    dictionary: &'a mut D,
    words: Vec<OneHotLayer>,
    temperature: f32,
    predict_amount: usize
}

impl<'a, D: NetworkDictionary> Predictor<'a, D>
{
    pub fn new(
        dictionary: &'a mut D,
        words: Vec<OneHotLayer>,
        temperature: f32,
        predict_amount: usize
    ) -> Self
    {
        Self{
            dictionary,
            words,
            temperature,
            predict_amount
        }
    }

    pub fn predict_into<N, O>(
        mut self,
        network: &mut Network<N, O>,
        mut out: impl Write
    )
    where
        N::Unit<O>: OptimizerUnit<O>,
        N::Unit<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
        for<'b> &'b N::Unit<DiffWrapper>: IntoIterator<Item=&'b DiffWrapper>,
        for<'b> &'b mut N::Unit<DiffWrapper>: IntoIterator<Item=&'b mut DiffWrapper>,
        N: UnitFactory
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

                let layer = self.dictionary.words_to_layer([word]);
                self.words.push(layer);

                let bytes = self.dictionary.word_to_bytes(word);

                out.write_all(&bytes).unwrap();
            }

            previous_state = Some(state);
        }

        out.flush().unwrap();
    }

    pub fn predict_bytes<N, O>(self, network: &mut Network<N, O>) -> Box<[u8]>
    where
        N::Unit<O>: OptimizerUnit<O>,
        N::Unit<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
        for<'b> &'b N::Unit<DiffWrapper>: IntoIterator<Item=&'b DiffWrapper>,
        for<'b> &'b mut N::Unit<DiffWrapper>: IntoIterator<Item=&'b mut DiffWrapper>,
        N: UnitFactory
    {
        let mut predicted = Vec::with_capacity(self.predict_amount);
        self.predict_into(network, &mut predicted);

        predicted.into_boxed_slice()
    }
}

type VectorizerType<'a, R, D> = WordVectorizer<<D as NetworkDictionary>::Adapter<BufReader<R>>, &'a mut D>;

pub enum StepsNum
{
    Steps(usize),
    StepsRange(Range<usize>)
}

impl From<usize> for StepsNum
{
    fn from(value: usize) -> Self
    {
        Self::Steps(value)
    }
}

impl fmt::Display for StepsNum
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let s = match self
        {
            Self::Steps(x) => x.to_string(),
            Self::StepsRange(Range{start, end}) =>
            {
                format!("{start} to {end}")
            }
        };

        write!(f, "{s}")
    }
}

impl StepsNum
{
    pub fn new(value: usize, deviation: f32) -> Self
    {
        let half_deviation = (value as f32 * deviation) / 2.0;

        let start = (value as f32 - half_deviation).round() as usize;
        let end = (value as f32 + half_deviation).round() as usize;

        Self::StepsRange(Range{start, end})
    }

    pub fn get(&self) -> usize
    {
        match self
        {
            Self::Steps(x) => *x,
            Self::StepsRange(range) => fastrand::usize(range.clone())
        }
    }

    pub fn mid(&self) -> usize
    {
        match self
        {
            Self::Steps(x) => *x,
            Self::StepsRange(Range{start, end}) =>
            {
                let mid = (end - start) / 2;

                start + mid
            }
        }
    }
}

pub struct TrainingInfo
{
    pub iterations: usize,
    pub batch_size: usize,
    pub steps_num: StepsNum,
    pub learning_rate: Option<f32>,
    pub loss_every: Option<usize>,
    pub calculate_loss: bool,
    pub calculate_accuracy: bool,
    pub less_info: bool
}

impl From<&Config> for TrainingInfo
{
    fn from(config: &Config) -> Self
    {
        TrainingInfo{
            iterations: config.iterations,
            batch_size: config.batch_size,
            steps_num: StepsNum::new(config.steps_num, config.steps_deviation),
            learning_rate: config.learning_rate,
            loss_every: config.loss_every,
            calculate_loss: config.calculate_loss,
            calculate_accuracy: config.calculate_accuracy,
            less_info: config.less_info
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork<N, O, D>
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<DiffWrapper>: NetworkUnit,
    for<'a> O::WeightParam: Serialize + Deserialize<'a>
{
    dictionary: D,
    network: Network<N, O::WeightParam>,
    optimizer: O,
    gradient_clip: Option<f32>
}

impl<N, O, D> NeuralNetwork<N, O, D>
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
    for<'b> &'b N::Unit<DiffWrapper>: IntoIterator<Item=&'b DiffWrapper>,
    for<'b> &'b mut N::Unit<DiffWrapper>: IntoIterator<Item=&'b mut DiffWrapper>,
    for<'a> O::WeightParam: Serialize + Deserialize<'a>,
    D: NetworkDictionary
{
    pub fn new(
        dictionary: D,
        sizes: LayerSizes,
        dropout_probability: f32,
        gradient_clip: Option<f32>
    ) -> Self
    where
        O::WeightParam: NewableLayer
    {
        let network = Network::new(sizes, dropout_probability);

        let optimizer = O::new();

        Self{dictionary, network, optimizer, gradient_clip}
    }

    // these trait bounds feel wrong somehow
    pub fn save<P: AsRef<Path>>(&mut self, path: P)
    where
        N: Serialize,
        O: Serialize,
        D: Serialize
    {
        self.network.assert_empty();

        let writer = File::create(path).unwrap();

        bincode::serialize_into(writer, self).unwrap();
    }

    pub fn load<P: AsRef<Path>>(path: P) -> bincode::Result<Self>
    where
        N: DeserializeOwned,
        O: DeserializeOwned,
        D: DeserializeOwned
    {
        let reader = File::open(path)?;

        bincode::deserialize_from(reader)
    }

    pub fn dictionary(&self) -> &D
    {
        &self.dictionary
    }

    #[allow(dead_code)]
    pub fn inner_network(&self) -> &Network<N, O::WeightParam>
    {
        &self.network
    }

    #[allow(dead_code)]
    pub fn inner_network_mut(&mut self) -> &mut Network<N, O::WeightParam>
    {
        &mut self.network
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
            &self.dictionary,
            inputs.iter()
        );

        self.network.disable_gradients();

        if calculate_accuracy
        {
            let accuracy = self.network.accuracy(input_outputs.clone());

            println!("accuracy: {}%", accuracy * 100.0);
        }

        if calculate_loss
        {
            let loss = self.network.feedforward(input_outputs);

            Self::print_loss(true, *loss.scalar() / inputs.len() as f32);
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

    // these trait bounds make me angry and i cant make them disappear cuz TRAITS SUCK
    pub fn train<const SURROUND: bool, RT, R>(
        &mut self,
        info: TrainingInfo,
        testing_reader: Option<RT>,
        reader: R
    )
    where
        RT: Read,
        R: Read,
        for<'b> VectorizerType<'b, RT, D>: Iterator<Item=VectorWord>,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        for<'b> &'b mut N::Unit<O::WeightParam>: IntoIterator<Item=&'b mut O::WeightParam>,
        N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam, Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
        N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam, Unit<LayerInnerType>=N::Unit<LayerInnerType>>,
        N::Unit<DiffWrapper>: NetworkUnit<Unit<LayerInnerType>=N::Unit<LayerInnerType>> + SubAssign + fmt::Debug,
        N::Unit<LayerInnerType>: DivAssign<f32> + AddAssign + Serialize + DeserializeOwned + IntoIterator<Item=LayerInnerType>,
        for<'b> &'b mut N::Unit<LayerInnerType>: IntoIterator<Item=&'b mut LayerInnerType>,
        for<'b> InputOutput<'b, SURROUND, D>: InputOutputable
    {
        if let Some(learning_rate) = info.learning_rate
        {
            self.optimizer.set_learning_rate(learning_rate);
        }

        // i dunno wuts the correct way to handle this stuff
        let batch_step = info.batch_size * info.steps_num.mid();

        let inputs: Vec<_> = self.vectorized(reader);
        let testing_inputs: Vec<_> = if !info.calculate_loss && !info.calculate_accuracy
        {
            Vec::new()
        } else
        {
            testing_reader
                .map(|reader| self.vectorized(reader))
                .unwrap_or_else(Vec::new)
        };

        let inputs_per_loss = info.loss_every.unwrap_or_else(||
        {
            (inputs.len() / batch_step).max(1)
        });

        let display_header = !info.less_info;
        let display_inner = !info.less_info;

        if display_header
        {
            println!("input vector size: {}", self.dictionary.words_amount());
            println!("parameters amount: {}", self.network.parameters_amount());
            println!("batch size: {}", info.batch_size);

            println!("steps amount: {}", info.steps_num);
        
            println!("calculate loss every ~{inputs_per_loss} inputs");
        }

        let output_loss = |network: &mut NeuralNetwork<_, _, _>|
        {
            if testing_inputs.is_empty()
            {
                return;
            }

            network.test_loss_inner(
                &testing_inputs,
                info.calculate_loss,
                info.calculate_accuracy
            );
        };

        for input_index in 0..info.iterations
        {
            if display_inner
            {
                eprintln!("iteration: {input_index}");
            }
            
            time_debug! {
                let steps_num = info.steps_num.get();

                let print_loss = (input_index % inputs_per_loss) == inputs_per_loss - 1;
                if print_loss
                {
                    output_loss(self);
                }

                let mut kahan_sum = KahanSum::new();

                let max_batch_start = inputs.len()
                    .saturating_sub(steps_num + (InputOutput::<SURROUND, D>::min_len() - 1));

                let mut gradients = (0..info.batch_size).map(|_|
                {
                    let batch_start = if max_batch_start == 0
                    {
                        0
                    } else
                    {
                        fastrand::usize(0..max_batch_start)
                    };

                    let values = InputOutput::<SURROUND, _>::values_slice(
                        &self.dictionary,
                        &inputs,
                        batch_start,
                        steps_num
                    );

                    let (loss, gradients): (f32, _) = self.network.gradients(values.iter());

                    kahan_sum.add(loss as f64 / info.batch_size as f64);

                    gradients
                }).reduce(|mut acc, this|
                {
                    acc.iter_mut().zip(this.into_iter()).for_each(|(acc, this)|
                    {
                        *acc += this;
                    });

                    acc
                }).expect("batch size must not be 0");

                gradients.iter_mut().for_each(|gradient| *gradient /= info.batch_size as f32);

                let batch_loss = kahan_sum.value() / steps_num as f64;

                if display_inner
                {
                    Self::print_loss(false, batch_loss as f32);
                }

                self.network.apply_gradients(gradients, &mut self.optimizer, self.gradient_clip);
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
        N::Unit<DiffWrapper>: NetworkUnit,
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
        N::Unit<DiffWrapper>: NetworkUnit,
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
        N::Unit<DiffWrapper>: NetworkUnit,
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
        F: FnOnce(Predictor<D>, &mut Network<N, O::WeightParam>) -> T
    {
        self.network.disable_gradients();

        let predictor = {
            // could do this without a collect but wheres the fun in that
            let words = self.vectorized(reader).into_iter().map(|v|
            {
                self.dictionary.words_to_layer([v])
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
        let mut test_layer = LayerInnerType::from_raw([1.0, 2.0, 8.0], 3, 1);

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
