use std::{
    f32,
    fmt,
    slice,
    marker::PhantomData,
    io::{self, Read, Write, BufReader, BufWriter},
    fs::File,
    path::Path,
    collections::VecDeque,
    ops::{Range, DivAssign, AddAssign, SubAssign}
};

use serde::{Serialize, Deserialize, de::DeserializeOwned};

//use network::{NetworkOutput, Network};

#[allow(unused_imports)]
use crate::{
    Config,
//    EmbeddingsUnitFactory,
    word_vectorizer::{
        ByteDictionary,
        CharDictionary,
        WordDictionary,
        EmbeddingsDictionary,
        NetworkDictionary,
        WordVectorizer,
        VectorWord,
        ReaderAdapter
    }
};

//use optimizers::*;

pub use network::LayerSizes;
//pub use optimizers::{NewableLayer, DecayFunction, Optimizer};
//pub use network_unit::{NetworkUnit, GenericUnit, UnitFactory, OptimizerUnit};
//pub use network::WeightsNamed;
pub use containers::{
    OperationsRecorder,
    LayerType,
    DiffWrapper,
    DiffTensor,
    DiffScalar,
    TensorIndex,
    ValueIndex,
    OneHotIndex,
    InputType,
    OneHotLayer,
    Softmaxer
};

//#[allow(unused_imports)]
// pub use network::WeightsSize;

/*#[allow(unused_imports)]
use gru::Gru;*/

//#[allow(unused_imports)]
//use lstm::Lstm;

//pub use embedding_unit::EmbeddingUnit;

//mod optimizers;
//mod network_unit;
// mod gru;
// mod lstm;
//mod embedding_unit;

pub mod network;
pub mod containers;

pub use neural_network_config::*;
mod neural_network_config;


/*#[allow(dead_code)]
#[derive(Debug)]
pub struct BincodeFormat;*/

#[allow(dead_code)]
#[derive(Debug)]
pub struct PostcardFormat;

#[allow(dead_code)]
#[derive(Debug)]
pub struct JsonFormat;

#[allow(dead_code)]
#[derive(Debug)]
pub struct MixedFormat<Save, Load>(PhantomData<(Save, Load)>);

pub trait SerializeFormat
{
    type Error: fmt::Debug + fmt::Display + From<io::Error>;

    fn serialize<T: Serialize>(writer: impl Write, value: &T) -> Result<(), Self::Error>;
    fn deserialize<T: for<'de> Deserialize<'de>>(reader: impl Read) -> Result<T, Self::Error>;
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum MixedError<Save: SerializeFormat, Load: SerializeFormat>
{
    SaveError(Save::Error),
    LoadError(Load::Error),
    Io(io::Error)
}

impl<Save: SerializeFormat, Load: SerializeFormat> From<io::Error> for MixedError<Save, Load>
{
    fn from(x: io::Error) -> Self
    {
        Self::Io(x)
    }
}

impl<Save: SerializeFormat, Load: SerializeFormat> fmt::Display for MixedError<Save, Load>
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result
    {
        match self
        {
            Self::SaveError(x) => x.fmt(fmt),
            Self::LoadError(x) => x.fmt(fmt),
            Self::Io(x) => x.fmt(fmt)
        }
    }
}

impl<Save: SerializeFormat + fmt::Debug, Load: SerializeFormat + fmt::Debug> SerializeFormat for MixedFormat<Save, Load>
{
    type Error = MixedError<Save, Load>;

    fn serialize<T: Serialize>(writer: impl Write, value: &T) -> Result<(), Self::Error>
    {
        Save::serialize(writer, value).map_err(MixedError::SaveError)
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(reader: impl Read) -> Result<T, Self::Error>
    {
        Load::deserialize(reader).map_err(MixedError::LoadError)
    }
}

/*impl SerializeFormat for BincodeFormat
{
    type Error = bincode::Error;

    fn serialize<T: Serialize>(writer: impl Write, value: &T) -> Result<(), Self::Error>
    {
        bincode::serialize_into(writer, value)
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(reader: impl Read) -> Result<T, Self::Error>
    {
        bincode::deserialize_from(reader)
    }
}*/

#[allow(dead_code)]
#[derive(Debug)]
pub enum PostcardError
{
    Io(io::Error),
    Postcard(postcard::Error)
}

impl fmt::Display for PostcardError
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result
    {
        match self
        {
            Self::Io(x) => x.fmt(fmt),
            Self::Postcard(x) => x.fmt(fmt)
        }
    }
}

impl From<io::Error> for PostcardError
{
    fn from(x: io::Error) -> Self
    {
        Self::Io(x)
    }
}

impl From<postcard::Error> for PostcardError
{
    fn from(x: postcard::Error) -> Self
    {
        Self::Postcard(x)
    }
}

impl SerializeFormat for PostcardFormat
{
    type Error = PostcardError;

    fn serialize<T: Serialize>(writer: impl Write, value: &T) -> Result<(), Self::Error>
    {
        Ok(postcard::to_io(value, writer).map(drop)?)
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(mut reader: impl Read) -> Result<T, Self::Error>
    {
        let mut v = Vec::new();
        reader.read_to_end(&mut v)?;

        Ok(postcard::from_bytes(&v)?)
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum JsonError
{
    Io(io::Error),
    Json(serde_json::Error)
}

impl fmt::Display for JsonError
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result
    {
        match self
        {
            Self::Io(x) => x.fmt(fmt),
            Self::Json(x) => x.fmt(fmt)
        }
    }
}

impl From<io::Error> for JsonError
{
    fn from(x: io::Error) -> Self
    {
        Self::Io(x)
    }
}

impl From<serde_json::Error> for JsonError
{
    fn from(x: serde_json::Error) -> Self
    {
        Self::Json(x)
    }
}

impl SerializeFormat for JsonFormat
{
    type Error = JsonError;

    fn serialize<T: Serialize>(writer: impl Write, value: &T) -> Result<(), Self::Error>
    {
        serde_json::to_writer(writer, value)?;

        Ok(())
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(reader: impl Read) -> Result<T, Self::Error>
    {
        let value = serde_json::from_reader(reader)?;

        Ok(value)
    }
}

#[allow(dead_code)]
pub enum AFType
{
    Tanh,
    LeakyRelu
}

#[allow(dead_code)]
pub enum EMType
{
    BagOfWords(usize),
    SkipGram(usize)
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

impl Default for KahanSum
{
    fn default() -> Self
    {
        Self::new()
    }
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

pub struct InputOutput<'a, const EMBEDDINGS: bool, D>
{
    dictionary: &'a D,
    values: &'a [VectorWord]
}

impl<'a, const EMBEDDINGS: bool, D> InputOutput<'a, EMBEDDINGS, D>
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

        assert!(this_slice.len() > min_slice_len);

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
        if EMBEDDINGS
        {
            let amount = match EMBEDDINGS_TYPE
            {
                EMType::BagOfWords(amount) => amount,
                EMType::SkipGram(amount) => amount
            };

            amount * 2 + 1
        } else
        {
            1
        }
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
    type Iter<'a>: Iterator<Item=(InputType, OneHotLayer)>
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
    type Iter<'b> = InputOutputEmbeddingsIter<'b, D, slice::Iter<'b, VectorWord>>
    where
        Self: 'b,
        D: 'b;

    fn iter(&self) -> Self::Iter<'_>
    {
        InputOutputEmbeddingsIter::new(self.dictionary, self.values.iter())
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
}

impl<'a, D, I> Iterator for InputOutputIter<'a, D, I>
where
    D: NetworkDictionary,
    I: Iterator<Item=&'a VectorWord>
{
    type Item = (InputType, OneHotLayer);

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.inputs.next()
        {
            None => None,
            Some(input) =>
            {
                let out = Some(
                    (
                        self.dictionary.words_to_layer([*self.previous]),
                        self.dictionary.words_to_onehot([*input])
                    )
                );

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

pub struct InputOutputEmbeddingsIter<'a, D, I>
{
    dictionary: &'a D,
    inputs: I,
    context: VecDeque<&'a VectorWord>
}

// why cant the macro figure this out :/
impl<'a, D, I> Clone for InputOutputEmbeddingsIter<'a, D, I>
where
    I: Clone
{
    fn clone(&self) -> Self
    {
        Self{
            dictionary: self.dictionary,
            inputs: self.inputs.clone(),
            context: self.context.clone()
        }
    }
}

impl<'a, D, I> InputOutputEmbeddingsIter<'a, D, I>
where
    I: Iterator<Item=&'a VectorWord>
{
    pub fn new(dictionary: &'a D, mut inputs: I) -> Self
    {
        let context_len = InputOutput::<true, D>::min_len() - 1;
        let context: VecDeque<_> = inputs.by_ref().take(context_len).collect();

        assert_eq!(context.len(), context_len);

        Self{
            dictionary,
            context,
            inputs
        }
    }

    fn around_window(&self, amount: usize) -> Vec<VectorWord>
    {
        let mut words = self.context.iter().take(amount)
            .chain(self.context.iter().rev().take(amount))
            .map(|v| **v)
            .collect::<Vec<_>>();

        words.sort_unstable();
        words.dedup();

        words
    }

    fn middle_word(&self, amount: usize) -> VectorWord
    {
        *self.context[amount]
    }

    fn next_bag_of_words(&mut self, amount: usize) -> (InputType, OneHotLayer)
    where
        D: NetworkDictionary
    {
        let this_input = self.dictionary.words_to_layer(self.around_window(amount));
        let this_output = self.dictionary.words_to_onehot([self.middle_word(amount)]);

        (this_input, this_output)
    }

    fn next_skip_gram(&mut self, amount: usize) -> (InputType, OneHotLayer)
    where
        D: NetworkDictionary
    {
        let this_input = self.dictionary.words_to_layer([self.middle_word(amount)]);
        let this_output = self.dictionary.words_to_onehot(self.around_window(amount));

        (this_input, this_output)
    }
}

impl<'a, D, I> Iterator for InputOutputEmbeddingsIter<'a, D, I>
where
    D: NetworkDictionary,
    I: Iterator<Item=&'a VectorWord>
{
    type Item = (InputType, OneHotLayer);

    fn next(&mut self) -> Option<Self::Item>
    {
        match self.inputs.next()
        {
            None => None,
            Some(input) =>
            {
                self.context.push_back(input);

                let output = match EMBEDDINGS_TYPE
                {
                    EMType::BagOfWords(amount) =>
                    {
                        self.next_bag_of_words(amount)
                    },
                    EMType::SkipGram(amount) =>
                    {
                        self.next_skip_gram(amount)
                    }
                };

                let _ = self.context.pop_front();

                Some(output)
            }
        }
    }
}

impl<'a, D, I> ExactSizeIterator for InputOutputEmbeddingsIter<'a, D, I>
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
    words: Vec<InputType>,
    sizes: LayerSizes,
    temperature: f32,
    predict_amount: usize
}

impl<'a, D: NetworkDictionary> Predictor<'a, D>
{
    pub fn new(
        dictionary: &'a mut D,
        words: Vec<InputType>,
        sizes: LayerSizes,
        temperature: f32,
        predict_amount: usize
    ) -> Self
    {
        Self{
            dictionary,
            words,
            sizes,
            temperature,
            predict_amount
        }
    }

const fix_me_0: () = ();
/*    pub fn predict_into<N, O>(
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

        let mut previous_word = None;
        let mut previous_state = None;

        let dropout_masks = network.create_dropout_masks(
            self.sizes.hidden,
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

                let bytes = self.dictionary.word_to_bytes(previous_word, word);
                previous_word = Some(word);

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
    }*/
}

type VectorizerType<'a, R, D> = WordVectorizer<<D as NetworkDictionary>::Adapter<BufReader<R>>, &'a mut D>;

#[derive(Clone)]
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

const fix_me_1: () = ();
/*trait FromGuesses<N, O>
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
    for<'b> &'b N::Unit<DiffWrapper>: IntoIterator<Item=&'b DiffWrapper>,
    for<'b> &'b mut N::Unit<DiffWrapper>: IntoIterator<Item=&'b mut DiffWrapper>
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam>,
        input_outputs: impl Iterator<Item=(InputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, Self)>;
}

impl<N, O> FromGuesses<N, O> for bool
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
    for<'b> &'b N::Unit<DiffWrapper>: IntoIterator<Item=&'b DiffWrapper>,
    for<'b> &'b mut N::Unit<DiffWrapper>: IntoIterator<Item=&'b mut DiffWrapper>
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam>,
        input_outputs: impl Iterator<Item=(InputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, Self)>
    {
        network.correct_guesses(input_outputs)
    }
}

impl<N, O> FromGuesses<N, O> for u32
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
    for<'b> &'b N::Unit<DiffWrapper>: IntoIterator<Item=&'b DiffWrapper>,
    for<'b> &'b mut N::Unit<DiffWrapper>: IntoIterator<Item=&'b mut DiffWrapper>
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam>,
        input_outputs: impl Iterator<Item=(InputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, Self)>
    {
        network.top_guesses(input_outputs)
    }
}

impl<N, O> FromGuesses<N, O> for f32
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
    for<'b> &'b N::Unit<DiffWrapper>: IntoIterator<Item=&'b DiffWrapper>,
    for<'b> &'b mut N::Unit<DiffWrapper>: IntoIterator<Item=&'b mut DiffWrapper>
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam>,
        input_outputs: impl Iterator<Item=(InputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, Self)>
    {
        network.certainty_guesses(input_outputs)
    }
}*/

#[derive(Clone)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraInfo
{
    pub iterations: u64
}

impl Default for ExtraInfo
{
    fn default() -> Self
    {
        Self{
            iterations: 0
        }
    }
}

const fix_me_4: () = ();
#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralNetwork<N, O, D>
/*where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<DiffWrapper>: NetworkUnit,
    for<'a> O::WeightParam: Serialize + Deserialize<'a>*/
{
    dictionary: D,
    network: N,
//    network: Network<N, O::WeightParam>,
    optimizer: O,
    gradient_clip: Option<f32>,
    #[serde(default)]
    extra_info: ExtraInfo,
    sizes: LayerSizes
}

// wut do u mean its not used??
//#[allow(dead_code)]
//pub type EN<T> = <EmbeddingsUnitFactory as UnitFactory>::Unit<T>;

const fix_me_2: () = ();
/*impl<O, D> NeuralNetwork<EmbeddingsUnitFactory, O, D>
where
    O: Optimizer,
    EN<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    EN<DiffWrapper>: NetworkUnit,
    for<'a> O::WeightParam: Serialize + Deserialize<'a>
{
    pub fn without_optimizer(self) -> NeuralNetwork<EmbeddingsUnitFactory, (), D>
    where
        EN<()>: OptimizerUnit<()>,
    {
        NeuralNetwork{
            dictionary: self.dictionary,
            network: self.network.without_optimizer(),
            optimizer: (),
            gradient_clip: self.gradient_clip,
            extra_info: self.extra_info,
            sizes: self.sizes
        }
    }
}*/

const fix_me_3: () = ();
impl<D> NeuralNetwork<(), (), D>
/*where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
    for<'b> &'b N::Unit<DiffWrapper>: IntoIterator<Item=&'b DiffWrapper>,
    for<'b> &'b mut N::Unit<DiffWrapper>: IntoIterator<Item=&'b mut DiffWrapper>,
    for<'a> O::WeightParam: Serialize + Deserialize<'a>,
    D: NetworkDictionary*/
{
    pub fn new(
        dictionary: D,
        sizes: LayerSizes,
        dropout_probability: f32,
        gradient_clip: Option<f32>
    ) -> Self
//    where
//        O::WeightParam: NewableLayer
    {
        //let network = Network::new(sizes, dropout_probability);
        let network = ();

        //let optimizer = O::new();
        let optimizer = ();

        let extra_info = ExtraInfo::default();

        Self{dictionary, network, optimizer, gradient_clip, extra_info, sizes}
    }

/*    pub fn into_embeddings_info(self) -> (D, Network<N, O::WeightParam>)
    {
        (self.dictionary, self.network)
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

        SaveFormat::serialize(BufWriter::new(writer), self).unwrap();
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, <SaveFormat as SerializeFormat>::Error>
    where
        N: DeserializeOwned,
        O: DeserializeOwned,
        D: DeserializeOwned
    {
        let reader = File::open(path)?;

        SaveFormat::deserialize(BufReader::new(reader))
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

    fn with_guesses<R, T: FromGuesses<N, O>>(&mut self, reader: R) -> Vec<(Box<[u8]>, T, Box<[u8]>)>
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        let inputs = self.vectorized(reader);

        let input_outputs = InputOutputIter::new(
            &self.dictionary,
            inputs.iter()
        );

        self.network.disable_gradients();

        // im only getting the guess info on the output, NOT the inputs, therefore skip the first one cuz it has no prediction for it
        inputs.iter().cloned().map(Some).zip(inputs.iter().skip(1).cloned()).map(|(previous_word, word)|
        {
            self.dictionary.word_to_bytes(previous_word, word)
        }).zip(T::from_guesses(&mut self.network, input_outputs.clone()).map(|(highest_index, certainty)|
        {
            (certainty, VectorWord::from_raw(highest_index))
        }).scan(None, |previous_word, (certainty, word)|
        {
            let output = Some((certainty, self.dictionary.word_to_bytes(*previous_word, word)));

            *previous_word = Some(word);

            output
        })).map(|(a, (b, c))| (a, b, c)).collect()
    }

    pub fn correct_guesses<R>(&mut self, reader: R) -> Vec<(Box<[u8]>, bool, Box<[u8]>)>
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        self.with_guesses(reader)
    }

    pub fn top_guesses<R>(&mut self, reader: R) -> Vec<(Box<[u8]>, u32, Box<[u8]>)>
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        self.with_guesses(reader)
    }

    pub fn certainty_guesses<R>(&mut self, reader: R) -> Vec<(Box<[u8]>, f32, Box<[u8]>)>
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        self.with_guesses(reader)
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
    }*/

    // these trait bounds make me angry and i cant make them disappear cuz TRAITS SUCK
    pub fn train<const EMBEDDINGS: bool, RT, R>(
        &mut self,
        info: TrainingInfo,
        testing_reader: Option<RT>,
        reader: R
    )
/*    where
        RT: Read,
        R: Read,
        for<'b> VectorizerType<'b, RT, D>: Iterator<Item=VectorWord>,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        for<'b> &'b mut N::Unit<O::WeightParam>: IntoIterator<Item=&'b mut O::WeightParam>,
        N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam, Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
        N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam, Unit<LayerType>=N::Unit<LayerType>>,
        N::Unit<DiffWrapper>: NetworkUnit<Unit<LayerType>=N::Unit<LayerType>> + SubAssign + fmt::Debug,
        N::Unit<LayerType>: DivAssign<f32> + AddAssign + Serialize + DeserializeOwned + IntoIterator<Item=LayerType>,
        for<'b> &'b mut N::Unit<LayerType>: IntoIterator<Item=&'b mut LayerType>,
        for<'b> InputOutput<'b, EMBEDDINGS, D>: InputOutputable*/
    {
        if let Some(learning_rate) = info.learning_rate
        {
//            self.optimizer.set_learning_rate(learning_rate);
        }

        // i dunno wuts the correct way to handle this stuff
/*        let batch_step = info.batch_size * info.steps_num.mid();

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
                let total_iterations = self.extra_info.iterations;

                eprintln!("total iteration: {total_iterations}, iteration: {input_index}");
            }

            self.extra_info.iterations = self.extra_info.iterations.saturating_add(1);

            time_debug! {
                let steps_num = info.steps_num.get();

                let print_loss = (input_index % inputs_per_loss) == inputs_per_loss - 1;
                if print_loss
                {
                    output_loss(self);
                }

                let mut kahan_sum = KahanSum::new();

                let max_batch_start = inputs.len()
                    .saturating_sub(steps_num + (InputOutput::<EMBEDDINGS, D>::min_len() - 1));

                let mut gradients = (0..info.batch_size).map(|_|
                {
                    let batch_start = if max_batch_start == 0
                    {
                        0
                    } else
                    {
                        fastrand::usize(0..max_batch_start)
                    };

                    let values = InputOutput::<EMBEDDINGS, _>::values_slice(
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

        output_loss(self);*/
    }

/*    pub fn predict_into<R>(
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

            Predictor::new(&mut self.dictionary, words, self.sizes, temperature, amount)
        };

        let predicted = f(predictor, &mut self.network);

        self.network.enable_gradients();

        predicted
    }*/
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
        let mut test_layer = LayerType::from_raw([1.0, 2.0, 8.0], 3, 1);

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
