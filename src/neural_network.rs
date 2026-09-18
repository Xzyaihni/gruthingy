use std::{
    f32,
    fmt,
    cell::RefCell,
    marker::PhantomData,
    io::{self, Read, Write, BufReader, BufWriter},
    fs::File,
    path::Path,
    ops::Range
};

use serde::{Serialize, Deserialize};

use network::Network;

#[allow(unused_imports)]
use crate::{
    Config,
    EmbeddingsUnitFactory,
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

use optimizers::*;

pub use optimizers::{NewableLayer, DecayFunction, Optimizer};

pub use network_unit::{
    DebugUnitInfo,
    NetworkStateSelectable,
    NetworkStateGettable,
    NetworkUnitParameterable,
    NetworkUnitNewable,
    NetworkUnit,
    GenericUnit,
    UnitFactory,
    OptimizerUnit
};

pub use network::{UnitState, SaveWeightType, LayerSizes, WeightsNamed, NetworkConfigInfo};
pub use containers::{
    PhiOtherSelectorRecordingIndex,
    OperationsRecorder,
    OperationsRecorderMemory,
    TensorShape,
    LayerType,
    LayerTypeRef,
    LayerTypeMut,
    DiffTensorPtr,
    DiffTensor,
    DiffScalar,
    LoopIndex,
    LoopInputs,
    ShapedTensorIndex,
    TensorPtr,
    OneHotIndex,
    InputType,
    InputTypePtr,
    DiffInputType,
    OwnedInputType,
    OneHotLayer,
    Softmaxer
};

#[allow(unused_imports)]
pub use network::{WeightInfo, WeightInfoPtr, WeightsSize};

#[allow(unused_imports)]
use gru::Gru;

#[allow(unused_imports)]
use lstm::Lstm;

pub use embedding_unit::EmbeddingUnit;

mod optimizers;
mod network_unit;
mod gru;
mod lstm;
mod embedding_unit;

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

pub struct OneHotEmbeddings;

pub struct BagOfWordsEmbeddings;

pub trait EmbeddingsTypeable
{
    fn min_len() -> usize;
}

impl EmbeddingsTypeable for OneHotEmbeddings
{
    fn min_len() -> usize { 0 }
}

impl EmbeddingsTypeable for BagOfWordsEmbeddings
{
    fn min_len() -> usize { BAG_OF_WORDS_EMBEDDINGS_COUNT * 2 }
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

#[derive(Debug)]
pub struct InputOutput<'a, D>
{
    dictionary: &'a D,
    values: &'a [VectorWord],
    batch_size: usize
}

impl<'a, D> InputOutput<'a, D>
{
    pub fn values_slice<EmbeddingsType: EmbeddingsTypeable>(
        dictionary: &'a D,
        values: &'a [VectorWord],
        inputs_count: usize
    ) -> Self
    {
        let min_slice_len = EmbeddingsType::min_len();

        // +1 for target output of last input
        let inputs_block = inputs_count + min_slice_len + 1;

        debug_assert!(values.len() >= inputs_block);
        debug_assert!(values.len() % inputs_block == 0, "values: (len {}) must be evenly divisible into blocks of len {inputs_block}", values.len());

        let batch_size = values.len() / inputs_block;

        Self::new(dictionary, values, batch_size)
    }

    pub fn new(dictionary: &'a D, values: &'a [VectorWord], batch_size: usize) -> Self
    {
        Self{
            dictionary,
            values,
            batch_size
        }
    }

    pub fn iter<EmbeddingsType>(&self) -> InputOutputEmbeddingsIter<'a, EmbeddingsType, D>
    {
        InputOutputEmbeddingsIter::new(self.dictionary, self.values, self.batch_size)
    }
}

pub struct InputOutputEmbeddingsIter<'a, EmbeddingsType, D>
{
    dictionary: &'a D,
    inputs: &'a [VectorWord],
    index: usize,
    batch_size: usize,
    _embeddings: PhantomData<EmbeddingsType>
}

impl<'a, EmbeddingsType, D> Clone for InputOutputEmbeddingsIter<'a, EmbeddingsType, D>
{
    fn clone(&self) -> Self
    {
        Self{
            dictionary: self.dictionary,
            inputs: self.inputs,
            index: self.index,
            batch_size: self.batch_size,
            _embeddings: PhantomData
        }
    }
}

impl<'a, EmbeddingsType, D> InputOutputEmbeddingsIter<'a, EmbeddingsType, D>
{
    pub fn new(dictionary: &'a D, inputs: &'a [VectorWord], batch_size: usize) -> Self
    {
        Self{
            dictionary,
            inputs,
            index: 0,
            batch_size,
            _embeddings: PhantomData
        }
    }

    fn around_window(context: &[VectorWord], amount: usize) -> Vec<VectorWord>
    {
        let mut words = context.iter().take(amount)
            .chain(context.iter().rev().take(amount))
            .copied()
            .collect::<Vec<_>>();

        words.sort_unstable();
        words.dedup();

        words
    }

    fn next_single(&mut self) -> Option<(OwnedInputType, OneHotLayer)>
    where
        D: NetworkDictionary
    {
        let inputs_per_batch = self.inputs.len() / self.batch_size;

        if (self.index + 1) == inputs_per_batch
        {
            return None;
        }

        let words_amount = self.dictionary.words_amount();

        let this_input = {
            let words = (0..self.batch_size).map(|batch_index|
            {
                [self.inputs[batch_index * inputs_per_batch + self.index].index()].into()
            }).collect::<Box<[_]>>();

            OneHotLayer::new(words, words_amount, self.batch_size)
        };

        let this_input: OwnedInputType = this_input.into();

        let this_output = {
            let words = (0..self.batch_size).map(|batch_index|
            {
                [self.inputs[batch_index * inputs_per_batch + self.index + 1].index()].into()
            }).collect::<Box<[_]>>();

            OneHotLayer::new(words, words_amount, self.batch_size)
        };

        self.index += 1;

        Some((this_input, this_output))
    }

    fn next_bag_of_words(&mut self, amount: usize) -> Option<(OwnedInputType, OneHotLayer)>
    where
        D: NetworkDictionary
    {
        let context = todo!();
        let middle_word = todo!();

        self.index += 1;

        let this_input = self.dictionary.words_to_layer(Self::around_window(context, amount));
        let this_output = self.dictionary.words_to_onehot([middle_word]);

        Some((this_input, this_output))
    }
}

impl<'a, D> Iterator for InputOutputEmbeddingsIter<'a, OneHotEmbeddings, D>
where
    D: NetworkDictionary
{
    type Item = (OwnedInputType, OneHotLayer);

    fn next(&mut self) -> Option<Self::Item>
    {
        self.next_single()
    }
}

impl<'a, D> Iterator for InputOutputEmbeddingsIter<'a, BagOfWordsEmbeddings, D>
where
    D: NetworkDictionary
{
    type Item = (OwnedInputType, OneHotLayer);

    fn next(&mut self) -> Option<Self::Item>
    {
        self.next_bag_of_words(BAG_OF_WORDS_EMBEDDINGS_COUNT)
    }
}

fn input_output_embeddings_iter_len<EmbeddingsType: EmbeddingsTypeable, D>(
    iter: &InputOutputEmbeddingsIter<'_, EmbeddingsType, D>
) -> usize
{
    let inputs_per_batch = iter.inputs.len() / iter.batch_size;

    inputs_per_batch - iter.index - (EmbeddingsType::min_len() + 1)
}

impl<'a, D> ExactSizeIterator for InputOutputEmbeddingsIter<'a, OneHotEmbeddings, D>
where
    D: NetworkDictionary
{
    fn len(&self) -> usize
    {
        input_output_embeddings_iter_len(self)
    }
}

impl<'a, D> ExactSizeIterator for InputOutputEmbeddingsIter<'a, BagOfWordsEmbeddings, D>
where
    D: NetworkDictionary
{
    fn len(&self) -> usize
    {
        input_output_embeddings_iter_len(self)
    }
}

struct Predictor<'a, D>
{
    dictionary: &'a mut D,
    words: RefCell<Vec<OwnedInputType>>,
    temperature: f32,
    predict_amount: usize
}

impl<'a, D: NetworkDictionary> Predictor<'a, D>
{
    pub fn new(
        dictionary: &'a mut D,
        words: Vec<OwnedInputType>,
        temperature: f32,
        predict_amount: usize
    ) -> Self
    {
        Self{
            dictionary,
            words: RefCell::new(words),
            temperature,
            predict_amount
        }
    }

    pub fn predict_into<N, O>(
        self,
        network: &mut Network<N, O>,
        mut out: impl Write
    )
    where
        N: UnitFactory,
        N::Unit<O>: OptimizerUnit<O>,
        N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
        N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
        N::Unit<WeightInfoPtr>: NetworkUnitNewable,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        UnitState<N, DiffTensorPtr>: Clone + NetworkStateSelectable<UnitState<N, PhiOtherSelectorRecordingIndex>>,
        UnitState<N, PhiOtherSelectorRecordingIndex>: NetworkStateGettable<UnitState<N, DiffTensorPtr>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        for<'b> &'b N::Unit<DiffTensor>: IntoIterator<Item=&'b DiffTensor>,
        for<'b> &'b mut N::Unit<DiffTensor>: IntoIterator<Item=&'b mut DiffTensor>
    {
        network.set_predict_mode();

        let input_amount = self.words.borrow().len();
        let mut previous_word = None;

        let mut current_input = 0;

        network.predict_temperature(self.temperature, (0..(input_amount + self.predict_amount)).map(|i|
        {
            self.words.borrow()[i].clone()
        }), |output|
        {
            if current_input >= (input_amount - 1)
            {
                let word = output.pick_weighed();
                let word = VectorWord::from_raw(word);

                let layer = self.dictionary.words_to_layer([word]);
                self.words.borrow_mut().push(layer);

                let bytes = self.dictionary.word_to_bytes(previous_word, word);
                previous_word = Some(word);

                out.write_all(&bytes).unwrap();
            }

            current_input += 1;
        });

        out.flush().unwrap();
    }

    pub fn predict_bytes<N, O>(self, network: &mut Network<N, O>) -> Box<[u8]>
    where
        N: UnitFactory,
        N::Unit<O>: OptimizerUnit<O>,
        N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
        N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
        N::Unit<WeightInfoPtr>: NetworkUnitNewable,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        UnitState<N, DiffTensorPtr>: Clone + NetworkStateSelectable<UnitState<N, PhiOtherSelectorRecordingIndex>>,
        UnitState<N, PhiOtherSelectorRecordingIndex>: NetworkStateGettable<UnitState<N, DiffTensorPtr>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        for<'b> &'b N::Unit<DiffTensor>: IntoIterator<Item=&'b DiffTensor>,
        for<'b> &'b mut N::Unit<DiffTensor>: IntoIterator<Item=&'b mut DiffTensor>
    {
        let mut predicted = Vec::with_capacity(self.predict_amount);
        self.predict_into(network, &mut predicted);

        predicted.into_boxed_slice()
    }
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

    pub fn highest(&self) -> usize
    {
        match self
        {
            Self::Steps(x) => *x,
            Self::StepsRange(range) => range.end
        }
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

trait FromGuesses<N: UnitFactory, O: Optimizer>
where
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam>,
        input_outputs: impl Iterator<Item=(OwnedInputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, Self)>;
}

impl<N, O> FromGuesses<N, O> for bool
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
    UnitState<N, DiffTensorPtr>: Clone + NetworkStateSelectable<UnitState<N, PhiOtherSelectorRecordingIndex>>,
    UnitState<N, PhiOtherSelectorRecordingIndex>: NetworkStateGettable<UnitState<N, DiffTensorPtr>>,
    for<'a> &'a N::Unit<WeightInfoPtr>: IntoIterator<Item=&'a WeightInfoPtr>,
    for<'a> &'a N::Unit<DiffTensor>: IntoIterator<Item=&'a DiffTensor>,
    for<'a> &'a mut N::Unit<DiffTensor>: IntoIterator<Item=&'a mut DiffTensor>
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam>,
        input_outputs: impl Iterator<Item=(OwnedInputType, OneHotLayer)>
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
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
    UnitState<N, DiffTensorPtr>: Clone + NetworkStateSelectable<UnitState<N, PhiOtherSelectorRecordingIndex>>,
    UnitState<N, PhiOtherSelectorRecordingIndex>: NetworkStateGettable<UnitState<N, DiffTensorPtr>>,
    for<'a> &'a N::Unit<WeightInfoPtr>: IntoIterator<Item=&'a WeightInfoPtr>,
    for<'a> &'a N::Unit<DiffTensor>: IntoIterator<Item=&'a DiffTensor>,
    for<'a> &'a mut N::Unit<DiffTensor>: IntoIterator<Item=&'a mut DiffTensor>
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam>,
        input_outputs: impl Iterator<Item=(OwnedInputType, OneHotLayer)>
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
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
    UnitState<N, DiffTensorPtr>: Clone + NetworkStateSelectable<UnitState<N, PhiOtherSelectorRecordingIndex>>,
    UnitState<N, PhiOtherSelectorRecordingIndex>: NetworkStateGettable<UnitState<N, DiffTensorPtr>>,
    for<'a> &'a N::Unit<WeightInfoPtr>: IntoIterator<Item=&'a WeightInfoPtr>,
    for<'a> &'a N::Unit<DiffTensor>: IntoIterator<Item=&'a DiffTensor>,
    for<'a> &'a mut N::Unit<DiffTensor>: IntoIterator<Item=&'a mut DiffTensor>
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam>,
        input_outputs: impl Iterator<Item=(OwnedInputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, Self)>
    {
        network.certainty_guesses(input_outputs)
    }
}

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

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "O: Serialize, O::WeightParam: Serialize + Clone, D: Serialize, N::Unit<SaveWeightType>: Serialize, N::Unit<WeightInfo>: Clone + GenericUnit<WeightInfo, Unit<SaveWeightType>=N::Unit<SaveWeightType>>, N::Unit<O::WeightParam>: Serialize + Clone, N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<SaveWeightType>=N::Unit<SaveWeightType>>", deserialize = "O: Deserialize<'de>, O::WeightParam: Deserialize<'de>, D: Deserialize<'de>, N::Unit<O::WeightParam>: Deserialize<'de>, N::Unit<SaveWeightType>: Deserialize<'de> + GenericUnit<SaveWeightType, Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>, N::Unit<O::WeightParam>: Deserialize<'de>, N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>, for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>"))]
pub struct NeuralNetwork<N, O, D>
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>
{
    dictionary: D,
    network: Network<N, O::WeightParam>,
    optimizer: O,
    gradient_clip: Option<f32>,
    extra_info: ExtraInfo,
    sizes: LayerSizes
}

pub type EN<T> = <EmbeddingsUnitFactory as UnitFactory>::Unit<T>;

// only use this for saving the network, it doesnt fully clone things!!
impl<O, D> Clone for NeuralNetwork<EmbeddingsUnitFactory, O, D>
where
    O: Optimizer,
    D: Clone,
    O::WeightParam: Clone
{
    fn clone(&self) -> Self
    {
        Self{
            dictionary: self.dictionary.clone(),
            network: self.network.clone(),
            optimizer: self.optimizer.clone(),
            gradient_clip: self.gradient_clip.clone(),
            extra_info: self.extra_info.clone(),
            sizes: self.sizes.clone()
        }
    }
}

impl<O, D> NeuralNetwork<EmbeddingsUnitFactory, O, D>
where
    O: Optimizer
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
}

impl<N, O, D> NeuralNetwork<N, O, D>
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    UnitState<N, DiffTensorPtr>: Clone + NetworkStateSelectable<UnitState<N, PhiOtherSelectorRecordingIndex>>,
    UnitState<N, PhiOtherSelectorRecordingIndex>: NetworkStateGettable<UnitState<N, DiffTensorPtr>>,
    for<'b> &'b N::Unit<DiffTensor>: IntoIterator<Item=&'b DiffTensor>,
    for<'b> &'b mut N::Unit<DiffTensor>: IntoIterator<Item=&'b mut DiffTensor>,
    D: NetworkDictionary
{
    pub fn new(
        dictionary: D,
        sizes: LayerSizes,
        config: NetworkConfigInfo,
        dropout_probability: f32,
        gradient_clip: Option<f32>
    ) -> Self
    where
        O::WeightParam: NewableLayer,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        debug_assert_eq!(sizes.input, dictionary.words_amount());

        let network = Network::new(sizes, dropout_probability, config);

        let optimizer = O::new();

        let extra_info = ExtraInfo::default();

        Self{dictionary, network, optimizer, gradient_clip, extra_info, sizes}
    }

    pub fn into_embeddings_info(self) -> (D, Network<N, O::WeightParam>)
    {
        (self.dictionary, self.network)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()>
    where
        O: Serialize,
        D: Serialize,
        O::WeightParam: Serialize + Clone,
        N::Unit<SaveWeightType>: Serialize,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<SaveWeightType>=N::Unit<SaveWeightType>>,
        N::Unit<WeightInfo>: Clone + GenericUnit<WeightInfo, Unit<SaveWeightType>=N::Unit<SaveWeightType>>,
        N::Unit<O::WeightParam>: Serialize + Clone
    {
        let writer = File::create(path)?;

        Ok(SaveFormat::serialize(BufWriter::new(writer), self).unwrap())
    }

    pub fn load<P: AsRef<Path>>(config: NetworkConfigInfo, path: P) -> Result<Self, <SaveFormat as SerializeFormat>::Error>
    where
        for<'de> O: Deserialize<'de>,
        for<'de> D: Deserialize<'de>,
        for<'de> O::WeightParam: Deserialize<'de>,
        for<'de> N::Unit<O::WeightParam>: Deserialize<'de>,
        for<'de> N::Unit<SaveWeightType>: Deserialize<'de>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        N::Unit<SaveWeightType>: GenericUnit<SaveWeightType, Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>
    {
        let reader = File::open(path)?;

        let mut this: Self = SaveFormat::deserialize(BufReader::new(reader))?;

        this.network.initialize_with_params(config);

        Ok(this)
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
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        /*let inputs = self.vectorized(reader);

        let input_outputs = InputOutputIter::new(
            &self.dictionary,
            inputs.iter()
        );

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
        })).map(|(a, (b, c))| (a, b, c)).collect()*/todo!()
    }

    pub fn correct_guesses<R>(&mut self, reader: R) -> Vec<(Box<[u8]>, bool, Box<[u8]>)>
    where
        R: Read,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        self.with_guesses(reader)
    }

    pub fn top_guesses<R>(&mut self, reader: R) -> Vec<(Box<[u8]>, u32, Box<[u8]>)>
    where
        R: Read,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        self.with_guesses(reader)
    }

    pub fn certainty_guesses<R>(&mut self, reader: R) -> Vec<(Box<[u8]>, f32, Box<[u8]>)>
    where
        R: Read,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
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
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
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
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        for<'b> InputOutputEmbeddingsIter<'b, OneHotEmbeddings, D>: ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>
    {
        let input_outputs = InputOutputEmbeddingsIter::new(&self.dictionary, inputs, 1);

        if calculate_accuracy
        {
            let accuracy = self.network.accuracy(input_outputs.clone());

            println!("accuracy: {}%", accuracy * 100.0);
        }

        if calculate_loss
        {
            let total_loss = self.network.feedforward_no_gradient(input_outputs);

            Self::print_loss(true, total_loss / inputs.len() as f32);
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

    pub fn train<EmbeddingType, RT, R>(
        &mut self,
        info: TrainingInfo,
        testing_reader: Option<RT>,
        reader: R
    )
    where
        EmbeddingType: EmbeddingsTypeable,
        RT: Read,
        R: Read,
        for<'b> VectorizerType<'b, RT, D>: Iterator<Item=VectorWord>,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        for<'b> &'b mut N::Unit<O::WeightParam>: IntoIterator<Item=&'b mut O::WeightParam>,
        N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam, Unit<DiffTensor>=N::Unit<DiffTensor>>,
        N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam, Unit<LayerType>=N::Unit<LayerType>>,
        N::Unit<WeightInfo>: NetworkUnitParameterable,
        N::Unit<WeightInfo>: GenericUnit<WeightInfo, Unit<LayerType>=N::Unit<LayerType>> + fmt::Debug,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        N::Unit<LayerType>: IntoIterator<Item=LayerType>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        for<'b> &'b mut N::Unit<LayerType>: IntoIterator<Item=&'b mut LayerType>,
        for<'b> &'b mut N::Unit<WeightInfo>: IntoIterator<Item=&'b mut WeightInfo>,
        for<'b> InputOutputEmbeddingsIter<'b, EmbeddingType, D>: ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>
    {
        self.network.set_train_mode();

        self.network.prepare(true);

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
                    .saturating_sub(steps_num + EmbeddingType::min_len() - 1);

                self.network.feedforward_setup_dropout();

                let batch_start = if max_batch_start == 0
                {
                    0
                } else
                {
                    fastrand::usize(0..max_batch_start)
                };

                let values = InputOutput::values_slice::<EmbeddingType>(
                    &self.dictionary,
                    &inputs[batch_start..],
                    steps_num
                );

                let (loss, gradients): (f32, _) = self.network.gradients(values.iter());

                kahan_sum.add(loss as f64 / info.batch_size as f64);

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
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
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
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
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
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
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
        let predictor = {
            // could do this without a collect but wheres the fun in that
            let words = self.vectorized(reader).into_iter().map(|v|
            {
                self.dictionary.words_to_layer([v])
            }).collect::<Vec<_>>();

            Predictor::new(&mut self.dictionary, words, temperature, amount)
        };

        let predicted = f(predictor, &mut self.network);

        predicted
    }
}

#[cfg(test)]
mod tests
{
    use std::iter;

    use super::*;

    use network::WeightsFullContainer;


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
        let mut test_layer = LayerType::from_boxed([1.0, 2.0, 8.0].into(), 3, 1);

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

    struct LstmUnitFactory;

    impl UnitFactory for LstmUnitFactory
    {
        type Unit<T> = Lstm<T>;
    }

    type Unit = LstmUnitFactory;

    fn gradient_with_batch_size(
        network: &mut NeuralNetwork<Unit, (), ByteDictionary>,
        inputs: &[VectorWord],
        steps_num: usize
    ) -> WeightsFullContainer<Unit, LayerType>
    {
        let values = InputOutput::values_slice::<OneHotEmbeddings>(
            &network.dictionary,
            inputs,
            steps_num
        );

        network.network.gradients(values.iter::<OneHotEmbeddings>()).1
    }

    #[test]
    fn batch_equivalent()
    {
        let put_me_batch_size_64 = ();
        let inputs_amount = 3;
        let batch_size = 2;

        let vector_word_size = ByteDictionary.words_amount();

        fastrand::seed(333);

        let inputs: Vec<_> = iter::repeat_with(||
        {
            VectorWord::from_raw(fastrand::usize(0..vector_word_size))
        }).take(batch_size * (inputs_amount + 1)).collect();

        let put_me_to_hidden_32 = ();
        let put_me_to_layers_3 = ();
        let layer_sizes = LayerSizes{
            hidden: 1,
            layers: 1,
            input: vector_word_size,
            output: vector_word_size
        };

        let dropout_probability = 0.5;
        let gradient_clip = Some(1.0);

        fastrand::seed(222);

        let mut network_single = NeuralNetwork::new(
            ByteDictionary,
            layer_sizes,
            NetworkConfigInfo{
                is_input_one_hot: true,
                is_multistep: true,
                print_optional_info: false,
                batch_size: 1
            },
            dropout_probability,
            gradient_clip
        );

        network_single.network.set_train_mode();

        network_single.network.prepare(true);

        fastrand::seed(111);

        network_single.network.feedforward_setup_dropout();

        let mut single_added_gradients = (0..batch_size).map(|batch_step|
        {
            let count = inputs_amount + 1;
            let start = batch_step * count;

            gradient_with_batch_size(&mut network_single, &inputs[start..(start + count)], inputs_amount)
        }).reduce(|mut acc, this|
        {
            acc.iter_mut().zip(this.into_iter()).for_each(|(acc, this)|
            {
                acc.add_inplace(this.as_ref());
            });

            acc
        }).expect("batch size must not be 0");

        single_added_gradients.iter_mut().for_each(|gradient| gradient.mul_scalar_inplace((batch_size as f32).recip()));

        eprintln!("calculated single_added_gradients");

        fastrand::seed(222);

        let mut network_batched = NeuralNetwork::new(
            ByteDictionary,
            layer_sizes,
            NetworkConfigInfo{
                is_input_one_hot: true,
                is_multistep: true,
                print_optional_info: false,
                batch_size
            },
            dropout_probability,
            gradient_clip
        );

        network_batched.network.set_train_mode();

        network_batched.network.prepare(true);

        fastrand::seed(111);

        network_batched.network.feedforward_setup_dropout();

        dbg!(&network_batched.network.recorder);

        let batched_gradients = gradient_with_batch_size(&mut network_batched, &inputs, inputs_amount);

        eprintln!("calculated batched_gradients");

        // eprintln!("single_added_gradients: {single_added_gradients:?}");
        // eprintln!("batched_gradients: {batched_gradients:?}");

        assert_eq!(single_added_gradients, batched_gradients);
    }
}
