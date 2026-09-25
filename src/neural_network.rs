use std::{
    f32,
    fmt,
    iter,
    borrow::Cow,
    cell::RefCell,
    marker::PhantomData,
    io::{self, Read, Write, BufReader, BufWriter},
    fs::File,
    path::Path,
    ops::Range
};

use serde::{Serialize, Deserialize};

#[allow(unused_imports)]
use crate::{
    Config,
    EmbeddingsUnitFactory,
    word_vectorizer::{
        ByteDictionary,
        CharDictionary,
        WordDictionary,
        BpeDictionary,
        EmbeddingsDictionary,
        NetworkDictionary,
        VectorWord,
        WordVectorizer,
        ReaderAdapter
    }
};

use optimizers::*;

pub use optimizers::{NewableLayer, DecayFunction, Optimizer};

pub use network_unit::{
    DebugUnitInfo,
    NetworkStateSelectable,
    NetworkStateGettable,
    NetworkUnitNewable,
    NetworkUnit,
    GenericUnit,
    UnitFactory,
    OptimizerUnit
};

pub use network::{PrecomputedRng, Network, SaveNetwork, UnitState, SaveWeightType, LayerSizes, WeightsNamed, NetworkConfigInfo};
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

#[allow(unused_imports)]
use star::Star;

pub use embedding_unit::EmbeddingUnit;

mod optimizers;
mod network_unit;
mod gru;
mod lstm;
mod star;
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

pub struct OneHotEmbeddings;

pub struct BagOfWordsEmbeddings;

pub trait EmbeddingsTypeable
{
    fn min_len() -> usize;

    fn name() -> &'static str;
}

impl EmbeddingsTypeable for OneHotEmbeddings
{
    fn min_len() -> usize { 1 }

    fn name() -> &'static str { "one hot" }
}

impl EmbeddingsTypeable for BagOfWordsEmbeddings
{
    fn min_len() -> usize { BAG_OF_WORDS_EMBEDDINGS_COUNT * 2 }

    fn name() -> &'static str { "bag of words" }
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

type VectorizerType<'a, R, D> = WordVectorizer<<D as NetworkDictionary>::Adapter<BufReader<R>>, &'a mut D>;

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
    values: Cow<'a, Vec<VectorWord>>,
    batch_starts: Cow<'a, Vec<usize>>,
    inputs_per_batch: usize
}

impl<'a, D> InputOutput<'a, D>
{
    pub fn new(
        dictionary: &'a D,
        values: &'a Vec<VectorWord>,
        batch_starts: &'a Vec<usize>,
        inputs_per_batch: usize
    ) -> Self
    where
        D: NetworkDictionary
    {
        let (batch_starts, values) = if dictionary.needs_reencoding()
        {
            let batch_size = batch_starts.len();
            let sequential_batch_starts: Vec<usize> = (0..batch_size).map(|x| x * inputs_per_batch).collect();

            let mut reencoded_values = Vec::new();

            fn reencode_at<D: NetworkDictionary>(
                reencoded_values: &mut Vec<VectorWord>,
                dictionary: &D,
                values: &[VectorWord],
                batch_start: usize,
                take_amount: usize,
                inputs_per_batch: usize
            )
            {
                let reencoded = dictionary.reencode(&values[batch_start..(batch_start + inputs_per_batch)]);

                if reencoded.len() >= inputs_per_batch
                {
                    reencoded_values.extend(&reencoded[..inputs_per_batch]);
                } else
                {
                    reencode_at(reencoded_values, dictionary, values, batch_start, take_amount * 2, inputs_per_batch)
                }
            }

            batch_starts.iter().copied().for_each(|batch_start|
            {
                reencode_at(&mut reencoded_values, dictionary, values, batch_start, inputs_per_batch, inputs_per_batch)
            });

            (Cow::Owned(sequential_batch_starts), Cow::Owned(reencoded_values))
        } else
        {
            (Cow::Borrowed(batch_starts), Cow::Borrowed(values))
        };

        Self{
            dictionary,
            values,
            batch_starts,
            inputs_per_batch
        }
    }

    pub fn iter<EmbeddingsType>(&self) -> InputOutputEmbeddingsIter<'_, EmbeddingsType, D>
    {
        InputOutputEmbeddingsIter::new(self.dictionary, &self.values, &self.batch_starts, self.inputs_per_batch)
    }
}

pub struct InputOutputEmbeddingsIter<'a, EmbeddingsType, D>
{
    dictionary: &'a D,
    inputs: &'a [VectorWord],
    batch_starts: &'a [usize],
    inputs_per_batch: usize,
    index: usize,
    _embeddings: PhantomData<EmbeddingsType>
}

impl<'a, EmbeddingsType, D> Clone for InputOutputEmbeddingsIter<'a, EmbeddingsType, D>
{
    fn clone(&self) -> Self
    {
        Self{
            dictionary: self.dictionary,
            inputs: self.inputs,
            batch_starts: self.batch_starts,
            inputs_per_batch: self.inputs_per_batch,
            index: self.index,
            _embeddings: PhantomData
        }
    }
}

impl<'a, EmbeddingsType, D> InputOutputEmbeddingsIter<'a, EmbeddingsType, D>
{
    pub fn new(
        dictionary: &'a D,
        inputs: &'a [VectorWord],
        batch_starts: &'a [usize],
        inputs_per_batch: usize
    ) -> Self
    {
        Self{
            dictionary,
            inputs,
            batch_starts,
            inputs_per_batch,
            index: 0,
            _embeddings: PhantomData
        }
    }

    fn around_window(context: &[VectorWord], amount: usize) -> Box<[usize]>
    {
        let mut words = context.iter().take(amount)
            .chain(context.iter().rev().take(amount))
            .map(VectorWord::index)
            .collect::<Vec<usize>>();

        words.sort_unstable();
        words.dedup();

        words.into_boxed_slice()
    }

    fn index_of_batch_inputs(&self, batch_index: usize) -> usize
    {
        self.batch_starts[batch_index]
    }

    fn batch_size(&self) -> usize
    {
        self.batch_starts.len()
    }

    fn next_single(&mut self) -> Option<(OwnedInputType, OneHotLayer)>
    where
        D: NetworkDictionary
    {
        if (self.index + 1) == self.inputs_per_batch
        {
            return None;
        }

        let words_amount = self.dictionary.words_amount();

        let this_input = {
            let words = (0..self.batch_size()).map(|batch_index|
            {
                [self.inputs[self.index_of_batch_inputs(batch_index) + self.index].index()].into()
            }).collect::<Box<[_]>>();

            OneHotLayer::new(words, words_amount, self.batch_size())
        };

        let this_input: OwnedInputType = self.dictionary.one_hot_to_input(this_input);

        let this_output = {
            let words = (0..self.batch_size()).map(|batch_index|
            {
                [self.inputs[self.index_of_batch_inputs(batch_index) + self.index + 1].index()].into()
            }).collect::<Box<[_]>>();

            OneHotLayer::new(words, words_amount, self.batch_size())
        };

        self.index += 1;

        Some((this_input, this_output))
    }

    fn next_bag_of_words(&mut self, amount: usize) -> Option<(OwnedInputType, OneHotLayer)>
    where
        D: NetworkDictionary
    {
        if (self.index + BAG_OF_WORDS_EMBEDDINGS_COUNT * 2) == self.inputs_per_batch
        {
            return None;
        }

        let words_amount = self.dictionary.words_amount();

        let this_input = {
            let words = (0..self.batch_size()).map(|batch_index|
            {
                let context_start = self.index_of_batch_inputs(batch_index) + self.index;
                let context = &self.inputs[context_start..(context_start + BAG_OF_WORDS_EMBEDDINGS_COUNT * 2 + 1)];

                Self::around_window(context, amount)
            }).collect::<Box<[_]>>();

            OneHotLayer::new(words, words_amount, self.batch_size())
        };

        let this_input: OwnedInputType = self.dictionary.one_hot_to_input(this_input);

        let this_output = {
            let words = (0..self.batch_size()).map(|batch_index|
            {
                [self.inputs[self.index_of_batch_inputs(batch_index) + self.index + BAG_OF_WORDS_EMBEDDINGS_COUNT].index()].into()
            }).collect::<Box<[_]>>();

            OneHotLayer::new(words, words_amount, self.batch_size())
        };

        self.index += 1;

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
    iter.inputs_per_batch - iter.index - EmbeddingsType::min_len()
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
        for<'b> &'b N::Unit<WeightInfo>: IntoIterator<Item=&'b WeightInfo>,
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
        for<'b> &'b N::Unit<WeightInfo>: IntoIterator<Item=&'b WeightInfo>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        for<'b> &'b N::Unit<DiffTensor>: IntoIterator<Item=&'b DiffTensor>,
        for<'b> &'b mut N::Unit<DiffTensor>: IntoIterator<Item=&'b mut DiffTensor>
    {
        let mut predicted = Vec::with_capacity(self.predict_amount);
        self.predict_into(network, &mut predicted);

        predicted.into_boxed_slice()
    }
}

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
            Self::StepsRange(range) => if range.start == range.end { range.start } else { fastrand::usize(range.clone()) }
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
    for<'a> &'a N::Unit<WeightInfo>: IntoIterator<Item=&'a WeightInfo>,
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
    for<'a> &'a N::Unit<WeightInfo>: IntoIterator<Item=&'a WeightInfo>,
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
    for<'a> &'a N::Unit<WeightInfo>: IntoIterator<Item=&'a WeightInfo>,
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
pub struct NeuralNetwork<NetworkType, O, D>
{
    dictionary: D,
    network: NetworkType,
    optimizer: O,
    gradient_clip: Option<f32>,
    extra_info: ExtraInfo,
    sizes: LayerSizes
}

pub type EN<T> = <EmbeddingsUnitFactory as UnitFactory>::Unit<T>;

// only use this for saving the network, it doesnt fully clone things!!
impl<O, D> Clone for NeuralNetwork<Network<EmbeddingsUnitFactory, O::WeightParam>, O, D>
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

impl<O, D> NeuralNetwork<Network<EmbeddingsUnitFactory, O::WeightParam>, O, D>
where
    O: Optimizer
{
    pub fn without_optimizer(self) -> NeuralNetwork<Network<EmbeddingsUnitFactory, ()>, (), D>
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
{
    #[allow(dead_code)]
    pub fn inner_network(&self) -> &N
    {
        &self.network
    }

    #[allow(dead_code)]
    pub fn inner_network_mut(&mut self) -> &mut N
    {
        &mut self.network
    }

    pub fn dictionary(&self) -> &D
    {
        &self.dictionary
    }
}

impl<N, O, D> NeuralNetwork<SaveNetwork<N, O::WeightParam>, O, D>
where
    N: UnitFactory,
    O: Optimizer
{
    pub fn load_data(path: &Path) -> Result<Self, <SaveFormat as SerializeFormat>::Error>
    where
        for<'de> O: Deserialize<'de>,
        for<'de> D: Deserialize<'de>,
        for<'de> O::WeightParam: Deserialize<'de>,
        for<'de> N::Unit<O::WeightParam>: Deserialize<'de>,
        for<'de> N::Unit<SaveWeightType>: Deserialize<'de>
    {
        let reader = File::open(path)?;

        SaveFormat::deserialize(BufReader::new(reader))
    }

    pub fn into_embeddings_info(self) -> (D, SaveNetwork<N, O::WeightParam>)
    {
        (self.dictionary, self.network)
    }
}

impl<N, O, D> NeuralNetwork<Network<N, O::WeightParam>, O, D>
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    UnitState<N, DiffTensorPtr>: Clone + NetworkStateSelectable<UnitState<N, PhiOtherSelectorRecordingIndex>>,
    UnitState<N, PhiOtherSelectorRecordingIndex>: NetworkStateGettable<UnitState<N, DiffTensorPtr>>,
    for<'b> &'b N::Unit<WeightInfo>: IntoIterator<Item=&'b WeightInfo>,
    for<'b> &'b N::Unit<DiffTensor>: IntoIterator<Item=&'b DiffTensor>,
    for<'b> &'b mut N::Unit<DiffTensor>: IntoIterator<Item=&'b mut DiffTensor>,
    D: NetworkDictionary
{
    pub fn new(
        dictionary: D,
        sizes: LayerSizes,
        config: NetworkConfigInfo,
        input_dropout_probability: f32,
        dropout_probability: f32,
        gradient_clip: Option<f32>
    ) -> Self
    where
        O::WeightParam: NewableLayer,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        let network = Network::new(sizes, input_dropout_probability, dropout_probability, config);

        let optimizer = O::new();

        let extra_info = ExtraInfo::default();

        Self{dictionary, network, optimizer, gradient_clip, extra_info, sizes}
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
        let path = path.as_ref();

        eprintln!("saved network at {}", path.display());

        let writer = File::create(path)?;

        Ok(SaveFormat::serialize(BufWriter::new(writer), self).unwrap())
    }

    pub fn load<P: AsRef<Path>>(
        config: NetworkConfigInfo,
        batch_size: usize,
        path: P
    ) -> Result<Self, <SaveFormat as SerializeFormat>::Error>
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
        let path = path.as_ref();

        eprintln!("loading network from {}", path.display());

        let this = NeuralNetwork::load_data(path)?;

        Ok(Self{
            dictionary: this.dictionary,
            network: Network::load(this.network, config, batch_size),
            optimizer: this.optimizer,
            gradient_clip: this.gradient_clip,
            extra_info: this.extra_info,
            sizes: this.sizes
        })
    }

    fn with_guesses<R, T: FromGuesses<N, O>>(&mut self, reader: R) -> Vec<(Box<[u8]>, T, Box<[u8]>)>
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>
    {
        self.network.set_predict_mode();

        let inputs = self.vectorized(reader);

        let input_outputs = InputOutputEmbeddingsIter::<NEmbeddings, D>::new(
            &self.dictionary,
            &inputs,
            &[0],
            inputs.len()
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
        })).map(|(a, (b, c))| (a, b, c)).collect()
    }

    pub fn correct_guesses<R>(&mut self, reader: R) -> Vec<(Box<[u8]>, bool, Box<[u8]>)>
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        self.with_guesses(reader)
    }

    pub fn top_guesses<R>(&mut self, reader: R) -> Vec<(Box<[u8]>, u32, Box<[u8]>)>
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        self.with_guesses(reader)
    }

    pub fn certainty_guesses<R>(&mut self, reader: R) -> Vec<(Box<[u8]>, f32, Box<[u8]>)>
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        self.with_guesses(reader)
    }

    pub fn test_loss<R>(
        &mut self,
        reader: R,
        calculate_accuracy: bool
    ) -> f32
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        let inputs = self.vectorized(reader);

        self.test_loss_inner(&inputs, calculate_accuracy)
    }

    fn test_loss_inner(
        &mut self,
        inputs: &[VectorWord],
        calculate_accuracy: bool
    ) -> f32
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        for<'b> InputOutputEmbeddingsIter<'b, OneHotEmbeddings, D>: ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>
    {
        let input_outputs = InputOutputEmbeddingsIter::new(&self.dictionary, inputs, &[0], inputs.len());

        if calculate_accuracy
        {
            self.network.set_predict_mode();

            let accuracy = self.network.accuracy(input_outputs.clone());

            println!("accuracy: {}%", accuracy * 100.0);

            accuracy
        } else
        {
            self.network.set_train_mode();

            let total_loss = self.network.feedforward_no_gradient(input_outputs);
            let loss = total_loss / inputs.len() as f32;

            Self::print_loss("testing".to_owned(), loss);

            loss
        }
    }

    fn print_loss(name: String, loss: f32)
    {
        println!("{name} loss: {loss}");
    }

    fn vectorized<R: Read>(&mut self, reader: R) -> Vec<VectorWord>
    where
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>
    {
        self.dictionary.vectorized(reader)
    }

    pub fn train<EmbeddingType, R>(
        &mut self,
        info: TrainingInfo,
        reader: R
    )
    where
        EmbeddingType: EmbeddingsTypeable,
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        for<'b> &'b mut N::Unit<O::WeightParam>: IntoIterator<Item=&'b mut O::WeightParam>,
        N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam, Unit<DiffTensor>=N::Unit<DiffTensor>>,
        N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam, Unit<LayerType>=N::Unit<LayerType>>,
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

        let average_inputs_per_batch = info.batch_size * info.steps_num.mid();

        let inputs: Vec<_> = self.vectorized(reader);

        let iterations_per_epoch = (inputs.len() / average_inputs_per_batch).max(1);

        let display_header = !info.less_info;
        let display_inner = !info.less_info;

        if display_header
        {
            println!("input vector size: {}", self.dictionary.words_amount());
            println!("parameters amount: {}", self.network.parameters_amount());
            println!("batch size: {}", info.batch_size);

            println!("using {} inputs", EmbeddingType::name());

            println!("steps amount: {}", info.steps_num);

            println!("iterations per epoch: ~{iterations_per_epoch}");
        }

        for input_index in 0..info.iterations
        {
            self.extra_info.iterations = self.extra_info.iterations.saturating_add(1);

            time_debug! {
                let steps_num = info.steps_num.get();

                let min_len: usize = EmbeddingType::min_len();
                let inputs_per_block = (steps_num + min_len) * info.batch_size;

                let max_batch_start = inputs.len().saturating_sub(inputs_per_block);

                let batch_starts: Vec<usize> = iter::repeat_with(||
                {
                    if max_batch_start == 0
                    {
                        0
                    } else
                    {
                        fastrand::usize(0..max_batch_start)
                    }
                }).take(info.batch_size).collect();

                let values = InputOutput::new(
                    &self.dictionary,
                    &inputs,
                    &batch_starts,
                    steps_num
                );

                let rng = fastrand::Rng::new();
                let (loss, gradients_batch): (f32, _) = self.network.gradients(rng, values.iter());

                let batch_loss = loss as f64 / steps_num as f64;

                if display_inner
                {
                    let total_iterations = self.extra_info.iterations;

                    Self::print_loss(format!("iteration {total_iterations} ({input_index}) training"), batch_loss as f32);
                }

                let gradients = gradients_batch.average_batch();
                self.network.apply_gradients(gradients, &mut self.optimizer, self.gradient_clip);
            }
        }
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
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
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
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
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
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
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

    use crate::word_vectorizer::InputData;

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

        Softmaxer::softmax(test_layer.as_mut());

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

    type ThisEmbeddings = BagOfWordsEmbeddings;

    type ThisDictionary = CharDictionary;

    fn gradient_with_batch_size(
        rng: PrecomputedRng,
        network: &mut NeuralNetwork<Network<Unit, ()>, (), ThisDictionary>,
        inputs: &Vec<VectorWord>,
        batch_starts: &Vec<usize>,
        inputs_per_batch: usize
    ) -> WeightsFullContainer<Unit, LayerType>
    {
        let values = InputOutput::new(
            &network.dictionary,
            inputs,
            batch_starts,
            inputs_per_batch
        );

        network.network.gradients(rng, values.iter::<ThisEmbeddings>()).1
    }

    #[test]
    fn batch_equivalent()
    {
        let inputs_amount = 5;
        let batch_size = 32;

        let dictionary = CharDictionary::new(InputData::String("abcde".to_owned()));

        let is_input_one_hot = ThisDictionary::is_input_one_hot();

        let input_size = dictionary.input_amount();
        let output_size = dictionary.words_amount();

        let inputs_per_batch = inputs_amount + ThisEmbeddings::min_len();

        fastrand::seed(333);

        let inputs: Vec<_> = iter::repeat_with(||
        {
            VectorWord::from_raw(fastrand::usize(0..output_size))
        }).take(batch_size * inputs_per_batch).collect();

        let layers = 3;

        let hidden = 32;

        let layer_sizes = LayerSizes{
            hidden,
            layers,
            initial_input: input_size,
            input: input_size,
            output: output_size,
            final_output: output_size,
            batch_size
        };

        let input_dropout_probability = 0.5;
        let dropout_probability = 0.5;
        let gradient_clip = Some(1.0);

        fastrand::seed(222);

        let mut network_single = NeuralNetwork::new(
            dictionary.clone(),
            LayerSizes{
                batch_size: 1,
                ..layer_sizes
            },
            NetworkConfigInfo{
                is_input_one_hot,
                is_multistep: true,
                print_optional_info: true
            },
            input_dropout_probability,
            dropout_probability,
            gradient_clip
        );

        network_single.network.set_train_mode();

        network_single.network.prepare(true);

        fastrand::seed(111);

        let embeddings_input_rolls = if USE_EMBEDDING_LAYER { input_size } else { 0 };
        let embeddings_output_rolls = if USE_EMBEDDING_LAYER { output_size } else { 0 };
        let output_rolls = output_size;

        let rolls_per_input_weight = hidden;
        let rolls_per_layer_weight = hidden;

        let rolls_per_hidden = hidden;

        dbg!(
            embeddings_input_rolls,
            embeddings_output_rolls,
            output_rolls,
            rolls_per_input_weight,
            rolls_per_layer_weight,
            rolls_per_hidden,
            layers
        );

        // hardcoded for lstm
        let weight_units = 4;
        let hidden_units = 4;

        let rolls_per_batch = embeddings_input_rolls
            + embeddings_output_rolls
            + output_rolls
            + rolls_per_input_weight * weight_units
            + rolls_per_layer_weight * weight_units * (layers - 1)
            + rolls_per_hidden * hidden_units * layers;

        dbg!(rolls_per_batch);

        let dropout_rng_values: Vec<f32> = {
            let total_rolls = batch_size * rolls_per_batch;

            dbg!(total_rolls);

            iter::repeat_with(||
            {
                fastrand::f32()
            }).take(total_rolls).collect()
        };

        let mut single_added_gradients = (0..batch_size).map(|batch_step|
        {
            let count = inputs_per_batch;
            let start = batch_step * count;

            let single_rng = {
                let mut index = 0;

                let mut values: Vec<f32> = Vec::new();

                let mut push_dropout_layer = |index: &mut usize, s: usize|
                {
                    for _ in 0..s
                    {
                        let offset = batch_step * s;

                        values.push(dropout_rng_values[*index + offset]);

                        *index += 1;
                    }

                    *index += (batch_size - 1) * s;
                };

                for i in 0..layers
                {
                    for _ in 0..weight_units
                    {
                        if i == 0
                        {
                            push_dropout_layer(&mut index, rolls_per_input_weight);
                        } else
                        {
                            push_dropout_layer(&mut index, rolls_per_layer_weight);
                        }
                    }

                    for _ in 0..hidden_units
                    {
                        push_dropout_layer(&mut index, rolls_per_hidden);
                    }
                }

                push_dropout_layer(&mut index, embeddings_output_rolls);
                push_dropout_layer(&mut index, output_rolls);

                push_dropout_layer(&mut index, embeddings_input_rolls);

                PrecomputedRng{
                    index: 0,
                    values
                }
            };

            let inputs = inputs[start..(start + count)].to_vec();
            gradient_with_batch_size(single_rng, &mut network_single, &inputs, &vec![0], inputs_per_batch)
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
            dictionary,
            layer_sizes,
            NetworkConfigInfo{
                is_input_one_hot,
                is_multistep: true,
                print_optional_info: true
            },
            input_dropout_probability,
            dropout_probability,
            gradient_clip
        );

        network_batched.network.set_train_mode();

        network_batched.network.prepare(true);

        let batched_rng = PrecomputedRng{
            index: 0,
            values: dropout_rng_values.clone()
        };

        let batch_starts: Vec<usize> = (0..batch_size).map(|i| i * inputs_per_batch).collect();
        let batched_gradients_batch = gradient_with_batch_size(batched_rng, &mut network_batched, &inputs, &batch_starts, inputs_per_batch);
        let batched_gradients = batched_gradients_batch.average_batch();

        eprintln!("calculated batched_gradients");

        // eprintln!("single_added_gradients: {single_added_gradients:?}");
        // eprintln!("batched_gradients: {batched_gradients:?}");

        single_added_gradients.iter().zip(batched_gradients.iter()).for_each(|(single_added_gradient, batched_gradient)|
        {
            let all_equal = single_added_gradient.iter().zip(batched_gradient.iter()).all(|(single, batched)|
            {
                *single == *batched
            });

            assert!(all_equal, "single:  {single_added_gradient:?}\nnot equal to\nbatched: {batched_gradient:?}");
        });
    }
}
