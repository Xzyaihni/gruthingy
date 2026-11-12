use std::{
    f32,
    fmt,
    sync::mpsc,
    borrow::Cow,
    cell::RefCell,
    marker::PhantomData,
    io::{self, Read, Write, BufReader, BufWriter},
    fs::File,
    path::Path,
    ops::Range
};

use serde::{Serialize, Deserialize};

pub use crate::config::ActivationFunction;

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

pub use network_unit::{
    NetworkUnitNewable,
    NetworkUnit,
    GenericUnit,
    UnitFactory
};

pub use network::{
    AccuracyCertaintyInfo,
    PrecomputedRng,
    Network,
    SaveNetwork,
    UnitState,
    SaveWeightType,
    LayerSizes,
    NetworkConfigInfo,
    NetworkArchitecture
};

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
    Softmaxer,
    SeedIndex,
    optimizers::{NewableLayer, Optimizer, OptimizerInfo, OptimizerInfoPtr}
};

#[allow(unused_imports)]
pub use containers::{
    close_enough,
    optimizers::{Sgd, PowerSign, Adam, AdamW}
};

#[allow(unused_imports)]
pub use network::{WeightInfo, WeightInfoPtr, WeightsFullInfo};

#[allow(unused_imports)]
use gru::Gru;

#[allow(unused_imports)]
use lstm::Lstm;

#[allow(unused_imports)]
use star::Star;

pub use embedding_unit::EmbeddingUnit;

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

    fn one_hot_count() -> usize { Self::min_len() }
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

            OneHotLayer::new(words, 1, words_amount, self.batch_size())
        };

        let this_input: OwnedInputType = self.dictionary.one_hot_to_input(this_input);

        let this_output = {
            let words = (0..self.batch_size()).map(|batch_index|
            {
                [self.inputs[self.index_of_batch_inputs(batch_index) + self.index + 1].index()].into()
            }).collect::<Box<[_]>>();

            OneHotLayer::new(words, 1, words_amount, self.batch_size())
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

            OneHotLayer::new(words, BAG_OF_WORDS_EMBEDDINGS_COUNT * 2, words_amount, self.batch_size())
        };

        let this_input: OwnedInputType = self.dictionary.one_hot_to_input(this_input);

        let this_output = {
            let words = (0..self.batch_size()).map(|batch_index|
            {
                [self.inputs[self.index_of_batch_inputs(batch_index) + self.index + BAG_OF_WORDS_EMBEDDINGS_COUNT].index()].into()
            }).collect::<Box<[_]>>();

            OneHotLayer::new(words, 1, words_amount, self.batch_size())
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
        N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
        N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
        N::Unit<WeightInfoPtr>: NetworkUnitNewable,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
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
                let (word, _times) = output.as_ref().batch_pick_weighed();
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
        N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
        N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
        N::Unit<WeightInfoPtr>: NetworkUnitNewable,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
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
    Self: Sized,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam<OptimizerInfoPtr>>,
        input_outputs: impl ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>,
        f: impl FnMut(usize, Self)
    );
}

impl<N, O> FromGuesses<N, O> for bool
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
    for<'a> &'a N::Unit<WeightInfo>: IntoIterator<Item=&'a WeightInfo>,
    for<'a> &'a N::Unit<WeightInfoPtr>: IntoIterator<Item=&'a WeightInfoPtr>,
    for<'a> &'a N::Unit<DiffTensor>: IntoIterator<Item=&'a DiffTensor>,
    for<'a> &'a mut N::Unit<DiffTensor>: IntoIterator<Item=&'a mut DiffTensor>
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam<OptimizerInfoPtr>>,
        input_outputs: impl ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>,
        f: impl FnMut(usize, Self)
    )
    {
        network.correct_guesses(input_outputs, f);
    }
}

impl<N, O> FromGuesses<N, O> for u32
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
    for<'a> &'a N::Unit<WeightInfo>: IntoIterator<Item=&'a WeightInfo>,
    for<'a> &'a N::Unit<WeightInfoPtr>: IntoIterator<Item=&'a WeightInfoPtr>,
    for<'a> &'a N::Unit<DiffTensor>: IntoIterator<Item=&'a DiffTensor>,
    for<'a> &'a mut N::Unit<DiffTensor>: IntoIterator<Item=&'a mut DiffTensor>
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam<OptimizerInfoPtr>>,
        input_outputs: impl ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>,
        f: impl FnMut(usize, Self)
    )
    {
        network.top_guesses(input_outputs, f);
    }
}

impl<N, O> FromGuesses<N, O> for AccuracyCertaintyInfo
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
    for<'a> &'a N::Unit<WeightInfo>: IntoIterator<Item=&'a WeightInfo>,
    for<'a> &'a N::Unit<WeightInfoPtr>: IntoIterator<Item=&'a WeightInfoPtr>,
    for<'a> &'a N::Unit<DiffTensor>: IntoIterator<Item=&'a DiffTensor>,
    for<'a> &'a mut N::Unit<DiffTensor>: IntoIterator<Item=&'a mut DiffTensor>
{
    fn from_guesses(
        network: &mut Network<N, O::WeightParam<OptimizerInfoPtr>>,
        input_outputs: impl ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>,
        f: impl FnMut(usize, Self)
    )
    {
        network.certainty_guesses(input_outputs, f);
    }
}

#[derive(Clone)]
pub struct TrainingInfo
{
    pub iterations: usize,
    pub batch_size: usize,
    pub steps_num: StepsNum,
    pub learning_rate: Option<f32>,
    pub lambda: Option<f32>,
    pub reset_state: bool,
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
            lambda: config.lambda,
            reset_state: !config.keep_state,
            calculate_accuracy: config.calculate_accuracy,
            less_info: config.less_info
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraInfo
{
    pub iterations: u64,
    pub last_lr_reduction: u64
}

impl Default for ExtraInfo
{
    fn default() -> Self
    {
        Self{
            iterations: 0,
            last_lr_reduction: 0
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
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

impl<O, D> NeuralNetwork<Network<EmbeddingsUnitFactory, O::WeightParam<OptimizerInfoPtr>>, O, D>
where
    O: Optimizer,
    D: Clone
{
    pub fn without_optimizer(&self) -> NeuralNetwork<SaveNetwork<EmbeddingsUnitFactory, ()>, (), D>
    {
        NeuralNetwork{
            dictionary: self.dictionary.clone(),
            network: self.network.without_optimizer::<O>(),
            optimizer: (),
            gradient_clip: self.gradient_clip.clone(),
            extra_info: self.extra_info.clone(),
            sizes: self.sizes.clone()
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

impl<N, O, D> NeuralNetwork<SaveNetwork<N, O::WeightParam<LayerType>>, O, D>
where
    N: UnitFactory,
    O: Optimizer
{
    pub fn load_data(path: &Path) -> Result<Self, <SaveFormat as SerializeFormat>::Error>
    where
        for<'de> O: Deserialize<'de>,
        for<'de> D: Deserialize<'de>,
        for<'de> O::WeightParam<LayerType>: Deserialize<'de>,
        for<'de> N::Unit<SaveWeightType>: Deserialize<'de>
    {
        let reader = File::open(path)?;

        SaveFormat::deserialize(BufReader::new(reader))
    }

    pub fn save_data(&self, path: &Path) -> io::Result<()>
    where
        O: Serialize,
        D: Serialize,
        O::WeightParam<LayerType>: Serialize,
        N::Unit<SaveWeightType>: Serialize,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<SaveWeightType>=N::Unit<SaveWeightType>>,
        N::Unit<WeightInfo>: GenericUnit<WeightInfo, Unit<SaveWeightType>=N::Unit<SaveWeightType>>
    {
        eprintln!("saved network at {}", path.display());

        let writer = File::create(path)?;

        Ok(SaveFormat::serialize(BufWriter::new(writer), self).unwrap())
    }

    pub fn into_embeddings_info(self) -> (D, SaveNetwork<N, O::WeightParam<LayerType>>)
    {
        (self.dictionary, self.network)
    }
}

impl<N, O, D> NeuralNetwork<Network<N, O::WeightParam<OptimizerInfoPtr>>, O, D>
where
    N: UnitFactory,
    O: Optimizer,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    for<'b> &'b N::Unit<WeightInfo>: IntoIterator<Item=&'b WeightInfo>,
    for<'b> &'b N::Unit<DiffTensor>: IntoIterator<Item=&'b DiffTensor>,
    for<'b> &'b mut N::Unit<DiffTensor>: IntoIterator<Item=&'b mut DiffTensor>,
    D: NetworkDictionary
{
    pub fn new(
        dictionary: D,
        sizes: LayerSizes,
        config: NetworkConfigInfo,
        architecture: NetworkArchitecture
    ) -> Self
    where
        O::WeightParam<OptimizerInfoPtr>: NewableLayer,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        let gradient_clip = architecture.gradient_clip;

        let network = Network::new(sizes, architecture, config);

        let optimizer = O::default();

        let extra_info = ExtraInfo::default();

        Self{dictionary, network, optimizer, gradient_clip, extra_info, sizes}
    }

    pub fn reset(&mut self, config: NetworkConfigInfo, batch_size: usize)
    where
        D: Clone,
        O::WeightParam<LayerType>: Clone,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<SaveWeightType>=N::Unit<SaveWeightType>>,
        N::Unit<WeightInfo>: GenericUnit<WeightInfo, Unit<SaveWeightType>=N::Unit<SaveWeightType>>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        N::Unit<SaveWeightType>: GenericUnit<SaveWeightType, Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>
    {
        let save_network: SaveNetwork<N, O::WeightParam<LayerType>> = SaveNetwork::from_network::<O, O::WeightParam<OptimizerInfoPtr>>(
            &self.network
        );

        self.network = Network::load::<O, O::WeightParam<LayerType>>(save_network, config, batch_size);
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()>
    where
        O: Serialize,
        D: Serialize + Clone,
        O::WeightParam<LayerType>: Serialize + Clone,
        N::Unit<SaveWeightType>: Serialize,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<SaveWeightType>=N::Unit<SaveWeightType>>,
        N::Unit<WeightInfo>: GenericUnit<WeightInfo, Unit<SaveWeightType>=N::Unit<SaveWeightType>>
    {
        let path = path.as_ref();

        let save_network: NeuralNetwork<SaveNetwork<N, O::WeightParam<LayerType>>, O, D> = NeuralNetwork{
            dictionary: self.dictionary.clone(),
            network: SaveNetwork::from_network::<O, O::WeightParam<OptimizerInfoPtr>>(&self.network),
            optimizer: self.optimizer.clone(),
            gradient_clip: self.gradient_clip,
            extra_info: self.extra_info.clone(),
            sizes: self.sizes.clone()
        };

        save_network.save_data(path)
    }

    pub fn load<P: AsRef<Path>>(
        config: NetworkConfigInfo,
        batch_size: usize,
        path: P
    ) -> Result<Self, <SaveFormat as SerializeFormat>::Error>
    where
        for<'de> O: Deserialize<'de>,
        for<'de> D: Deserialize<'de>,
        for<'de> O::WeightParam<LayerType>: Deserialize<'de>,
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
            network: Network::load::<O, O::WeightParam<LayerType>>(this.network, config, batch_size),
            optimizer: this.optimizer,
            gradient_clip: this.gradient_clip,
            extra_info: this.extra_info,
            sizes: this.sizes
        })
    }

    fn with_guesses<R, T: FromGuesses<N, O>>(
        &mut self,
        reader: R,
        mut f: impl FnMut(Box<[u8]>, T, Box<[u8]>)
    )
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

        let mut current_input_index = 0;
        let mut previous_word = None;
        T::from_guesses(&mut self.network, input_outputs, |highest_index, certainty|
        {
            let correct = self.dictionary.word_to_bytes(Some(inputs[current_input_index]), inputs[current_input_index + 1]);
            current_input_index += 1;

            let word = VectorWord::from_raw(highest_index);

            let output = self.dictionary.word_to_bytes(previous_word, word);

            previous_word = Some(word);

            f(correct, certainty, output);
        });
    }

    pub fn correct_guesses<R>(&mut self, reader: R, f: impl FnMut(Box<[u8]>, bool, Box<[u8]>))
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        self.with_guesses(reader, f)
    }

    pub fn top_guesses<R>(&mut self, reader: R, f: impl FnMut(Box<[u8]>, u32, Box<[u8]>))
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        self.with_guesses(reader, f)
    }

    pub fn certainty_guesses<R>(&mut self, reader: R, f: impl FnMut(Box<[u8]>, AccuracyCertaintyInfo, Box<[u8]>))
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        self.with_guesses(reader, f)
    }

    pub fn test_loss<R>(
        &mut self,
        reader: R,
        inputs_in_flight: Option<usize>,
        calculate_accuracy: bool,
        calculate_perplexity: bool
    ) -> f32
    where
        R: Read,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        let inputs = self.vectorized(reader);

        self.test_loss_inner(&inputs, inputs_in_flight, calculate_accuracy, calculate_perplexity)
    }

    fn test_loss_inner(
        &mut self,
        inputs: &[VectorWord],
        inputs_in_flight: Option<usize>,
        calculate_accuracy: bool,
        calculate_perplexity: bool
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
        } else if calculate_perplexity
        {
            self.network.set_predict_mode();

            let perplexity = self.network.perplexity(input_outputs.clone());

            println!("perplexity: {perplexity}");

            perplexity
        } else
        {
            self.network.set_test_mode();

            let total_loss = self.network.feedforward_no_gradient(OneHotEmbeddings::one_hot_count(), inputs_in_flight, input_outputs);
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

    pub fn extra_info(&self) -> &ExtraInfo
    {
        &self.extra_info
    }

    pub fn extra_info_mut(&mut self) -> &mut ExtraInfo
    {
        &mut self.extra_info
    }

    pub fn get_learning_rate(&self) -> f32
    {
        self.optimizer.get_learning_rate()
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.optimizer.set_learning_rate(learning_rate);
    }

    pub fn train<EmbeddingType, R>(
        &mut self,
        info: TrainingInfo,
        reader: R
    ) -> Option<f64>
    where
        EmbeddingType: EmbeddingsTypeable,
        R: Read,
        O::WeightParam<OptimizerInfoPtr>: Clone,
        for<'b> VectorizerType<'b, R, D>: Iterator<Item=VectorWord>,
        N::Unit<WeightInfo>: GenericUnit<WeightInfo, Unit<LayerType>=N::Unit<LayerType>> + fmt::Debug,
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        N::Unit<LayerType>: IntoIterator<Item=LayerType>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        for<'b> &'b mut N::Unit<LayerType>: IntoIterator<Item=&'b mut LayerType>,
        for<'b> &'b mut N::Unit<WeightInfo>: IntoIterator<Item=&'b mut WeightInfo>,
        for<'b> InputOutputEmbeddingsIter<'b, EmbeddingType, D>: ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>
    {
        self.network.set_train_mode();

        if let Some(learning_rate) = info.learning_rate
        {
            self.optimizer.set_learning_rate(learning_rate);
        }

        if let Some(lambda) = info.lambda
        {
            self.optimizer.set_lambda(lambda);
        }

        self.network.prepare_with_gradient(self.optimizer.clone(), self.gradient_clip, EmbeddingType::one_hot_count());

        let average_inputs_per_batch = info.batch_size * info.steps_num.mid();

        let inputs: Vec<_> = self.vectorized(reader);

        let iterations_per_epoch = (inputs.len() / average_inputs_per_batch).max(1);

        let display_header = !info.less_info;
        let display_inner = !info.less_info;

        if display_header
        {
            println!("input vector size: {}", self.dictionary.words_amount());

            #[cfg(not(test))]
            println!("parameters amount: {}", self.network.parameters_amount());

            println!("batch size: {}", info.batch_size);

            println!("using {} inputs", EmbeddingType::name());

            println!("optimizer: {:?}", &self.optimizer);

            println!("steps amount: {}", info.steps_num);

            println!("iterations per epoch: ~{iterations_per_epoch}");
        }

        if info.iterations == 0
        {
            return None;
        }

        let (last_loss_tx, last_loss_rx) = mpsc::channel();

        let inputs_per_block = (info.steps_num.highest() + EmbeddingType::min_len()) * info.batch_size;

        let max_batch_start = inputs.len().saturating_sub(inputs_per_block);
        let mut batch_starts: Vec<usize> = (0..info.batch_size).map(|_batch_index|
        {
            if max_batch_start == 0
            {
                0
            } else
            {
                fastrand::usize(0..max_batch_start)
            }
        }).collect();

        for input_index in 0..info.iterations
        {
            self.extra_info.iterations = self.extra_info.iterations.saturating_add(1);

            time_debug! {
                let steps_num = info.steps_num.get();

                let values = InputOutput::new(
                    &self.dictionary,
                    &inputs,
                    &batch_starts,
                    steps_num
                );

                let total_iterations = self.extra_info.iterations;
                let last_loss_tx = last_loss_tx.clone();

                let is_last_step = (input_index + 1) == info.iterations;

                if cfg!(feature="printinputsoutputs")
                {
                    assert_eq!(EmbeddingType::one_hot_count(), 1);

                    values.iter().for_each(|(inputs, outputs)|
                    {
                        (0..info.batch_size).for_each(|batch_index|
                        {
                            let f = |x: &OneHotLayer|
                            {
                                let vector_word = VectorWord::from_raw(x.positions[batch_index] as usize);
                                String::from_utf8_lossy(&self.dictionary.word_to_bytes(None, vector_word)).into_owned()
                            };

                            eprintln!("batch #{batch_index}: `{}` expects `{}` next", f(&inputs.clone().into_one_hot()), f(&outputs));
                        });
                    });
                }

                if info.reset_state
                {
                    self.network.clear_hidden();
                }

                let rng = fastrand::Rng::new();
                self.network.gradient_step_once(rng, total_iterations, is_last_step, values.iter(), move |loss|
                {
                    let batch_loss = loss as f64 / steps_num as f64;

                    if display_inner
                    {
                        Self::print_loss(format!("iteration {total_iterations} ({input_index}) training"), batch_loss as f32);
                    }

                    if is_last_step
                    {
                        last_loss_tx.send(batch_loss).unwrap();
                    }
                });

                batch_starts.iter_mut().enumerate().for_each(|(batch_index, batch_start)|
                {
                    *batch_start += steps_num;

                    if *batch_start >= max_batch_start
                    {
                        *batch_start = 0;

                        self.network.clear_hidden_batch(batch_index);
                    }
                });
            }
        }

        Some(last_loss_rx.recv().unwrap())
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
        F: FnOnce(Predictor<D>, &mut Network<N, O::WeightParam<OptimizerInfoPtr>>) -> T
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
    use std::{iter, io::Cursor};

    use super::*;

    use crate::word_vectorizer::{bpe_from_bytes, InputData, BpeLimit};


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

    struct GruUnitFactory;
    impl UnitFactory for GruUnitFactory
    {
        type Unit<T> = Gru<T>;
    }

    struct LstmUnitFactory;
    impl UnitFactory for LstmUnitFactory
    {
        type Unit<T> = Lstm<T>;
    }

    struct StarUnitFactory;
    impl UnitFactory for StarUnitFactory
    {
        type Unit<T> = Star<T>;
    }

    type ThisDictionary = CharDictionary;

    enum TestStrategy
    {
        None,
        KeepState,
        Gelu,
        Lstm,
        Gru,
        Bpe,
        NonBatch,
        LongDistance,
        MultiLayer(usize),
        InputDropout(f32),
        Clip(f32),
        Dropout(f32)
    }

    fn default_dictionary() -> CharDictionary
    {
        CharDictionary::new(InputData::String("abcdefg".to_owned()))
    }

    #[test]
    fn simple_data_overfit_char_sgd_simple()
    {
        simple_data_overfit_with::<Sgd, _>(default_dictionary(), 15000, TestStrategy::None);
    }

    #[test]
    fn simple_data_overfit_char_sgd_keep_state()
    {
        simple_data_overfit_with::<Sgd, _>(default_dictionary(), 15000, TestStrategy::KeepState);
    }

    #[test]
    fn simple_data_overfit_char_sgd_lstm()
    {
        simple_data_overfit_with::<Sgd, _>(default_dictionary(), 18000, TestStrategy::Lstm);
    }

    #[test]
    fn simple_data_overfit_char_sgd_gru()
    {
        simple_data_overfit_with::<Sgd, _>(default_dictionary(), 14000, TestStrategy::Gru);
    }

    #[test]
    fn simple_data_overfit_char_sgd_gelu()
    {
        simple_data_overfit_with::<Sgd, _>(default_dictionary(), 15000, TestStrategy::Gelu);
    }

    #[test]
    fn simple_data_overfit_char_sgd_non_batched()
    {
        simple_data_overfit_with::<Sgd, _>(default_dictionary(), 17000, TestStrategy::NonBatch);
    }

    #[test]
    fn simple_data_overfit_char_powersign()
    {
        simple_data_overfit_with::<PowerSign, _>(default_dictionary(), 6000, TestStrategy::None);
    }

    #[test]
    fn simple_data_overfit_char_adam_normal()
    {
        simple_data_overfit_with::<Adam, _>(default_dictionary(), 1500, TestStrategy::None);
    }

    #[test]
    fn simple_data_overfit_char_adam_multi_layer()
    {
        simple_data_overfit_with::<Adam, _>(default_dictionary(), 3000, TestStrategy::MultiLayer(3));
    }

    #[test]
    fn simple_data_overfit_char_adam_clip()
    {
        simple_data_overfit_with::<Adam, _>(default_dictionary(), 1000, TestStrategy::Clip(0.1));
    }

    #[test]
    fn simple_data_overfit_char_adam_long_distance()
    {
        let (_last_loss, test_loss) = simple_data_overfit_inner::<Adam, _>(
            default_dictionary(),
            1000,
            TestStrategy::LongDistance
        );

        assert!(test_loss < 0.01, "{test_loss} must be close to 0");
    }

    #[test]
    fn simple_data_overfit_char_adam_dropout()
    {
        let (_last_loss, test_loss) = simple_data_overfit_inner::<Adam, _>(
            default_dictionary(),
            1500,
            TestStrategy::Dropout(0.2)
        );

        assert!(test_loss < 0.05, "{test_loss} must be close to 0");
    }

    #[test]
    fn simple_data_overfit_char_adam_input_dropout()
    {
        let (_last_loss, test_loss) = simple_data_overfit_inner::<Adam, _>(
            default_dictionary(),
            2000,
            TestStrategy::InputDropout(0.1)
        );

        assert!(test_loss < 0.1, "{test_loss} must be close to 0");
    }

    #[test]
    fn simple_data_overfit_bpe_single()
    {
        let dictionary = bpe_from_bytes(
            BpeLimit::Static(0),
            0.5,
            true,
            b"12345".iter().copied() // numbers so that the input letters are always bytes
        );

        simple_data_overfit_with::<Sgd, _>(dictionary, 9000, TestStrategy::None);
    }

    #[test]
    fn simple_data_overfit_bpe_pairs()
    {
        let dictionary = bpe_from_bytes(
            BpeLimit::Static(2),
            0.0,
            true,
            b"ab12ef34ab32ab5ef1abef".iter().copied()
        );

        simple_data_overfit_with::<Sgd, _>(dictionary, 5000, TestStrategy::Bpe);
    }

    fn simple_data_overfit_with<OptimizerType: Optimizer, D: NetworkDictionary>(
        dictionary: D,
        iterations: usize,
        strategy: TestStrategy
    )
    where
        for<'de> OptimizerType: Serialize + Deserialize<'de>,
        for<'de> D: Serialize + Clone + Deserialize<'de>,
        for<'de> OptimizerType::WeightParam<LayerType>: Serialize + Clone + Deserialize<'de>,
        OptimizerType::WeightParam<OptimizerInfoPtr>: NewableLayer,
        OptimizerType::WeightParam<OptimizerInfoPtr>: Clone,
        for<'b> VectorizerType<'b, Cursor<String>, D>: Iterator<Item=VectorWord>,
        for<'b> InputOutputEmbeddingsIter<'b, OneHotEmbeddings, D>: ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>
    {
        let (last_loss, test_loss) = simple_data_overfit_inner::<OptimizerType, D>(dictionary, iterations, strategy);

        assert!(last_loss < 0.001, "train loss {last_loss} must be close to 0");
        assert!(test_loss < 0.001, "test loss {test_loss} must be close to 0");
    }

    fn simple_data_overfit_inner<OptimizerType: Optimizer, D: NetworkDictionary>(
        dictionary: D,
        iterations: usize,
        strategy: TestStrategy
    ) -> (f64, f64)
    where
        for<'de> OptimizerType: Serialize + Deserialize<'de>,
        for<'de> D: Serialize + Clone + Deserialize<'de>,
        for<'de> OptimizerType::WeightParam<LayerType>: Serialize + Clone + Deserialize<'de>,
        OptimizerType::WeightParam<OptimizerInfoPtr>: NewableLayer,
        OptimizerType::WeightParam<OptimizerInfoPtr>: Clone,
        for<'b> VectorizerType<'b, Cursor<String>, D>: Iterator<Item=VectorWord>,
        for<'b> InputOutputEmbeddingsIter<'b, OneHotEmbeddings, D>: ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>
    {
        let batch_size = if let TestStrategy::NonBatch = strategy { 1 } else { 2 };

        let is_input_one_hot = ThisDictionary::is_input_one_hot();

        eprintln!("dictionary: {dictionary:#?}");

        let input_size = dictionary.input_amount();
        let output_size = dictionary.words_amount();

        let inputs: String = {
            let mut input_index = 0_u8;

            iter::repeat_with(||
            {
                let index = match strategy
                {
                    TestStrategy::LongDistance =>
                    {
                        input_index = (input_index + 1) % ("abcdefg".len() as u8 * 2);

                        if input_index % 2 == 1 { 0 } else { input_index / 2 }
                    },
                    _ =>
                    {
                        input_index = (input_index + 1) % ("abcdefg".len() as u8);

                        input_index
                    }
                };

                char::from(b'a' + index)
            }).take(1000).collect()
        };

        eprintln!("training input: {inputs}");

        let layers = if let TestStrategy::MultiLayer(layers) = strategy { layers } else { 1 };
        let hidden = 8;

        let layer_sizes = LayerSizes{
            hidden,
            layers,
            initial_input: input_size,
            input: input_size,
            output: output_size,
            final_output: output_size,
            batch_size
        };

        let input_dropout_probability = if let TestStrategy::InputDropout(dropout) = strategy { dropout } else { 0.0 };
        let dropout_probability = if let TestStrategy::Dropout(dropout) = strategy { dropout } else { 0.0 };

        type NetworkType<ThisUnit, OptimizerType> = Network<ThisUnit, <OptimizerType as Optimizer>::WeightParam<OptimizerInfoPtr>>;

        let start_config = NetworkConfigInfo{
            is_input_one_hot,
            print_batch_losses: false,
            print_optional_info: false
        };

        let clip = if let TestStrategy::Clip(clip) = strategy { Some(clip) } else { None };

        let steps_num = match strategy
        {
            TestStrategy::LongDistance
            | TestStrategy::Dropout(_)
            | TestStrategy::InputDropout(_) => 50,
            _ => 4
        };

        let info = TrainingInfo{
            batch_size,
            calculate_accuracy: false,
            iterations,
            learning_rate: Some(0.01),
            lambda: None,
            reset_state: !(matches!(strategy, TestStrategy::KeepState)),
            less_info: false,
            steps_num: StepsNum::Steps(steps_num)
        };

        let test_input = String::from_utf8_lossy((if let TestStrategy::Bpe = strategy { &inputs[1..] } else { &inputs }).as_bytes()).into_owned();

        let config = NetworkConfigInfo{
            print_optional_info: false,
            print_batch_losses: true,
            is_input_one_hot
        };

        let test_batches = 64;

        let activation_function = if let TestStrategy::Gelu = strategy { ActivationFunction::GaussianRelu } else { ActivationFunction::LeakyRelu };

        macro_rules! with_unit
        {
            ($unit:ident) =>
            {
                {
                    let mut network: NeuralNetwork<NetworkType<$unit, OptimizerType>, OptimizerType, _> = NeuralNetwork::new(
                        dictionary,
                        layer_sizes,
                        start_config,
                        NetworkArchitecture{
                            input_dropout_probability,
                            dropout_probability,
                            gradient_clip: clip,
                            activation_function
                        }
                    );

                    let last_train_loss = network.train::<OneHotEmbeddings, _>(info, Cursor::new(inputs)).unwrap();

                    network.reset(config.clone(), test_batches);

                    let test_loss = network.test_loss(Cursor::new(test_input.clone()), None, false, false);

                    network.reset(config, test_batches);

                    let accuracy = network.test_loss(Cursor::new(test_input), None, true, false);

                    assert!(accuracy > 0.9);

                    (last_train_loss, test_loss as f64)
                }
            }
        }

        match strategy
        {
            TestStrategy::Lstm => with_unit!(LstmUnitFactory),
            TestStrategy::Gru => with_unit!(GruUnitFactory),
            _ => with_unit!(StarUnitFactory)
        }
    }
}
