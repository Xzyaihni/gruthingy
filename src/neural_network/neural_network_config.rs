#[allow(unused_imports)]
use super::{
    DecayFunction,
    Sgd,
    Adam,
    AdamX,
    PowerSign,
    AFType,
    OneHotEmbeddings,
    BagOfWordsEmbeddings,
    Gru,
    Lstm,
    CharDictionary,
    WordDictionary,
    BpeDictionary,
    EmbeddingsDictionary,
    ByteDictionary,
    PostcardFormat,
    JsonFormat,
    MixedFormat
};


pub const DROPCONNECT_PROBABILITY: f32 = 0.5;

// options: Power, Division
pub const DECAY_FUNCTION: DecayFunction = DecayFunction::Power;

// options: Tanh, LeakyRelu
pub const LAYER_ACTIVATION: AFType = AFType::LeakyRelu;

// options: OneHotEmbeddings, BagOfWordsEmbeddings
pub type NEmbeddings = OneHotEmbeddings;

// only applies to BagOfWordsEmbeddings
pub const BAG_OF_WORDS_EMBEDDINGS_COUNT: usize = 2;

// options: Lstm, Gru
pub type NUnit<T> = Lstm<T>;

// options: EmbeddingsDictionary, BpeDictionary, WordDictionary, CharDictionary, ByteDictionary
pub type NDictionary = ByteDictionary;

pub const USE_EMBEDDING_LAYER: bool = false;

// only applies to EmbeddingsDictionary and WordDictionary
pub const LOWERCASE_ONLY: bool = true;

// options: Sgd, Adam, AdamX, PowerSign
pub type NOptimizer = AdamX;

// options: PostcardFormat, JsonFormat, MixedFormat<Save, Load>
pub type SaveFormat = PostcardFormat;
