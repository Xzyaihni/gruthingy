#[allow(unused_imports)]
use super::{
    DecayFunction,
    Sgd,
    Adam,
    AdamX,
    PowerSign,
    OneHotEmbeddings,
    BagOfWordsEmbeddings,
    Gru,
    Lstm,
    Star,
    CharDictionary,
    WordDictionary,
    BpeDictionary,
    EmbeddingsDictionary,
    ByteDictionary,
    PostcardFormat,
    JsonFormat,
    MixedFormat
};


// options: Power, Division
pub const DECAY_FUNCTION: DecayFunction = DecayFunction::Power;

// options: OneHotEmbeddings, BagOfWordsEmbeddings
pub type NEmbeddings = OneHotEmbeddings;

// only applies to BagOfWordsEmbeddings
pub const BAG_OF_WORDS_EMBEDDINGS_COUNT: usize = 2;

// options: Star, Lstm, Gru
pub type NUnit<T> = Star<T>;

// options: EmbeddingsDictionary, BpeDictionary, WordDictionary, CharDictionary, ByteDictionary
pub type NDictionary = BpeDictionary;

pub const USE_EMBEDDING_LAYER: bool = true;

// only applies to EmbeddingsDictionary and WordDictionary
pub const LOWERCASE_ONLY: bool = true;

// options: Sgd, Adam, AdamX, PowerSign
pub type NOptimizer = AdamX;

// options: PostcardFormat, JsonFormat, MixedFormat<Save, Load>
pub type SaveFormat = PostcardFormat;
