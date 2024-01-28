#[allow(unused_imports)]
use super::{
    DecayFunction,
    Sgd,
    Adam,
    AdamX,
    PowerSign,
    AFType,
    EMType,
    Gru,
    Lstm,
    CharDictionary,
    WordDictionary,
    EmbeddingsDictionary,
    ByteDictionary
};


pub const DROPCONNECT_PROBABILITY: f32 = 0.5;

// options: Power, Division
pub const DECAY_FUNCTION: DecayFunction = DecayFunction::Power;

// options: Tanh, LeakyRelu
pub const LAYER_ACTIVATION: AFType = AFType::LeakyRelu;

// options: BagOfWords, SkipGram (its broken cuz im bad)
pub const EMBEDDINGS_TYPE: EMType = EMType::BagOfWords(2);

// options: Lstm, Gru
pub type NUnit<T> = Lstm<T>;

// options: EmbeddingsDictionary, WordDictionary, CharDictionary, ByteDictionary
pub type NDictionary = EmbeddingsDictionary;

// only applies to EmbeddingsDictionary and WordDictionary
pub const LOWERCASE_ONLY: bool = true;

// options: Sgd, Adam, AdamX, PowerSign
pub type NOptimizer = AdamX;
