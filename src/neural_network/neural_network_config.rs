#[allow(unused_imports)]
use super::{
    DecayFunction,
    Sgd,
    Adam,
    AdamX,
    PowerSign,
    AFType,
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

// options: Lstm, Gru
pub type NUnit<T> = Lstm<T>;

// options: EmbeddingsDictionary, WordDictionary, CharDictionary, ByteDictionary
pub type NDictionary = CharDictionary;

// options: Sgd, Adam, AdamX, PowerSign
pub type NOptimizer = AdamX;
