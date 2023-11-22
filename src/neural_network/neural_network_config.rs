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
    ByteDictionary
};


pub const HIDDEN_AMOUNT: usize = 512;
pub const LAYERS_AMOUNT: usize = 3;

pub const DROPCONNECT_PROBABILITY: f32 = 0.5;
pub const DROPOUT_PROBABILITY: f32 = 0.5;

pub const GRADIENT_CLIP: f32 = 1.0;

// options: Power, Division
pub const DECAY_FUNCTION: DecayFunction = DecayFunction::Power;

// options: Sgd, Adam, AdamX, PowerSign (garbage (maybe i did it wrong))
pub type CurrentOptimizer = AdamX;

// options: Tanh, LeakyRelu
pub const LAYER_ACTIVATION: AFType = AFType::LeakyRelu;

// options: Gru, Lstm
pub type CurrentNetworkUnit = Lstm;

// WordDictionary, CharDictionary uses a dictionary and ByteDictionary doesnt
pub const USES_DICTIONARY: bool = true;
pub const DICTIONARY_TEXT: &str = include_str!("../../ascii_dictionary.txt");

pub const INPUT_SIZE: usize = DictionaryType::words_amount();

// options: WordDictionary, ByteDictionary, CharDictionary
pub type DictionaryType = CharDictionary;
