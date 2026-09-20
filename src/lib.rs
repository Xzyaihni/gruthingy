#![allow(clippy::suspicious_else_formatting)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::match_like_matches_macro)]

use std::{process, io::Cursor, path::Path};

use serde::{Serialize, Deserialize};

use neural_network::{
    NetworkConfigInfo,
    NeuralNetwork,
    NUnit,
    NOptimizer,
    NDictionary,
    UnitFactory,
    EmbeddingUnit,
    Optimizer,
    NewableLayer
};

use word_vectorizer::{NetworkDictionary, WordDictionary};

pub use config::Config;

mod config;
mod word_vectorizer;

pub mod neural_network;


#[derive(Serialize, Deserialize)]
struct NUnitFactory;

impl UnitFactory for NUnitFactory
{
    type Unit<T> = NUnit<T>;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EmbeddingsUnitFactory;

impl UnitFactory for EmbeddingsUnitFactory
{
    type Unit<T> = EmbeddingUnit<T>;
}

pub fn load_embeddings<O>(
    _path: Option<&Path>,
    _config: Option<&mut Config>,
    _auto_create: bool
) -> NeuralNetwork<EmbeddingsUnitFactory, O, WordDictionary>
where
    O: Optimizer,
    for<'a> O::WeightParam: Clone + NewableLayer + Serialize + Deserialize<'a>
{
    unreachable!()
}

pub fn complain<S>(message: S) -> !
where
    S: Into<String>
{
    eprintln!("{}", message.into());

    process::exit(1)
}

pub fn predict(path: impl AsRef<Path>, text: String, amount: usize, temperature: f32) -> String
{
    let path = path.as_ref();

    let network_config = NetworkConfigInfo{
        is_multistep: true,
        is_input_one_hot: NDictionary::is_input_one_hot(),
        print_optional_info: false
    };

    let mut network: NeuralNetwork<NUnitFactory, NOptimizer, NDictionary> =
        NeuralNetwork::load(network_config, path).unwrap_or_else(|err|
        {
            complain(format!("could not load network at {} ({err})", path.display()))
        });

    let text = Cursor::new(text.as_bytes());

    let predicted = network.predict_bytes(text, amount, temperature);

    String::from_utf8_lossy(&predicted).into_owned()
}
