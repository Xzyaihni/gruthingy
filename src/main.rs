use std::{
    env,
    process,
    fs::File
};

use neural_network::NeuralNetwork;

#[allow(unused_imports)]
use word_vectorizer::{CharDictionary, WordDictionary};

mod neural_network;
mod word_vectorizer;


const DEFAULT_NETWORK_NAME: &'static str = "network.nn";

fn complain(message: &str) -> !
{
    eprintln!("{message}");

    process::exit(1)
}

struct TrainConfig
{
    epochs: usize,
    batch_size: usize,
    ignore_loss: bool,
    network_path: String
}

impl TrainConfig
{
    pub fn parse(mut args: impl Iterator<Item=String>) -> Self
    {
        let mut epochs = 1;
        let mut batch_size = 2_usize.pow(14);
        let mut ignore_loss = false;
        let mut network_path = DEFAULT_NETWORK_NAME.to_owned();

        while let Some(arg) = args.next()
        {
            match arg.as_str()
            {
                "-e" | "--epochs" =>
                {
                    epochs = args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        }).parse()
                        .unwrap_or_else(|err|
                        {
                            complain(&format!("cant parse the epochs: {err:?}"))
                        });
                },
                "-b" | "--batch" =>
                {
                    batch_size = args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        }).parse()
                        .unwrap_or_else(|err|
                        {
                            complain(&format!("cant parse the batch size: {err:?}"))
                        });
                },
                "-p" | "--path" =>
                {
                    network_path = args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        });
                },
                "-i" | "--ignore-loss" =>
                {
                    ignore_loss = true;
                },
                x => complain(&format!("cant parse arg: {x}"))
            }
        }

        Self{
            epochs,
            batch_size,
            ignore_loss,
            network_path
        }
    }
}

fn train_new(mut args: impl Iterator<Item=String>)
{
    /*let dictionary_path = args.next()
        .unwrap_or_else(|| complain("give path to a file with text for a dictionary"));

    let dictionary_file = File::open(dictionary_path)
        .unwrap_or_else(|err| complain(&format!("give a valid file plz ({err})")));*/

    let text_path = args.next()
        .unwrap_or_else(|| complain("give path to a file with text training data"));

    let text_file = File::open(&text_path)
        .unwrap_or_else(|err|
        {
            let err_msg = format!("give a valid file plz, cant open {text_path} ({err})");
            complain(&err_msg)
        });
    
    let config = TrainConfig::parse(args);

    // let dictionary = WordDictionary::build(dictionary_file);
    let dictionary = CharDictionary::new();

    let mut network = NeuralNetwork::new(dictionary);

    network.train(config.epochs, config.batch_size, config.ignore_loss, text_file);

    network.save(config.network_path);
}

fn train(mut args: impl Iterator<Item=String>)
{
    let text_path = args.next()
        .unwrap_or_else(|| complain("give path to a file with text training data"));

    let text_file = File::open(&text_path)
        .unwrap_or_else(|err|
        {
            let err_msg = format!("give a valid file plz, cant open {text_path} ({err})");
            complain(&err_msg)
        });
    
    let config = TrainConfig::parse(args);
    
    let mut network: NeuralNetwork<CharDictionary> =
        NeuralNetwork::load(&config.network_path).unwrap();

    network.train(config.epochs, config.batch_size, config.ignore_loss, text_file);

    network.save(config.network_path);
}

struct RunConfig
{
    tokens_amount: usize,
    temperature: f64,
    network_path: String
}

impl RunConfig
{
    pub fn parse(mut args: impl Iterator<Item=String>) -> Self
    {
        let mut tokens_amount = 25;
        let mut temperature = 1.0;
        let mut network_path = DEFAULT_NETWORK_NAME.to_owned();

        while let Some(arg) = args.next()
        {
            match arg.as_str()
            {
                "-n" =>
                {
                    tokens_amount = args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        }).parse()
                        .unwrap_or_else(|err|
                        {
                            complain(&format!("cant parse the amount: {err:?}"))
                        });
                },
                "-t" | "--temperature" =>
                {
                    temperature = args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        }).parse()
                        .unwrap_or_else(|err|
                        {
                            complain(&format!("cant parse the temperature: {err:?}"))
                        });
                },
                "-p" | "--path" =>
                {
                    network_path = args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        });
                },
                x => complain(&format!("cant parse arg: {x}"))
            }
        }

        Self{
            tokens_amount,
            temperature,
            network_path
        }
    }
}

fn run(mut args: impl Iterator<Item=String>)
{
    let text = args.next()
        .unwrap_or_else(|| complain("pls give the text to predict"));

    let config = RunConfig::parse(args);

    let mut network: NeuralNetwork<CharDictionary> =
        NeuralNetwork::load(config.network_path).unwrap();
    
    let predicted = network.predict(&text, config.tokens_amount, config.temperature);

    println!("{predicted}");
}

fn main()
{
    let mut args = env::args().skip(1);

    let mode = args.next()
        .unwrap_or_else(|| complain("pls give a mode"))
        .trim().to_lowercase();

    match mode.as_str()
    {
        "train_new" => train_new(args),
        "train" => train(args),
        "run" => run(args),
        x => complain(&format!("plz give a valid mode!! {x} isnt a valid mode!!!!"))
    }
}
