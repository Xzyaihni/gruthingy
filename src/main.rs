use std::{
    env,
    process,
    fs::File,
    ops::{Div, Mul, Sub}
};

use serde::{Serialize, de::DeserializeOwned};

#[allow(unused_imports)]
use neural_network::{
    TrainingInfo,
    NeuralNetwork,
    MatrixWrapper,
    ArrayWrapper,
    GenericContainer,
    NetworkType
};

#[allow(unused_imports)]
use word_vectorizer::{NetworkDictionary, CharDictionary, WordDictionary};

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
    batch_start: usize,
    batch_size: usize,
    steps_num: usize,
    learning_rate: f32,
    calculate_accuracy: bool,
    ignore_loss: bool,
    use_gpu: bool,
    testing_data: Option<String>,
    network_path: String
}

impl TrainConfig
{
    pub fn parse(mut args: impl Iterator<Item=String>) -> Self
    {
        let mut epochs = 1;
        let mut batch_start = 0;
        let mut batch_size = 2_usize.pow(6);
        let mut steps_num = 64;
        let mut learning_rate = 0.001;
        let mut calculate_accuracy = false;
        let mut ignore_loss = false;
        let mut use_gpu = false;
        let mut testing_data = None;
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
                "-s" | "--steps" =>
                {
                    steps_num = args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        }).parse()
                        .unwrap_or_else(|err|
                        {
                            complain(&format!("cant parse the steps amount: {err:?}"))
                        });
                },
                "--start" =>
                {
                    batch_start = args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        }).parse()
                        .unwrap_or_else(|err|
                        {
                            complain(&format!("cant parse the start offset: {err:?}"))
                        });
                },
                "-l" | "--learning-rate" =>
                {
                    learning_rate = args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        }).parse()
                        .unwrap_or_else(|err|
                        {
                            complain(&format!("cant parse the learning rate: {err:?}"))
                        });
                },
                "-t" | "--testing" =>
                {
                    testing_data = Some(args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        }));
                },
                "-p" | "--path" =>
                {
                    network_path = args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        });
                },
                "-g" | "--gpu" =>
                {
                    use_gpu = true;
                },
                "-a" | "--accuracy" =>
                {
                    calculate_accuracy = true;
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
            batch_start,
            batch_size,
            steps_num,
            learning_rate,
            calculate_accuracy,
            ignore_loss,
            use_gpu,
            testing_data,
            network_path
        }
    }
}

fn test_loss(mut args: impl Iterator<Item=String>)
{
    let text_path = args.next()
        .unwrap_or_else(|| complain("give path to a file with text testing data"));
    
    let text_file = File::open(&text_path)
        .unwrap_or_else(|err|
        {
            let err_msg = format!("give a valid file plz, cant open {text_path} ({err})");
            complain(&err_msg)
        });
    
    let config = TrainConfig::parse(args);

    if config.use_gpu
    {
        let mut network: NeuralNetwork<ArrayWrapper, CharDictionary> =
            NeuralNetwork::load(&config.network_path).unwrap();

        network.test_loss(text_file, config.calculate_accuracy);
    } else
    {
        let mut network: NeuralNetwork<MatrixWrapper, CharDictionary> =
            NeuralNetwork::load(&config.network_path).unwrap();

        network.test_loss(text_file, config.calculate_accuracy);
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
    
    let config = TrainConfig::parse(args);

    // let dictionary = WordDictionary::build(dictionary_file);
    let dictionary = CharDictionary::new();

    if config.use_gpu
    {
        let network = NeuralNetwork::<ArrayWrapper, _>::new(dictionary);

        train_inner(network, text_path, config);
    } else
    {
        let network = NeuralNetwork::<MatrixWrapper, _>::new(dictionary);

        train_inner(network, text_path, config);
    }
}

fn train_inner<T, D>(
    mut network: NeuralNetwork<T, D>,
    text_path: String,
    config: TrainConfig
)
where
    T: NetworkType,
    for<'a> &'a T: Mul<f32, Output=T> + Mul<&'a T, Output=T> + Mul<T, Output=T>,
    for<'a> &'a T: Div<f32, Output=T>,
    for<'a> &'a T: Sub<Output=T>,
    D: NetworkDictionary + DeserializeOwned + Serialize
{
    let text_file = File::open(&text_path)
        .unwrap_or_else(|err|
        {
            let err_msg = format!("give a valid file plz, cant open {text_path} ({err})");
            complain(&err_msg)
        });

    let training_info = TrainingInfo{
        epochs: config.epochs,
        batch_start: config.batch_start,
        batch_size: config.batch_size,
        steps_num: config.steps_num,
        learning_rate: config.learning_rate,
        calculate_accuracy: config.calculate_accuracy,
        ignore_loss: config.ignore_loss
    };

    let test_file = config.testing_data.map(|test_path|
    {
        File::open(&test_path)
            .unwrap_or_else(|err|
            {
                let err_msg = format!("give a valid file plz, cant open {test_path} ({err})");
                complain(&err_msg)
            })
    });

    network.train(training_info, test_file, text_file);

    network.save(config.network_path);
}

fn train(mut args: impl Iterator<Item=String>)
{
    let text_path = args.next()
        .unwrap_or_else(|| complain("give path to a file with text training data"));
    
    let config = TrainConfig::parse(args);
    
    if config.use_gpu
    {
        let network: NeuralNetwork<ArrayWrapper, CharDictionary> =
            NeuralNetwork::load(&config.network_path).unwrap();

        train_inner(network, text_path, config);
    } else
    {
        let network: NeuralNetwork<MatrixWrapper, CharDictionary> =
            NeuralNetwork::load(&config.network_path).unwrap();

        train_inner(network, text_path, config);
    }
}

struct RunConfig
{
    tokens_amount: usize,
    temperature: f32,
    use_gpu: bool,
    network_path: String
}

impl RunConfig
{
    pub fn parse(mut args: impl Iterator<Item=String>) -> Self
    {
        let mut tokens_amount = 100;
        let mut temperature = 1.0;
        let mut use_gpu = false;
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
                "-g" | "--gpu" =>
                {
                    use_gpu = true;
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
            use_gpu,
            network_path
        }
    }
}

fn run(mut args: impl Iterator<Item=String>)
{
    let text = args.next()
        .unwrap_or_else(|| complain("pls give the text to predict"));

    let config = RunConfig::parse(args);

    let predicted = if config.use_gpu
    {
        let mut network: NeuralNetwork<ArrayWrapper, CharDictionary> =
            NeuralNetwork::load(config.network_path).unwrap();
        
        network.predict(&text, config.tokens_amount, config.temperature)
    } else
    {
        let mut network: NeuralNetwork<MatrixWrapper, CharDictionary> =
            NeuralNetwork::load(config.network_path).unwrap();
        
        network.predict(&text, config.tokens_amount, config.temperature)
    };

    println!("{predicted}");
}

fn debug_network(mut args: impl Iterator<Item=String>)
{
    let network_path = args.next()
        .unwrap_or_else(|| complain("give path to network"));
    
    let network: NeuralNetwork<MatrixWrapper, CharDictionary> =
        NeuralNetwork::load(&network_path).unwrap();

    println!("{network:#?}");
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
        "test" => test_loss(args),
        "dbg" => debug_network(args),
        x => complain(&format!("plz give a valid mode!! {x} isnt a valid mode!!!!"))
    }
}
