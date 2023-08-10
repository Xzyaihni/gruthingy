use std::{
    env,
    process,
    path::Path,
    io::{self, Write},
    fs::File,
    ops::{Index, IndexMut, Div, Mul, Sub}
};

use serde::{Serialize, de::DeserializeOwned};

#[allow(unused_imports)]
use neural_network::{
    TrainingInfo,
    NeuralNetwork,
    MatrixWrapper,
    ArrayWrapper,
    GenericContainer,
    NetworkType,
    HIDDEN_AMOUNT,
    LAYERS_AMOUNT
};

#[allow(unused_imports)]
use word_vectorizer::{NetworkDictionary, CharDictionary, WordDictionary};

mod neural_network;
mod word_vectorizer;

type DictionaryType = CharDictionary;


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
        let mut network: NeuralNetwork<ArrayWrapper, DictionaryType> =
            NeuralNetwork::load(&config.network_path).unwrap();

        network.test_loss(text_file, config.calculate_accuracy);
    } else
    {
        let mut network: NeuralNetwork<MatrixWrapper, DictionaryType> =
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
    D: NetworkDictionary + DeserializeOwned + Serialize + Send + Sync
{
    let text_file = File::open(&text_path)
        .unwrap_or_else(|err|
        {
            let err_msg = format!("give a valid file plz, cant open {text_path} ({err})");
            complain(&err_msg)
        });

    let training_info = TrainingInfo{
        epochs: config.epochs,
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
        let network: NeuralNetwork<ArrayWrapper, DictionaryType> =
            NeuralNetwork::load(&config.network_path).unwrap();

        train_inner(network, text_path, config);
    } else
    {
        let network: NeuralNetwork<MatrixWrapper, DictionaryType> =
            NeuralNetwork::load(&config.network_path).unwrap();

        train_inner(network, text_path, config);
    }
}

struct RunConfig
{
    tokens_amount: usize,
    temperature: f32,
    use_gpu: bool,
    replace_invalid: bool,
    save_path: Option<String>,
    network_path: String
}

impl RunConfig
{
    pub fn parse(mut args: impl Iterator<Item=String>) -> Self
    {
        let mut tokens_amount = 100;
        let mut temperature = 1.0;
        let mut use_gpu = false;
        let mut replace_invalid = true;
        let mut save_path = None;
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
                "-r" | "--raw" =>
                {
                    replace_invalid = false;
                },
                "-g" | "--gpu" =>
                {
                    use_gpu = true;
                },
                "-o" | "--output" =>
                {
                    save_path = Some(args.next().unwrap_or_else(||
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
                x => complain(&format!("cant parse arg: {x}"))
            }
        }

        Self{
            tokens_amount,
            temperature,
            use_gpu,
            replace_invalid,
            save_path,
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
        let mut network: NeuralNetwork<ArrayWrapper, DictionaryType> =
            NeuralNetwork::load(config.network_path).unwrap();
        
        network.predict_bytes(&text, config.tokens_amount, config.temperature)
    } else
    {
        let mut network: NeuralNetwork<MatrixWrapper, DictionaryType> =
            NeuralNetwork::load(config.network_path).unwrap();
        
        network.predict_bytes(&text, config.tokens_amount, config.temperature)
    };

    let f = config.save_path.map(|filepath|
    {
        File::create(&filepath)
            .unwrap_or_else(|err|
            {
                complain(&format!("couldnt create a file at {filepath}: {err}"))
            })
    });

    if config.replace_invalid
    {
        let s = String::from_utf8_lossy(&predicted);

        if let Some(mut f) = f
        {
            f.write_all(s.as_bytes()).unwrap();
        } else
        {
            println!("{s}");
        }
    } else
    {
        let mut f = f.unwrap_or_else(||
        {
            complain("u must provide a file to save to for a file that doesnt replace invalid unicode")
        });

        f.write_all(&predicted).unwrap();
    };
}

fn debug_network(mut args: impl Iterator<Item=String>)
{
    let network_path = args.next()
        .unwrap_or_else(|| complain("give path to network"));
    
    let network: NeuralNetwork<MatrixWrapper, DictionaryType> =
        NeuralNetwork::load(&network_path).unwrap();

    println!("{network:#?}");
}

#[derive(Clone, Copy)]
struct Color
{
    pub r: u8,
    pub g: u8,
    pub b: u8
}

impl Color
{
    pub fn black() -> Self
    {
        Self{r: 0, g: 0, b: 0}
    }

    pub fn gradient_lerp(gradient: &[Self], amount: f32) -> Self
    {
        let colors_amount = gradient.len();

        let amount = amount * (colors_amount - 1) as f32;

        let amount_lower = (amount.floor() as usize).min(colors_amount.saturating_sub(2));

        gradient[amount_lower].lerp(gradient[amount_lower + 1], amount - amount_lower as f32)
    }

    pub fn lerp(self, other: Self, amount: f32) -> Self
    {
        Self{
            r: Self::lerp_single(self.r, other.r, amount),
            g: Self::lerp_single(self.g, other.g, amount),
            b: Self::lerp_single(self.b, other.b, amount)
        }
    }

    fn lerp_single(a: u8, b: u8, lerp: f32) -> u8
    {
        ((a as f32) * (1.0 - lerp) + (b as f32) * lerp) as u8
    }
}

struct PPMImage
{
    data: Vec<Color>,
    width: usize,
    height: usize
}

impl PPMImage
{
    pub fn new(width: usize, height: usize) -> Self
    {
        Self{data: vec![Color::black(); width * height], width, height}
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()>
    {
        let mut f = File::create(path)?;

        let header = format!("P6\n{} {}\n255\n", self.width, self.height);

        f.write_all(header.as_bytes())?;

        let data = self.data.iter().flat_map(|c| [c.r, c.g, c.b]).collect::<Vec<u8>>();
        f.write_all(&data)
    }

    fn index(&self, pos: (usize, usize)) -> usize
    {
        assert!(pos.1 < self.height);
        assert!(pos.0 < self.width);

        pos.0 + pos.1 * self.width
    }
}

impl Index<(usize, usize)> for PPMImage
{
    type Output = Color;

    fn index(&self, index: (usize, usize)) -> &Self::Output
    {
        &self.data[self.index(index)]
    }
}

impl IndexMut<(usize, usize)> for PPMImage
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output
    {
        let index = self.index(index);
        &mut self.data[index]
    }
}

fn weights_image(mut args: impl Iterator<Item=String>)
{
    let network_path = args.next()
        .unwrap_or_else(|| complain("give path to network"));

    let display_type = args.next()
        .unwrap_or_else(||
        {
            let mut options = String::new();

            let mut add_option = |name|
            {
                options += "    ";
                options += name;
                options.push(',');
                options.push('\n');
            };

            add_option("input_update");
            add_option("input_reset");
            add_option("input_activation");
            add_option("hidden_update");
            add_option("hidden_reset");
            add_option("hidden_activation");
            add_option("output");
            add_option("biases");

            complain(&format!("give wut to display\noptions:\n{options}"))
        });
    
    let network: NeuralNetwork<MatrixWrapper, DictionaryType> =
        NeuralNetwork::load(&network_path).unwrap();

    let negative_color = Color{r: 255, g: 0, b: 0};
    let none_color = Color{r: 0, g: 0, b: 0};
    let positive_color = Color{r: 0, g: 0, b: 255};

    let words_amount = network.words_amount();
    
    let (width, weights_per_hidden) = match display_type.as_ref()
    {
        "biases" => (HIDDEN_AMOUNT, 3),
        "input_update" => (words_amount, HIDDEN_AMOUNT),
        "input_reset" => (words_amount, HIDDEN_AMOUNT),
        "input_activation" => (words_amount, HIDDEN_AMOUNT),
        "output" => (HIDDEN_AMOUNT, words_amount),
        "hidden_update" => (HIDDEN_AMOUNT, HIDDEN_AMOUNT),
        "hidden_reset" => (HIDDEN_AMOUNT, HIDDEN_AMOUNT),
        "hidden_activation" => (HIDDEN_AMOUNT, HIDDEN_AMOUNT),
        x => panic!("invalid display type: {}", x)
    };

    let height = weights_per_hidden * LAYERS_AMOUNT;

    let mut image = PPMImage::new(width, height);

    let mut line_num = 0;
    for layer in &network.inner_network().layers
    {
        let mut display_weights = |weights: &[f32]|
        {
            for (i, weight) in weights.iter().enumerate()
            {
                let weight_num = i;
                let weight_value = weight;

                let a = ((weight_value + 1.0) / 2.0).max(0.0).min(1.0);

                let weight_color = Color::gradient_lerp(
                    &[negative_color, none_color, positive_color],
                    a
                );

                image[(weight_num, line_num)] = weight_color;
            }

            line_num += 1;
        };

        if display_type == "biases"
        {
            display_weights(&layer.update_biases.as_vec());
            display_weights(&layer.reset_biases.as_vec());
            display_weights(&layer.activation_biases.as_vec());
        }

        let mut display_weights_m = |weights: Vec<f32>, previous_size: usize|
        {
            let this_size = weights.len() / previous_size;
            let mut this_start = 0;

            for _ in 0..this_size
            {
                let this_end = this_start + previous_size;
                display_weights(&weights[this_start..this_end]);

                this_start += previous_size;
            }
        };

        if display_type == "input_update"
        {
            display_weights_m(layer.input_update_weights.as_vec(), words_amount);
        }

        if display_type == "input_reset"
        {
            display_weights_m(layer.input_reset_weights.as_vec(), words_amount);
        }

        if display_type == "input_activation"
        {
            display_weights_m(layer.input_activation_weights.as_vec(), words_amount);
        }

        if display_type == "hidden_update"
        {
            display_weights_m(layer.hidden_update_weights.as_vec(), HIDDEN_AMOUNT);
        }

        if display_type == "hidden_reset"
        {
            display_weights_m(layer.hidden_reset_weights.as_vec(), HIDDEN_AMOUNT);
        }

        if display_type == "hidden_activation"
        {
            display_weights_m(layer.hidden_activation_weights.as_vec(), HIDDEN_AMOUNT);
        }
        
        if display_type == "output"
        {
            display_weights_m(layer.output_weights.as_vec(), HIDDEN_AMOUNT);
        }
    }

    image.save(format!("{display_type}.ppm")).unwrap();
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
        "weightsimage" => weights_image(args),
        x => complain(&format!("plz give a valid mode!! {x} isnt a valid mode!!!!"))
    }
}
