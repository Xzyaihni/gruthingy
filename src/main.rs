use std::{
    env,
    process,
    marker::PhantomData,
    path::{PathBuf, Path},
    io::{self, Write, BufReader},
    fs::{self, File},
    collections::HashSet,
    ops::{AddAssign, DivAssign, Index, IndexMut}
};

use serde::{Serialize, Deserialize};

#[allow(unused_imports)]
use neural_network::{
    TrainingInfo,
    NeuralNetwork,
    WeightsNamed,
    LayerInnerType,
    NetworkUnit,
    Optimizer,
    LayerType,
    UnitContainer,
    GradientableUnitContainer,
    HIDDEN_AMOUNT,
    LAYERS_AMOUNT
};

use word_vectorizer::{CharsAdapter, ReaderAdapter, NetworkDictionary, WORD_SEPARATORS};

mod neural_network;
mod word_vectorizer;


const DEFAULT_NETWORK_NAME: &str = "network.nn";

fn complain(message: &str) -> !
{
    eprintln!("{message}");

    process::exit(1)
}

struct Config
{
    epochs: usize,
    batch_size: usize,
    steps_num: usize,
    learning_rate: Option<f32>,
    calculate_loss: bool,
    calculate_accuracy: bool,
    ignore_loss: bool,
    testing_data: Option<String>,
    network_path: Option<String>
}

impl Config
{
    pub fn parse(mut args: impl Iterator<Item=String>) -> Self
    {
        let mut epochs = 1;
        let mut batch_size = 32;
        let mut steps_num = 64;
        let mut learning_rate = None;
        let mut calculate_loss = true;
        let mut calculate_accuracy = false;
        let mut ignore_loss = false;
        let mut testing_data = None;
        let mut network_path = None;

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
                    learning_rate = Some(args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        }).parse()
                        .unwrap_or_else(|err|
                        {
                            complain(&format!("cant parse the learning rate: {err:?}"))
                        }));
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
                    network_path = Some(args.next().unwrap_or_else(||
                        {
                            complain(&format!("expected value after {arg}"))
                        }));
                },
                "-a" | "--accuracy" =>
                {
                    calculate_loss = false;
                    calculate_accuracy = true;
                },
                "--and-accuracy" =>
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
            calculate_loss,
            calculate_accuracy,
            ignore_loss,
            testing_data,
            network_path
        }
    }
}

struct NetworkTypeBuilder<C, N, O, D>
{
    c: PhantomData<C>,
    n: PhantomData<N>,
    o: PhantomData<O>,
    d: PhantomData<D>
}

impl NetworkTypeBuilder<(), (), (), ()>
{
    pub fn new() -> Self
    {
        Self::recreate()
    }
}

impl<C, N, O, D> NetworkTypeBuilder<C, N, O, D>
{
    fn recreate() -> Self
    {
        Self{c: PhantomData, n: PhantomData, o: PhantomData, d: PhantomData}
    }

    pub fn with_container<NewC>(self) -> NetworkTypeBuilder<NewC, N, O, D>
    {
        NetworkTypeBuilder::recreate()
    }

    pub fn with_network_unit<NewN>(self) -> NetworkTypeBuilder<C, NewN, O, D>
    {
        NetworkTypeBuilder::recreate()
    }

    pub fn with_optimizer<NewO>(self) -> NetworkTypeBuilder<C, N, NewO, D>
    {
        NetworkTypeBuilder::recreate()
    }

    pub fn with_dictionary<NewD>(self) -> NetworkTypeBuilder<C, N, O, NewD>
    {
        NetworkTypeBuilder::recreate()
    }

    /*pub fn build_load<P>(self, path: P) -> NeuralNetwork<C, N, O, D>
    where
        P: AsRef<Path>,
        C: GradientableUnitContainer,
        N: NetworkUnit,
        for<'a> &'a mut N: UnitContainer<Item=&'a mut LayerType>,
        O: Optimizer<WeightParam=C::Item>,
        D: NetworkDictionary,
        N::UnitContainer<LayerInnerType>:
            DivAssign<f32>
            + AddAssign<N::UnitContainer<LayerInnerType>>,
        for<'a> &'a N: UnitContainer<Item=&'a LayerType>,
        for<'a> &'a mut N: UnitContainer<Item=&'a mut LayerType>
    {
        NeuralNetwork::load(path).unwrap()
    }*/
}

// holy trait bounds
fn load_network<P, C, N, O, D>(path: P, config: Config) -> NeuralNetwork<C, N, O, D>
where
    P: AsRef<Path>,
    C: GradientableUnitContainer,
    N: NetworkUnit,
    for<'a> &'a mut N: UnitContainer<Item=&'a mut LayerType>,
    O: Optimizer<WeightParam=C::Item>,
    D: NetworkDictionary,
    N::UnitContainer<LayerInnerType>:
        DivAssign<f32>
        + AddAssign<N::UnitContainer<LayerInnerType>>,
    for<'a> &'a N: UnitContainer<Item=&'a LayerType>,
    for<'a> &'a mut N: UnitContainer<Item=&'a mut LayerType>
{
    let builder = NetworkTypeBuilder::new();

    todo!();
    // builder.build_load(path)
}

fn test_loss(mut args: impl Iterator<Item=String>)
{
    todo!();
    /*let text_path = args.next()
        .unwrap_or_else(|| complain("give path to a file with text testing data"));
    
    let text_file = File::open(&text_path)
        .unwrap_or_else(|err|
        {
            let err_msg = format!("give a valid file plz, cant open {text_path} ({err})");
            complain(&err_msg)
        });
    
    let config = Config::parse(args);

    let network_path = config.network_path.unwrap_or_else(|| DEFAULT_NETWORK_NAME.to_owned());

    let mut network: NeuralNetwork =
        NeuralNetwork::load(&network_path).unwrap();

    network.test_loss(text_file, config.calculate_loss, config.calculate_accuracy);*/
}

fn train_inner<C, N: NetworkUnit, O: Optimizer, D: NetworkDictionary>(
    mut network: NeuralNetwork<C, N, O, D>,
    text_path: String,
    config: Config
)
{
    todo!();
    /*let text_file = File::open(&text_path)
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
        calculate_loss: config.calculate_loss,
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

    let network_path = config.network_path.unwrap_or_else(|| DEFAULT_NETWORK_NAME.to_owned());

    network.save(network_path);*/
}

fn train(mut args: impl Iterator<Item=String>)
{
    todo!();
    /*let text_path = args.next()
        .unwrap_or_else(|| complain("give path to a file with text training data"));
    
    let config = Config::parse(args);
 
    let network_path = config.network_path.clone()
        .unwrap_or_else(|| DEFAULT_NETWORK_NAME.to_owned());

    let network: NeuralNetwork = if PathBuf::from(&network_path).exists()
    {
        NeuralNetwork::load(&network_path).unwrap_or_else(|err|
        {
            panic!("couldnt load network at {network_path} (error: {})", err);
        })
    } else
    {
        let dictionary = if USES_DICTIONARY
        {
            DictionaryType::build(DICTIONARY_TEXT)
        } else
        {
            DictionaryType::new()
        };

        NeuralNetwork::new(dictionary)
    };

    train_inner(network, text_path, config);*/
}

struct RunConfig
{
    tokens_amount: usize,
    temperature: f32,
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
            replace_invalid,
            save_path,
            network_path
        }
    }
}

fn run(mut args: impl Iterator<Item=String>)
{
    todo!();
    /*let text = args.next()
        .unwrap_or_else(|| complain("pls give the text to predict"));

    let config = RunConfig::parse(args);

    let mut network: NeuralNetwork =
        NeuralNetwork::load(config.network_path).unwrap();

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
        let predicted =
            network.predict_bytes(&text, config.tokens_amount, config.temperature);

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

        network.predict_into(&text, config.tokens_amount, config.temperature, &mut f);
    };*/
}

fn debug_network(mut args: impl Iterator<Item=String>)
{
    todo!();
    /*let network_path = args.next()
        .unwrap_or_else(|| complain("give path to network"));
    
    let network: NeuralNetwork =
        NeuralNetwork::load(network_path).unwrap();

    println!("{network:#?}");*/
}

#[derive(Clone, Copy)]
struct Color
{
    pub r: u8,
    pub g: u8,
    pub b: u8
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

fn weight_color(value: f32) -> Color
{
    let negative_color = Color{r: 255, g: 0, b: 0};
    let none_color = Color{r: 0, g: 0, b: 0};
    let positive_color = Color{r: 0, g: 0, b: 255};

    let a = ((value + 1.0) / 2.0).max(0.0).min(1.0);

    Color::gradient_lerp(
        &[negative_color, none_color, positive_color],
        a
    )
}

fn weights_image(mut args: impl Iterator<Item=String>)
{
    todo!();
    /*let network_path = args.next()
        .unwrap_or_else(|| complain("give path to network"));

    let output_folder = args.next()
        .unwrap_or_else(|| "output".to_owned());
    
    let network = load_network(&network_path, Config::parse(args));

    let weights = network.inner_network().weights_info();

    let output_folder = PathBuf::from(output_folder);

    for (layer_index, layer) in weights.into_iter().enumerate()
    {
        let layer_name = format!("layer{layer_index}");
        let layer_folder = output_folder.join(layer_name);
        fs::create_dir_all(&layer_folder).unwrap();

        layer.for_each_weight(|WeightsNamed{name, weights_size}|
        {
            let mut image = PPMImage::new(weights_size.previous_size, weights_size.current_size);

            for (index, weight) in weights_size.weights.as_vec().into_iter().enumerate()
            {
                let color = weight_color(weight);

                let x = index % weights_size.previous_size;
                let y = index / weights_size.previous_size;

                image[(x, y)] = color;
            }

            let name = name.chars().filter(|c|
            {
                c.is_ascii_alphanumeric()
            }).collect::<String>();

            let filename = format!("{name}.ppm");
            let full_path = layer_folder.join(filename);

            image.save(full_path).unwrap();
        });
    }*/
}

fn create_word_dictionary(mut args: impl Iterator<Item=String>)
{
    let text_path = args.next().unwrap_or_else(||
    {
        complain("give path to text file")
    });

    let text_file = BufReader::new(File::open(text_path).unwrap());

    let config = Config::parse(args);
 
    let dictionary_path = config.network_path.clone()
        .unwrap_or_else(|| "dictionary.txt".to_owned());

    let mut words: HashSet<String> = HashSet::new();

    let mut chars_reader = CharsAdapter::adapter(text_file);

    loop
    {
        let mut current_word = String::new();

        while let Some(c) = chars_reader.next()
        {
            if WORD_SEPARATORS.contains(&c)
            {
                if current_word.is_empty()
                {
                    continue;
                }

                break;
            }

            current_word.push(c);
        }

        if current_word.is_empty()
        {
            break;
        }

        words.insert(current_word);
    }

    let mut dictionary_file = File::create(dictionary_path).unwrap();
    for (index, word) in words.into_iter().enumerate()
    {
        if index != 0
        {
            dictionary_file.write(&[b'\n']).unwrap();
        }

        dictionary_file.write(word.as_bytes()).unwrap();
    }

    dictionary_file.flush().unwrap();
}

fn main()
{
    let mut args = env::args().skip(1);

    let mode = args.next()
        .unwrap_or_else(|| complain("pls give a mode"))
        .trim().to_lowercase();

    if LayerInnerType::is_arrayfire()
    {
        panic!("what");
        /* arrayfire::set_device(0);

        #[cfg(not(test))]
        {
            arrayfire::info();

            let device_info = arrayfire::device_info();
            eprintln!(
                "name: {}, platform: {}, toolkit: {}, compute: {}",
                device_info.0, device_info.1, device_info.2, device_info.3
            );
        } */
    }

    match mode.as_str()
    {
        "train" => train(args),
        "run" => run(args),
        "test" => test_loss(args),
        "dbg" => debug_network(args),
        "createdictionary" => create_word_dictionary(args),
        "weightsimage" => weights_image(args),
        x => complain(&format!("plz give a valid mode!! {x} isnt a valid mode!!!!"))
    }
}
