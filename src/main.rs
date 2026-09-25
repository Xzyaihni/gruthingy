// >putting else on a different line is suspicious
// why?
#![allow(clippy::suspicious_else_formatting)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::match_like_matches_macro)]

use std::{
    env,
    iter,
    process,
    error::Error,
    num::ParseFloatError,
    thread::{self, JoinHandle},
    path::{PathBuf, Path},
    io::{self, ErrorKind, Read, Write, BufReader, BufWriter, Cursor},
    fs::{self, File},
    collections::HashSet,
    ops::{Index, IndexMut}
};

use serde::{Serialize, Deserialize, de::DeserializeOwned};

#[allow(unused_imports)]
use neural_network::{
    USE_EMBEDDING_LAYER,
    NetworkConfigInfo,
    PhiOtherSelectorRecordingIndex,
    NetworkStateSelectable,
    NetworkStateGettable,
    TrainingInfo,
    NeuralNetwork,
    UnitState,
    WeightsNamed,
    WeightsSize,
    LayerType,
    SaveWeightType,
    NetworkUnit,
    NetworkUnitNewable,
    OptimizerUnit,
    GenericUnit,
    Optimizer,
    DiffTensor,
    DiffTensorPtr,
    WeightInfo,
    WeightInfoPtr,
    UnitFactory,
    NUnit,
    EmbeddingUnit,
    NewableLayer,
    NEmbeddings,
    NOptimizer,
    NDictionary,
    Network,
    SaveNetwork,
    LayerSizes
};

use config::{Config, ProgramMode};

use word_vectorizer::{
    bpe_from_bytes,
    CharsAdapter,
    ReaderAdapter,
    NetworkDictionary,
    WordDictionary,
    ScaffoldedIndex,
    BpeLimit,
    VectorWord,
    PathType,
    InputDataType,
    InputData
};

mod config;
mod word_vectorizer;

pub mod neural_network;


pub fn complain<S>(message: S) -> !
where
    S: Into<String>
{
    eprintln!("{}", message.into());

    process::exit(1)
}

struct SizesInfo
{
    pub hidden: usize,
    pub layers: usize,
    pub batch_size: usize
}

impl From<&Config> for SizesInfo
{
    fn from(value: &Config) -> Self
    {
        Self{hidden: value.hidden_size, layers: value.layers_amount, batch_size: value.batch_size}
    }
}

struct NUnitFactory;

impl UnitFactory for NUnitFactory
{
    type Unit<T> = NUnit<T>;
}

fn load_network(
    config: &Config,
    sizes: Option<SizesInfo>,
    auto_create: bool
) -> NeuralNetwork<Network<NUnitFactory, <NOptimizer as Optimizer>::WeightParam>, NOptimizer, NDictionary>
{
    load_network_with(config.network_path.as_ref(), Some(config), sizes, true, auto_create)
}

pub fn load_embeddings<O>(
    path: Option<&Path>,
    mut config: Option<&mut Config>,
    auto_create: bool
) -> NeuralNetwork<Network<EmbeddingsUnitFactory, O::WeightParam>, O, WordDictionary>
where
    O: Optimizer + DeserializeOwned,
    <EmbeddingsUnitFactory as UnitFactory>::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    for<'a> O::WeightParam: NewableLayer + Serialize + Deserialize<'a>
{
    let sizes = config.as_mut().map(|config|
    {
        SizesInfo{hidden: config.embeddings_size, layers: 1, batch_size: config.batch_size}
    });

    let config = config.map(|x| &*x);

    let path = path.unwrap_or_else(||
    {
        config.expect("config must be provided if the path is none")
            .network_path
            .as_ref()
    });

    load_network_with(path, config, sizes, false, auto_create)
}

fn load_network_with<N, O, D>(
    path: &Path,
    config: Option<&Config>,
    sizes: Option<SizesInfo>,
    is_multistep: bool,
    auto_create: bool
) -> NeuralNetwork<Network<N, O::WeightParam>, O, D>
where
    for<'de> O: Optimizer + Deserialize<'de>,
    for<'de> N: UnitFactory,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    N::Unit<<NOptimizer as Optimizer>::WeightParam>: OptimizerUnit<<NOptimizer as Optimizer>::WeightParam>,
    UnitState<N, DiffTensorPtr>: Clone + NetworkStateSelectable<UnitState<N, PhiOtherSelectorRecordingIndex>>,
    UnitState<N, PhiOtherSelectorRecordingIndex>: NetworkStateGettable<UnitState<N, DiffTensorPtr>>,
    for<'de> N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam> + Deserialize<'de>,
    for<'de> O::WeightParam: NewableLayer + Serialize + Deserialize<'de>,
    for<'de> N::Unit<SaveWeightType>: GenericUnit<SaveWeightType, Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>> + Deserialize<'de>,
    for<'b> &'b N::Unit<WeightInfo>: IntoIterator<Item=&'b WeightInfo>,
    for<'b> &'b N::Unit<DiffTensor>: IntoIterator<Item=&'b DiffTensor>,
    for<'b> &'b mut N::Unit<DiffTensor>: IntoIterator<Item=&'b mut DiffTensor>,
    N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
    for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
    for<'de> D: NetworkDictionary + Deserialize<'de>
{
    let network_config = NetworkConfigInfo{
        is_multistep,
        is_input_one_hot: D::is_input_one_hot(),
        print_optional_info: config.as_ref().map(|x| x.optional_info).unwrap_or(false)
    };

    if path.exists()
    {
        let batch_size = sizes.or_else(|| config.map(|config| SizesInfo::from(config))).map(|x| x.batch_size).unwrap_or(1);

        NeuralNetwork::load(network_config, batch_size, path).unwrap_or_else(|err|
        {
            complain(format!("could not load network at {} ({err})", path.display()))
        })
    } else if auto_create
    {
        let config = config.expect("config must be provided for autocreate");

        let data = match D::input_data()
        {
            InputDataType::String => {
                let dictionary_path = &config.dictionary_path;

                InputData::String(fs::read_to_string(dictionary_path).unwrap_or_else(|err|
                {
                    complain(format!(
                        "could not load dictionary at {} ({err})",
                        dictionary_path.display()
                    ))
                }))
            },
            InputDataType::None => InputData::None,
            InputDataType::Path(PathType::Dictionary) =>
            {
                if !config.dictionary_path.exists()
                {
                    eprintln!("bpe dictionary doesnt exist, creating one...");

                    create_bpe(config);
                }

                InputData::Path(config.dictionary_path.clone())
            },
            InputDataType::Path(PathType::Embeddings) =>
            {
                InputData::Path(config.embeddings_path.clone())
            }
        };

        let dictionary = D::new(data);

        let sizes = sizes.unwrap_or_else(|| SizesInfo::from(config));

        let initial_input = dictionary.input_amount();
        let final_output = dictionary.words_amount();

        let sizes = LayerSizes{
            initial_input,
            input: if USE_EMBEDDING_LAYER { config.embeddings_size } else { initial_input },
            output: if USE_EMBEDDING_LAYER { config.output_embeddings_size } else { final_output },
            final_output,
            hidden: sizes.hidden,
            layers: sizes.layers,
            batch_size: sizes.batch_size
        };

        NeuralNetwork::new(
            dictionary,
            sizes,
            network_config,
            config.input_dropout_probability,
            config.dropout_probability,
            config.gradient_clip
        )
    } else
    {
        complain(format!("cant load the network at: {}", path.display()))
    }
}

fn train_until_best(config: Config)
{
    let mut test_config = config.clone();
    test_config.batch_size = 5;
    test_config.calculate_accuracy = false;

    let losses_path = config.network_path.with_extension("losses");
    let best_path = config.network_path.with_extension("best");

    let mut network = load_network(&config, None, true);

    let training_info = TrainingInfo::from(&config);

    let mut test_thread: Option<JoinHandle<Result<bool, (&'static str, Box<dyn Error + Send>)>>> = None;

    loop
    {
        let text_file = config.get_input_file();
        let test_file = config.get_test_file().unwrap_or_else(|| complain(format!("--test-path must be provided")));

        network.train::<NEmbeddings, _>(training_info.clone(), text_file);

        if let Some(test_thread) = test_thread.take()
        {
            match test_thread.join().unwrap()
            {
                Err((place, err)) => complain(format!("{place}: {err}")),
                Ok(true) => return eprintln!("achieved best possible loss"),
                Ok(false) => ()
            }
        }

        try_save_network(&network, &config.network_path);

        test_thread = {
            let losses_path = losses_path.clone();
            let best_path = best_path.clone();
            let test_config = test_config.clone();

            Some(thread::spawn(move ||
            {
                let mut network = load_network(&test_config, None, false);

                let loss = network.test_loss(test_file, test_config.calculate_accuracy);

                let file = match File::create_new(&losses_path)
                {
                    Err(err) if err.kind() == ErrorKind::AlreadyExists => File::options().read(true).append(true).open(&losses_path),
                    Ok(x) =>
                    {
                        eprintln!("creating losses file at: {}", losses_path.display());

                        Ok(x)
                    },
                    x => x
                };

                fn err_mapper<T, E: Error + Send + 'static>(
                    err: Result<T, E>,
                    name: &'static str
                ) -> Result<T, (&'static str, Box<dyn Error + Send>)>
                {
                    err.map_err(|err| -> (&'static str, Box<dyn Error + Send>)
                    {
                        (name, Box::new(err))
                    })
                }

                let mut file = err_mapper(file, "opening losses file")?;

                {
                    let mut losses = String::new();
                    err_mapper(BufReader::new(&file).read_to_string(&mut losses), "reading losses file")?;

                    let lines_count = losses.lines().count();

                    let (lowest_line_index, lowest_loss) = err_mapper(losses.lines()
                        .map(|x| x.parse::<f32>())
                        .enumerate()
                        .try_fold(None, |acc, (line_index, loss)| -> Result<Option<(usize, f32)>, ParseFloatError>
                        {
                            let loss = loss?;

                            Ok(if let Some((_line_index, lowest_loss)) = acc
                            {
                                if loss < lowest_loss
                                {
                                    Some((line_index, loss))
                                } else
                                {
                                    acc
                                }
                            } else
                            {
                                Some((line_index, loss))
                            })
                        }), "finding lowest loss")?
                        .unwrap_or((0, f32::INFINITY));

                    if loss < lowest_loss
                    {
                        eprintln!("copying network to best at: {}", best_path.display());

                        err_mapper(fs::copy(&test_config.network_path, &best_path), "best copy")?;
                    } else if lines_count.saturating_sub(lowest_line_index + 1) > test_config.validation_attempts
                    {
                        return Ok(true);
                    }
                }

                err_mapper(file.write_all((loss.to_string() + "\n").as_bytes()), "writing loss")?;

                Ok(false)
            }))
        };
    }
}

fn test_loss(config: Config)
{
    let text_file = config.get_input_file();

    let mut network = load_network(&config, None, false);

    network.test_loss(text_file, config.calculate_accuracy);
}

fn train(config: Config)
{
    let mut network = load_network(&config, None, true);

    let mut run_this = |training_info|
    {
        let text_file = config.get_input_file();

        network.train::<NEmbeddings, _>(training_info, text_file);

        try_save_network(&network, &config.network_path);
    };

    let training_info = TrainingInfo::from(&config);

    if config.infinite_loop
    {
        loop
        {
            run_this(training_info.clone());
        }
    } else
    {
        run_this(training_info);
    }
}

fn run(config: Config)
{
    let mut network = load_network(&config, None, false);

    let f = config.output.as_ref().map(|filepath|
    {
        File::create(filepath)
            .unwrap_or_else(|err|
            {
                complain(format!("couldnt create a file at {filepath}: {err}"))
            })
    });

    let text = Cursor::new(config.get_input());

    if config.replace_invalid
    {
        let predicted = network.predict_bytes(text, config.tokens_amount, config.temperature);

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

        network.predict_into(text, config.tokens_amount, config.temperature, &mut f);
    };
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

fn weights_image(config: Config)
{
    let network = load_network(&config, None, false);

    let weights = network.inner_network().weights_info();

    let output_folder = PathBuf::from(config.output.unwrap_or_else(|| "output".to_owned()));

    let layer_name = |index|
    {
        format!("layer{index}")
    };

    for WeightsNamed{
        name,
        layer,
        weights_size: WeightsSize{
            weights,
            previous_size,
            this_size,
            ..
        }
    } in weights.into_iter()
    {
        let layer_folder = output_folder.join(layer_name(layer));
        fs::create_dir_all(&layer_folder).unwrap();

        let mut image = PPMImage::new(this_size, previous_size);

        let weights = weights.as_vec();

        assert_eq!(this_size * previous_size, weights.len(), "{name} size doesnt match");

        for (index, weight) in weights.into_iter().enumerate()
        {
            let color = weight_color(weight);

            let x = index % this_size;
            let y = index / this_size;

            image[(x, y)] = color;
        }

        let name = name.chars().enumerate().flat_map(|(index, c)|
        {
            if index != 0 && c.is_uppercase()
            {
                iter::once('_').chain(c.to_lowercase()).collect::<Vec<_>>()
            } else
            {
                c.to_lowercase().collect()
            }
        }).collect::<String>();

        let filename = format!("{name}.ppm");
        let full_path = layer_folder.join(filename);

        image.save(full_path).unwrap();
    }
}

fn create_word_dictionary(config: Config)
{
    let text_file = BufReader::new(File::open(config.get_input()).unwrap());

    let mut words: HashSet<String> = HashSet::new();

    let mut chars_reader = CharsAdapter::adapter(text_file);

    loop
    {
        let (separator, word) = WordDictionary::read_word(&mut chars_reader);

        if separator.is_none() && word.is_empty()
        {
            break;
        }

        words.insert(word);
    }

    let mut dictionary_file = File::create(&config.dictionary_path).unwrap();
    for (index, word) in words.into_iter().enumerate()
    {
        if index != 0
        {
            dictionary_file.write_all(&[b'\n']).unwrap();
        }

        dictionary_file.write_all(word.as_bytes()).unwrap();
    }

    dictionary_file.flush().unwrap();

    eprintln!("created word dictionary at {}", config.dictionary_path.display());
}

fn create_bpe(config: &Config)
{
    let mut text_file_reader = BufReader::new(File::open(config.get_input()).unwrap());

    fn handle_io(err: io::Error) -> ! { complain(format!("bpe io error: {err}")) }

    let mut text_file: Vec<u8> = Vec::new();
    text_file_reader.read_to_end(&mut text_file).unwrap_or_else(|err| handle_io(err));

    let dictionary = bpe_from_bytes(
        config.bpe_limit.map(BpeLimit::Static).unwrap_or(BpeLimit::Dynamic(0.95, 100)),
        config.bpe_dropout_probability,
        config.optional_info,
        text_file
    );

    if let Some(longest_ngram) = dictionary.pairs.iter().map(|x|
    {
        dictionary.word_to_bytes_scaffolded_single(ScaffoldedIndex(x.output))
    }).max_by_key(|x| x.len())
    {
        println!("the longest is a {}-gram: {}", longest_ngram.len(), String::from_utf8_lossy(&longest_ngram));
    }

    if config.optional_info
    {
        dictionary.print_all_tokens();
    }

    let output_file = File::create(&config.dictionary_path).unwrap_or_else(|err| handle_io(err));

    postcard::to_io(&dictionary, output_file).unwrap_or_else(|err|
    {
        complain(format!("bpe serialization error: {err}"));
    });

    eprintln!("created bpe dictionary at {}", config.dictionary_path.display());
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EmbeddingsUnitFactory;

impl UnitFactory for EmbeddingsUnitFactory
{
    type Unit<T> = EmbeddingUnit<T>;
}

fn try_save_network<N, O, D>(
    network: &NeuralNetwork<Network<N, O::WeightParam>, O, D>,
    path: &PathBuf
)
where
    N: UnitFactory,
    O: Optimizer + Serialize,
    D: NetworkDictionary + Serialize,
    N::Unit<O::WeightParam>: OptimizerUnit<O::WeightParam>,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    UnitState<N, DiffTensorPtr>: Clone + NetworkStateSelectable<UnitState<N, PhiOtherSelectorRecordingIndex>>,
    UnitState<N, PhiOtherSelectorRecordingIndex>: NetworkStateGettable<UnitState<N, DiffTensorPtr>>,
    for<'b> &'b N::Unit<WeightInfo>: IntoIterator<Item=&'b WeightInfo>,
    for<'b> &'b N::Unit<DiffTensor>: IntoIterator<Item=&'b DiffTensor>,
    for<'b> &'b mut N::Unit<DiffTensor>: IntoIterator<Item=&'b mut DiffTensor>,
    O::WeightParam: Serialize + Clone,
    N::Unit<SaveWeightType>: Serialize,
    N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<SaveWeightType>=N::Unit<SaveWeightType>>,
    N::Unit<WeightInfo>: Clone + GenericUnit<WeightInfo, Unit<SaveWeightType>=N::Unit<SaveWeightType>>,
    N::Unit<O::WeightParam>: Serialize + Clone
{
    if let Err(err) = network.save(path)
    {
        complain(format!("couldnt save to {}: {err}", path.display()));
    }
}

fn train_embeddings(mut config: Config)
{
    let mut network = load_embeddings::<NOptimizer>(
        None,
        Some(&mut config),
        true
    );

    let run_this = |
        network: &mut NeuralNetwork<Network<EmbeddingsUnitFactory, <NOptimizer as Optimizer>::WeightParam>, NOptimizer, WordDictionary>,
        training_info
    |
    {
        let text_file = config.get_input_file();

        network.train::<NEmbeddings, _>(training_info, text_file);

        try_save_network(network, &config.network_path);
    };

    let training_info = TrainingInfo{
        steps_num: 1.into(),
        ..TrainingInfo::from(&config)
    };

    if config.infinite_loop
    {
        loop
        {
            run_this(&mut network, training_info.clone());

            try_save_network(&network.clone().without_optimizer(), &config.embeddings_path);
        }
    } else
    {
        run_this(&mut network, training_info);

        try_save_network(&network.clone().without_optimizer(), &config.embeddings_path);
    }
}

fn closest_embeddings(config: Config)
{
    let mut network: NeuralNetwork<SaveNetwork<_, ()>, (), _> = {
        let path = config.embeddings_path.as_ref();

        NeuralNetwork::load_data(path).unwrap_or_else(|err|
        {
            complain(format!("could not load embeddings at {} ({err})", path.display()))
        })
    };

    let input = config.get_input();

    let to_vector_word = |network: &NeuralNetwork<SaveNetwork<_, _>, _, WordDictionary>, s|
    {
        network.dictionary().str_to_word(s)
            .unwrap_or_else(|| complain(format!("\"{input}\" isnt a valid word")))
    };

    let embeddings_of = |network: &mut NeuralNetwork<SaveNetwork<_, _>, _, WordDictionary>, word|
    {
        let input = network.dictionary().words_to_layer([word]);

        network.inner_network().embeddings(&input.into_one_hot())
    };

    let this_word = to_vector_word(&network, input);
    let this_index = this_word.index();

    let this_embeddings = embeddings_of(&mut network, this_word);

    let mut word_similarities = (0..network.dictionary().words_amount())
        .filter(|v| *v != this_index)
        .map(|i|
        {
            let other_word = VectorWord::from_raw(i);

            let other_embeddings = embeddings_of(&mut network, other_word);

            let similarity = this_embeddings.cosine_similarity(other_embeddings.as_ref());

            (other_word, similarity)
        }).collect::<Vec<_>>();

    word_similarities.sort_unstable_by(|this, other| this.1.partial_cmp(&other.1).unwrap());

    let closest_amount = 5.min(word_similarities.len());

    println!("closest {closest_amount} embeddings:");
    for i in 0..closest_amount
    {
        let (vector_word, _similarity) = word_similarities.pop()
            .expect("closest amount must be less or equal to len");

        let word_bytes = network.dictionary().word_to_bytes(None, vector_word);

        let word = String::from_utf8_lossy(&word_bytes);

        println!("{}: {word}", i + 1);
    }
}

fn accuracy_data(config: Config)
{
    if config.certainty && config.top_guesses
    {
        eprintln!("certainty and top-guesses are contradictory, choose only one");
        return;
    }

    let text_file = config.get_input_file();

    let mut network = load_network(&config, None, false);

    fn to_data_with<T: Serialize>(
        config: Config,
        metadata: Option<(Box<[u8]>, T, Box<[u8]>)>,
        mut correct_guesses: Vec<(Box<[u8]>, T, Box<[u8]>)>
    ) -> Result<(), serde_json::Error>
    {
        let path = config.output.clone().unwrap_or_else(|| "output.json".to_owned());

        let file = File::create(path).unwrap();
        let writer = BufWriter::new(file);

        if let Some(metadata) = metadata
        {
            correct_guesses.push(metadata);
        }

        if !config.replace_invalid
        {
            serde_json::to_writer_pretty(writer, &correct_guesses)
        } else
        {
            serde_json::to_writer_pretty(writer, &correct_guesses.into_iter().map(|(word, value, predicted)|
            {
                (String::from_utf8_lossy(&word).into_owned(), value, String::from_utf8_lossy(&predicted).into_owned())
            }).collect::<Vec<_>>())
        }
    }

    let result = if config.certainty
    {
        to_data_with(config, None, network.certainty_guesses(text_file))
    } else if config.top_guesses
    {
        to_data_with(
            config,
            Some((Box::from(b"_WORDS_AMOUNT".clone()), network.dictionary().words_amount() as u32, Box::new([]))),
            network.top_guesses(text_file)
        )
    } else
    {
        to_data_with(config, None, network.correct_guesses(text_file))
    };

    if let Err(err) = result
    {
        eprintln!("error saving accuracy data: {err}");
    }
}

fn main()
{
    #[cfg(debug_assertions)]
    {
        fastrand::seed(7777);
    }

    let config = Config::parse(env::args().skip(1));

    match config.mode
    {
        ProgramMode::TrainUntilBest => train_until_best(config),
        ProgramMode::Train => train(config),
        ProgramMode::Run => run(config),
        ProgramMode::Test => test_loss(config),
        ProgramMode::CreateDictionary => create_word_dictionary(config),
        ProgramMode::CreateBpe => create_bpe(&config),
        ProgramMode::ClosestEmbeddings => closest_embeddings(config),
        ProgramMode::TrainEmbeddings => train_embeddings(config),
        ProgramMode::WeightsImage => weights_image(config),
        ProgramMode::AccuracyData => accuracy_data(config)
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    use word_vectorizer::{BpeDictionary, BpeMapping};


    #[test]
    fn correct_bpe()
    {
        // a = 97
        // b = 98
        // c = 99
        // d = 100
        // o = 111

        let bytes = b"aaoabdaaabac";
        let dictionary = bpe_from_bytes(BpeLimit::Static(2), 0.0, true, bytes.into_iter().copied());

        assert_eq!(dictionary, BpeDictionary{
            pairs: vec![BpeMapping{
                pair: (b'a' as u32, b'a' as u32),
                output: 256,
                frequency: 2,
                is_scaffold: false
            }, BpeMapping{
                pair: (b'a' as u32, b'b' as u32),
                output: 257,
                frequency: 2,
                is_scaffold: false
            }],
            dropout_probability: 0.0,
            cached: None
        });
    }

    #[test]
    fn buncha_bpe_stuff()
    {
        // h = 104
        // s = 115

        let bytes = b"ahsshshshdhshshhshshahahahhahshdhdhshdahsdajkshdjashsshshdajskajshdasjhdasjkjksakjdhasjkdhsaasdhdhsaasdjdhsajkdhsjskahdajshdjkdjkjjkjkhsdhjasjkdaksjjsshdajsjsdhhdhdjskakaksjdhhdhdsjkajdshadhasjdhaskdhasdjhksadjksahdkjashdjksahdajkhsdajsdhasjkkdjhasjdshadkjash";
        bpe_from_bytes(BpeLimit::Static(20), 0.0, true, bytes.into_iter().copied());
    }
}
