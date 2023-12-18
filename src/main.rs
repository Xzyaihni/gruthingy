// >putting else on a different line is suspicious
// why?
#![allow(clippy::suspicious_else_formatting)]

use std::{
    env,
    iter,
    process,
    path::{PathBuf, Path},
    io::{self, Write, BufReader, Cursor},
    fs::{self, File},
    collections::HashSet,
    ops::{Index, IndexMut}
};

use serde::{Serialize, Deserialize, de::DeserializeOwned};

#[allow(unused_imports)]
use neural_network::{
    TrainingInfo,
    NeuralNetwork,
    WeightsNamed,
    WeightsSize,
    LayerType,
    NetworkUnit,
    OptimizerUnit,
    GenericUnit,
    Optimizer,
    DiffWrapper,
    UnitFactory,
    NUnit,
    EmbeddingUnit,
    NOptimizer,
    NDictionary,
    LayerSizes
};

use config::{Config, ProgramMode};

use word_vectorizer::{
    CharsAdapter,
    ReaderAdapter,
    NetworkDictionary,
    WordDictionary,
    VectorWord,
    InputDataType,
    InputData,
    WORD_SEPARATORS
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
    pub layers: usize
}

impl From<&Config> for SizesInfo
{
    fn from(value: &Config) -> Self
    {
        Self{hidden: value.hidden_size, layers: value.layers_amount}
    }
}

// this is definitely unneeded in theory, but in practice serde macros r stupid
#[derive(Serialize, Deserialize)]
struct NUnitFactory;

impl UnitFactory for NUnitFactory
{
    type Unit<T> = NUnit<T>;
}

fn load_network(
    config: &Config,
    sizes: Option<SizesInfo>,
    auto_create: bool
) -> NeuralNetwork<NUnitFactory, NOptimizer, NDictionary>
{
    load_network_with(config.network_path.as_ref(), Some(config), sizes, auto_create)
}

pub fn load_embeddings(
    path: Option<&Path>,
    mut config: Option<&mut Config>,
    auto_create: bool
) -> NeuralNetwork<EmbeddingsUnitFactory, NOptimizer, WordDictionary>
{
    // &mut &mut, im not sure wut im doing wrong
    let sizes = config.as_mut().map(|config|
    {
        // the whole point of embeddings is to overfit right?
        config.dropout_probability = 0.0;

        // and uhh i dont think it can NaN out on me with just 1 layer
        config.gradient_clip = None;

        SizesInfo{hidden: config.embeddings_size, layers: 1}
    });

    let config = config.map(|x| &*x);

    let path = path.unwrap_or_else(||
    {
        config.expect("config must be provided if the path is none")
            .network_path
            .as_ref()
    });

    load_network_with(path, config, sizes, auto_create)
}

fn load_network_with<N, D>(
    path: &Path,
    config: Option<&Config>,
    sizes: Option<SizesInfo>,
    auto_create: bool
) -> NeuralNetwork<N, NOptimizer, D>
where
    N: UnitFactory + DeserializeOwned,
    N::Unit<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
    N::Unit<<NOptimizer as Optimizer>::WeightParam>: OptimizerUnit<<NOptimizer as Optimizer>::WeightParam>,
    for<'b> &'b N::Unit<DiffWrapper>: IntoIterator<Item=&'b DiffWrapper>,
    for<'b> &'b mut N::Unit<DiffWrapper>: IntoIterator<Item=&'b mut DiffWrapper>,
    D: NetworkDictionary + DeserializeOwned
{
    if path.exists()
    {
        NeuralNetwork::load(path).unwrap_or_else(|err|
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
                    complain(format!("could not load dictionary at {dictionary_path} ({err})"))
                }))
            },
            InputDataType::None => InputData::None,
            InputDataType::Path =>
            {
                InputData::Path(config.dictionary_path.as_str().into())
            }
        };

        let dictionary = D::new(data);

        let sizes = sizes.unwrap_or_else(|| SizesInfo::from(config));

        let sizes = LayerSizes{
            input: dictionary.input_amount(),
            output: dictionary.words_amount(),
            hidden: sizes.hidden,
            layers: sizes.layers
        };

        NeuralNetwork::new(dictionary, sizes, config.dropout_probability, config.gradient_clip)
    } else
    {
        complain(format!("cant load the network at: {}", path.display()))
    }
}

fn test_loss(config: Config)
{
    let text_file = config.get_input_file();

    let mut network = load_network(&config, None, false);

    network.test_loss(text_file, config.calculate_loss, config.calculate_accuracy);
}

fn train(config: Config)
{
    let mut network = load_network(&config, None, true);

    let text_file = config.get_input_file();

    let training_info = TrainingInfo::from(&config);

    let test_file = config.test_file();

    network.train::<false, _, _>(training_info, test_file, text_file);

    network.save(&config.network_path);
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
        let predicted =
            network.predict_bytes(text, config.tokens_amount, config.temperature);

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
            current_size,
            ..
        }
    } in weights.into_iter()
    {
        let layer_folder = output_folder.join(layer_name(layer));
        fs::create_dir_all(&layer_folder).unwrap();

        let mut image = PPMImage::new(previous_size, current_size);

        for (index, weight) in weights.as_vec().into_iter().enumerate()
        {
            let color = weight_color(weight);

            let x = index % previous_size;
            let y = index / previous_size;

            image[(x, y)] = color;
        }

        let name = name.chars().filter(|c|
        {
            c.is_ascii_alphanumeric()
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
        let mut current_word = String::new();

        for c in chars_reader.by_ref()
        {
            if WORD_SEPARATORS.iter().chain(iter::once(&' ')).any(|v| *v == c)
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

    let mut dictionary_file = File::create(config.dictionary_path).unwrap();
    for (index, word) in words.into_iter().enumerate()
    {
        if index != 0
        {
            dictionary_file.write_all(&[b'\n']).unwrap();
        }

        dictionary_file.write_all(word.as_bytes()).unwrap();
    }

    dictionary_file.flush().unwrap();
}

#[derive(Serialize, Deserialize)]
pub struct EmbeddingsUnitFactory;

impl UnitFactory for EmbeddingsUnitFactory
{
    type Unit<T> = EmbeddingUnit<T>;
}

fn train_embeddings(mut config: Config)
{
    let mut network = load_embeddings(
        None,
        Some(&mut config),
        true
    );

    let text_file = config.get_input_file();

    let test_file = config.test_file();

    let training_info = TrainingInfo{
        steps_num: 1.into(),
        ..TrainingInfo::from(&config)
    };

    network.train::<true, _, _>(training_info, test_file, text_file);

    network.save(&config.network_path);
}

fn closest_embeddings(mut config: Config)
{
    let mut network = load_embeddings(
        None,
        Some(&mut config),
        false
    );

    let input = config.get_input();

    network.inner_network_mut().disable_gradients();

    let to_vector_word = |network: &NeuralNetwork<_, _, WordDictionary>, s|
    {
        network.dictionary().str_to_word(s)
            .unwrap_or_else(|| complain(format!("\"{input}\" isnt a valid word")))
    };

    let embeddings_of = |network: &mut NeuralNetwork<_, _, WordDictionary>, word|
    {
        let input = network.dictionary().words_to_layer([word]);

        network.inner_network_mut().embeddings(input.into_one_hot())
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

            let similarity = this_embeddings.cosine_similarity(other_embeddings);

            (other_word, similarity)
        }).collect::<Vec<_>>();

    word_similarities.sort_unstable_by(|this, other| this.1.partial_cmp(&other.1).unwrap());

    let closest_amount = 5.min(word_similarities.len());

    println!("closest {closest_amount} embeddings:");
    for i in 0..closest_amount
    {
        let (vector_word, _similarity) = word_similarities.pop()
            .expect("closest amount must be less or equal to len");

        let word_bytes = network.dictionary().word_to_bytes(vector_word);

        let word = String::from_utf8_lossy(&word_bytes);

        println!("{}: {word}", i + 1);
    }
}

fn main()
{
    if LayerType::is_arrayfire()
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
    
    let config = Config::parse(env::args().skip(1));

    match config.mode
    {
        ProgramMode::Train => train(config),
        ProgramMode::Run => run(config),
        ProgramMode::Test => test_loss(config),
        ProgramMode::CreateDictionary => create_word_dictionary(config),
        ProgramMode::ClosestEmbeddings => closest_embeddings(config),
        ProgramMode::TrainEmbeddings => train_embeddings(config),
        ProgramMode::WeightsImage => weights_image(config)
    }
}
