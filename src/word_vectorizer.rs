use std::{
    str,
    fs::File,
    fmt::{self, Debug},
    hash::Hash,
    borrow::Borrow,
    ops::Deref,
    path::PathBuf,
    collections::{HashMap, HashSet},
    io::{
        self,
        BufReader,
        Bytes,
        Read
    }
};

use unicode_reader::CodePoints;

use serde::{Serialize, Deserialize};

use crate::{complain, EmbeddingsUnitFactory, NeuralNetwork};

use super::neural_network::{
    LayerType,
    OwnedInputType,
    OneHotLayer,
    LOWERCASE_ONLY,
    SaveNetwork,
    PostcardFormat,
    SerializeFormat
};


#[allow(dead_code)]
fn debug_bytes(bytes: &[u8]) -> String
{
    bytes.iter().flat_map(|byte|
    {
        let c = char::from(*byte);
        let is_ascii = c.is_ascii_graphic();

        if is_ascii
        {
            c.to_string()
        } else
        {
            format!("{byte:#x}")
        }.chars().collect::<Vec<_>>()
    }).collect()
}

#[derive(Debug, Clone, Copy, Default, PartialOrd, Ord, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorWord(usize);

impl VectorWord
{
    fn new(index: usize) -> Self
    {
        Self(index)
    }

    pub fn from_raw(index: usize) -> Self
    {
        Self(index)
    }

    pub fn index(&self) -> usize
    {
        self.0
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Bimap<K, V>
where
    K: Hash + Eq,
    V: Hash + Eq
{
    k_map: HashMap<K, V>,
    v_map: HashMap<V, K>
}

impl<K, V> Bimap<K, V>
where
    K: Hash + Eq,
    V: Hash + Eq
{
    pub fn by_key<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized
    {
        self.k_map.get(key)
    }

    pub fn by_value<Q>(&self, value: &Q) -> Option<&K>
    where
        V: Borrow<Q>,
        Q: Hash + Eq + ?Sized
    {
        self.v_map.get(value)
    }

    pub fn len(&self) -> usize
    {
        self.k_map.len()
    }
}

impl<K, V> FromIterator<(K, V)> for Bimap<K, V>
where
    K: Hash + Eq + Clone,
    V: Hash + Eq + Clone
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item=(K, V)>
    {
        let (k_map, v_map): (HashMap<_, _>, HashMap<_, _>) = iter.into_iter().map(|(key, value)|
        {
            ((key.clone(), value.clone()), (value, key))
        }).unzip();

        Self{
            k_map,
            v_map
        }
    }
}

pub enum PathType
{
    Dictionary,
    Embeddings
}

pub enum InputDataType
{
    None,
    String,
    Path(PathType)
}

pub enum InputData
{
    None,
    String(String),
    Path(PathBuf)
}

pub trait NetworkDictionary: Debug
{
    type Adapter<R: Read>: ReaderAdapter<R>;


    fn new(data: InputData) -> Self;

    fn word_to_bytes(&self, previous_word: Option<VectorWord>, word: VectorWord) -> Box<[u8]>;
    fn words_amount(&self) -> usize;

    fn is_input_one_hot() -> bool;
    fn input_data() -> InputDataType;

    fn vectorized<R: Read>(&mut self, reader: R) -> Vec<VectorWord>
    where
        for<'a> WordVectorizer<Self::Adapter<BufReader<R>>, &'a mut Self>: Iterator<Item=VectorWord>
    {
        WordVectorizer::<Self::Adapter<BufReader<R>>, &mut Self>::new(self, reader).collect()
    }

    fn input_amount(&self) -> usize
    {
        self.words_amount()
    }

    fn one_hot_to_input(&self, layer: OneHotLayer) -> OwnedInputType
    {
        layer.into()
    }

    fn words_to_layer(&self, words: impl IntoIterator<Item=VectorWord>) -> OwnedInputType
    {
        self.words_to_onehot(words).into()
    }

    fn words_to_onehot(&self, words: impl IntoIterator<Item=VectorWord>) -> OneHotLayer
    {
        OneHotLayer::new(
            [words.into_iter().map(|word| word.index()).collect::<Box<[_]>>()].into(),
            self.words_amount(),
            1
        )
    }

    fn layer_to_word(&self, layer: LayerType) -> VectorWord
    {
        let index = layer.pick_weighed();

        VectorWord::new(index)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteDictionary;

impl NetworkDictionary for ByteDictionary
{
    type Adapter<R: Read> = DefaultAdapter<R>;


    fn new(_data: InputData) -> Self
    {
        Self{}
    }

    fn is_input_one_hot() -> bool { true }

    fn input_data() -> InputDataType
    {
        InputDataType::None
    }

    fn word_to_bytes(&self, _previous_word: Option<VectorWord>, word: VectorWord) -> Box<[u8]>
    {
        Box::new([word.index() as u8])
    }

    fn words_amount(&self) -> usize
    {
        u8::MAX as usize + 1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharDictionary
{
    dictionary: Bimap<char, VectorWord>
}

impl CharDictionary
{
    fn character_match(&self, c: char) -> VectorWord
    {
        let index = self.dictionary.by_key(&c).copied();

        index.unwrap_or_else(||
        {
            // replacement character VectorWord
            VectorWord::new(self.dictionary.len())
        })
    }
}

impl NetworkDictionary for CharDictionary
{
    type Adapter<R: Read> = CharsAdapter<R>;


    fn new(data: InputData) -> Self
    {
        let s = match data
        {
            InputData::String(value) => value,
            _ => unreachable!()
        };

        let unique_chars: HashSet<_> = s.chars().collect();

        let dictionary = unique_chars.into_iter().enumerate().map(|(index, c)|
        {
            (c, VectorWord::new(index))
        }).collect::<Bimap<_, _>>();

        Self{dictionary}
    }

    fn is_input_one_hot() -> bool { true }

    fn input_data() -> InputDataType
    {
        InputDataType::String
    }

    fn word_to_bytes(&self, _previous_word: Option<VectorWord>, word: VectorWord) -> Box<[u8]>
    {
        let c = self.dictionary.by_value(&word).cloned()
            .unwrap_or(char::REPLACEMENT_CHARACTER);

        let mut s = [0_u8; 4];
        let encoded = c.encode_utf8(&mut s);

        encoded.as_bytes().into()
    }

    fn words_amount(&self) -> usize
    {
        // +1 for replacement character
        self.dictionary.len() + 1
    }
}

enum SpaceInfo
{
    Left,
    Right,
    Both,
    None
}

impl SpaceInfo
{
    pub fn right(self) -> bool
    {
        match self
        {
            Self::Right | Self::Both => true,
            _ => false
        }
    }

    pub fn left(self) -> bool
    {
        match self
        {
            Self::Left | Self::Both => true,
            _ => false
        }
    }
}

pub const WORD_SEPARATORS: [char; 32] = [
    '>',
    '<',
    ':',
    '\n',
    '.',
    ',',
    '-',
    '(',
    ')',
    '{',
    '}',
    '[',
    ']',
    '=',
    '!',
    '?',
    '/',
    '*',
    '\'',
    '\\',
    '"',
    '_',
    // all the numbers r separators cuz i dont want a billion different words for numbers
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9'
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordDictionary
{
    dictionary: Bimap<String, VectorWord>,
    leftover_separator: Option<usize>
}

impl WordDictionary
{
    fn separator_word(&self, index: usize) -> VectorWord
    {
        let index = self.dictionary.len() + index;

        VectorWord::new(index)
    }

    pub fn read_word(reader: impl Iterator<Item=char>) -> (Option<usize>, String)
    {
        let mut leftover_separator = None;
        let mut word = String::new();

        for c in reader
        {
            if c == ' '
            {
                if !word.is_empty()
                {
                    break;
                } else
                {
                    continue;
                }
            }

            if let Some(pos) = WORD_SEPARATORS.iter().position(|v| c == *v)
            {
                leftover_separator = Some(pos);

                break;
            } else
            {
                // i kinda hate clippy cuz of these stupid suggestions
                #[allow(clippy::collapsible_else_if)]
                if LOWERCASE_ONLY
                {
                    c.to_lowercase().for_each(|c|
                    {
                        word.push(c);
                    })
                } else
                {
                    word.push(c);
                }
            }
        }

        (leftover_separator, word)
    }

    fn next_word<R: Read>(&mut self, reader: &mut CharsAdapter<R>) -> Option<VectorWord>
    {
        if let Some(separator_index) = self.leftover_separator
        {
            self.leftover_separator = None;

            return Some(self.separator_word(separator_index));
        }

        let (leftover_separator, word) = Self::read_word(reader);

        self.leftover_separator = leftover_separator;

        if word.is_empty()
        {
            return self.leftover_separator.take().map(|i|
            {
                self.separator_word(i)
            });
        }

        Some(self.dictionary.by_key(&word).cloned().unwrap_or_else(||
        {
            eprintln!("unknown word: {word}");
            VectorWord::new(self.words_amount() - 1)
        }))
    }

    fn word_as_separator(&self, word: VectorWord) -> Option<char>
    {
        let index = word.index();
        let words_amount = self.dictionary.len();

        if index >= words_amount
        {
            let c = if index == (self.words_amount() - 1)
            {
                char::REPLACEMENT_CHARACTER
            } else
            {
                WORD_SEPARATORS[index - words_amount]
            };

            Some(c)
        } else
        {
            None
        }
    }

    fn word_to_bytes_inner(&self, word: VectorWord) -> Vec<u8>
    {
        if let Some(separator) = self.word_as_separator(word)
        {
            return separator.to_string().into_bytes();
        }

        self.dictionary.by_value(&word)
            .cloned()
            .unwrap()
            .into_bytes()
    }

    fn space_info(separator: char) -> Option<SpaceInfo>
    {
        match separator
        {
            ')'|']'|'}' => Some(SpaceInfo::Right),
            '('|'['|'{' => Some(SpaceInfo::Left),
            '>'|'<' => Some(SpaceInfo::None),
            ':' => Some(SpaceInfo::Right),
            '\n' => Some(SpaceInfo::None),
            '.'|','|'!'|'?' => Some(SpaceInfo::Right),
            '-' => Some(SpaceInfo::None),
            '=' => Some(SpaceInfo::Both),
            '/' => Some(SpaceInfo::None),
            '*' => Some(SpaceInfo::Both),
            '\'' => Some(SpaceInfo::None),
            '\\' => Some(SpaceInfo::None),
            '"' => Some(SpaceInfo::Both),
            '_' => Some(SpaceInfo::None),
            char::REPLACEMENT_CHARACTER => Some(SpaceInfo::Both),
            _ => None
        }
    }

    fn needs_space(&self, previous_word: Option<VectorWord>, word: VectorWord) -> bool
    {
        if let Some(previous_word) = previous_word
        {
            let previous_word = self.word_as_separator(previous_word);
            let word = self.word_as_separator(word);

            let is_previous_digit = previous_word.map(|x| x.is_ascii_digit()).unwrap_or(false);

            let right_space = previous_word.map(|x|
            {
                if let Some(info) = Self::space_info(x)
                {
                    info.right()
                } else
                {
                    false
                }
            }).unwrap_or(true);

            let left_space = word.map(|x|
            {
                if let Some(info) = Self::space_info(x)
                {
                    info.left()
                } else
                {
                    false
                }
            }).unwrap_or(true);

            if word.map(|x| x.is_ascii_digit()).unwrap_or(false)
            {
                if let Some(':') = previous_word
                {
                    return false;
                }

                return right_space && !is_previous_digit;
            }

            if is_previous_digit
            {
                return left_space;
            }

            right_space && left_space
        } else
        {
            false
        }
    }

    pub fn str_to_word(&self, s: &str) -> Option<VectorWord>
    {
        if s.len() == 1
        {
            let c = s.chars().next().unwrap();
            if let Some(pos) = WORD_SEPARATORS.iter().position(|v| c == *v)
            {
                return Some(self.separator_word(pos));
            }
        }

        self.dictionary.by_key(s).copied()
    }
}

impl NetworkDictionary for WordDictionary
{
    type Adapter<R: Read> = CharsAdapter<R>;


    fn new(data: InputData) -> Self
    {
        let s = match data
        {
            InputData::String(value) => value,
            _ => unreachable!()
        };

        let dictionary: Bimap<_, _> = s.split('\n').enumerate().map(|(index, word)|
        {
            (word.to_owned(), VectorWord::new(index))
        }).collect();

        Self{dictionary, leftover_separator: None}
    }

    fn is_input_one_hot() -> bool { true }

    fn input_data() -> InputDataType
    {
        InputDataType::String
    }

    fn word_to_bytes(&self, previous_word: Option<VectorWord>, word: VectorWord) -> Box<[u8]>
    {
        let needs_space = self.needs_space(previous_word, word);

        let mut bytes = self.word_to_bytes_inner(word);

        if needs_space
        {
            bytes.insert(0, b' ');
        }

        bytes.into()
    }

    fn words_amount(&self) -> usize
    {
        // +1 for unknown token
        self.dictionary.len() + WORD_SEPARATORS.len() + 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BpeMapping
{
    pub pair: (u32, u32),
    pub output: u32,
    pub is_scaffold: bool
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaffoldedIndex(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenIndex(u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BpeDictionaryCache
{
    words_to_bytes: Vec<Box<[u8]>>
}

impl BpeDictionaryCache
{
    pub fn new(dictionary: &BpeDictionary) -> Self
    {
        let mut words_to_bytes: Vec<Box<[u8]>> = (0..=u8::MAX).map(|x| -> Box<[u8]> { Box::new([x]) }).collect();

        dictionary.pairs.iter().for_each(|mapping|
        {
            let (a, b) = mapping.pair;

            let combined_bytes = words_to_bytes[a as usize].iter().chain(&words_to_bytes[b as usize]).copied().collect();
            words_to_bytes.push(combined_bytes);
        });

        Self{
            words_to_bytes
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BpeDictionary
{
    pub pairs: Vec<BpeMapping>,
    #[serde(skip)]
    pub cached: Option<BpeDictionaryCache>
}

#[allow(dead_code)]
impl BpeDictionary
{
    fn word_to_pair_index_known(word: u32) -> usize
    {
        (word - u8::MAX as u32 - 1) as usize
    }

    pub fn word_to_pair_index(word: u32) -> Option<usize>
    {
        if word <= u8::MAX as u32
        {
            None
        } else
        {
            Some(Self::word_to_pair_index_known(word))
        }
    }

    fn token_to_scaffolded(&self, word: TokenIndex) -> ScaffoldedIndex
    {
        if let Some(word) = Self::word_to_pair_index(word.0)
        {
            let (index, _mapping) = self.pairs.iter()
                .enumerate()
                .filter(|(_, x)| !x.is_scaffold)
                .nth(word as usize)
                .unwrap();

            ScaffoldedIndex(index as u32 + u8::MAX as u32 + 1)
        } else
        {
            ScaffoldedIndex(word.0)
        }
    }

    fn require_cache(&mut self)
    {
        self.cached = Some(BpeDictionaryCache::new(self));
    }

    pub fn word_to_bytes_scaffolded_single(&self, word: ScaffoldedIndex) -> Box<[u8]>
    {
        if let Some(pair_index) = Self::word_to_pair_index(word.0)
        {
            let pair = self.pairs[pair_index].pair;

            self.word_to_bytes_scaffolded_single(ScaffoldedIndex(pair.0))
                .into_iter()
                .chain(self.word_to_bytes_scaffolded_single(ScaffoldedIndex(pair.1)))
                .collect()
        } else
        {
            Box::new([word.0 as u8])
        }
    }

    pub fn word_to_bytes_single(&self, word: TokenIndex) -> Box<[u8]>
    {
        self.cached.as_ref().unwrap().words_to_bytes[self.token_to_scaffolded(word).0 as usize].clone()
    }

    pub fn combine_pair(ngrams: &mut Vec<u32>, mapping: BpeMapping, mut on_replace: impl FnMut(Option<u32>, Option<u32>))
    {
        let mut any_replaced = false;

        // this gets constantly reallocated so its not a big deal that its a bit bigger
        let mut new_ngrams = Vec::with_capacity(ngrams.len());

        let mut i = 0;
        while i < ngrams.len().saturating_sub(1)
        {
            if (ngrams[i], ngrams[i + 1]) == mapping.pair
            {
                any_replaced = true;

                on_replace(new_ngrams.last().copied(), ngrams.get(i + 2).copied());

                new_ngrams.push(mapping.output);

                i += 1;
            } else
            {
                new_ngrams.push(ngrams[i]);
            }

            i += 1;
        }

        if !any_replaced
        {
            return;
        }

        if (i + 1) == ngrams.len()
        {
            new_ngrams.push(ngrams[ngrams.len() - 1]);
        }

        *ngrams = new_ngrams;
    }

    fn next_word<R: Read>(&mut self, _reader: &mut DefaultAdapter<R>) -> Option<VectorWord>
    {
        unreachable!()
    }
}

#[allow(dead_code)]
pub fn bpe_from_bytes(
    limit: usize,
    optional_info: bool,
    bytes: impl IntoIterator<Item=u8>
) -> BpeDictionary
{
    let mut ngrams: Vec<u32> = bytes.into_iter().map(u32::from).collect();

    if ngrams.is_empty()
    {
        complain("the input is empty, cant create bpe");
    }

    let mut dictionary = BpeDictionary{pairs: Vec::new(), cached: None};

    fn pair_frequencies_of(ngrams: &[u32]) -> HashMap<(u32, u32), usize>
    {
        let mut pair_frequencies: HashMap<(u32, u32), usize> = HashMap::new();

        ngrams.windows(2).for_each(|x|
        {
            let pair = (x[0], x[1]);

            *pair_frequencies.entry(pair).or_insert(0) += 1;
        });

        pair_frequencies
    }

    let mut pair_frequencies: HashMap<(u32, u32), usize> = pair_frequencies_of(&ngrams);

    fn most_common_of(pair_frequencies: &HashMap<(u32, u32), usize>) -> ((u32, u32), usize)
    {
        pair_frequencies.iter()
            .max_by_key(|(_key, value)| *value)
            .map(|(a, b)| (*a, *b))
            .expect("must exist")
    }

    let mut most_common = most_common_of(&pair_frequencies);

    loop
    {
        let (most_common_pair, occurred_times) = most_common;

        let mapping = BpeMapping{
            pair: most_common_pair,
            output: u8::MAX as u32 + 1 + dictionary.pairs.len() as u32,
            is_scaffold: false
        };

        if let Some(previous_mapping) = dictionary.pairs.iter_mut().find(|x| x.pair == mapping.pair)
        {
            previous_mapping.is_scaffold = false;

            continue;
        }

        dictionary.pairs.push(mapping);

        let before_length = ngrams.len();
        let mut replaced_times = 0;

        BpeDictionary::combine_pair(&mut ngrams, mapping, |u, v|
        {
            replaced_times += 1;

            let decrease_pair = |pair_frequencies: &mut HashMap<_, _>, x: u32, y: u32|
            {
                let key = (x, y);

                let value = pair_frequencies.get_mut(&key)
                    .unwrap_or_else(|| panic!("pair ({x},{y}) must exist"));

                *value -= 1;

                if *value == 0
                {
                    pair_frequencies.remove(&key);
                }
            };

            let increase_pair = |pair_frequencies: &mut HashMap<_, _>, x: u32, y: u32|
            {
                *pair_frequencies.entry((x, y)).or_insert(0) += 1;
            };

            let t = mapping.output;

            let (a, b) = mapping.pair;

            if let Some(u) = u
            {
                decrease_pair(&mut pair_frequencies, u, a);
                increase_pair(&mut pair_frequencies, u, t);
            }

            if let Some(v) = v
            {
                decrease_pair(&mut pair_frequencies, b, v);
                increase_pair(&mut pair_frequencies, t, v);
            }
        });

        debug_assert_eq!(before_length - replaced_times, ngrams.len());

        pair_frequencies.remove(&mapping.pair);

        most_common = most_common_of(&pair_frequencies);

        fn pair_number_to_index(number: u32) -> Option<usize>
        {
            BpeDictionary::word_to_pair_index(number)
        }

        {
            let a = pair_number_to_index(mapping.pair.0);
            let b = pair_number_to_index(mapping.pair.1);

            let mut mark_if_scaffold = |mapping: &mut BpeMapping|
            {
                if mapping.is_scaffold
                {
                    return;
                }

                let standalone_frequency = pair_frequencies.get(&mapping.pair).copied().unwrap_or(0);

                if standalone_frequency < most_common.1
                {
                    mapping.is_scaffold = true;

                    pair_frequencies.insert(mapping.pair, standalone_frequency);
                }
            };

            if let Some(a) = a
            {
                mark_if_scaffold(&mut dictionary.pairs[a]);
            }

            if let Some(b) = b
            {
                mark_if_scaffold(&mut dictionary.pairs[b]);
            }
        }

        let scaffold_count = dictionary.pairs.iter().filter(|x| x.is_scaffold).count();
        let used_count = dictionary.pairs.len() - scaffold_count;

        if optional_info
        {
            println!("({scaffold_count} scaffold) {used_count}/{limit} replaced pair that occurs {occurred_times} times");
        }

        if used_count == limit
        {
            let ngram = dictionary.word_to_bytes_scaffolded_single(ScaffoldedIndex(mapping.output));
            println!("least common ngram occurs {occurred_times} times: {}", String::from_utf8_lossy(&ngram));

            return dictionary;
        }
    }
}

impl NetworkDictionary for BpeDictionary
{
    type Adapter<R: Read> = DefaultAdapter<R>;


    fn new(data: InputData) -> Self
    {
        let path = match data
        {
            InputData::Path(value) => value,
            _ => unreachable!()
        };

        let file = File::open(&path).unwrap_or_else(|err|
        {
            complain(format!("error opening bpe file ({}): {err}", path.display()))
        });

        let mut this: Self = PostcardFormat::deserialize(file).unwrap_or_else(|err|
        {
            complain(format!("error loading bpe: {err}"))
        });

        this.require_cache();

        this
    }

    fn is_input_one_hot() -> bool { true }

    fn vectorized<R: Read>(&mut self, reader: R) -> Vec<VectorWord>
    {
        self.require_cache();

        let mut ngrams: Vec<u32> = {
            let mut reader = BufReader::new(reader);

            let mut bytes: Vec<u8> = Vec::new();
            reader.read_to_end(&mut bytes).unwrap();

            bytes.into_iter().map(u32::from).collect()
        };

        self.pairs.iter().for_each(|pair|
        {
            Self::combine_pair(&mut ngrams, *pair, |_, _| {});
        });

        let mut word_to_used: Vec<Box<[VectorWord]>> = (0..=u8::MAX as usize)
            .map(|x| -> Box<[VectorWord]> { Box::new([VectorWord::new(x)]) })
            .collect();

        self.pairs.iter().fold(0, |mut current_word, mapping|
        {
            let word = current_word + u8::MAX as usize + 1;

            if !mapping.is_scaffold
            {
                current_word += 1;

                word_to_used.push(Box::new([VectorWord::new(word)]));
            } else
            {
                let (a, b) = mapping.pair;

                let combined_word = word_to_used[a as usize].iter().chain(&word_to_used[b as usize]).copied().collect();
                word_to_used.push(combined_word);
            }

            current_word
        });

        let mut output = Vec::with_capacity(ngrams.len());

        ngrams.into_iter().for_each(|x|
        {
            output.extend(&word_to_used[x as usize]);
        });

        output
    }

    fn input_data() -> InputDataType
    {
        InputDataType::Path(PathType::Dictionary)
    }

    fn word_to_bytes(&self, _previous_word: Option<VectorWord>, word: VectorWord) -> Box<[u8]>
    {
        self.word_to_bytes_single(TokenIndex(word.0 as u32))
    }

    fn words_amount(&self) -> usize
    {
        self.pairs.iter().filter(|x| !x.is_scaffold).count() + u8::MAX as usize + 1
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EmbeddingsDictionary
{
    word_dictionary: WordDictionary,
    network: SaveNetwork<EmbeddingsUnitFactory, ()>,
    embeddings_size: usize
}

impl Debug for EmbeddingsDictionary
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        f.debug_struct("EmbeddingsDictionary")
            .field("word_dictionary", &self.word_dictionary)
            .field("embeddings_size", &self.embeddings_size)
            .finish()
    }
}

impl NetworkDictionary for EmbeddingsDictionary
{
    type Adapter<R: Read> = CharsAdapter<R>;


    fn new(data: InputData) -> Self
    {
        let path = match data
        {
            InputData::Path(value) => value,
            _ => unreachable!()
        };

        let neural_network: NeuralNetwork<SaveNetwork<_, ()>, (), _> = NeuralNetwork::load_data(path.as_ref()).unwrap_or_else(|err|
        {
            complain(format!("could not load embeddings at {} ({err})", path.display()))
        });

        let (word_dictionary, network) = neural_network.into_embeddings_info();

        let embeddings_size = network.sizes().hidden;

        Self{word_dictionary, network, embeddings_size}
    }

    fn is_input_one_hot() -> bool { false }

    fn input_data() -> InputDataType
    {
        InputDataType::Path(PathType::Embeddings)
    }

    fn one_hot_to_input(&self, layer: OneHotLayer) -> OwnedInputType
    {
        self.network.embeddings(&layer).into()
    }

    fn words_to_layer(&self, words: impl IntoIterator<Item=VectorWord>) -> OwnedInputType
    {
        self.one_hot_to_input(self.words_to_onehot(words))
    }

    fn word_to_bytes(&self, previous_word: Option<VectorWord>, word: VectorWord) -> Box<[u8]>
    {
        self.word_dictionary.word_to_bytes(previous_word, word)
    }

    fn words_amount(&self) -> usize
    {
        self.word_dictionary.words_amount()
    }

    fn input_amount(&self) -> usize
    {
        self.embeddings_size
    }
}

pub trait ReaderAdapter<R>
{
    fn adapter(reader: R) -> Self;
}

pub struct DefaultAdapter<R>
{
    reader: R
}

impl<R> ReaderAdapter<R> for DefaultAdapter<R>
{
    fn adapter(reader: R) -> Self
    {
        Self{reader}
    }
}

pub struct CharsAdapter<R: Read>
{
    code_points: CodePoints<Bytes<R>>
}

impl<R: Read> ReaderAdapter<R> for CharsAdapter<R>
{
    fn adapter(reader: R) -> Self
    {
        Self{code_points: CodePoints::from(reader)}
    }
}

impl<R: Read> Iterator for CharsAdapter<R>
{
    type Item = char;

    fn next(&mut self) -> Option<Self::Item>
    {
        self.code_points.next().map(|c|
        {
            match c
            {
                Ok(c) => c,
                Err(err)
                    if err.kind() == io::ErrorKind::InvalidData
                    || err.kind() == io::ErrorKind::UnexpectedEof =>
                {
                    char::REPLACEMENT_CHARACTER
                }
                Err(x) => panic!("{}", x)
            }
        })
    }
}

pub struct WordVectorizer<A, D>
{
    adapter: A,
    dictionary: D
}

impl<A, D> WordVectorizer<A, D>
{
    pub fn new<R>(dictionary: D, reader: R) -> Self
    where
        R: Read,
        A: ReaderAdapter<BufReader<R>>
    {
        let adapter = A::adapter(BufReader::new(reader));

        Self{adapter, dictionary}
    }
}

impl<A, D> Deref for WordVectorizer<A, D>
{
    type Target = D;

    fn deref(&self) -> &Self::Target
    {
        &self.dictionary
    }
}

impl<R: Read> Iterator for WordVectorizer<DefaultAdapter<R>, &mut ByteDictionary>
{
    type Item = VectorWord;

    fn next(&mut self) -> Option<Self::Item>
    {
        let reader = &mut self.adapter.reader;

        reader.bytes()
            .next()
            .map(|b| VectorWord::new(b.expect("io error? wow") as usize))
    }
}

impl<R: Read> Iterator for WordVectorizer<CharsAdapter<R>, &mut CharDictionary>
{
    type Item = VectorWord;

    fn next(&mut self) -> Option<Self::Item>
    {
        let c = self.adapter.next()?;

        Some(self.dictionary.character_match(c))
    }
}

impl<R: Read> Iterator for WordVectorizer<CharsAdapter<R>, &mut WordDictionary>
{
    type Item = VectorWord;

    fn next(&mut self) -> Option<Self::Item>
    {
        self.dictionary.next_word(self.adapter.by_ref())
    }
}

impl<R: Read> Iterator for WordVectorizer<DefaultAdapter<R>, &mut BpeDictionary>
{
    type Item = VectorWord;

    fn next(&mut self) -> Option<Self::Item>
    {
        self.dictionary.next_word(&mut self.adapter)
    }
}

impl<R: Read> Iterator for WordVectorizer<CharsAdapter<R>, &mut EmbeddingsDictionary>
{
    type Item = VectorWord;

    fn next(&mut self) -> Option<Self::Item>
    {
        self.dictionary.word_dictionary.next_word(self.adapter.by_ref())
    }
}

#[cfg(test)]
mod tests
{
    #[allow(unused_imports)]
    use super::*;

    use std::io::Cursor;

    fn original_text() -> &'static str
    {
        "hello world im testing a COOL encoder (not rly) fake and gay"
    }

    fn reader() -> impl Read
    {
        Cursor::new(original_text())
    }

    #[test]
    fn encodes_decodes()
    {
        let s = if LOWERCASE_ONLY
        {
            "cool\ngay\nbro\nhello\nrly\nworld\na\nnot"
        } else
        {
            "COOL\ngay\nbro\nhello\nrly\nworld\na\nnot"
        };

        let mut dictionary = WordDictionary::new(InputData::String(s.into()));

        encode_decode_test_lossy(
            dictionary.clone(),
            WordVectorizer::new(&mut dictionary, reader()),
            if LOWERCASE_ONLY
            {
                "hello world � � a cool � (not rly) � � gay"
            } else
            {
                "hello world � � a COOL � (not rly) � � gay"
            }
        );
    }

    #[test]
    fn encodes_decodes_bpe()
    {
        let s = b"hellohellohellohelloworldimgayworldwor";
        let mut dictionary = bpe_from_bytes(2, true, s.into_iter().copied());

        let encoded = dictionary.vectorized(reader());

        let decoded: Vec<Box<[u8]>> = encoded.into_iter().map(|word| dictionary.word_to_bytes(None, word)).collect();

        let decoded: Vec<u8> = decoded.into_iter().flatten().collect();

        assert_eq!(
            decoded, original_text().bytes().collect::<Vec<u8>>(),
            "expected: {}, got: {}",
            original_text(),
            String::from_utf8_lossy(&decoded)
        );
    }

    #[test]
    fn encodes_decodes_char()
    {
        let s = "h elow / im tsngaCLcdr()lyfk)";

        let mut dictionary = CharDictionary::new(InputData::String(s.into()));

        encode_decode_test_lossy(
            dictionary.clone(),
            WordVectorizer::new(&mut dictionary, reader()),
            "hello world im testing a C��L encoder (not rly) fake and gay"
        );
    }

    #[test]
    fn encodes_decodes_bytes()
    {
        let mut dictionary = ByteDictionary::new(InputData::None);

        encode_decode_test_lossy(
            dictionary.clone(),
            WordVectorizer::new(&mut dictionary, reader()),
            original_text()
        );
    }

    #[allow(dead_code)]
    fn encode_decode_test_lossy<D, V>(dictionary: D, vectorizer: V, expected: &str)
    where
        D: NetworkDictionary,
        V: Iterator<Item=VectorWord>
    {
        let mut previous_word = None;

        let decoded_bytes = vectorizer.map(|word|
        {
            let output = (previous_word, word);

            previous_word = Some(word);

            output
        }).flat_map(|(previous_word, word)|
        {
            let layer = dictionary.words_to_layer([word]);
            let word = dictionary.layer_to_word(layer.into_one_hot().into_layer());

            dictionary.word_to_bytes(previous_word, word).into_vec().into_iter()
        }).collect::<Vec<u8>>();

        assert_eq!(
            decoded_bytes,
            expected.bytes().collect::<Vec<u8>>(),
            "decoded: {}, expected: {expected}",
            &String::from_utf8_lossy(&decoded_bytes)
        );
    }
}
