use std::{
    str,
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

use crate::{load_embeddings, EmbeddingsUnitFactory};

use super::neural_network::{
    LayerType,
    InputType,
    OneHotLayer,
    LOWERCASE_ONLY,
    network::Network
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

pub enum InputDataType
{
    None,
    String,
    Path
}

pub enum InputData
{
    None,
    String(String),
    Path(PathBuf)
}

pub trait NetworkDictionary
{
    type Adapter<R: Read>: ReaderAdapter<R>;


    fn new(data: InputData) -> Self;

    fn word_to_bytes(&self, previous_word: Option<VectorWord>, word: VectorWord) -> Box<[u8]>;
    fn words_amount(&self) -> usize;

    fn input_data() -> InputDataType;

    fn input_amount(&self) -> usize
    {
        self.words_amount()
    }

    fn words_to_layer(&self, words: impl IntoIterator<Item=VectorWord>) -> InputType
    {
        self.words_to_onehot(words).into()
    }
    
    fn words_to_onehot(&self, words: impl IntoIterator<Item=VectorWord>) -> OneHotLayer
    {
        OneHotLayer::new(
            words.into_iter().map(|word| word.index()).collect::<Box<[_]>>(),
            self.words_amount()
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

            let is_previous_digit = previous_word.map(|x| x.is_digit(10)).unwrap_or(false);

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

            if word.map(|x| x.is_digit(10)).unwrap_or(false)
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

#[derive(Serialize, Deserialize)]
pub struct EmbeddingsDictionary
{
    word_dictionary: WordDictionary,
    network: Network<EmbeddingsUnitFactory, ()>,
    embeddings_size: usize
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

        let neural_network = load_embeddings::<()>(
            Some(path.as_ref()),
            None,
            false
        );

        let (word_dictionary, network) = neural_network.into_embeddings_info();

        let embeddings_size = network.sizes().hidden;

        Self{word_dictionary, network, embeddings_size}
    }

    fn input_data() -> InputDataType
    {
        InputDataType::Path
    }
    
    fn words_to_layer(&self, words: impl IntoIterator<Item=VectorWord>) -> InputType
    {
        self.network.embeddings(self.words_to_onehot(words)).into()
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
        let s = "COOL\ngay\nbro\nhello\nrly\nworld\na\nnot";

        let mut dictionary = WordDictionary::new(InputData::String(s.into()));

        encode_decode_test_lossy(
            dictionary.clone(),
            WordVectorizer::new(&mut dictionary, reader()),
            "hello world � � a COOL � (not rly) � � gay"
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
