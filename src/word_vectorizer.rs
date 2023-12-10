use std::{
    str,
    hash::Hash,
    borrow::Borrow,
    ops::Deref,
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

use super::neural_network::LayerInnerType;


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

pub trait NetworkDictionary
{
    type Adapter<R: Read>: ReaderAdapter<R>;


    fn new(data: Option<&str>) -> Self;

    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>;
    fn words_amount(&self) -> usize;

    fn needs_data() -> bool
    {
        true
    }
    
    fn words_to_layer(&self, w0: VectorWord, w1: VectorWord) -> LayerInnerType
    {
        let mut layer = vec![0.0; self.words_amount()];

        layer[w0.index()] = 1.0;
        layer[w1.index()] = 1.0;

        LayerInnerType::from_raw(layer, self.words_amount(), 1)
    }
    
    fn word_to_layer(&self, word: VectorWord) -> LayerInnerType
    {
        let mut layer = vec![0.0; self.words_amount()];

        layer[word.index()] = 1.0;

        LayerInnerType::from_raw(layer, self.words_amount(), 1)
    }

    fn layer_to_word(&self, layer: LayerInnerType) -> VectorWord
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


    fn new(_data: Option<&str>) -> Self
    {
        Self{}
    }

    fn needs_data() -> bool
    {
        false
    }

    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>
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


    fn new(data: Option<&str>) -> Self
    {
        let unique_chars: HashSet<_> = data.unwrap().chars().collect();

        let dictionary = unique_chars.into_iter().enumerate().map(|(index, c)|
        {
            (c, VectorWord::new(index))
        }).collect::<Bimap<_, _>>();

        Self{dictionary}
    }

    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>
    {
        let c = self.dictionary.by_value(&word).cloned()
            .unwrap_or(char::REPLACEMENT_CHARACTER);

        let mut s = [0_u8; 4];
        let s = c.encode_utf8(&mut s);

        s.as_bytes().into()
    }

    fn words_amount(&self) -> usize
    {
        // +1 for replacement character
        self.dictionary.len() + 1
    }
}

pub const WORD_SEPARATORS: [char; 33] = [
    '>',
    '<',
    ' ',
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
}

impl NetworkDictionary for WordDictionary
{
    type Adapter<R: Read> = CharsAdapter<R>;


    fn new(data: Option<&str>) -> Self
    {
        let dictionary: Bimap<_, _> = data.unwrap().split('\n').enumerate().map(|(index, word)|
        {
            (word.to_owned(), VectorWord::new(index))
        }).collect();

        Self{dictionary, leftover_separator: None}
    }

    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>
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

            // WHYYYYYYYY i wanna rework all of this garbage but i dont want to touch this
            // stupid project anymore either (mostly cuz i dont wanna bugfix it)
            return c.to_string().into_bytes().into_boxed_slice();
        }

        // why?
        self.dictionary.by_value(&word).cloned().unwrap()
            .into_bytes()
            .into_boxed_slice()
    }

    fn words_amount(&self) -> usize
    {
        // +1 for unknown token
        self.dictionary.len() + WORD_SEPARATORS.len() + 1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsDictionary
{
    word_dictionary: WordDictionary,
    network: ()
}

impl NetworkDictionary for EmbeddingsDictionary
{
    type Adapter<R: Read> = CharsAdapter<R>;


    fn new(data: Option<&str>) -> Self
    {
        let word_dictionary = WordDictionary::new(data);

        let network = ();

        Self{word_dictionary, network}
    }

    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>
    {
        self.word_dictionary.word_to_bytes(word)
    }

    fn words_amount(&self) -> usize
    {
        todo!();
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
        if let Some(separator_index) = self.dictionary.leftover_separator
        {
            self.dictionary.leftover_separator = None;

            return Some(self.dictionary.separator_word(separator_index));
        }

        let mut word = String::new();

        while let Some(c) = self.adapter.next()
        {
            if let Some(pos) = WORD_SEPARATORS.iter().position(|v| c == *v)
            {
                self.dictionary.leftover_separator = Some(pos);

                break;
            } else
            {
                word.push(c);
            }
        }

        if word.is_empty()
        {
            return self.dictionary.leftover_separator.take().map(|i|
            {
                self.dictionary.separator_word(i)
            });
        }

        Some(self.dictionary.dictionary.by_key(&word).cloned().unwrap_or_else(||
        {
            eprintln!("unknown word: {word}");
            VectorWord::new(self.dictionary.words_amount() - 1)
        }))
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
        let mut dictionary = WordDictionary::new(Some("COOL\ngay\nbro\nhello\nrly\nworld\na\nnot"));

        encode_decode_test_lossy(
            dictionary.clone(),
            WordVectorizer::new(&mut dictionary, reader()),
            "hello world � � a COOL � (not rly) � � gay"
        );
    }

    #[test]
    fn encodes_decodes_char()
    {
        let mut dictionary = CharDictionary::new(Some("h elow / im tsngaCLcdr()lyfk)"));

        encode_decode_test_lossy(
            dictionary.clone(),
            WordVectorizer::new(&mut dictionary, reader()),
            "hello world im testing a C��L encoder (not rly) fake and gay"
        );
    }

    #[test]
    fn encodes_decodes_bytes()
    {
        let mut dictionary = ByteDictionary::new(None);

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
        let decoded_bytes = vectorizer.flat_map(|word|
        {
            let layer = dictionary.word_to_layer(word);
            let word = dictionary.layer_to_word(layer);

            dictionary.word_to_bytes(word).into_vec().into_iter()
        }).collect::<Vec<u8>>();

        assert_eq!(
            decoded_bytes,
            expected.bytes().collect::<Vec<u8>>(),
            "decoded: {}, original: {}",
            &String::from_utf8_lossy(&decoded_bytes),
            original_text()
        );
    }
}
