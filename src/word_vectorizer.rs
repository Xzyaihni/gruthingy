use std::{
    str,
    hash::Hash,
    borrow::Borrow,
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

use super::neural_network::{
    LayerInnerType,
    DICTIONARY_TEXT
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
#[derive(Debug, Serialize, Deserialize)]
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
    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>;
    fn words_amount_trait(&self) -> usize;
    
    fn word_to_layer(&self, word: VectorWord) -> LayerInnerType
    {
        let mut layer = vec![0.0; self.words_amount_trait()];

        layer[word.index()] = 1.0;

        LayerInnerType::from_raw(layer, self.words_amount_trait(), 1)
    }

    fn layer_to_word(&self, layer: LayerInnerType) -> VectorWord
    {
        let index = layer.pick_weighed();

        VectorWord::new(index)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ByteDictionary;

impl ByteDictionary
{
    #[allow(dead_code)]
    pub fn build(_: &'static str) -> Self
    {
        unimplemented!();
    }

    #[allow(dead_code)]
    pub fn new() -> Self
    {
        Self{}
    }

    pub const fn words_amount() -> usize
    {
        u8::MAX as usize + 1
    }
}

impl NetworkDictionary for ByteDictionary
{
    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>
    {
        Box::new([word.index() as u8])
    }

    fn words_amount_trait(&self) -> usize
    {
        Self::words_amount()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CharDictionary
{
    dictionary: Bimap<char, VectorWord>
}

impl CharDictionary
{
    #[allow(dead_code)]
    pub fn build(s: &'static str) -> Self
    {
        let mut unique_chars = s.chars().collect::<Vec<char>>();
        unique_chars.sort_unstable();

        let mut repeated = Vec::new();
        let mut chars_set = HashSet::new();

        for c in DICTIONARY_TEXT.chars()
        {
            if !chars_set.insert(c)
            {
                repeated.push(c);
            }
        }

        assert!(
            repeated.is_empty(),
            "remove these repeating characters from the dictionary: {}",
            repeated.into_iter().fold(String::new(), |acc: String, c: char|
            {
                let fmt_c = |c|
                {
                    format!("'{c}' (hex {:#x})", c as u32)
                };

                if acc.is_empty()
                {
                    fmt_c(c)
                } else
                {
                    format!("{acc}, {}", fmt_c(c))
                }
            })
        );
        assert_eq!(Self::words_amount(), unique_chars.len() + 1);

        let dictionary = unique_chars.into_iter().enumerate().map(|(index, c)|
        {
            (c, VectorWord::new(index))
        }).collect::<Bimap<_, _>>();

        Self{dictionary}
    }

    #[allow(dead_code)]
    pub fn new() -> Self
    {
        unimplemented!();
    }

    const fn chars_amount(s: &'static str) -> usize
    {
        let s = s.as_bytes();

        let mut amount = 0;

        let mut b = 0;
        while b < s.len()
        {
            let is_continuation = (s[b] & 0b1100_0000) == 0b1000_0000;

            if !is_continuation
            {
                amount += 1;
            }

            b += 1;
        }

        amount
    }

    pub const fn words_amount() -> usize
    {
        // +1 for replacement character
        Self::chars_amount(DICTIONARY_TEXT) + 1
    }

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
    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>
    {
        let c = self.dictionary.by_value(&word).cloned()
            .unwrap_or(char::REPLACEMENT_CHARACTER);

        let mut s = [0_u8; 4];
        let s = c.encode_utf8(&mut s);

        s.as_bytes().into()
    }

    fn words_amount_trait(&self) -> usize
    {
        Self::words_amount()
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

#[derive(Debug, Serialize, Deserialize)]
pub struct WordDictionary
{
    dictionary: Bimap<String, VectorWord>,
    leftover_separator: Option<usize>
}

impl WordDictionary
{
    #[allow(dead_code)]
    pub fn build(s: &'static str) -> Self
    {
        let dictionary: Bimap<_, _> = s.split('\n').enumerate().map(|(index, word)|
        {
            (word.to_owned(), VectorWord::new(index))
        }).collect();

        assert_eq!(dictionary.len(), Self::words_amount_raw());

        Self{dictionary, leftover_separator: None}
    }

    #[allow(dead_code)]
    pub fn new() -> Self
    {
        unimplemented!();
    }

    const fn words_amount_inner(s: &'static str) -> usize
    {
        let s = s.as_bytes();

        let mut amount = 0;

        let mut is_trailing = true;

        let mut b = 0;
        while b < s.len()
        {
            is_trailing = false;

            if s[b] == b'\n'
            {
                is_trailing = true;

                amount += 1;
            }

            b += 1;
        }

        if is_trailing
        {
            amount
        } else
        {
            // add 1 word cuz one of them doesnt have a last newline
            amount + 1
        }
    }

    const fn words_amount_raw() -> usize
    {
        Self::words_amount_inner(DICTIONARY_TEXT)
    }

    pub const fn words_amount() -> usize
    {
        // +1 for unknown token
        Self::words_amount_raw() + WORD_SEPARATORS.len() + 1
    }

    fn separator_word(index: usize) -> VectorWord
    {
        let index = Self::words_amount_raw() + index;

        VectorWord::new(index)
    }
}

impl NetworkDictionary for WordDictionary
{
    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>
    {
        let index = word.index();
        let words_amount = Self::words_amount_raw();

        if index >= words_amount
        {
            let c = if index == (Self::words_amount() - 1)
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

    fn words_amount_trait(&self) -> usize
    {
        Self::words_amount()
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

            return Some(WordDictionary::separator_word(separator_index));
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
            return self.dictionary.leftover_separator.take().map(WordDictionary::separator_word);
        }

        Some(self.dictionary.dictionary.by_key(&word).cloned().unwrap_or_else(||
        {
            eprintln!("unknown word: {word}");
            VectorWord::new(WordDictionary::words_amount() - 1)
        }))
    }
}

#[cfg(test)]
mod tests
{
    #[allow(unused_imports)]
    use super::*;

    /*#[test]
    fn encodes_decodes()
    {
        let dictionary = WordDictionary::build("cool vocab bro hello rly".as_bytes());

        encode_decode_test(dictionary);
    }

    #[test]
    fn encodes_decodes_char()
    {
        let dictionary = CharDictionary::build("h elow / im tsngaCLcdr()lyfk");

        encode_decode_test_lossy(
            dictionary,
            "hello world im testing a C��L encoder (not rly) fake and gay"
        );
    }

    #[test]
    fn encodes_decodes_bytes()
    {
        let dictionary = ByteDictionary::new();

        encode_decode_test(dictionary);
    }*/

    /*#[allow(dead_code)]
    fn encode_decode_test(dictionary: impl NetworkDictionary)
    {
        encode_decode_test_lossy(
            dictionary,
            "hello world im testing a COOL encoder (not rly) fake and gay"
        );
    }

    #[allow(dead_code)]
    fn encode_decode_test_lossy(mut dictionary: impl NetworkDictionary, expected: &str)
    {
        let original_bytes = "hello world im testing a COOL encoder (not rly) fake and gay";

        let vectorizer = WordVectorizer::new(&mut dictionary, original_bytes.as_bytes());

        let decoded_bytes = vectorizer.collect::<Vec<_>>();
        let decoded_bytes = decoded_bytes.into_iter().flat_map(|word|
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
            original_bytes
        );
    }*/
}
