use std::{
    hash::Hash,
    borrow::Borrow,
    iter::Peekable,
    collections::{HashMap, HashSet},
    io::{
        Read,
        BufRead,
        BufReader
    }
};

use serde::{Serialize, Deserialize};

use super::neural_network::{SoftmaxedLayer, LayerContainer};


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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorWord(usize);

impl VectorWord
{
    fn new(index: usize) -> Self
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
    pub fn by_key<Q: ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq
    {
        self.k_map.get(key)
    }

    pub fn by_value<Q: ?Sized>(&self, value: &Q) -> Option<&K>
    where
        V: Borrow<Q>,
        Q: Hash + Eq
    {
        self.v_map.get(value)
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
    fn word_to_layer(&self, word: VectorWord) -> LayerContainer;
    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>;
    fn words_amount(&self) -> usize;
    
    fn next_word(&mut self, bytes: impl BufRead) -> Option<VectorWord>;
    
    fn layer_to_word(&self, layer: &SoftmaxedLayer, temperature: f64) -> VectorWord
    {
        let index = layer.pick_weighed(temperature);

        VectorWord::new(index)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CharDictionary
{

}

impl CharDictionary
{
    pub fn new() -> Self
    {
        Self{}
    }
}

impl NetworkDictionary for CharDictionary
{
    fn word_to_layer(&self, word: VectorWord) -> LayerContainer
    {
        let mut layer = vec![0.0; u8::MAX as usize];

        layer[word.index()] = 1.0;

        layer.into()
    }

    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>
    {
        Box::new([word.index() as u8])
    }

    fn words_amount(&self) -> usize
    {
        u8::MAX as usize
    }

    fn next_word(&mut self, bytes: impl BufRead) -> Option<VectorWord>
    {
        bytes.bytes().next().map(|b| VectorWord::new(b.expect("io error? wow") as usize))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WordDictionary
{
    dictionary: Bimap<Box<[u8]>, VectorWord>,
    words_amount: usize,
    longest_word: usize,
    c_word: Vec<u8>
}

impl WordDictionary
{
    #[allow(dead_code)]
    pub fn build(words: impl Read) -> Self
    {
        let default_words = (0..u8::MAX).map(|i|
        {
            vec![i].into_boxed_slice()
        }).collect::<HashSet<_>>();

        Self::build_inner(default_words, words)
    }

    #[allow(dead_code)]
    pub fn no_defaults(words: impl Read) -> Self
    {
        Self::build_inner(HashSet::new(), words)
    }

    fn build_inner(defaults: HashSet<Box<[u8]>>, words: impl Read) -> Self
    {
        let all_words = Self::unique_words(defaults, words);
        let words_amount = all_words.len();

        let mut longest_word = 1;
        let dictionary = all_words.into_iter().enumerate().map(|(i, bytes)|
        {
            if bytes.len() > longest_word
            {
                longest_word = bytes.len();
            }

            (bytes, VectorWord::new(i as usize))
        }).collect::<Bimap<_, _>>();

        Self{dictionary, words_amount, longest_word, c_word: Vec::new()}
    }

    fn unique_words(default_words: HashSet<Box<[u8]>>, words: impl Read) -> HashSet<Box<[u8]>>
    {
        let mut unique_words = default_words;

        let mut words = words.bytes().map(|b| b.unwrap());

        while let Some(word) = Self::read_word(words.by_ref().peekable())
        {
            unique_words.insert(word);
        }

        unique_words
    }

    pub fn read_word<I>(mut bytes: Peekable<I>) -> Option<Box<[u8]>>
    where
        I: Iterator<Item=u8>
    {
        let separators = [' ', ',', '\n', '.', ':', '!', '?', '\'', '"', '-'];

        let mut word = Vec::new();

        while let Some(&b) = bytes.peek()
        {
            let c = char::from_u32(b as u32).unwrap();
            if separators.contains(&c)
            {
                if !word.is_empty()
                {
                    return Some(word.into_boxed_slice());
                } else
                {
                    bytes.next();
                    return Some(vec![b].into_boxed_slice());
                }
            }

            bytes.next();
            word.push(b);
        }

        (!word.is_empty()).then(|| word.into_boxed_slice())
    }

    fn part_bytes_to_word(&mut self, matched_word: &[u8]) -> VectorWord
    {
        for i in 1..matched_word.len()
        {
            let new_len = matched_word.len() - i;
            
            let word_part = &matched_word[..new_len];
            let part_matched = self.bytes_to_word(word_part);

            if let Some(part_matched) = part_matched
            {
                self.c_word = self.c_word[new_len..].to_vec();

                return part_matched;
            }
        }

        unreachable!()
    }

    pub fn bytes_to_word(&self, bytes: &[u8]) -> Option<VectorWord>
    {
        self.dictionary.by_key(bytes).cloned()
    }

    #[allow(dead_code)]
    pub fn print_vector_word(&self, word: VectorWord) -> String
    {
        let display_bytes = self.word_to_bytes(word);

        String::from_utf8_lossy(&display_bytes).to_string()
    }
}

impl NetworkDictionary for WordDictionary
{
    fn word_to_layer(&self, word: VectorWord) -> LayerContainer
    {
        let mut layer = vec![0.0; self.words_amount];

        layer[word.index()] = 1.0;

        layer.into()
    }

    fn word_to_bytes(&self, word: VectorWord) -> Box<[u8]>
    {
        self.dictionary.by_value(&word).cloned().unwrap()
    }

    fn words_amount(&self) -> usize
    {
        self.words_amount
    }

    fn next_word(&mut self, mut bytes: impl BufRead) -> Option<VectorWord>
    {
        let mut consume_amount = 0;
        if self.c_word.len() < self.longest_word
        {
            let buffer = bytes.fill_buf().expect("io error, tough luck lmao");
            self.c_word.extend_from_slice(buffer);
            
            consume_amount = buffer.len();
        }

        let matched_word = WordDictionary::read_word(self.c_word.iter().cloned().peekable())?;
        
        bytes.consume(consume_amount);

        let vector_word = match self.bytes_to_word(&matched_word)
        {
            Some(vector_word) => vector_word,
            None =>
            {
                let word = self.part_bytes_to_word(&matched_word);
                return Some(word);
            }
        };

        if matched_word.len() >= self.c_word.len()
        {
            self.c_word = Vec::new();
        } else
        {
            self.c_word = self.c_word[matched_word.len()..].to_vec();
        }

        Some(vector_word)
    }
}

pub struct WordVectorizer<'a, R: Read, D: NetworkDictionary>
{
    bytes: BufReader<R>,
    dictionary: &'a mut D
}

impl<'a, R: Read, D: NetworkDictionary> WordVectorizer<'a, R, D>
{
    pub fn new(dictionary: &'a mut D, reader: R) -> Self
    {
        let bytes = BufReader::new(reader);

        Self{bytes, dictionary}
    }
}

impl<'a, R: Read, D: NetworkDictionary> Iterator for WordVectorizer<'a, R, D>
{
    type Item = VectorWord;

    fn next(&mut self) -> Option<Self::Item>
    {
        let word = self.dictionary.next_word(&mut self.bytes);

        // word.map(|word| eprintln!("word: {}", self.dictionary.print_vector_word(word)));

        word
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn encodes_decodes()
    {
        let mut dictionary = WordDictionary::build("cool vocab bro hello rly".as_bytes());

        let original_bytes = "hello world im testing a COOL encoder (not rly) fake and gay";

        let vectorizer = WordVectorizer::new(&mut dictionary, original_bytes.as_bytes());

        let decoded_bytes = vectorizer.collect::<Vec<_>>();
        let decoded_bytes = decoded_bytes.into_iter().flat_map(|word|
        {
            let layer = dictionary.word_to_layer(word);
            let word = dictionary.layer_to_word(&SoftmaxedLayer::from_raw(layer), 1.0);

            dictionary.word_to_bytes(word).into_vec().into_iter()
        }).collect::<Vec<u8>>();

        assert_eq!(
            decoded_bytes,
            original_bytes.bytes().collect::<Vec<u8>>(),
            "decoded: {}, original: {}",
            &String::from_utf8_lossy(&decoded_bytes),
            original_bytes
        );
    }
}
