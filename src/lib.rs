#![allow(dead_code)]

use std::ffi::{CStr, CString, c_char};

use neural_network::*;
use word_vectorizer::*;

mod word_vectorizer;
mod neural_network;


#[no_mangle]
pub extern "C" fn predict(
    network_path: *const c_char,
    text: *const c_char,
    amount: u32,
    temperature: f32
) -> *mut c_char
{
    let output = if text.is_null() || network_path.is_null()
    {
        String::new()
    } else
    {
        let network_path = unsafe{ CStr::from_ptr(network_path) };
        let network_path = network_path.to_str().unwrap();

        let text = unsafe{ CStr::from_ptr(text) };
        let text = text.to_str().unwrap();

        let mut network: NeuralNetwork<MatrixWrapper, CharDictionary> =
            NeuralNetwork::load(&network_path).unwrap();

        network.predict_text(text, amount as usize, temperature)
    };

    CString::new(output).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn predict_free(text: *mut c_char)
{
    unsafe{ CString::from_raw(text) };
}
