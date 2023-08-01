use std::{
    f32,
    mem,
    slice,
    borrow::Borrow,
    io::{self, Read},
    fs::File,
    path::Path
};

use arrayfire::Array;

use serde::{Serialize, Deserialize, de::DeserializeOwned};

// #[allow(unused_imports)]
// use rnn::{RNN, RNNGradients};

#[allow(unused_imports)]
use gru::{GRU, GRUGradients, GPUGradientsInfo, GPUGradientInfo, GRUOutput, GPUGRU};

use super::word_vectorizer::{NetworkDictionary, WordVectorizer, VectorWord, WordDictionary};

pub use containers::{
    WeightsContainer,
    WeightsIterValue,
    LayerContainer,
    SoftmaxedLayer,
    SoftmaxedArray
};

// mod rnn;
mod gru;

pub mod containers;


pub const HIDDEN_AMOUNT: usize = 10;

impl Into<GPUGradientInfo> for &GradientInfo<LayerContainer>
{
    fn into(self) -> GPUGradientInfo
    {
        let m = self.m.as_arrayfire();
        let v = self.v.as_arrayfire();

        GPUGradientInfo{m, v}
    }
}

impl GradientInfo<LayerContainer>
{
    pub fn copy_gradients_from(&mut self, value: &GPUGradientInfo)
    {
        debug_assert!(self.m.len() == value.m.elements());
        debug_assert!(self.v.len() == value.v.elements());

        let GPUGradientInfo{
            m,
            v
        } = value;

        self.m = m.clone().into();
        self.v = v.clone().into();
    }
}

impl Into<GPUGradientInfo> for &GradientInfo<WeightsContainer>
{
    fn into(self) -> GPUGradientInfo
    {
        let m = self.m.as_arrayfire();
        let v = self.v.as_arrayfire();

        GPUGradientInfo{m, v}
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct GradientInfo<T>
{
    m: T,
    v: T
}

impl GradientInfo<WeightsContainer>
{
    pub fn new_weights(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            m: WeightsContainer::new(previous_size, this_size),
            v: WeightsContainer::new(previous_size, this_size)
        }
    }
}

impl GradientInfo<WeightsContainer>
{
    pub fn copy_gradients_from(&mut self, value: &GPUGradientInfo)
    {
        debug_assert!(self.m.total_len() == value.m.elements());
        debug_assert!(self.v.total_len() == value.v.elements());

        let GPUGradientInfo{
            m,
            v
        } = value;

        self.m = self.m.new_from(m);
        self.v = self.v.new_from(v);
    }
}

impl GradientInfo<LayerContainer>
{
    pub fn new_layers(size: usize) -> Self
    {
        Self{
            m: LayerContainer::new(size),
            v: LayerContainer::new(size)
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct GradientsInfo
{
    pub input_update_gradients: GradientInfo<WeightsContainer>,
    pub input_reset_gradients: GradientInfo<WeightsContainer>,
    pub input_activation_gradients: GradientInfo<WeightsContainer>,
    pub hidden_update_gradients: GradientInfo<WeightsContainer>,
    pub hidden_reset_gradients: GradientInfo<WeightsContainer>,
    pub hidden_activation_gradients: GradientInfo<WeightsContainer>,
    pub update_bias_gradients: GradientInfo<LayerContainer>,
    pub reset_bias_gradients: GradientInfo<LayerContainer>,
    pub activation_bias_gradients: GradientInfo<LayerContainer>,
    pub output_gradients: GradientInfo<WeightsContainer>
}

impl GradientsInfo
{
    pub fn new(word_vector_size: usize) -> Self
    {
        Self{
        	input_update_gradients: GradientInfo::new_weights(word_vector_size, HIDDEN_AMOUNT),
        	input_reset_gradients: GradientInfo::new_weights(word_vector_size, HIDDEN_AMOUNT),
        	input_activation_gradients: GradientInfo::new_weights(word_vector_size, HIDDEN_AMOUNT),
        	hidden_update_gradients: GradientInfo::new_weights(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
        	hidden_reset_gradients: GradientInfo::new_weights(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
        	hidden_activation_gradients: GradientInfo::new_weights(HIDDEN_AMOUNT, HIDDEN_AMOUNT),
            update_bias_gradients: GradientInfo::new_layers(HIDDEN_AMOUNT),
            reset_bias_gradients: GradientInfo::new_layers(HIDDEN_AMOUNT),
            activation_bias_gradients: GradientInfo::new_layers(HIDDEN_AMOUNT),
            output_gradients: GradientInfo::new_weights(HIDDEN_AMOUNT, word_vector_size)
        }
    }

    pub fn as_arrayfire(&self) -> GPUGradientsInfo
    {
        GPUGradientsInfo{
			input_update_gradients: (&self.input_update_gradients).into(),
			input_reset_gradients: (&self.input_reset_gradients).into(),
			input_activation_gradients: (&self.input_activation_gradients).into(),
			hidden_update_gradients: (&self.hidden_update_gradients).into(),
			hidden_reset_gradients: (&self.hidden_reset_gradients).into(),
			hidden_activation_gradients: (&self.hidden_activation_gradients).into(),
			update_bias_gradients: (&self.update_bias_gradients).into(),
			reset_bias_gradients: (&self.reset_bias_gradients).into(),
			activation_bias_gradients: (&self.activation_bias_gradients).into(),
			output_gradients: (&self.output_gradients).into()
        }
    }

    fn gradient_to_change(
        gradient_info: &mut GradientInfo<WeightsContainer>,
        gradient: WeightsContainer,
        hyper: &AdamHyperparams
    ) -> WeightsContainer
    {
        gradient_info.m = &gradient_info.m * hyper.b1 + &gradient * (1.0 - hyper.b1);
        gradient_info.v = &gradient_info.v * hyper.b2 + (&gradient * &gradient) * (1.0 - hyper.b2);

        let a_t = hyper.a * hyper.one_minus_b2_t.sqrt() / hyper.one_minus_b1_t;

        (&gradient_info.m * -a_t) / (gradient_info.v.sqrt() + hyper.epsilon)
    }

    fn gradient_bias_to_change(
        gradient_info: &mut GradientInfo<LayerContainer>,
        gradient: LayerContainer,
        hyper: &AdamHyperparams
    ) -> LayerContainer
    {
        gradient_info.m = &gradient_info.m * hyper.b1 + &gradient * (1.0 - hyper.b1);
        gradient_info.v = &gradient_info.v * hyper.b2 + (&gradient * &gradient) * (1.0 - hyper.b2);

        let a_t = hyper.a * hyper.one_minus_b2_t.sqrt() / hyper.one_minus_b1_t;

        (&gradient_info.m * -a_t) / (gradient_info.v.sqrt() + hyper.epsilon)
    }
}

pub struct InputOutput<T>
{
    container: Vec<T>
}

impl InputOutput<Array<f32>>
{
    pub fn batch<V, F>(
        values: &[V],
        mut f: F,
        mut start: usize,
        size: usize,
        steps: usize
    ) -> (Array<f32>, Array<f32>)
    where
        V: Copy + Default,
        F: FnMut(&V) -> Array<f32>
    {
        let max_len = values.len();
        let advance = |start: &mut usize| -> bool
        {
            *start += steps;

            *start >= max_len
        };

        let mut output = Self::batch_slice(values, &mut f, start, steps);
        if advance(&mut start)
        {
            return output;
        }

        for _ in 0..(size - 1)
        {
            let values = Self::batch_slice(values, &mut f, start, steps);

            output = Self::tuple_joiner(output, values, 2);

            if advance(&mut start)
            {
                break;
            }
        }

        output
    }

    fn joiner(acc: &Array<f32>, v: &Array<f32>, dim: i32) -> Array<f32>
    {
        arrayfire::join(dim, acc, v)
    }

    fn tuple_joiner(
        acc: (Array<f32>, Array<f32>),
        v: (Array<f32>, Array<f32>),
        dim: i32
    ) -> (Array<f32>, Array<f32>) 
    {
        let (acc_a, acc_b) = acc;
        let (v_a, v_b) = v;

        (Self::joiner(&acc_a, &v_a, dim), Self::joiner(&acc_b, &v_b, dim))
    }

    fn batch_slice<V, F>(
        values: &[V],
        mut f: F,
        start: usize,
        steps: usize
    ) -> (Array<f32>, Array<f32>)
    where
        V: Copy + Default,
        F: FnMut(&V) -> Array<f32>
    {
        let full_length = steps + 1;

        let slice_end = (start + full_length).min(values.len());
        let this_slice = &values[start..slice_end];

        let tuple_joiner_one = |acc, v|
        {
            Self::tuple_joiner(acc, v, 1)
        };

        if this_slice.len() == full_length
        {
            InputOutputIter::new(this_slice.iter().map(f)).reduce(tuple_joiner_one).unwrap()
        } else
        {
            let pad_amount = full_length - this_slice.len();

            let iter = this_slice.iter().copied().chain((0..pad_amount)
                .map(|_| V::default()))
                .map(|v| f(&v));

            InputOutputIter::new(iter).reduce(tuple_joiner_one).unwrap()
        }
    }
}

impl<T> InputOutput<T>
{
    #[allow(dead_code)]
    pub fn values_slice<V, F>(
        values: &[V],
        f: F,
        start: usize,
        size: usize
    ) -> Self
    where
        F: FnMut(&V) -> T
    {
        let slice_end = (start + size + 1).min(values.len());
        let this_slice = &values[start..slice_end];

        debug_assert!(this_slice.len() > 1);

        Self::new(this_slice.iter().map(f).collect())
    }

    pub fn new(container: Vec<T>) -> Self
    {
        Self{container}
    }

    #[allow(dead_code)]
    pub fn iter<'a>(&'a self) -> InputOutputIter<slice::Iter<'a, T>, &T>
    {
        InputOutputIter::new(self.container.iter())
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize
    {
        self.container.len() - 1
    }
}

#[derive(Clone)]
pub struct InputOutputIter<I, T>
{
    previous: T,
    inputs: I
}

mod input_output_associated
{
    use std::borrow::Borrow;
    use arrayfire::Array;

    pub fn join_array<IterT, Iter>(mut iter: Iter) -> (Array<f32>, Array<f32>)
    where
        IterT: Borrow<Array<f32>>,
        Iter: Iterator<Item=(IterT, IterT)>
    {
        let (a, b) = iter.next().unwrap();
        let (mut a, mut b): (Array<f32>, Array<f32>) = (a.borrow().clone(), b.borrow().clone());

        while let Some((new_a, new_b)) = iter.next()
        {
            a = arrayfire::join(1, &a, new_a.borrow());
            b = arrayfire::join(1, &b, new_b.borrow());
        }

        (a, b)
    }
}

impl<I, T> InputOutputIter<I, T>
where
    I: Iterator<Item=T>
{
    pub fn new(mut inputs: I) -> Self
    {
        Self{
            previous: inputs.next().expect("input must not be empty"),
            inputs
        }
    }
}

impl<I, T> InputOutputIter<I, T>
where
    T: Borrow<Array<f32>> + Clone,
    I: Iterator<Item=T>
{
    pub fn join_array(self) -> (Array<f32>, Array<f32>)
    {
        input_output_associated::join_array(self)
    }
}

impl<I, T> Iterator for InputOutputIter<I, T>
where
    T: Clone,
    I: Iterator<Item=T>
{
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item>
    {
        let input = self.inputs.next();

        match input
        {
            None => None,
            Some(input) =>
            {
                let out = (mem::replace(&mut self.previous, input.clone()), input);

                Some(out)
            }
        }
    }
}

impl<I, T> ExactSizeIterator for InputOutputIter<I, T>
where
    T: Clone,
    I: Iterator<Item=T> + ExactSizeIterator
{
    fn len(&self) -> usize
    {
        self.inputs.len()
    }
}

struct Predictor<'a, D>
{
    dictionary: &'a mut D,
    words: Vec<LayerContainer>,
    predicted: Vec<u8>,
    temperature: f32,
    predict_amount: usize
}

impl<'a, D> Predictor<'a, D>
where
    D: NetworkDictionary
{
    pub fn new(
        dictionary: &'a mut D,
        words: Vec<LayerContainer>,
        temperature: f32,
        predict_amount: usize
    ) -> Self
    {
        Self{
            dictionary,
            words,
            predicted: Vec::with_capacity(predict_amount),
            temperature,
            predict_amount
        }
    }

    // uses the cpu
    pub fn predict_bytes(mut self, network: &GRU) -> Box<[u8]>
    {
        let input_amount = self.words.len();

        let mut previous_hidden = LayerContainer::new(self.dictionary.words_amount());
        for i in 0..(input_amount + self.predict_amount)
        {
            debug_assert!(self.words.len() < i);
            let this_input = unsafe{ self.words.get_unchecked(i) };

            let GRUOutput{
                output,
                hidden,
                ..
            } = network.feedforward_cpu_single(&previous_hidden, this_input);
            previous_hidden = hidden;

            if i >= (input_amount - 1)
            {
                let word = SoftmaxedLayer::pick_weighed_associated(&output, self.temperature);
                let word = VectorWord::from_raw(word);

                self.words.push(self.dictionary.word_to_layer(word));

                self.predicted.extend(self.dictionary.word_to_bytes(word).into_iter());
            }
        }

        self.predicted.into_boxed_slice()
    }
}

pub struct TrainingInfo
{
    pub epochs: usize,
    pub batch_start: usize,
    pub batch_size: usize,
    pub steps_num: usize,
    pub calculate_accuracy: bool,
    pub ignore_loss: bool
}

#[derive(Serialize, Deserialize)]
pub struct AdamHyperparams
{
    pub a: f32,
    pub b1: f32,
    pub b2: f32,
    pub epsilon: f32,
    pub t: i32,
    pub one_minus_b1_t: f32,
    pub one_minus_b2_t: f32

}

impl AdamHyperparams
{
    pub fn new() -> Self
    {
        let mut this = Self{
            a: 0.001,
            b1: 0.9,
            b2: 0.999,
            epsilon: 10e-8,
            t: 1,
            one_minus_b1_t: 0.0,
            one_minus_b2_t: 0.0
        };

        this.update_t_vars();

        this
    }

    fn update_t_vars(&mut self)
    {
        self.one_minus_b1_t = 1.0 - self.b1.powi(self.t);
        self.one_minus_b2_t = 1.0 - self.b2.powi(self.t);
    }

    pub fn advance_time(&mut self)
    {
        self.t += 1;

        self.update_t_vars();
    }
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork<D=WordDictionary>
{
    dictionary: D,
    network: GRU,
    gradients_info: GradientsInfo,
    hyper: AdamHyperparams
}

impl<D: NetworkDictionary> NeuralNetwork<D>
where
    D: Serialize + DeserializeOwned
{
    pub fn new(dictionary: D) -> Self
    {
        let words_vector_size = dictionary.words_amount();
        let network = GRU::new(words_vector_size);

        let gradients_info = GradientsInfo::new(words_vector_size);

        let hyper = AdamHyperparams::new();

        Self{dictionary, network, gradients_info, hyper}
    }

    pub fn save<P: AsRef<Path>>(&self, path: P)
    {
        let writer = File::create(path).unwrap();

        ciborium::into_writer(self, writer).unwrap();
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ciborium::de::Error<io::Error>>
    {
        let reader = File::open(path)?;

        ciborium::from_reader(reader)
    }

    pub fn apply_gradients(&mut self, gradients: GRUGradients)
    {
        let GRUGradients{
            input_update_gradients,
            input_reset_gradients,
            input_activation_gradients,
            hidden_update_gradients,
            hidden_reset_gradients,
            hidden_activation_gradients,
            update_bias_gradients,
            reset_bias_gradients,
            activation_bias_gradients,
            output_gradients
        } = gradients;

        let hyper = &mut self.hyper;

        self.network.input_update_weights += GradientsInfo::gradient_to_change(
            &mut self.gradients_info.input_update_gradients,
            input_update_gradients,
            hyper
        );

        self.network.input_reset_weights += GradientsInfo::gradient_to_change(
			&mut self.gradients_info.input_reset_gradients,
			input_reset_gradients,
            hyper
		);
        
        self.network.input_activation_weights += GradientsInfo::gradient_to_change(
			&mut self.gradients_info.input_activation_gradients,
			input_activation_gradients,
            hyper
		);

        self.network.hidden_update_weights += GradientsInfo::gradient_to_change(
			&mut self.gradients_info.hidden_update_gradients,
			hidden_update_gradients,
            hyper
		);

        self.network.hidden_reset_weights += GradientsInfo::gradient_to_change(
			&mut self.gradients_info.hidden_reset_gradients,
			hidden_reset_gradients,
            hyper
		);
        
        self.network.hidden_activation_weights += GradientsInfo::gradient_to_change(
			&mut self.gradients_info.hidden_activation_gradients,
			hidden_activation_gradients,
            hyper
		);
        
        self.network.update_biases += GradientsInfo::gradient_bias_to_change(
			&mut self.gradients_info.update_bias_gradients,
			update_bias_gradients,
            hyper
		);
        
        self.network.reset_biases += GradientsInfo::gradient_bias_to_change(
			&mut self.gradients_info.reset_bias_gradients,
			reset_bias_gradients,
            hyper
		);
        
        self.network.activation_biases += GradientsInfo::gradient_bias_to_change(
			&mut self.gradients_info.activation_bias_gradients,
			activation_bias_gradients,
            hyper
		);
        
        self.network.output_weights += GradientsInfo::gradient_to_change(
			&mut self.gradients_info.output_gradients,
			output_gradients,
            hyper
		);

        hyper.advance_time();
    }

    pub fn input_expected_from_text(
        &mut self,
        text: impl Read
    ) -> Vec<VectorWord>
    {
        let word_vectorizer = WordVectorizer::new(&mut self.dictionary, text);

        word_vectorizer.collect()
    }

    pub fn test_loss(&mut self, file: impl Read, calculate_accuracy: bool)
    {
        let inputs = self.input_expected_from_text(file);

        let gpu_adapter = self.network.gpu_adapter(&self.gradients_info);

        self.test_loss_inner(&gpu_adapter, &inputs, calculate_accuracy);
    }

    pub fn test_loss_cpu(&mut self, file: impl Read, calculate_accuracy: bool)
    {
        let inputs = self.input_expected_from_text(file);

        self.test_loss_cpu_inner(&inputs, calculate_accuracy);
    }


    fn test_loss_cpu_inner(
        &self,
        inputs: &[VectorWord],
        calculate_accuracy: bool
    )
    {
        let input_outputs = InputOutputIter::new(
            inputs.iter().map(|word|
            {
                self.dictionary.word_to_layer(*word)
            })
        );

        if calculate_accuracy
        {
            let accuracy = self.network.accuracy(input_outputs);

            println!("accuracy: {}%", accuracy * 100.0);
        } else
        {
            let loss = self.network.loss(input_outputs);

            println!("loss: {loss}");
        }
    }

    fn test_loss_inner(
        &self,
        gpu_adapter: &GPUGRU,
        testing_inputs: &[VectorWord],
        calculate_accuracy: bool
    )
    {
        let input_outputs = InputOutputIter::new(
            testing_inputs.iter().map(|word|
            {
                self.dictionary.word_to_array(*word)
            })
        ).join_array();

        if calculate_accuracy
        {
            let accuracy = gpu_adapter.accuracy(input_outputs);

            println!("accuracy: {}%", accuracy * 100.0);
        } else
        {
            let loss = gpu_adapter.loss(input_outputs);

            println!("loss: {loss}");
        }
    }

    pub fn train<R: Read>(
        &mut self,
        info: TrainingInfo,
        testing_data: Option<R>,
        text: impl Read
    )
    {
        let TrainingInfo{
            batch_start,
            batch_size,
            steps_num,
            epochs,
            calculate_accuracy,
            ignore_loss
        } = info;

        let batch_step = batch_size * steps_num;
        let mut batch_start = batch_start * batch_step;

        let mut gpu_adapter = self.network.gpu_adapter(&self.gradients_info);

        let inputs = self.input_expected_from_text(text);
        let testing_inputs = if info.ignore_loss
        {
            Vec::new()
        } else
        {
            match testing_data
            {
                None => inputs.clone(),
                Some(testing_data) =>
                {
                    self.input_expected_from_text(testing_data)
                }
            }
        };

        println!("batch size: {batch_size}");
        println!("steps amount: {steps_num}");
        
        let epochs_per_input = (inputs.len() / batch_step).max(1);
        println!("calculate loss every {epochs_per_input} epochs");

        let output_loss = |network: &NeuralNetwork<D>, gpu_adapter: &GPUGRU|
        {
            if ignore_loss
            {
                return;
            }

            network.test_loss_inner(gpu_adapter, &testing_inputs, calculate_accuracy);
        };

        // whats an epoch? cool word is wut it is
        // at some point i found out wut it was (going over the whole training data once)
        // but i dont rly feel like changing a silly thing like that
        for epoch in 0..epochs
        {
            eprintln!("epoch: {epoch}");

            let input_vectorizer = |dictionary: &D, word: &VectorWord|
            {
                dictionary.word_to_array(*word)
            };

            let print_loss = (epoch % epochs_per_input) == epochs_per_input - 1;
            if print_loss
            {
                output_loss(self, &gpu_adapter);
            }

            let mut batch_gradients = None;

            let values = InputOutput::batch(
                &inputs,
                |word| input_vectorizer(&self.dictionary, word),
                batch_start,
                batch_size,
                steps_num
            );

            // it may not be the full batch_size cuz the file ended
            for b_i in 0..(values.0.dims()[2] as i64)
            {
                // eprintln!("({}, {})", values.0.dims(), values.1.dims());
                let values = (arrayfire::slice(&values.0, b_i), arrayfire::slice(&values.1, b_i));

                let gradients = gpu_adapter.gradients::<true>(values);

                if batch_gradients.is_none()
                {
                    batch_gradients = Some(gradients);
                } else
                {
                    batch_gradients.as_mut().map(|batch_gradients| *batch_gradients += gradients);
                }
            }

            batch_start += batch_size * steps_num;
            if batch_start >= (inputs.len() - 1)
            {
                batch_start = 0;
            }

            let gradients = batch_gradients.unwrap() / batch_size as f32;

            gpu_adapter.apply_gradients(gradients, &mut self.hyper);
        }

        output_loss(self, &gpu_adapter);

        self.transfer_gradient_info(&gpu_adapter);
        self.network.transfer_weights(gpu_adapter);
    }

    pub fn train_cpu<R: Read>(
        &mut self,
        info: TrainingInfo,
        testing_data: Option<R>,
        text: impl Read
    )
    {
        let TrainingInfo{
            batch_start,
            batch_size,
            steps_num,
            epochs,
            calculate_accuracy,
            ignore_loss
        } = info;

        let batch_step = batch_size * steps_num;
        let mut batch_start = batch_start * batch_step;

        let inputs = self.input_expected_from_text(text);
        let testing_inputs = if info.ignore_loss
        {
            Vec::new()
        } else
        {
            match testing_data
            {
                None => inputs.clone(),
                Some(testing_data) =>
                {
                    self.input_expected_from_text(testing_data)
                }
            }
        };

        println!("batch size: {batch_size}");
        println!("steps amount: {steps_num}");
        
        let epochs_per_input = (inputs.len() / batch_step).max(1);
        println!("calculate loss every {epochs_per_input} epochs");

        let output_loss = |network: &NeuralNetwork<D>|
        {
            if ignore_loss
            {
                return;
            }

            network.test_loss_cpu_inner(&testing_inputs, calculate_accuracy);
        };

        // whats an epoch? cool word is wut it is
        // at some point i found out wut it was (going over the whole training data once)
        // but i dont rly feel like changing a silly thing like that
        for epoch in 0..epochs
        {
            eprintln!("epoch: {epoch}");

            let print_loss = (epoch % epochs_per_input) == epochs_per_input - 1;
            if print_loss
            {
                output_loss(self);
            }

            let mut batch_gradients = None;

            for _ in 0..batch_size
            {
                let values = InputOutput::values_slice(
                    &inputs,
                    |word| self.dictionary.word_to_layer(*word),
                    batch_start,
                    steps_num
                );

                let gradients = self.network.gradients_cpu::<true>(values.iter());

                if batch_gradients.is_none()
                {
                    batch_gradients = Some(gradients);
                } else
                {
                    batch_gradients.as_mut().map(|batch_gradients| *batch_gradients += gradients);
                }


                batch_start += steps_num;
                if batch_start >= (inputs.len() - 1)
                {
                    batch_start = 0;
                }
            }

            let mut gradients = batch_gradients.unwrap();
            gradients /= batch_size as f32;

            self.apply_gradients(gradients);
        }

        output_loss(self);
    }

    fn transfer_gradient_info(&mut self, gpugru: &GPUGRU)
    {
        let gradients = gpugru.gradients_info();

		self.gradients_info.input_update_gradients.copy_gradients_from(
			&gradients.input_update_gradients
		);

		self.gradients_info.input_reset_gradients.copy_gradients_from(
			&gradients.input_reset_gradients
		);

		self.gradients_info.input_activation_gradients.copy_gradients_from(
			&gradients.input_activation_gradients
		);

		self.gradients_info.hidden_update_gradients.copy_gradients_from(
			&gradients.hidden_update_gradients
		);

		self.gradients_info.hidden_reset_gradients.copy_gradients_from(
			&gradients.hidden_reset_gradients
		);

		self.gradients_info.hidden_activation_gradients.copy_gradients_from(
			&gradients.hidden_activation_gradients
		);

		self.gradients_info.update_bias_gradients.copy_gradients_from(
			&gradients.update_bias_gradients
		);

		self.gradients_info.reset_bias_gradients.copy_gradients_from(
			&gradients.reset_bias_gradients
		);

		self.gradients_info.activation_bias_gradients.copy_gradients_from(
			&gradients.activation_bias_gradients
		);

		self.gradients_info.output_gradients.copy_gradients_from(
			&gradients.output_gradients
		);
    }

    #[allow(dead_code)]
    pub fn predict(&mut self, text: &str, amount: usize, temperature: f32) -> String
    {
        let word_vectorizer = WordVectorizer::new(&mut self.dictionary, text.as_bytes());

        let predictor = {
            let words = word_vectorizer.collect::<Vec<_>>();

            let words = words.into_iter().map(|v|
            {
                self.dictionary.word_to_layer(v)
            }).collect::<Vec<_>>();

            Predictor::new(&mut self.dictionary, words, temperature, amount)
        };

        let output = predictor.predict_bytes(&self.network).into_iter().copied()
            .filter(|&c| c != b'\0').collect::<Vec<_>>();
        
        String::from_utf8_lossy(&output).to_string()
    }
}

#[cfg(test)]
mod tests
{
    use super::*;
    
    #[allow(unused_imports)]
    use arrayfire::af_print;
    
    fn close_enough(a: f32, b: f32, epsilon: f32) -> bool
    {
        if (a == b) || ((a.min(b) == -0.0) && (a.max(b) == 0.0))
        {
            return true;
        }

        ((a - b).abs() / (a.abs() + b.abs())) < epsilon
    }

    fn test_dictionary() -> WordDictionary
    {
        let _words = "test.testing.tester.epic.cool.true";

        // WordDictionary::build(words.as_bytes())
        
        let words = "testing test - a b c 6 1 A o l d e f g h i j s o r s m";
        WordDictionary::no_defaults(words.as_bytes())
    }

    fn test_network() -> NeuralNetwork
    {
        NeuralNetwork::new(test_dictionary())
    }

    fn test_texts_many() -> Vec<&'static str>
    {
        vec![
            "testing tests or sm",
            "abcdefghij",
            "coolllllll",
            "AAAAAAAAAA"
        ]
    }

    fn test_texts_one() -> Vec<&'static str>
    {
        vec!["testing", "a", "6", "A"]
    }

    fn test_texts_two() -> Vec<&'static str>
    {
        vec!["testing-", "ab", "61", "AA"]
    }

    fn test_texts_three() -> Vec<&'static str>
    {
        vec!["testing test", "abc", "611", "AAA"]
    }

    fn test_input_words(
        test_texts: Vec<&'static str>,
        network: &mut NeuralNetwork
    ) -> Vec<VectorWord>
    {
        test_texts.into_iter().flat_map(|text|
        {
            network.input_expected_from_text(text.as_bytes())
        }).collect()
    }

    fn test_input_outputs(
        test_texts: Vec<&'static str>,
        network: &mut NeuralNetwork
    ) -> Vec<InputOutput<LayerContainer>>
    {
        test_texts.into_iter().map(|text|
        {
            let inputs = network.input_expected_from_text(text.as_bytes());

            InputOutput::new(inputs.iter().map(|word: &VectorWord|
            {
                network.dictionary.word_to_layer(*word)
            }).collect())
        }).collect()
    }

    #[test]
    fn adam_correct()
    {
        let mut old_weight = vec![3.21, 7.65];

        let mut m = vec![0.0, 0.0];
        let mut v = vec![0.0, 0.0];
        
        let mut g = vec![3.1_f32, -0.8_f32];

        let mut t = 1;

        for _ in 0..2
        {
            let a = 0.001;
            let b1 = 0.9;
            let b2 = 0.999;

            let epsilon = 10e-8;

            let adam_g = {
                let mut gradient_info = GradientInfo{
                    m: WeightsContainer::from_raw(m.clone().into_boxed_slice(), 2, 1),
                    v: WeightsContainer::from_raw(v.clone().into_boxed_slice(), 2, 1)
                };

                let gradient = WeightsContainer::from_raw(g.clone().into_boxed_slice(), 2, 1);

                let hyper = AdamHyperparams{
                    a,
                    b1,
                    b2,
                    epsilon,
                    t,
                    one_minus_b1_t: 1.0 - b1.powi(t),
                    one_minus_b2_t: 1.0 - b2.powi(t)
                };

                let change = GradientsInfo::gradient_to_change(
                    &mut gradient_info,
                    gradient.clone(),
                    &hyper
                );

                WeightsContainer::from_raw(old_weight.clone().into_boxed_slice(), 2, 1) + change
            };

            m = vec![
                b1 * m[0] + (1.0 - b1) * g[0],
                b1 * m[1] + (1.0 - b1) * g[1]
            ];

            v = vec![
                b2 * v[0] + (1.0 - b2) * g[0].powi(2),
                b2 * v[1] + (1.0 - b2) * g[1].powi(2)
            ];

            let m_hat = vec![
                m[0] / (1.0 - b1.powi(t)),
                m[1] / (1.0 - b1.powi(t))
            ];

            let v_hat = vec![
                v[0] / (1.0 - b2.powi(t)),
                v[1] / (1.0 - b2.powi(t))
            ];

            let new_weight = vec![
                old_weight[0] - a * m_hat[0] / (v_hat[0].sqrt() + epsilon),
                old_weight[1] - a * m_hat[1] / (v_hat[1].sqrt() + epsilon)
            ];

            if t == 1
            {
                assert_eq!(new_weight[0], 3.2090000000322581);
                assert_eq!(new_weight[1], 7.6509999998750000);

                let mut adam_g = adam_g.iter();
                assert_eq!(new_weight[0], *adam_g.next().unwrap());
                assert_eq!(new_weight[1], *adam_g.next().unwrap());
            } else
            {
                assert_eq!(new_weight[0], 3.2080334761008376);
                assert_eq!(new_weight[1], 7.6518864757914067);

                let mut adam_g = adam_g.iter();
                assert_eq!(new_weight[0], *adam_g.next().unwrap());
                assert_eq!(new_weight[1], *adam_g.next().unwrap());
            }

            t += 1;

            old_weight = new_weight;
            g = vec![
                g[0] - 1.1,
                g[1] - 2.34
            ];
        }
    }

    #[test]
    fn matrix_multiplication()
    {
        let v = LayerContainer::from(vec![5.0, 0.0, 5.0, 7.0]);

        let m = WeightsContainer::from_raw(vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        ].into_boxed_slice(), 4, 4);

        assert_eq!(
            m.mul(&v),
            LayerContainer::from(vec![48.0, 116.0, 184.0, 252.0])
        );

        assert_eq!(
            m.mul_transposed(&v),
            LayerContainer::from(vec![141.0, 158.0, 175.0, 192.0])
        );
    }

    #[test]
    fn softmax()
    {
        let test_layer = LayerContainer::from_iter([1.0, 2.0, 8.0].iter().cloned());

        let softmaxed = SoftmaxedLayer::new(test_layer);

        softmaxed.0.iter().zip([0.001, 0.002, 0.997].iter()).for_each(|(softmaxed, correct)|
        {
            assert!(
                close_enough(*softmaxed, *correct, 0.2),
                "softmaxed: {}, correct: {}",
                *softmaxed,
                *correct
            );
        });
    }

    #[test]
    fn loss()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_many(), &mut network);

        let l = inputs.len() as f32;
        let this_loss = inputs.into_iter().map(|input|
        {
            network.network.loss(input.iter().map(|(a, b)| (a.clone(), b.clone())))
        }).sum::<f32>() / l;

        let predicted_loss = (network.dictionary.words_amount() as f32).ln();

        assert!(
            close_enough(this_loss, predicted_loss, 0.1),
            "this_loss: {this_loss}, predicted_loss: {predicted_loss}"
        );
    }

    #[ignore]
    #[test]
    fn gradients_check_many()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_many(), &mut network);

        gradients_check(&mut network, inputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_one()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_one(), &mut network);

        gradients_check(&mut network, inputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_two()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_two(), &mut network);

        gradients_check(&mut network, inputs);
    }

    #[ignore]
    #[test]
    fn gradients_check_three()
    {
        let mut network = test_network();
        let inputs = test_input_outputs(test_texts_three(), &mut network);

        gradients_check(&mut network, inputs);
    }

    #[test]
    fn batching_one()
    {
        let mut network = test_network();

        let texts = vec![
            "testing tests or sm",
            "abcdefghij",
            "coolllllll",
            "AAAAAAAAAA"
        ];

        let inputs = test_input_words(texts, &mut network);

        let steps_num = 20;

        let values = InputOutput::batch(
            &inputs,
            |word| network.dictionary.word_to_array(*word),
            0,
            1,
            steps_num
        );

        let values_slice = InputOutput::values_slice(
            &inputs,
            |word| network.dictionary.word_to_array(*word),
            0,
            steps_num
        ).iter().join_array();

        let v = |a: Array<f32>|
        {
            let mut v = vec![0.0_f32; a.elements()];

            a.host(&mut v);

            v
        };

        eprintln!("inputs");
        v(values.0).into_iter().zip(v(values_slice.0).into_iter()).for_each(|(b, nb)|
        {
            assert_eq!(b, nb, "b: {b}, nb: {nb}");
        });

        eprintln!("outputs");
        v(values.1).into_iter().zip(v(values_slice.1).into_iter()).for_each(|(b, nb)|
        {
            assert_eq!(b, nb, "b: {b}, nb: {nb}");
        });
    }

    #[test]
    fn batching_many()
    {
        let mut network = test_network();
        let texts = vec![
            "testing tests or sm",
            "abcdefghij",
            "coolllllll",
            "AAAAAAAAAA"
        ];

        let inputs = test_input_words(texts, &mut network);

        let steps_num = 19;

        let values = InputOutput::batch(
            &inputs,
            |word| network.dictionary.word_to_array(*word),
            0,
            2,
            steps_num
        );

        let values_slice = {
            let v = |s| InputOutput::values_slice(
                &inputs,
                |word| network.dictionary.word_to_array(*word),
                s,
                steps_num
            ).iter().join_array();

            let (a0, b0) = v(0);
            let (a1, b1) = v(steps_num);

            (
                arrayfire::join(2, &a0, &a1),
                arrayfire::join(2, &b0, &b1)
            )
        };

        let v = |a: Array<f32>|
        {
            let mut v = vec![0.0_f32; a.elements()];

            a.host(&mut v);

            v
        };

        // af_print!("b0: {}", values.0);
        // af_print!("nb0: {}", values_slice.0);
        eprintln!("inputs");
        v(values.0).into_iter().zip(v(values_slice.0).into_iter()).for_each(|(b, nb)|
        {
            assert_eq!(b, nb, "b: {b}, nb: {nb}");
        });

        // af_print!("b1: {}", values.1);
        // af_print!("nb1: {}", values_slice.1);
        eprintln!("outputs");
        v(values.1).into_iter().zip(v(values_slice.1).into_iter()).for_each(|(b, nb)|
        {
            assert_eq!(b, nb, "b: {b}, nb: {nb}");
        });
    }

    fn print_status(description: &str, f: impl FnOnce())
    {
        print!("checking {description}: ⛔ ");
        f();
        println!("\rchecking {description}: ✔️");
    }

    fn gradients_check(
        network: &mut NeuralNetwork,
        inputs: Vec<InputOutput<LayerContainer>>
    )
    {
        let network = &mut network.network;
        inputs.into_iter().for_each(|input|
        {
            print_status("output gradients", ||
            {
                gru::tests::output_gradients_check(network, input.iter())
            });

            print_status("hidden update gradients", ||
            {
                gru::tests::hidden_update_gradients_check(network, input.iter())
            });

            print_status("hidden reset gradients", ||
            {
                gru::tests::hidden_reset_gradients_check(network, input.iter())
            });

            print_status("hidden activation gradients", ||
            {
                gru::tests::hidden_activation_gradients_check(network, input.iter())
            });

            print_status("update bias gradients", ||
            {
                gru::tests::update_bias_gradients_check(network, input.iter())
            });

            print_status("reset bias gradients", ||
            {
                gru::tests::reset_bias_gradients_check(network, input.iter())
            });

            print_status("activation bias gradients", ||
            {
                gru::tests::activation_bias_gradients_check(network, input.iter())
            });

            print_status("input update gradients", ||
            {
                gru::tests::input_update_gradients_check(network, input.iter())
            });

            print_status("input reset gradients", ||
            {
                gru::tests::input_reset_gradients_check(network, input.iter())
            });

            print_status("input activation gradients", ||
            {
                gru::tests::input_activation_gradients_check(network, input.iter())
            });
        });
    }
}
