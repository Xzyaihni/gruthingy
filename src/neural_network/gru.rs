use std::{
    f32,
    borrow::Borrow,
    collections::VecDeque,
    ops::{DivAssign, AddAssign}
};

use serde::{Serialize, Deserialize};

use crate::neural_network::{
    Softmaxer,
    AFType,
    LayerType,
    ScalarType,
    LayerInnerType,
    HIDDEN_AMOUNT,
    LAYERS_AMOUNT,
    LAYER_ACTIVATION
};


pub type GRUState<H> = Vec<H>;

pub struct GRUOutput<H, O>
{
    pub hidden: H,
    pub output: O
}

impl<H, O> GRUOutput<Vec<H>, O>
{
    pub fn into_state(self) -> GRUState<H>
    {
        self.hidden
    }
}

#[derive(Debug)]
pub struct GRUGradients
{
    pub input_update_gradients: LayerInnerType,
    pub input_reset_gradients: LayerInnerType,
    pub input_activation_gradients: LayerInnerType,
    pub hidden_update_gradients: LayerInnerType,
    pub hidden_reset_gradients: LayerInnerType,
    pub hidden_activation_gradients: LayerInnerType,
    pub update_bias_gradients: LayerInnerType,
    pub reset_bias_gradients: LayerInnerType,
    pub activation_bias_gradients: LayerInnerType,
    pub output_gradients: LayerInnerType
}

impl DivAssign<f32> for GRUGradients
{
    fn div_assign(&mut self, rhs: f32)
    {
		self.input_update_gradients /= rhs;
		self.input_reset_gradients /= rhs;
		self.input_activation_gradients /= rhs;
		self.hidden_update_gradients /= rhs;
		self.hidden_reset_gradients /= rhs;
		self.hidden_activation_gradients /= rhs;
		self.update_bias_gradients /= rhs;
		self.reset_bias_gradients /= rhs;
		self.activation_bias_gradients /= rhs;
		self.output_gradients /= rhs;
    }
}

impl AddAssign for GRUGradients
{
    fn add_assign(&mut self, rhs: Self)
    {
		self.input_update_gradients += rhs.input_update_gradients;
		self.input_reset_gradients += rhs.input_reset_gradients;
		self.input_activation_gradients += rhs.input_activation_gradients;
		self.hidden_update_gradients += rhs.hidden_update_gradients;
		self.hidden_reset_gradients += rhs.hidden_reset_gradients;
		self.hidden_activation_gradients += rhs.hidden_activation_gradients;
		self.update_bias_gradients += rhs.update_bias_gradients;
		self.reset_bias_gradients += rhs.reset_bias_gradients;
		self.activation_bias_gradients += rhs.activation_bias_gradients;
		self.output_gradients += rhs.output_gradients;
    }
}

#[derive(Debug)]
pub struct GRUFullGradients(pub VecDeque<GRUGradients>);

impl DivAssign<f32> for GRUFullGradients
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.0.iter_mut().for_each(|v| *v /= rhs);
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GRULayer
{
	pub input_update_weights: LayerType,
	pub input_reset_weights: LayerType,
	pub input_activation_weights: LayerType,
	pub hidden_update_weights: LayerType,
	pub hidden_reset_weights: LayerType,
	pub hidden_activation_weights: LayerType,
	pub update_biases: LayerType,
	pub reset_biases: LayerType,
	pub activation_biases: LayerType,
	pub output_weights: LayerType
}

impl GRULayer
{
    pub fn new(word_vector_size: usize) -> Self
    {
        let weights_init = |previous: f32|
        {
            let v = 1.0 / previous.sqrt();

            (fastrand::f32() * 2.0 - 1.0) * v
        };

        Self{
        	input_update_weights: LayerType::new_diff(LayerInnerType::new_with(
                HIDDEN_AMOUNT,
                word_vector_size,
                || weights_init(word_vector_size as f32)
            )),
        	input_reset_weights: LayerType::new_diff(LayerInnerType::new_with(
				HIDDEN_AMOUNT,
				word_vector_size,
				|| weights_init(word_vector_size as f32)
			)),
        	input_activation_weights: LayerType::new_diff(LayerInnerType::new_with(
				HIDDEN_AMOUNT,
				word_vector_size,
				|| weights_init(word_vector_size as f32)
			)),
        	hidden_update_weights: LayerType::new_diff(LayerInnerType::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f32)
			)),
        	hidden_reset_weights: LayerType::new_diff(LayerInnerType::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f32)
			)),
        	hidden_activation_weights: LayerType::new_diff(LayerInnerType::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f32)
			)),
            // initialize biases to 0 cuz i read somewhere thats good
            update_biases: LayerType::new_diff(LayerInnerType::new(HIDDEN_AMOUNT, 1)),
            reset_biases: LayerType::new_diff(LayerInnerType::new(HIDDEN_AMOUNT, 1)),
            activation_biases: LayerType::new_diff(LayerInnerType::new(HIDDEN_AMOUNT, 1)),
            output_weights: LayerType::new_diff(LayerInnerType::new_with(
                word_vector_size,
                HIDDEN_AMOUNT,
                || weights_init(HIDDEN_AMOUNT as f32)
            ))
        }
    }

    fn for_weights(&mut self, mut f: impl FnMut(&mut LayerType))
    {
		f(&mut self.input_update_weights);
		f(&mut self.input_reset_weights);
		f(&mut self.input_activation_weights);
		f(&mut self.hidden_update_weights);
		f(&mut self.hidden_reset_weights);
		f(&mut self.hidden_activation_weights);
		f(&mut self.update_biases);
		f(&mut self.reset_biases);
		f(&mut self.activation_biases);
		f(&mut self.output_weights);
    }

    pub fn clear(&mut self)
    {
        self.for_weights(|v| v.clear());
    }

    pub fn enable_gradients(&mut self)
    {
        self.for_weights(|v| v.enable_gradients());
    }

    pub fn disable_gradients(&mut self)
    {
        self.for_weights(|v| v.disable_gradients());
    }

    fn feedforward_single_untrans(
        &mut self,
        previous_hidden: Option<&LayerType>,
        input: &LayerType
    ) -> GRUOutput<LayerType, LayerType>
    {
        let mut update_gate = self.input_update_weights.matmul(input) + &self.update_biases;

        if let Some(previous_hidden) = previous_hidden
        {
            update_gate += self.hidden_update_weights.matmul(previous_hidden);
        }

        update_gate.sigmoid();

        let mut reset_gate = self.input_reset_weights.matmul(input) + &self.reset_biases;

        if let Some(previous_hidden) = previous_hidden
        {
            reset_gate += self.hidden_reset_weights.matmul(previous_hidden);
        }

        reset_gate.sigmoid();

        let mut activation_gate = self.input_activation_weights.matmul(input)
            + &self.activation_biases;

        if let Some(previous_hidden) = previous_hidden
        {
            let activation_v = &reset_gate * previous_hidden;
            activation_gate += self.hidden_activation_weights.matmul(activation_v);
        }

        activation_gate.tanh();

        let this_activation = &activation_gate * &update_gate;

        let hidden = if let Some(previous_hidden) = previous_hidden
        {
            ScalarType::new(1.0) - &update_gate * previous_hidden + this_activation
        } else
        {
            this_activation + ScalarType::new(1.0)
        };

        let output_untrans = self.output_weights.matmul(&hidden);

        GRUOutput{
            hidden,
            output: output_untrans
        }
    }

    pub fn feedforward_single(
        &mut self,
        previous_hidden: Option<&LayerType>,
        input: &LayerType
    ) -> GRUOutput<LayerType, LayerType>
    {
        let mut output = self.feedforward_single_untrans(previous_hidden, input);

        match LAYER_ACTIVATION
        {
            AFType::LeakyRelu =>
            {
                output.output.leaky_relu();
            },
            AFType::Tanh =>
            {
                output.output.tanh();
            }
        }

        output
    }

    pub fn feedforward_single_last(
        &mut self,
        previous_hidden: Option<&LayerType>,
        input: &LayerType,
        targets: LayerInnerType
    ) -> GRUOutput<LayerType, ScalarType>
    {
        let GRUOutput{
            hidden,
            output
        } = self.feedforward_single_untrans(previous_hidden, input);

        GRUOutput{
            hidden,
            output: output.softmax_cross_entropy(targets)
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GRU
{
    pub layers: Vec<GRULayer>,
    word_vector_size: usize
}

impl GRU
{
    pub fn new(word_vector_size: usize) -> Self
    {
        Self{
            layers: (0..LAYERS_AMOUNT).map(|_| GRULayer::new(word_vector_size)).collect(),
            word_vector_size
        }
    }

    pub fn clear(&mut self)
    {
        self.layers.iter_mut().for_each(|layer| layer.clear());
    }

    // oh my god wut am i even doing at this point its so over
    pub fn enable_gradients(&mut self)
    {
        self.layers.iter_mut().for_each(|layer|
        {
            layer.enable_gradients();
        });
    }

    pub fn disable_gradients(&mut self)
    {
        self.layers.iter_mut().for_each(|layer|
        {
            layer.disable_gradients();
        });
    }

    #[allow(dead_code)]
    pub fn parameters_amount(&self) -> u128
    {
        let i = self.word_vector_size as u128;
        let h = HIDDEN_AMOUNT as u128;
        let l = LAYERS_AMOUNT as u128;

        // i hope i calculated this right
        ((4 * i * h) + (3 * h * h) + (3 * h)) * l
    }

    #[allow(dead_code)]
    pub fn accuracy(
        &mut self,
        input: impl Iterator<Item=(LayerType, LayerType)>
    ) -> f32
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let amount = input.len();

        let f_output = self.predict(input.into_iter());

        Self::correct_guesses(
            f_output.into_iter(),
            output.into_iter()
        ) as f32 / amount as f32
    }

    fn correct_guesses<P, T>(
        predicted: impl Iterator<Item=P>,
        target: impl Iterator<Item=T>
    ) -> usize
    where
        P: Borrow<LayerInnerType>,
        T: Borrow<LayerType>
    {
        predicted.zip(target).map(|(predicted, target)|
        {
            let target_index = target.borrow().highest_index();
            if predicted.borrow().highest_index() == target_index
            {
                1
            } else
            {
                0
            }
        }).sum()
    }

    fn feedforward_single_inner<F, T>(
        &mut self,
        last_f: F,
        previous_hiddens: Option<Vec<LayerType>>,
        input: &LayerType
    ) -> GRUOutput<Vec<LayerType>, T>
    where
        F: FnOnce(&mut GRULayer, Option<&LayerType>, &LayerType) -> GRUOutput<LayerType, T>
    {
        let mut output: Option<T> = None;
        let mut last_output: Option<LayerType> = None;

        let mut hiddens = Vec::with_capacity(LAYERS_AMOUNT);

        for l_i in 0..LAYERS_AMOUNT
        {
            let input = last_output.as_ref().unwrap_or(input);

            debug_assert!(l_i < self.layers.len());
            let layer = unsafe{ self.layers.get_unchecked_mut(l_i) };

            let previous_hidden = unsafe{
                previous_hiddens.as_ref().map(|previous_hidden|
                {
                    previous_hidden.get_unchecked(l_i)
                })
            };

            if l_i == (LAYERS_AMOUNT - 1)
            {
                // last layer
                let GRUOutput{
                    hidden,
                    output: this_output
                } = last_f(layer, previous_hidden, input);

                output = Some(this_output);

                hiddens.push(hidden);

                // i like how rust cant figure out that the last index is the last iteration
                // without this
                break;
            } else
            {
                let GRUOutput{
                    hidden,
                    output: this_output
                } = layer.feedforward_single(
                    previous_hidden,
                    input
                );

                last_output = Some(this_output);

                hiddens.push(hidden);
            }
        }

        GRUOutput{
            hidden: hiddens,
            output: output.unwrap()
        }
    }

    pub fn feedforward_single(
        &mut self,
        previous_hiddens: Option<Vec<LayerType>>,
        input: &LayerType,
        targets: LayerInnerType
    ) -> GRUOutput<Vec<LayerType>, ScalarType>
    {
        self.feedforward_single_inner(|layer, previous_hidden, input|
        {
            layer.feedforward_single_last(
                previous_hidden,
                input,
                targets
            )
        }, previous_hiddens, input)
    }

    pub fn predict_single(
        &mut self,
        previous_hiddens: Option<Vec<LayerType>>,
        input: &LayerType,
        temperature: f32
    ) -> GRUOutput<Vec<LayerType>, LayerInnerType>
    {
        self.feedforward_single_inner(|layer, previous_hidden, input|
        {
            let GRUOutput{
                hidden,
                mut output
            } = layer.feedforward_single(
                previous_hidden,
                input
            );

            let mut output = output.value_take();

            Softmaxer::softmax_temperature(&mut output, temperature);

            GRUOutput{
                hidden,
                output
            }
        }, previous_hiddens, input)
    }

    #[allow(dead_code)]
    pub fn predict(
        &mut self,
        input: impl Iterator<Item=LayerType> + ExactSizeIterator
    ) -> Vec<LayerInnerType>
    {
        let mut outputs: Vec<LayerInnerType> = Vec::with_capacity(input.len());
        let mut previous_hiddens: Option<Vec<LayerType>> = None;

        for this_input in input
        {
            let GRUOutput{
                hidden,
                output: this_output
            } = self.predict_single(
                previous_hiddens.take(),
                &this_input,
                1.0
            );

            outputs.push(this_output);
            previous_hiddens = Some(hidden);
        }

        outputs
    }

    #[allow(dead_code)]
    pub fn feedforward(
        &mut self,
        input: impl Iterator<Item=(LayerType, LayerType)> + ExactSizeIterator
    ) -> ScalarType
    {
        let mut output: Option<ScalarType> = None;
        let mut previous_hiddens: Option<Vec<LayerType>> = None;

        for (this_input, mut this_output) in input
        {
            let GRUOutput{
                hidden,
                output: this_output
            } = self.feedforward_single(
                previous_hiddens.take(),
                &this_input,
                this_output.value_take()
            );

            if let Some(output) = output.as_mut()
            {
                *output += this_output;
            } else
            {
                output = Some(this_output)
            }

            previous_hiddens = Some(hidden);
        }

        output.unwrap()
    }

    pub fn gradients(
        &mut self,
        input: impl Iterator<Item=(LayerType, LayerType)> + ExactSizeIterator
    ) -> (f32, GRUFullGradients)
    {
        self.clear();

        let loss = self.feedforward(input);

        let loss_value = loss.value_clone();

        loss.calculate_gradients();

        let gradients = GRUFullGradients(
            self.layers.iter_mut().map(|layer|
            {
                GRUGradients{
                    input_update_gradients:
                        layer.input_update_weights.take_gradient(),
                    hidden_update_gradients:
                        layer.hidden_update_weights.take_gradient(),
                    update_bias_gradients:
                        layer.update_biases.take_gradient(),
                    input_reset_gradients:
                        layer.input_reset_weights.take_gradient(),
                    hidden_reset_gradients:
                        layer.hidden_reset_weights.take_gradient(),
                    reset_bias_gradients:
                        layer.reset_biases.take_gradient(),
                    input_activation_gradients:
                        layer.input_activation_weights.take_gradient(),
                    hidden_activation_gradients:
                        layer.hidden_activation_weights.take_gradient(),
                    activation_bias_gradients:
                        layer.activation_biases.take_gradient(),
                    output_gradients:
                        layer.output_weights.take_gradient(),
                }
            }).collect()
        );

        (loss_value, gradients)
    }
}

#[cfg(test)]
pub mod tests
{
    use super::*;

    #[allow(dead_code)]
    pub fn close_enough(a: f32, b: f32, epsilon: f32) -> bool
    {
        if (a == b) || ((a.min(b) == -0.0) && (a.max(b) == 0.0))
        {
            return true;
        }

        if a.signum() != b.signum()
        {
            return false;
        }

        ((a - b).abs() / (a.abs() + b.abs())) < epsilon
    }

    pub fn close_enough_loose(a: f32, b: f32, epsilon: f32) -> bool
    {
        if a == 0.0 || a == -0.0
        {
            return b.abs() < epsilon;
        }

        if b == 0.0 || b == -0.0
        {
            return a.abs() < epsilon;
        }

        ((a - b).abs() / (a.abs() + b.abs())) < epsilon
    }

    #[allow(dead_code)]
    pub fn close_enough_abs(a: f32, b: f32, epsilon: f32) -> bool
    {
        (a - b).abs() < epsilon
    }
}
