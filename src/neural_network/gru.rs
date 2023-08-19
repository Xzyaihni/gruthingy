use std::{
    f32,
    borrow::Borrow,
    collections::VecDeque,
    ops::{DivAssign, AddAssign}
};

use serde::{Serialize, Deserialize};

use crate::neural_network::{
    SoftmaxedLayer,
    AFType,
    LayerType,
    ScalarType,
    LayerInnerType,
    HIDDEN_AMOUNT,
    LAYERS_AMOUNT,
    LAYER_ACTIVATION,
    USE_DROPOUT
};


#[derive(Debug)]
pub struct GRUOutput
{
    pub hidden: LayerType,
    pub output: LayerType
}

pub struct GRUOutputLayer(pub Vec<GRUOutput>);

impl GRUOutputLayer
{
    pub fn with_capacity(capacity: usize) -> Self
    {
        Self(Vec::with_capacity(capacity))
    }

    pub fn push(&mut self, value: GRUOutput)
    {
        self.0.push(value);
    }

    pub fn last_output_ref(&self) -> &LayerType
    {
        &self.0.iter().rev().next().unwrap().output
    }

    pub fn last_output(self) -> LayerType
    {
        self.0.into_iter().rev().next().unwrap().output
    }

    pub fn hiddens(self) -> Vec<LayerType>
    {
        self.0.into_iter().map(|output|
        {
            let GRUOutput{
                hidden,
                ..
            } = output;

            hidden
        }).collect()
    }

    pub fn hiddens_ref(&self) -> Vec<&LayerType>
    {
        self.0.iter().map(|output|
        {
            &output.hidden
        }).collect()
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

pub struct DropoutMasksSingle
{
    pub hidden: LayerType,
    pub output: LayerType
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

    pub fn clear(&mut self)
    {
        self.input_update_weights.clear();
        self.input_reset_weights.clear();
        self.input_activation_weights.clear();
        self.hidden_update_weights.clear();
        self.hidden_reset_weights.clear();
        self.hidden_activation_weights.clear();
        self.update_biases.clear();
        self.reset_biases.clear();
        self.activation_biases.clear();
        self.output_weights.clear();
    }

    pub fn feedforward_single<const LAST_LAYER: bool, FO>(
        &mut self,
        previous_hidden: Option<&LayerType>,
        input: &LayerType,
        dropout_mask: &DropoutMasksSingle,
        output_activation: FO
    ) -> GRUOutput
    where
        FO: FnOnce(&mut LayerType)
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
            this_activation
        };

        let hidden = if USE_DROPOUT
        {
            hidden * &dropout_mask.hidden
        } else
        {
            hidden
        };

        let output_untrans = self.output_weights.matmul(&hidden);

        let output_untrans = if LAST_LAYER || !USE_DROPOUT
        {
            output_untrans
        } else
        {
            output_untrans * &dropout_mask.output
        };

        let mut output_gate = output_untrans.clone();
        output_activation(&mut output_gate);

        GRUOutput{
            hidden,
            output: output_gate
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

    #[allow(dead_code)]
    pub fn accuracy(
        &mut self,
        input: impl Iterator<Item=(LayerType, LayerType)>
    ) -> f32
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let amount = input.len();

        let f_output = self.feedforward(&self.create_empty_dropout(), input.into_iter());

        Self::correct_guesses(
            f_output.into_iter().map(|output| output.last_output()),
            output.into_iter()
        ) as f32 / amount as f32
    }

    fn correct_guesses<P, T>(
        predicted: impl Iterator<Item=P>,
        target: impl Iterator<Item=T>
    ) -> usize
    where
        P: Borrow<LayerType>,
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


    pub fn loss(
        &mut self,
        input: impl Iterator<Item=(LayerType, LayerType)> + ExactSizeIterator
    ) -> f32
    {
        let amount = input.len();

        self.loss_unscaled(input) / amount as f32
    }

    #[allow(dead_code)]
    pub fn loss_unscaled_with_dropout<L>(
        &mut self,
        dropout: &[DropoutMasksSingle],
        input: impl Iterator<Item=(L, L)>
    ) -> ScalarType
    where
        L: Borrow<LayerType>
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();

        let f_output = self.feedforward(dropout, input.into_iter());

        Self::cross_entropy(
            f_output.into_iter().map(|output| output.last_output()),
            output.into_iter().map(|x| x.borrow().clone())
        )
    }

    pub fn loss_unscaled<L>(
        &mut self,
        input: impl Iterator<Item=(L, L)>
    ) -> f32
    where
        L: Borrow<LayerType>
    {
        let dropout = self.create_empty_dropout();
        *self.loss_unscaled_with_dropout(&dropout, input.into_iter()).value()
    }

    fn cross_entropy(
        predicted: impl Iterator<Item=LayerType>,
        target: impl Iterator<Item=LayerType>
    ) -> ScalarType
    {
        let s: ScalarType = predicted.zip(target).map(|(mut predicted, target)|
        {
            predicted.ln();

            predicted.dot(target)
        }).sum();

        -s
    }

    #[inline(always)]
    pub fn feedforward_single<B>(
        &mut self,
        previous_hiddens: Option<&[B]>,
        input: &LayerType,
        dropout_masks: &[DropoutMasksSingle]
    ) -> GRUOutputLayer
    where
        B: Borrow<LayerType>
    {
        if let Some(previous_hiddens) = previous_hiddens
        {
            debug_assert!(previous_hiddens.len() == LAYERS_AMOUNT);
        }

        let mut output: GRUOutputLayer = GRUOutputLayer::with_capacity(LAYERS_AMOUNT);

        for l_i in 0..LAYERS_AMOUNT
        {
            let input = if l_i == 0
            {
                input
            } else
            {
                let index = l_i - 1;

                debug_assert!(index < output.0.len());
                unsafe{ &output.0.get_unchecked(index).output }
            };

            debug_assert!(l_i < self.layers.len());
            let layer = unsafe{ self.layers.get_unchecked_mut(l_i) };

            let previous_hidden = unsafe{
                previous_hiddens.map(|ph| ph.get_unchecked(l_i).borrow())
            };

            debug_assert!(l_i < dropout_masks.len());
            let dropout_mask = unsafe{ dropout_masks.get_unchecked(l_i) };

            let this_output = if l_i == (LAYERS_AMOUNT - 1)
            {
                // last layer
                layer.feedforward_single::<true, _>(
                    previous_hidden,
                    input,
                    dropout_mask,
                    SoftmaxedLayer::softmax
                )
            } else
            {
                match LAYER_ACTIVATION
                {
                    AFType::LeakyRelu =>
                    {
                        layer.feedforward_single::<false, _>(
                            previous_hidden,
                            input,
                            dropout_mask,
                            LayerType::leaky_relu
                        )
                    },
                    AFType::Tanh =>
                    {
                        layer.feedforward_single::<false, _>(
                            previous_hidden,
                            input,
                            dropout_mask,
                            LayerType::tanh
                        )
                    }
                }
            };

            output.push(this_output);
        }

        output
    }

    #[allow(dead_code)]
    pub fn feedforward<L>(
        &mut self,
        dropout_masks: &[DropoutMasksSingle],
        input: impl Iterator<Item=L> + ExactSizeIterator
    ) -> Vec<GRUOutputLayer>
    where
        L: Borrow<LayerType>
    {
        let mut outputs: Vec<GRUOutputLayer> = Vec::with_capacity(input.len());

        for (t, this_input) in input.enumerate()
        {
            let this_input = this_input.borrow();

            let output = if t == 0
            {
                self.feedforward_single::<&LayerType>(None, this_input, &dropout_masks)
            } else
            {
                let previous_hidden = unsafe{ &outputs.get_unchecked(t - 1).hiddens_ref() };
                self.feedforward_single(Some(previous_hidden), this_input, &dropout_masks)
            };

            outputs.push(output);
        }

        outputs
    }

    pub fn gradients<L>(
        &mut self,
        input: impl Iterator<Item=(L, L)> + ExactSizeIterator
    ) -> (f32, GRUFullGradients)
    where
        L: Borrow<LayerType>
    {
        self.clear();

        let inputs_amount = input.len();

        let dropout = self.create_dropout();
        let loss = self.loss_unscaled_with_dropout(&dropout, input);

        let loss_value = loss.value_clone() / inputs_amount as f32;

        loss.calculate_gradients();

        let gradients = GRUFullGradients(
            self.layers.iter_mut().map(|layer|
            {
                GRUGradients{
                    input_update_gradients:
                        -layer.input_update_weights.take_gradient_tensor(),
                    hidden_update_gradients:
                        -layer.hidden_update_weights.take_gradient_tensor(),
                    update_bias_gradients:
                        -layer.update_biases.take_gradient_tensor(),
                    input_reset_gradients:
                        -layer.input_reset_weights.take_gradient_tensor(),
                    hidden_reset_gradients:
                        -layer.hidden_reset_weights.take_gradient_tensor(),
                    reset_bias_gradients:
                        -layer.reset_biases.take_gradient_tensor(),
                    input_activation_gradients:
                        -layer.input_activation_weights.take_gradient_tensor(),
                    hidden_activation_gradients:
                        -layer.hidden_activation_weights.take_gradient_tensor(),
                    activation_bias_gradients:
                        -layer.activation_biases.take_gradient_tensor(),
                    output_gradients:
                        -layer.output_weights.take_gradient_tensor(),
                }
            }).collect()
        );

        (loss_value, gradients)
    }

    pub fn create_dropout(&self) -> Vec<DropoutMasksSingle>
    {
        #[cfg(test)]
        {
            fastrand::seed(12345);
        }

        self.create_dropout_with(||
        {
            // p = 0.5 (dropout chance is 50%)
            let dropout = fastrand::bool();
            if dropout
            {
                0.0
            } else
            {
                1.0
            }
        })
    }

    pub fn create_empty_dropout(&self) -> Vec<DropoutMasksSingle>
    {
        self.create_dropout_with(|| 0.5)
    }

    fn create_dropout_with(&self, mut f: impl FnMut() -> f32) -> Vec<DropoutMasksSingle>
    {
        (0..LAYERS_AMOUNT).map(|_|
        {
            let output = LayerType::new_with(self.word_vector_size, 1, &mut f);
            let hidden = LayerType::new_with(HIDDEN_AMOUNT, 1, &mut f);

            DropoutMasksSingle{hidden, output}
        }).collect::<Vec<_>>()
    }
}

#[cfg(test)]
pub mod tests
{
    use super::*;
    use crate::neural_network::LayerType;

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

    #[test]
    fn loss_correct()
    {
        let c = |v: Vec<f32>|
        {
            let len = v.len();
            LayerType::from_raw(v, len, 1)
        };

        let predicted = vec![
            c(vec![0.25, 0.25, 0.25, 0.25]),
            c(vec![0.01, 0.01, 0.01, 0.96])
        ];

        let target = vec![
            c(vec![0.0, 0.0, 0.0, 1.0]),
            c(vec![0.0, 0.0, 0.0, 1.0])
        ];

        let amount = target.len() as f32;

        let loss = GRU::cross_entropy(
            predicted.into_iter(),
            target.into_iter()
        ).value_clone() / amount;

        assert!(
            close_enough(loss, 0.71355817782, 0.000001),
            "loss: {loss}"
        );
    }
}
