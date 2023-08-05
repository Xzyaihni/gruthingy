use std::{
    f32,
    iter,
    borrow::Borrow,
    collections::VecDeque,
    ops::{DivAssign, AddAssign, Mul, Div, Sub}
};

use serde::{Serialize, Deserialize};

use crate::neural_network::{
    SoftmaxedLayer,
    NetworkType,
    HIDDEN_AMOUNT,
    LAYERS_AMOUNT
};


#[derive(Debug)]
pub struct GRUOutput<T>
{
    pub update: T,
    pub reset: T,
    pub activation: T,
    pub hidden: T,
    pub output_ut: T,
    pub output: T
}

pub struct GRUOutputLayer<T>(pub Vec<GRUOutput<T>>);

impl<T> GRUOutputLayer<T>
{
    pub fn with_capacity(capacity: usize) -> Self
    {
        Self(Vec::with_capacity(capacity))
    }

    pub fn push(&mut self, value: GRUOutput<T>)
    {
        self.0.push(value);
    }

    pub fn last_output_ref(&self) -> &T
    {
        &self.0.iter().rev().next().unwrap().output
    }

    pub fn last_output(self) -> T
    {
        self.0.into_iter().rev().next().unwrap().output
    }

    pub fn hiddens(self) -> Vec<T>
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

    pub fn hiddens_ref(&self) -> Vec<&T>
    {
        self.0.iter().map(|output|
        {
            &output.hidden
        }).collect()
    }
}

#[derive(Debug)]
pub struct GRUGradients<T>
{
    pub input_update_gradients: T,
    pub input_reset_gradients: T,
    pub input_activation_gradients: T,
    pub hidden_update_gradients: T,
    pub hidden_reset_gradients: T,
    pub hidden_activation_gradients: T,
    pub update_bias_gradients: T,
    pub reset_bias_gradients: T,
    pub activation_bias_gradients: T,
    pub output_gradients: T
}

impl<T> DivAssign<f32> for GRUGradients<T>
where
    T: NetworkType,
    for<'a> &'a T: Mul<f32, Output=T> + Mul<&'a T, Output=T> + Mul<T, Output=T>,
    for<'a> &'a T: Div<f32, Output=T>,
    for<'a> &'a T: Sub<Output=T>
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

impl<T> AddAssign for GRUGradients<T>
where
    T: NetworkType,
    for<'a> &'a T: Mul<f32, Output=T> + Mul<&'a T, Output=T> + Mul<T, Output=T>,
    for<'a> &'a T: Div<f32, Output=T>,
    for<'a> &'a T: Sub<Output=T>
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
pub struct GRUFullGradients<T>(pub VecDeque<GRUGradients<T>>);

impl<T> DivAssign<f32> for GRUFullGradients<T>
where
    T: NetworkType,
    for<'a> &'a T: Mul<f32, Output=T> + Mul<&'a T, Output=T> + Mul<T, Output=T>,
    for<'a> &'a T: Div<f32, Output=T>,
    for<'a> &'a T: Sub<Output=T>
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.0.iter_mut().for_each(|v| *v /= rhs);
    }
}

impl<T> iter::Sum for GRUFullGradients<T>
where
    T: NetworkType,
    for<'a> &'a T: Mul<f32, Output=T> + Mul<&'a T, Output=T> + Mul<T, Output=T>,
    for<'a> &'a T: Div<f32, Output=T>,
    for<'a> &'a T: Sub<Output=T>
{
    fn sum<I: Iterator<Item=Self>>(iter: I) -> Self
    {
        iter.reduce(|mut acc, this|
        {
            acc.0.iter_mut().zip(this.0.into_iter()).for_each(|(acc, this)|
            {
                *acc += this;
            });

            acc
        }).expect("must not be called on an empty iterator (im too lazy)")
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GRULayer<T>
{
	pub input_update_weights: T,
	pub input_reset_weights: T,
	pub input_activation_weights: T,
	pub hidden_update_weights: T,
	pub hidden_reset_weights: T,
	pub hidden_activation_weights: T,
	pub update_biases: T,
	pub reset_biases: T,
	pub activation_biases: T,
	pub output_weights: T
}

impl<N> GRULayer<N>
where
    N: NetworkType,
    for<'a> &'a N: Mul<f32, Output=N> + Mul<&'a N, Output=N> + Mul<N, Output=N>,
    for<'a> &'a N: Div<f32, Output=N>,
    for<'a> &'a N: Sub<Output=N>
{
    pub fn new(word_vector_size: usize) -> Self
    {
        let weights_init = |previous: f32|
        {
            let v = 1.0 / previous.sqrt();

            (fastrand::f32() * 2.0 - 1.0) * v
        };

        Self{
        	input_update_weights: N::new_with(
                word_vector_size,
                HIDDEN_AMOUNT,
                || weights_init(word_vector_size as f32)
            ),
        	input_reset_weights: N::new_with(
				word_vector_size,
				HIDDEN_AMOUNT,
				|| weights_init(word_vector_size as f32)
			),
        	input_activation_weights: N::new_with(
				word_vector_size,
				HIDDEN_AMOUNT,
				|| weights_init(word_vector_size as f32)
			),
        	hidden_update_weights: N::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f32)
			),
        	hidden_reset_weights: N::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f32)
			),
        	hidden_activation_weights: N::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f32)
			),
            // initialize biases to 0 cuz i read somewhere thats good
            update_biases: N::new(HIDDEN_AMOUNT, 1),
            reset_biases: N::new(HIDDEN_AMOUNT, 1),
            activation_biases: N::new(HIDDEN_AMOUNT, 1),
            output_weights: N::new_with(
                HIDDEN_AMOUNT,
                word_vector_size,
                || weights_init(HIDDEN_AMOUNT as f32)
            )
        }
    }

    #[inline(always)]
    pub fn zeroed_gradients(word_vector_size: usize) -> GRUGradients<N>
    {
        let output_gradients = N::new(HIDDEN_AMOUNT, word_vector_size);

        let input_update_gradients = N::new(word_vector_size, HIDDEN_AMOUNT);
        let input_reset_gradients = N::new(word_vector_size, HIDDEN_AMOUNT);
        let input_activation_gradients = N::new(word_vector_size, HIDDEN_AMOUNT);

        let hidden_update_gradients = N::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT);
        let hidden_reset_gradients = N::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT);
        let hidden_activation_gradients = N::new(HIDDEN_AMOUNT, HIDDEN_AMOUNT);

        let update_bias_gradients = N::new(HIDDEN_AMOUNT, 1);
        let reset_bias_gradients = N::new(HIDDEN_AMOUNT, 1);
        let activation_bias_gradients = N::new(HIDDEN_AMOUNT, 1);

        GRUGradients{
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
        }
    }

    // needs to know if its the first layer so it doesnt apply the transfer function
    pub fn gradients_with_hidden<const FIRST_LAYER: bool>(
        &self,
        word_vector_size: usize,
        starting_hidden: &N,
        input_ut: &[&N],
        input: &[&N],
        output_gradient: Vec<N>,
        f_output: &[&GRUOutput<N>]
    ) -> (Vec<N>, GRUGradients<N>)
    {
        let mut gradients = Self::zeroed_gradients(word_vector_size);

        let mut input_gradients = vec![N::new(word_vector_size, 1); input.len()];

        for t in 0..input.len()
        {
            let hidden = unsafe{ &f_output.get_unchecked(t).hidden };

            let output_gradient = unsafe{ output_gradient.get_unchecked(t) };

            gradients.output_gradients.add_outer_product(output_gradient, hidden);

            let output_gradient = self.output_weights.matmul_transposed(output_gradient);

            let mut d3 = output_gradient;

            for b_t in (0..=t).rev()
            {
                let previous_hidden = if b_t == 0
                {
                    starting_hidden
                } else
                {
                    unsafe{ &f_output.get_unchecked(b_t - 1).hidden }
                };

                let this_update = unsafe{ &f_output.get_unchecked(b_t).update };
                let this_reset = unsafe{ &f_output.get_unchecked(b_t).reset };
                let this_activation = unsafe{ &f_output.get_unchecked(b_t).activation };

                let d4 = this_update.clone().one_minus_this() * &d3;

                let d5 = previous_hidden * &d3;
                let d6 = d5 * -1.0;
                let d7 = this_activation * &d3;
                let d8 = this_update * &d3;
                let d9 = d7 + d6;

                // d10
                let activation_gate_derivative =
                    (this_activation * this_activation).one_minus_this() * d8;

                // d11
                let update_gate_derivative =
                    &d9 * (this_update * this_update.clone().one_minus_this());

                let d12 =
                    self.input_activation_weights.matmul_transposed(&activation_gate_derivative);

                let d13 =
                    self.hidden_activation_weights.matmul_transposed(&activation_gate_derivative);

                let d14 = self.input_update_weights.matmul_transposed(&update_gate_derivative);
                let d15 = self.hidden_update_weights.matmul_transposed(&update_gate_derivative);
                let d16 = previous_hidden * &d13;
                let d17 = d13 * this_reset;

                // d18
                let reset_gate_derivative =
                    (this_reset * this_reset.clone().one_minus_this()) * &d16;

                let d19 = d17 + d4;
                let d20 = self.input_reset_weights.matmul_transposed(&reset_gate_derivative);
                let d21 = self.hidden_reset_weights.matmul_transposed(&reset_gate_derivative);
                let d22 = d21 + d15;

                gradients.hidden_update_gradients
                    .add_outer_product(&update_gate_derivative, previous_hidden);

                gradients.hidden_reset_gradients
                    .add_outer_product(&reset_gate_derivative, previous_hidden);
                
                {
                    let previous_hidden = previous_hidden * this_reset;
                    gradients.hidden_activation_gradients
                        .add_outer_product(&activation_gate_derivative, previous_hidden);
                }

                let this_input = unsafe{ *input.get_unchecked(b_t) };

                gradients.input_update_gradients
                    .add_outer_product(&update_gate_derivative, this_input);

                gradients.input_reset_gradients
                    .add_outer_product(&reset_gate_derivative, this_input);
                
                gradients.input_activation_gradients
                    .add_outer_product(&activation_gate_derivative, this_input);

                gradients.update_bias_gradients += update_gate_derivative;
                gradients.reset_bias_gradients += reset_gate_derivative;
                gradients.activation_bias_gradients += activation_gate_derivative;

                let d23 = d19 + d22;
                let d24 = d12 + d14 + d20;

                let mut this_input_d = unsafe{ *input_ut.get_unchecked(b_t) }.clone();

                if !FIRST_LAYER
                {
                    this_input_d.leaky_relu_d();
                }

                let input_gradient = this_input_d * d24;
                *unsafe{ input_gradients.get_unchecked_mut(b_t) } += input_gradient;

                d3 = d23;
            }
        }

        (input_gradients, gradients)
    }

    pub fn feedforward_single<FO>(
        &self,
        previous_hidden: &N,
        input: &N,
        output_activation: FO
    ) -> GRUOutput<N>
    where
        FO: FnOnce(&mut N)
    {
        let mut update_gate =
            self.hidden_update_weights.matmul(previous_hidden)
            + self.input_update_weights.matmul(input)
            + &self.update_biases;

        update_gate.sigmoid();

        let mut reset_gate =
            self.hidden_reset_weights.matmul(previous_hidden)
            + self.input_reset_weights.matmul(input)
            + &self.reset_biases;

        reset_gate.sigmoid();

        let activation_v = &reset_gate * previous_hidden;
        let mut activation_gate =
            self.hidden_activation_weights.matmul(activation_v)
            + self.input_activation_weights.matmul(input)
            + &self.activation_biases;

        activation_gate.tanh();

        let this_activation = &activation_gate * &update_gate;
        let hidden = update_gate.clone().one_minus_this() * previous_hidden + this_activation;

        let output_untrans = self.output_weights.matmul(&hidden);

        let mut output_gate = output_untrans.clone();
        output_activation(&mut output_gate);

        GRUOutput{
            update: update_gate,
            reset: reset_gate,
            activation: activation_gate,
            hidden,
            output_ut: output_untrans,
            output: output_gate
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GRU<T>
{
    pub layers: Vec<GRULayer<T>>,
    word_vector_size: usize
}

impl<N> GRU<N>
where
    N: NetworkType,
    for<'a> &'a N: Mul<f32, Output=N> + Mul<&'a N, Output=N> + Mul<N, Output=N>,
    for<'a> &'a N: Div<f32, Output=N>,
    for<'a> &'a N: Sub<Output=N>
{
    pub fn new(word_vector_size: usize) -> Self
    {
        Self{
            layers: (0..LAYERS_AMOUNT).map(|_| GRULayer::new(word_vector_size)).collect(),
            word_vector_size
        }
    }

    #[allow(dead_code)]
    pub fn accuracy(
        &self,
        input: impl Iterator<Item=(N, N)>
    ) -> f32
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let amount = input.len();

        let f_output = self.feedforward(input.into_iter());

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
        P: Borrow<N>,
        T: Borrow<N>
    {
        predicted.zip(target).map(|(predicted, target)|
        {
            let target_index = target.borrow().highest_index();
            if SoftmaxedLayer::pick_weighed_associated(predicted.borrow(), 1.0) == target_index
            {
                1
            } else
            {
                0
            }
        }).sum()
    }


    pub fn loss(
        &self,
        input: impl Iterator<Item=(N, N)> + ExactSizeIterator
    ) -> f32
    {
        let amount = input.len();

        self.loss_unscaled(input) / amount as f32
    }

    pub fn loss_unscaled(
        &self,
        input: impl Iterator<Item=(N, N)>
    ) -> f32
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();

        let f_output = self.feedforward(input.into_iter());

        Self::cross_entropy(
            f_output.into_iter().map(|output| output.last_output()),
            output.into_iter()
        )
    }

    fn cross_entropy(
        predicted: impl Iterator<Item=N>,
        target: impl Iterator<Item=N>
    ) -> f32
    {
        let s: f32 = predicted.zip(target).map(|(mut predicted, target)|
        {
            predicted.ln();

            predicted.dot(target)
        }).sum();

        -s
    }

    // deriving this hell made me appreciate how simple the rnn.rs derivation was
    // nevermind found somebody else deriving it again (mine were wrong ;-;)
    // https://cran.r-project.org/web/packages/rnn/vignettes/GRU_units.html
    // found the derivatives here ^
    // BUT a lot of them r wrong
    // list of the derivatives that r wrong there and the correct ones i calculated:
    // d6 isnt 1 - d5, its instead d5 * -1.0
    // d9 isnt d7 + d8, its instead d6 + d7
    // d18 isnt d17 * derivative of sigmoid(reset_gate), its instead d16 * derivative of sigmoid(reset_gate)
    // also the U and W letters r swapped in the derivatives compared to the picture
    // also derivatives of reset gate things use d18 not d10 (and actiavtions use d10)
    #[allow(dead_code)]
    pub fn gradients<'a, const ONE_HOT_ENCODED: bool>(
        &self,
        input: impl Iterator<Item=(&'a N, &'a N)>
    ) -> GRUFullGradients<N>
    where
        N: 'a
    {
        let first_hiddens = vec![N::new(HIDDEN_AMOUNT, 1); LAYERS_AMOUNT];
        self.gradients_with_hidden::<ONE_HOT_ENCODED>(&first_hiddens, input)
    }

    pub fn gradients_with_hidden<'a, const ONE_HOT_ENCODED: bool>(
        &self,
        starting_hiddens: &[N],
        input: impl Iterator<Item=(&'a N, &'a N)>
    ) -> GRUFullGradients<N>
    where
        N: 'a
    {
        let (starting_input, output): (Vec<_>, Vec<_>) = input.unzip();
        let f_output = self.feedforward_with_hidden(
            starting_hiddens,
            starting_input.iter().map(|v| *v)
        );

        let mut gradients = VecDeque::with_capacity(LAYERS_AMOUNT);

        let mut this_output: Option<Vec<N>> = None;

        for l_i in (0..LAYERS_AMOUNT).rev()
        {
            debug_assert!(l_i < starting_hiddens.len());
            let starting_hidden = unsafe{ starting_hiddens.get_unchecked(l_i) };

            debug_assert!(l_i < self.layers.len());
            let layer = unsafe{ self.layers.get_unchecked(l_i) };

            let this_f_output = f_output.iter().map(|layer|
            {
                debug_assert!(l_i < layer.0.len());
                unsafe{ layer.0.get_unchecked(l_i) }
            }).collect::<Vec<_>>();

            let is_last_layer = l_i == (LAYERS_AMOUNT - 1);

            let output_gradients = if is_last_layer
            {
                // if its the last layer
                (0..starting_input.len()).map(|t|
                {
                    let predicted_output = unsafe{ &this_f_output.get_unchecked(t).output };

                    let expected_output = unsafe{ *output.get_unchecked(t) };

                    if ONE_HOT_ENCODED
                    {
                        predicted_output - expected_output
                    } else
                    {
                        predicted_output * expected_output.sum() - expected_output
                    }
                }).collect::<Vec<_>>()
            } else
            {
                this_output.take().unwrap()
            };

            let (input_gradients, this_gradient) = if l_i == 0
            {
                // if its the first layer
                layer.gradients_with_hidden::<true>(
                    self.word_vector_size,
                    starting_hidden,
                    &starting_input,
                    &starting_input,
                    output_gradients,
                    &this_f_output
                )
            } else
            {
                let (input_ut, input): (Vec<_>, Vec<_>) = f_output.iter().map(|layer|
                {
                    let index = l_i - 1;

                    debug_assert!(index < layer.0.len());
                    let prev_layer = unsafe{ layer.0.get_unchecked(index) };
                    (&prev_layer.output_ut, &prev_layer.output)
                }).unzip();

                layer.gradients_with_hidden::<false>(
                    self.word_vector_size,
                    starting_hidden,
                    &input_ut,
                    &input,
                    output_gradients,
                    &this_f_output
                )
            };

            this_output = Some(input_gradients);

            gradients.push_front(this_gradient);
        }

        GRUFullGradients(gradients)
    }

    #[inline(always)]
    pub fn feedforward_single<B>(
        &self,
        previous_hiddens: &[B],
        input: &N
    ) -> GRUOutputLayer<N>
    where
        B: Borrow<N>
    {
        debug_assert!(previous_hiddens.len() == LAYERS_AMOUNT);
        let mut output: GRUOutputLayer<N> = GRUOutputLayer::with_capacity(LAYERS_AMOUNT);

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
            let layer = unsafe{ self.layers.get_unchecked(l_i) };

            let previous_hidden = unsafe{ previous_hiddens.get_unchecked(l_i).borrow() };

            let this_output = if l_i == (LAYERS_AMOUNT - 1)
            {
                // last layer
                layer.feedforward_single(
                    previous_hidden,
                    input,
                    SoftmaxedLayer::softmax
                )
            } else
            {
                layer.feedforward_single(
                    previous_hidden,
                    input,
                    N::leaky_relu
                )
            };

            output.push(this_output);
        }

        output
    }

    #[allow(dead_code)]
    pub fn feedforward<L>(
        &self,
        input: impl Iterator<Item=L> + ExactSizeIterator
    ) -> Vec<GRUOutputLayer<N>>
    where
        L: Borrow<N>
    {
        let first_hiddens = vec![N::new(HIDDEN_AMOUNT, 1); LAYERS_AMOUNT];

        self.feedforward_with_hidden(&first_hiddens, input)
    }

    #[allow(dead_code)]
    pub fn feedforward_with_hidden<L, B>(
        &self,
        first_hiddens: &[B],
        input: impl Iterator<Item=L> + ExactSizeIterator
    ) -> Vec<GRUOutputLayer<N>>
    where
        B: Borrow<N>,
        L: Borrow<N>
    {
        let mut outputs: Vec<GRUOutputLayer<N>> = Vec::with_capacity(input.len());

        for (t, this_input) in input.enumerate()
        {
            let this_input = this_input.borrow();

            let output = if t == 0
            {
                self.feedforward_single(first_hiddens, this_input)
            } else
            {
                let previous_hidden = unsafe{ &outputs.get_unchecked(t - 1).hiddens_ref() };
                self.feedforward_single(previous_hidden, this_input)
            };

            outputs.push(output);
        }

        outputs
    }
}

#[cfg(test)]
pub mod tests
{
    use super::*;
    use crate::neural_network::{
        MatrixWrapper,
        GenericContainer,
        NetworkType,
        WeightsIterValue,
        InputOutputIter
    };

    fn close_enough(a: f32, b: f32, epsilon: f32) -> bool
    {
        if (a == b) || ((a.min(b) == -0.0) && (a.max(b) == 0.0))
        {
            return true;
        }

        ((a - b).abs() / (a.abs() + b.abs())) < epsilon
    }

    fn close_enough_abs(a: f32, b: f32, epsilon: f32) -> bool
    {
        (a - b).abs() < epsilon
    }

    #[allow(dead_code)]
    // #[test]
    fn forwardprop()
    {
        assert_eq!(HIDDEN_AMOUNT, 2);
        assert_eq!(LAYERS_AMOUNT, 2);
        
        let single_match = |correct, calculated|
        {
            assert!(
                close_enough_abs(calculated, correct, 0.000001),
                "correct: {correct}, calculated: {calculated}"
            );
        };

        let layer_match = |correct: &GenericContainer, calculated: &GenericContainer|
        {
            correct.iter().zip(calculated.iter()).for_each(|(correct, calculated)|
            {
                single_match(*correct, *calculated);
            });
        };

        let v = |values: Vec<f32>|
        {
            let size = values.len();
            GenericContainer::from_raw(values, size, 1)
        };

        let input = vec![
            v(vec![fastrand::f32() * 20.0 - 10.0, fastrand::f32() * 20.0 - 10.0])
        ];

        let gru = GRU::<GenericContainer>::new(2);

        let mut z =
            gru.layers[0].input_update_weights.matmul(&input[0])
            + gru.layers[0].hidden_update_weights.matmul(v(vec![0.0, 0.0]))
            + &gru.layers[0].update_biases;

        z.sigmoid();

        let mut r =
            gru.layers[0].input_reset_weights.matmul(&input[0])
            + gru.layers[0].hidden_reset_weights.matmul(v(vec![0.0, 0.0]))
            + &gru.layers[0].reset_biases;

        r.sigmoid();

        let av = v(vec![0.0, 0.0]) * &r;

        let mut a =
            gru.layers[0].input_activation_weights.matmul(&input[0])
            + gru.layers[0].hidden_activation_weights.matmul(av)
            + &gru.layers[0].activation_biases;

        a.tanh();

        let h_out = z.clone().one_minus_this() * v(vec![0.0, 0.0]) + &z * &a;

        let mut o_out = gru.layers[0].output_weights.matmul(&h_out);
        o_out.sigmoid();

        let f_output = gru.feedforward(input.into_iter());

        let l_i = 0;

        eprintln!("update");
        layer_match(&z, &f_output[0].0[l_i].update);
        eprintln!("reset");
        layer_match(&r, &f_output[0].0[l_i].reset);
        eprintln!("activation");
        layer_match(&a, &f_output[0].0[l_i].activation);
        eprintln!("hidden");
        layer_match(&h_out, &f_output[0].0[l_i].hidden);
        eprintln!("output");
        layer_match(&o_out, &f_output[0].0[l_i].output);

        let mut z =
            gru.layers[1].input_update_weights.matmul(&o_out)
            + gru.layers[1].hidden_update_weights.matmul(v(vec![0.0, 0.0]))
            + &gru.layers[1].update_biases;

        z.sigmoid();

        let mut r =
            gru.layers[1].input_reset_weights.matmul(&o_out)
            + gru.layers[1].hidden_reset_weights.matmul(v(vec![0.0, 0.0]))
            + &gru.layers[1].reset_biases;

        r.sigmoid();

        let av = v(vec![0.0, 0.0]) * &r;

        let mut a =
            gru.layers[1].input_activation_weights.matmul(&o_out)
            + gru.layers[1].hidden_activation_weights.matmul(av)
            + &gru.layers[1].activation_biases;

        a.tanh();

        let h_out = z.clone().one_minus_this() * v(vec![0.0, 0.0]) + &z * &a;

        let mut o_out = gru.layers[1].output_weights.matmul(&h_out);
        SoftmaxedLayer::softmax(&mut o_out);

        let l_i = 1;

        eprintln!("update");
        layer_match(&z, &f_output[0].0[l_i].update);
        eprintln!("reset");
        layer_match(&r, &f_output[0].0[l_i].reset);
        eprintln!("activation");
        layer_match(&a, &f_output[0].0[l_i].activation);
        eprintln!("hidden");
        layer_match(&h_out, &f_output[0].0[l_i].hidden);
        eprintln!("output");
        layer_match(&o_out, &f_output[0].0[l_i].output);
    }

    fn test_values(amount: usize) -> Vec<f32>
    {
        (0..amount).map(|_| fastrand::f32()).collect::<Vec<_>>()
    }

    #[test]
    fn backprop_match()
    {
        let inputs_amount = 5;
        let inputs_size = 10;

        fastrand::seed(12345);
        let gru_correct: GRU<GenericContainer> = GRU::new(inputs_size);

        fastrand::seed(12345);
        let gru_nalgebra: GRU<MatrixWrapper> = GRU::new(inputs_size);

        let values = test_values(inputs_amount * inputs_size);

        let inputs = (0..inputs_amount).map(|i|
        {
            (0..inputs_size).map(|j|
            {
                let index = i * inputs_size + j;

                values[index]
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();

        let input_correct = (0..inputs_amount).map(|i|
        {
            GenericContainer::from_raw(inputs[i].clone(), inputs_size, 1)
        }).collect::<Vec<_>>();

        let input_nalgebra = (0..inputs_amount).map(|i|
        {
            MatrixWrapper::from_raw(inputs[i].clone(), inputs_size, 1)
        }).collect::<Vec<_>>();

        let input_correct = InputOutputIter::new(input_correct.iter());
        let output_correct = gru_correct.gradients::<false>(input_correct);

        let input_nalgebra = InputOutputIter::new(input_nalgebra.iter());
        let output_nalgebra = gru_nalgebra.gradients::<false>(input_nalgebra);

        for (output_correct, output_nalgebra) in output_correct.0.into_iter()
            .zip(output_nalgebra.0.into_iter())
        {
            let single_match = |correct, calculated, index|
            {
                assert!(
                    close_enough_abs(calculated, correct, 0.00001),
                    "correct: {correct}, calculated: {calculated}, index: {index}"
                );
            };

            let layer_match = |correct: GenericContainer, calculated: MatrixWrapper|
            {
                for y in 0..correct.this_size()
                {
                    for x in 0..correct.previous_size()
                    {
                        let index = y * correct.previous_size() + x;

                        let correct = correct.as_vec()[index];
                        let calculated = calculated.as_vec()[index];

                        single_match(correct, calculated, index);
                    }
                }
            };

            eprintln!("input update gradients");
            layer_match(
                output_correct.input_update_gradients,
                output_nalgebra.input_update_gradients
            );

            eprintln!("input reset gradients");
            layer_match(
                output_correct.input_reset_gradients,
                output_nalgebra.input_reset_gradients
            );

            eprintln!("input activation gradients");
            layer_match(
                output_correct.input_activation_gradients,
                output_nalgebra.input_activation_gradients
            );

            eprintln!("hidden update gradients");
            layer_match(
                output_correct.hidden_update_gradients,
                output_nalgebra.hidden_update_gradients
            );

            eprintln!("hidden reset gradients");
            layer_match(
                output_correct.hidden_reset_gradients,
                output_nalgebra.hidden_reset_gradients
            );

            eprintln!("hidden activation gradients");
            layer_match(
                output_correct.hidden_activation_gradients,
                output_nalgebra.hidden_activation_gradients
            );

            eprintln!("update bias gradients");
            layer_match(
                output_correct.update_bias_gradients,
                output_nalgebra.update_bias_gradients
            );

            eprintln!("reset bias gradients");
            layer_match(
                output_correct.reset_bias_gradients,
                output_nalgebra.reset_bias_gradients
            );

            eprintln!("activation bias gradients");
            layer_match(
                output_correct.activation_bias_gradients,
                output_nalgebra.activation_bias_gradients
            );

            eprintln!("output gradients");
            layer_match(
                output_correct.output_gradients,
                output_nalgebra.output_gradients
            );
        }
    }

    /*
    #[allow(dead_code)]
    // #[test]
    fn backprop_smol()
    {
        assert_eq!(HIDDEN_AMOUNT, 2);
        assert_eq!(LAYERS_AMOUNT, 1);

        let input_update_weights = GenericContainer::from_raw(
            vec![4.63, -2.64, 4.76, 3.63].into_boxed_slice(),
            2,
            HIDDEN_AMOUNT
        );

        let input_reset_weights = GenericContainer::from_raw(
            vec![-8.29, 9.96, -4.78, 2.24].into_boxed_slice(),
            2,
            HIDDEN_AMOUNT
        );

        let input_activation_weights = GenericContainer::from_raw(
            vec![-5.09, 1.99, 1.15, 4.63].into_boxed_slice(),
            2,
            HIDDEN_AMOUNT
        );

        let hidden_update_weights = GenericContainer::from_raw(
            vec![-0.48, 8.48, -6.14, 2.42].into_boxed_slice(),
            HIDDEN_AMOUNT,
            HIDDEN_AMOUNT
        );

        let hidden_reset_weights = GenericContainer::from_raw(
            vec![-5.74, -2.66, -6.25, -9.21].into_boxed_slice(),
            HIDDEN_AMOUNT,
            HIDDEN_AMOUNT
        );

        let hidden_activation_weights = GenericContainer::from_raw(
            vec![-3.95, -6.07, 6.36, -5.36].into_boxed_slice(),
            HIDDEN_AMOUNT,
            HIDDEN_AMOUNT
        );

        let update_biases = GenericContainer::from_raw(vec![-2.00, -0.87], 2, 1);
        let reset_biases = GenericContainer::from_raw(vec![-8.36, -8.16], 2, 1);
        let activation_biases = GenericContainer::from_raw(vec![3.47, 3.52], 2, 1);

        let output_weights = GenericContainer::from_raw(
            vec![8.59, -1.08, -7.31, -7.97].into_boxed_slice(),
            HIDDEN_AMOUNT,
            2
        );

        let gru = GRU{
            input_update_weights,
            input_reset_weights,
            input_activation_weights,
            hidden_update_weights,
            hidden_reset_weights,
            hidden_activation_weights,
            update_biases,
            reset_biases,
            activation_biases,
            output_weights
        };

        let input = vec![
            GenericContainer::from_raw(vec![-5.90, 1.78], 2, 1),
            GenericContainer::from_raw(vec![-1.23, 4.56], 2, 1),
            GenericContainer::from_raw(vec![9.99, -1.02], 2, 1),
            GenericContainer::from_raw(vec![0.01, 10.0], 2, 1)
        ];

        let input = InputOutputIter::new(input.iter());

        let output = gru.gradients::<false>(input);

        let single_match = |correct, calculated|
        {
            assert!(
                close_enough_abs(calculated, correct, 0.00001),
                "correct: {correct}, calculated: {calculated}"
            );
        };

        let layer_match = |correct: [f32; 4], calculated: GenericContainer|
        {
            correct.iter().zip(calculated.iter()).for_each(|(correct, calculated)|
            {
                single_match(*correct, *calculated);
            });
        };

        let bias_match = |correct: [f32; 2], calculated: GenericContainer|
        {
            correct.iter().zip(calculated.iter()).for_each(|(correct, calculated)|
            {
                single_match(*correct, *calculated);
            });
        };

        layer_match([
            5.416323632651452e-8,
            -2.008012961152585e-7,
            0.0004701980745839663,
            -0.0017432416518287585,
        ], output.input_update_gradients);

        layer_match([
            -3.2916534783598936e-62,
            3.3608473953224137e-63,
            -4.376070777317816e-36,
            1.6512401517820905e-35,
        ], output.input_reset_gradients);

        layer_match([
            1.2026719991784244e-23,
            -4.4586864359785484e-23,
            -2.815569864494384e-9,
            2.8823767498944457e-10,
        ], output.input_activation_gradients);

        layer_match([
            -7.425982242345576e-23,
            -7.50496131841157e-18,
            -6.446831266464838e-19,
            -6.515396576798509e-14,
        ], output.hidden_update_gradients);

        layer_match([
            -8.867181401246852e-72,
            -3.2947707711316803e-63,
            2.1594354702353152e-47,
            8.02152580715418e-39,
        ], output.hidden_reset_gradients);

        layer_match([
            -1.6488992976193476e-38,
            -1.6658391023269182e-33,
            -5.1807365201048425e-64,
            -1.496633901915441e-39,
        ], output.hidden_activation_gradients);

        bias_match([
            -4.4035341451853064e-8,
            -0.000382287763564198,
        ], output.update_bias_gradients);

        bias_match([
            -3.29494842678668e-63,
            3.630957026725113e-36,
        ], output.reset_bias_gradients);

        bias_match([
            -9.777821131531905e-24,
            -2.8080948683310396e-10,
        ], output.activation_bias_gradients);

        layer_match([
            0.008777106286423676,
            -1.0378464082942112,
            -0.008777106286423085,
            1.037846408294211,
        ], output.output_gradients);
    }*/

    #[test]
    fn loss_correct()
    {
        let c = |v: Vec<f32>|
        {
            let len = v.len();
            GenericContainer::from_raw(v, len, 1)
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
        ) / amount;

        assert!(
            close_enough(loss, 0.71355817782, 0.000001),
            "loss: {loss}"
        );
    }

    pub fn input_update_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        layer_index: usize,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.layers[layer_index].input_update_weights,
            input.clone()
        );

        let calculated_gradient = &network.gradients::<false>(input).0[layer_index];

        true_gradient.iter_pos().zip(calculated_gradient.input_update_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn input_reset_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        layer_index: usize,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.layers[layer_index].input_reset_weights,
            input.clone()
        );

        let calculated_gradient = &network.gradients::<false>(input).0[layer_index];

        true_gradient.iter_pos().zip(calculated_gradient.input_reset_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn input_activation_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        layer_index: usize,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.layers[layer_index].input_activation_weights,
            input.clone()
        );

        let calculated_gradient = &network.gradients::<false>(input).0[layer_index];

        true_gradient.iter_pos().zip(calculated_gradient.input_activation_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_update_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        layer_index: usize,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.layers[layer_index].hidden_update_weights,
            input.clone()
        );

        let calculated_gradient = &network.gradients::<false>(input).0[layer_index];

        true_gradient.iter_pos().zip(calculated_gradient.hidden_update_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_reset_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        layer_index: usize,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.layers[layer_index].hidden_reset_weights,
            input.clone()
        );

        let calculated_gradient = &network.gradients::<false>(input).0[layer_index];

        true_gradient.iter_pos().zip(calculated_gradient.hidden_reset_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_activation_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        layer_index: usize,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.layers[layer_index].hidden_activation_weights,
            input.clone()
        );

        let calculated_gradient = &network.gradients::<false>(input).0[layer_index];

        true_gradient.iter_pos().zip(calculated_gradient.hidden_activation_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn update_bias_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        layer_index: usize,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.layers[layer_index].update_biases,
            input.clone()
        );

        let calculated_gradient = &network.gradients::<false>(input).0[layer_index];

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.update_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn reset_bias_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        layer_index: usize,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.layers[layer_index].reset_biases,
            input.clone()
        );

        let calculated_gradient = &network.gradients::<false>(input).0[layer_index];

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.reset_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn activation_bias_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        layer_index: usize,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.layers[layer_index].activation_biases, input.clone()
        );

        let calculated_gradient = &network.gradients::<false>(input).0[layer_index];

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.activation_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn output_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        layer_index: usize,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.layers[layer_index].output_weights, input.clone()
        );

        let calculated_gradient = &network.gradients::<false>(input).0[layer_index];

        true_gradient.iter_pos().zip(calculated_gradient.output_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    fn check_single_gradient(pair: (WeightsIterValue<f32>, WeightsIterValue<f32>))
    {
        let (true_gradient, calculated_gradient) = pair;
        let (previous, this) = (true_gradient.previous, true_gradient.this);

        let true_gradient = true_gradient.value;
        let calculated_gradient = calculated_gradient.value;

        // println!("comparing {true_gradient} and {calculated_gradient}");
        assert!(
            close_enough_abs(true_gradient, calculated_gradient, 0.002),
            "true_gradient: {true_gradient}, calculated_gradient: {calculated_gradient}, previous_index: {previous}, this_index: {this}"
        );
    }

    fn check_single_bias_gradient(pair: ((usize, &f32), (usize, &f32)))
    {
        let (true_gradient, calculated_gradient) = pair;
        let index = true_gradient.0;

        let true_gradient = true_gradient.1;
        let calculated_gradient = calculated_gradient.1;

        assert!(
            close_enough_abs(*true_gradient, *calculated_gradient, 0.002),
            "true_gradient: {true_gradient}, calculated_gradient: {calculated_gradient}, index: {index}"
        );
    }

    pub fn gradient_check<'a>(
        network: &mut GRU<GenericContainer>,
        mut weights_member: impl FnMut(&mut GRU<GenericContainer>) -> &mut GenericContainer,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    ) -> GenericContainer
    {
        let input = input.map(|(a, b)| (a.clone(), b.clone()));

        let weights = weights_member(network).iter_pos().map(|weight|
        {
            weight.to_owned()
        }).collect::<Vec<WeightsIterValue<_>>>();

        let weights = weights.into_iter().map(|weight|
        {
            let WeightsIterValue{value: weight, previous, this} = weight;
            let epsilon = 0.003;

            let mut set_this_weight = |network: &mut GRU<GenericContainer>, value|
            {
                *weights_member(network).weight_mut(previous, this) = value;
            };

            let gradient = {
                let f = |network: &GRU<_>|
                {
                    network.loss_unscaled(input.clone())
                };

                let mut change = |c|
                {
                    set_this_weight(network, weight + c);

                    f(&network)
                };

                let loss_0 = change(2.0 * epsilon);
                let loss_1 = change(epsilon);
                let loss_2 = change(-epsilon);
                let loss_3 = change(-2.0 * epsilon);

                (-loss_0 + 8.0 * loss_1 - 8.0 * loss_2 + loss_3) / (12.0 * epsilon)
            };

            set_this_weight(network, weight);

            gradient
        }).collect::<Box<[_]>>();

        GenericContainer::from_raw(
            weights,
            weights_member(network).previous_size(),
            weights_member(network).this_size()
        )
    }
}
