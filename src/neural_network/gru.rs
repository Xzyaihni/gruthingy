use std::{
    f32,
    borrow::Borrow,
    ops::{DivAssign, AddAssign, Mul, Div}
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
    pub output: T
}

impl<T> AddAssign for GRUOutput<T>
where
    T: NetworkType,
    for<'a> &'a T: Mul<f32, Output=T> + Mul<&'a T, Output=T> + Mul<T, Output=T>,
    for<'a> &'a T: Div<f32, Output=T>
{
    fn add_assign(&mut self, rhs: Self)
    {
        let Self{
            update,
            reset,
            activation,
            hidden,
            output
        } = rhs;

		self.update += update;
		self.reset += reset;
		self.activation += activation;
		self.hidden += hidden;
		self.output += output;
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
    for<'a> &'a T: Div<f32, Output=T>
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
    for<'a> &'a T: Div<f32, Output=T>
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

#[derive(Debug, Serialize, Deserialize)]
pub struct GRU<T>
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

impl<N> GRU<N>
where
    N: NetworkType,
    for<'a> &'a N: Mul<f32, Output=N> + Mul<&'a N, Output=N> + Mul<N, Output=N>,
    for<'a> &'a N: Div<f32, Output=N>
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
            f_output.into_iter().map(|output| output.output),
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
            f_output.into_iter().map(|output| output.output),
            output.into_iter()
        )
    }

    fn cross_entropy(
        predicted: impl Iterator<Item=N>,
        target: impl Iterator<Item=N>
    ) -> f32
    {
        let s: f32 = predicted.zip(target).map(|(predicted, target)|
        {
            predicted.ln().dot(target)
        }).sum();

        -s
    }

    #[inline(always)]
    pub fn zeroed_gradients(&self) -> GRUGradients<N>
    {
        let output_gradients = N::zeroed_copy(&self.output_weights);

        let input_update_gradients = N::zeroed_copy(&self.input_update_weights);
        let input_reset_gradients = N::zeroed_copy(&self.input_reset_weights);
        let input_activation_gradients = N::zeroed_copy(&self.input_activation_weights);

        let hidden_update_gradients = N::zeroed_copy(&self.hidden_update_weights);
        let hidden_reset_gradients = N::zeroed_copy(&self.hidden_reset_weights);
        let hidden_activation_gradients = N::zeroed_copy(&self.hidden_activation_weights);

        let update_bias_gradients = N::zeroed_copy(&self.update_biases);
        let reset_bias_gradients = N::zeroed_copy(&self.reset_biases);
        let activation_bias_gradients = N::zeroed_copy(&self.activation_biases);

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
    ) -> GRUGradients<N>
    where
        N: 'a
    {
        self.gradients_with_hidden::<ONE_HOT_ENCODED>(
            N::new(HIDDEN_AMOUNT, 1),
            input
        )
    }

    pub fn gradients_with_hidden<'a, const ONE_HOT_ENCODED: bool>(
        &self,
        starting_hidden: N,
        input: impl Iterator<Item=(&'a N, &'a N)>
    ) -> GRUGradients<N>
    where
        N: 'a
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let f_output = self.feedforward_with_hidden(
            &starting_hidden,
            input.iter().map(|v| *v)
        );

        let mut gradients = self.zeroed_gradients();

        for t in (0..output.len()).rev()
        {
            let predicted_output = unsafe{ &f_output.get_unchecked(t).output };

            let expected_output = unsafe{ *output.get_unchecked(t) };
            let hidden = unsafe{ &f_output.get_unchecked(t).hidden };

            let expected_sum: f32 = if ONE_HOT_ENCODED
            {
                1.0
            } else
            {
                expected_output.sum()
            };

            let diff = predicted_output * expected_sum - expected_output;

            gradients.output_gradients.add_outer_product(&diff, hidden);

            let mut d3 = self.output_weights.matmul_transposed(diff);

            for b_t in (0..(t + 1)).rev()
            {
                let previous_hidden = if b_t == 0
                {
                    &starting_hidden
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

                let d13 =
                    self.hidden_activation_weights.matmul_transposed(&activation_gate_derivative);

                let d15 = self.hidden_update_weights.matmul_transposed(&update_gate_derivative);
                let d16 = previous_hidden * &d13;
                let d17 = d13 * this_reset;

                // d18
                let reset_gate_derivative =
                    (this_reset * this_reset.clone().one_minus_this()) * &d16;

                let d19 = d17 + d4;

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

                d3 = d23;
            }
        }

        gradients
    }

    #[inline(always)]
    pub fn feedforward_single(
        &self,
        previous_hidden: &N,
        input: &N
    ) -> GRUOutput<N>
    {
        let update_gate =
            self.hidden_update_weights.matmul(previous_hidden)
            + self.input_update_weights.matmul(input)
            + &self.update_biases;

        let update_gate = update_gate.sigmoid();

        let reset_gate =
            self.hidden_reset_weights.matmul(previous_hidden)
            + self.input_reset_weights.matmul(input)
            + &self.reset_biases;

        let reset_gate = reset_gate.sigmoid();

        let activation_v = &reset_gate * previous_hidden;
        let activation_gate =
            self.hidden_activation_weights.matmul(activation_v)
            + self.input_activation_weights.matmul(input)
            + &self.activation_biases;

        let activation_gate = activation_gate.tanh();

        let this_activation = &activation_gate * &update_gate;
        let hidden = update_gate.clone().one_minus_this() * previous_hidden + this_activation;

        let output_gate = SoftmaxedLayer::softmax(self.output_weights.matmul(&hidden));

        GRUOutput{
            update: update_gate,
            reset: reset_gate,
            activation: activation_gate,
            hidden,
            output: output_gate
        }
    }

    #[allow(dead_code)]
    pub fn feedforward<L>(&self, input: impl Iterator<Item=L>) -> Vec<GRUOutput<N>>
    where
        L: Borrow<N>
    {
        let first_hidden = N::new(HIDDEN_AMOUNT, 1);

        self.feedforward_with_hidden(&first_hidden, input)
    }

    #[allow(dead_code)]
    pub fn feedforward_with_hidden<L>(
        &self,
        first_hidden: &N,
        input: impl Iterator<Item=L>
    ) -> Vec<GRUOutput<N>>
    where
        L: Borrow<N>
    {
        let (lower_bound, upper_bound) = input.size_hint();
        let time_total = upper_bound.unwrap_or(lower_bound);

        let mut outputs: Vec<GRUOutput<N>> = Vec::with_capacity(time_total);

        for (t, this_input) in input.enumerate()
        {
            let previous_hidden = if t == 0
            {
                first_hidden
            } else
            {
                unsafe{ &outputs.get_unchecked(t - 1).hidden }
            };

            let this_input = this_input.borrow();

            let output = self.feedforward_single(previous_hidden, this_input);

            outputs.push(output);
        }

        outputs
    }
}

#[cfg(test)]
pub mod tests
{
    use std::iter;

    use super::*;
    use crate::neural_network::{GenericContainer, WeightsIterValue, InputOutputIter};

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

    #[ignore]
    #[allow(dead_code)]
    // #[test]
    fn forwardprop()
    {
        assert_eq!(HIDDEN_AMOUNT, 2);

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

        let input = GenericContainer::from_raw(vec![-5.90, 1.78], 2, 1);
        let f_output = gru.feedforward(iter::once(input));

        let output = &f_output[0].output;

        let epsilon = 0.00001;

        let z = vec![1.686366804463125e-15, 1.7044644055950665e-10];
        assert!(close_enough(z[0], *f_output[0].update.weight(0, 0), epsilon));
        assert!(close_enough(z[1], *f_output[0].update.weight(1, 0), epsilon));

        let r = vec![1.0, 0.9999999999633351];
        assert!(close_enough(r[0], *f_output[0].reset.weight(0, 0), epsilon));
        assert!(close_enough(r[1], *f_output[0].reset.weight(1, 0), epsilon));

        let h = vec![1.0, 0.9999048161632378];
        assert!(close_enough(h[0], *f_output[0].activation.weight(0, 0), epsilon));
        assert!(close_enough(h[1], *f_output[0].activation.weight(1, 0), epsilon));

        let hidden = vec![1.686366804463125e-15, 1.7043021681333174e-10];
        assert!(close_enough(hidden[0], *f_output[0].hidden.weight(0, 0), epsilon));
        assert!(close_enough(hidden[1], *f_output[0].hidden.weight(1, 0), epsilon));

        let o = vec![0.5000000000460197, 0.4999999999539802];
        let (correct, calculated) = (o[0], *output.weight(0, 0));
        assert!(
            close_enough(correct, calculated, epsilon),
            "correct: {correct}, calculated: {calculated}"
        );

        let (correct, calculated) = (o[1], *output.weight(1, 0));
        assert!(
            close_enough(correct, calculated, epsilon),
            "correct: {correct}, calculated: {calculated}"
        );
    }

    #[ignore]
    #[allow(dead_code)]
    // #[test]
    fn backprop_smol()
    {
        assert_eq!(HIDDEN_AMOUNT, 2);

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
    }

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
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.input_update_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients::<false>(input);

        true_gradient.iter_pos().zip(calculated_gradient.input_update_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn input_reset_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.input_reset_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients::<false>(input);

        true_gradient.iter_pos().zip(calculated_gradient.input_reset_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn input_activation_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.input_activation_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients::<false>(input);

        true_gradient.iter_pos().zip(calculated_gradient.input_activation_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_update_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.hidden_update_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients::<false>(input);

        true_gradient.iter_pos().zip(calculated_gradient.hidden_update_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_reset_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.hidden_reset_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients::<false>(input);

        true_gradient.iter_pos().zip(calculated_gradient.hidden_reset_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_activation_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.hidden_activation_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients::<false>(input);

        true_gradient.iter_pos().zip(calculated_gradient.hidden_activation_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn update_bias_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.update_biases,
            input.clone()
        );

        let calculated_gradient = network.gradients::<false>(input);

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.update_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn reset_bias_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.reset_biases,
            input.clone()
        );

        let calculated_gradient = network.gradients::<false>(input);

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.reset_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn activation_bias_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.activation_biases, input.clone()
        );

        let calculated_gradient = network.gradients::<false>(input);

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.activation_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn output_gradients_check<'a>(
        network: &mut GRU<GenericContainer>,
        input: impl Iterator<Item=(&'a GenericContainer, &'a GenericContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.output_weights, input.clone()
        );

        let calculated_gradient = network.gradients::<false>(input);
        true_gradient.iter_pos().zip(calculated_gradient.output_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    fn check_single_gradient(pair: (WeightsIterValue<f32>, WeightsIterValue<f32>))
    {
        let (true_gradient, calculated_gradient) = pair;
        let (previous, this) = (true_gradient.previous, true_gradient.this);

        let true_gradient = true_gradient.value;
        let calculated_gradient = calculated_gradient.value;

        assert!(
            close_enough_abs(true_gradient, calculated_gradient, 0.001),
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
            close_enough_abs(*true_gradient, *calculated_gradient, 0.001),
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
            let epsilon = 0.01;

            let mut set_this_weight = |network: &mut GRU<GenericContainer>, value|
            {
                *weights_member(network).weight_mut(previous, this) = value;
            };

            // ; ;
            set_this_weight(network, weight - epsilon);
            let under_loss = network.loss_unscaled(input.clone());
            
            set_this_weight(network, weight + epsilon);
            let over_loss = network.loss_unscaled(input.clone());

            set_this_weight(network, weight);

            (over_loss - under_loss) / (2.0 * epsilon)
        }).collect::<Box<[_]>>();

        GenericContainer::from_raw(
            weights,
            weights_member(network).previous_size(),
            weights_member(network).this_size()
        )
    }
}
