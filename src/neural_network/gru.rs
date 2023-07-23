use std::{
    f64,
    borrow::Borrow
};

use serde::{Serialize, Deserialize};

use crate::neural_network::{
    LayerContainer,
    WeightsContainer,
    SoftmaxedLayer,
    InputOutputIter,
    HIDDEN_AMOUNT
};


#[derive(Debug)]
pub struct GRUOutput
{
    pub update: Vec<LayerContainer>,
    pub reset: Vec<LayerContainer>,
    pub activation: Vec<LayerContainer>,
    pub hiddens: Vec<LayerContainer>,
    pub outputs: Vec<SoftmaxedLayer>
}

#[derive(Debug)]
pub struct GRUSingleOutput
{
    pub update: LayerContainer,
    pub reset: LayerContainer,
    pub activation: LayerContainer,
    pub hidden: LayerContainer,
    pub output: SoftmaxedLayer
}

#[derive(Debug)]
pub struct GRUGradients
{
    pub input_update_gradients: WeightsContainer,
    pub input_reset_gradients: WeightsContainer,
    pub input_activation_gradients: WeightsContainer,
    pub hidden_update_gradients: WeightsContainer,
    pub hidden_reset_gradients: WeightsContainer,
    pub hidden_activation_gradients: WeightsContainer,
    pub update_bias_gradients: LayerContainer,
    pub reset_bias_gradients: LayerContainer,
    pub activation_bias_gradients: LayerContainer,
    pub output_gradients: WeightsContainer
}

struct D3Info<'a>
{
    output_derivative: LayerContainer,
    hidden_derivative: &'a LayerContainer
}

struct D4Info<'a>
{
    update_gate: LayerContainer,
    d3: &'a LayerContainer
}

struct D6Info<'a>
{
    previous_hidden: LayerContainer,
    d3: &'a LayerContainer
}

struct D8Info<'a>
{
    update_gate: LayerContainer,
    d3: &'a LayerContainer
}

struct ActivationGateDerivativeInfo<'a>
{
    activation_gate: LayerContainer,
    d8: &'a LayerContainer
}

struct UpdateGateDerivativeInfo<'a>
{
    update_gate: LayerContainer,
    activation_gate: LayerContainer,
    d3: &'a LayerContainer,
    d6: &'a LayerContainer
}

struct D13Info<'a>
{
    d10: &'a LayerContainer
}

struct ResetGateDerivativeInfo<'a>
{
    reset_gate: LayerContainer,
    previous_hidden: LayerContainer,
    d13: &'a LayerContainer
}

struct HiddenDerivativeInfo<'a>
{
    reset_gate: LayerContainer,
    d4: &'a LayerContainer,
    d11: &'a LayerContainer,
    d13: LayerContainer,
    d18: &'a LayerContainer
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GRU
{
    input_update_weights: WeightsContainer,
    input_reset_weights: WeightsContainer,
    input_activation_weights: WeightsContainer,
    hidden_update_weights: WeightsContainer,
    hidden_reset_weights: WeightsContainer,
    hidden_activation_weights: WeightsContainer,
    update_biases: LayerContainer,
    reset_biases: LayerContainer,
    activation_biases: LayerContainer,
    output_weights: WeightsContainer
}

impl GRU
{
    pub fn new(word_vector_size: usize) -> Self
    {
        let weights_init = |previous: f64|
        {
            let v = 1.0 / previous.sqrt();

            (fastrand::f64() * 2.0 - 1.0) * v
        };

        Self{
        	input_update_weights: WeightsContainer::new_with(
				word_vector_size,
				HIDDEN_AMOUNT,
				|| weights_init(word_vector_size as f64)
			),
        	input_reset_weights: WeightsContainer::new_with(
				word_vector_size,
				HIDDEN_AMOUNT,
				|| weights_init(word_vector_size as f64)
			),
        	input_activation_weights: WeightsContainer::new_with(
				word_vector_size,
				HIDDEN_AMOUNT,
				|| weights_init(word_vector_size as f64)
			),
        	hidden_update_weights: WeightsContainer::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f64)
			),
        	hidden_reset_weights: WeightsContainer::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f64)
			),
        	hidden_activation_weights: WeightsContainer::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f64)
			),
            // initialize biases to 0 cuz i read somewhere thats good
            update_biases: LayerContainer::new(HIDDEN_AMOUNT),
            reset_biases: LayerContainer::new(HIDDEN_AMOUNT),
            activation_biases: LayerContainer::new(HIDDEN_AMOUNT),
            output_weights: WeightsContainer::new_with(
                HIDDEN_AMOUNT,
                word_vector_size,
                || weights_init(HIDDEN_AMOUNT as f64)
            )
        }
    }

    pub fn adjust_weights(&mut self, gradients: GRUGradients)
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
        
        self.input_update_weights.iter_mut().zip(input_update_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });
        
        self.input_reset_weights.iter_mut().zip(input_reset_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });
        
        self.input_activation_weights.iter_mut().zip(input_activation_gradients.iter())
            .for_each(|(this, g)|
            {
                *this += g;
            });

        self.hidden_update_weights.iter_mut().zip(hidden_update_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });
        
        self.hidden_reset_weights.iter_mut().zip(hidden_reset_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });
        
        self.hidden_activation_weights.iter_mut().zip(hidden_activation_gradients.iter())
            .for_each(|(this, g)|
            {
                *this += g;
            });

        self.update_biases.iter_mut().zip(update_bias_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });

        self.reset_biases.iter_mut().zip(reset_bias_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });

        self.activation_biases.iter_mut().zip(activation_bias_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });
        
        self.output_weights.iter_mut().zip(output_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });
    }

    pub fn accuracy<B: Borrow<LayerContainer>>(
        &self,
        input: impl Iterator<Item=(B, B)>
    ) -> f64
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let amount = input.len();

        let f_output = self.feedforward(input.into_iter());

        let predicted = f_output.outputs;

        Self::correct_guesses(
            predicted.into_iter(),
            output.into_iter()
        ) as f64 / amount as f64
    }

    fn correct_guesses<P, T>(
        predicted: impl Iterator<Item=P>,
        target: impl Iterator<Item=T>
    ) -> usize
    where
        P: Borrow<SoftmaxedLayer>,
        T: Borrow<LayerContainer>
    {
        predicted.zip(target).map(|(predicted, target)|
        {
            let target_index = target.borrow().highest_index();
            if predicted.borrow().pick_weighed(1.0) == target_index
            {
                1
            } else
            {
                0
            }
        }).sum()
    }

    pub fn loss<B: Borrow<LayerContainer>>(&self, input: impl Iterator<Item=(B, B)>) -> f64
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();

        let f_output = self.feedforward(input.into_iter());

        let predicted = f_output.outputs;

        Self::cross_entropy(
            predicted.into_iter().map(|v| v.0.into_iter()),
            output.into_iter().map(|v| v.borrow().clone().into_iter())
        )
    }

    fn cross_entropy<PIter, TIter>(
        predicted: impl Iterator<Item=PIter>,
        target: impl Iterator<Item=TIter>
    ) -> f64
    where
        PIter: Iterator<Item=f64>,
        TIter: Iterator<Item=f64>
    {
        let mut count = 0;

        let s: f64 = predicted.zip(target).map(|(predicted, target)|
        {
            count += 1;

            Self::cross_entropy_single(predicted, target)
        }).sum();

        -s / count as f64
    }

    fn cross_entropy_single(
        predicted: impl Iterator<Item=f64>,
        target: impl Iterator<Item=f64>
    ) -> f64
    {
        predicted.into_iter().zip(target.into_iter()).map(|(p, t)|
        {
            t * p.ln()
        }).sum()
    }

    fn d3(info: D3Info) -> LayerContainer
    {
        info.output_derivative + info.hidden_derivative
    }

    fn d4(info: D4Info) -> LayerContainer
    {
        info.update_gate.one_minus_this() * info.d3
    }

    fn d6(info: D6Info) -> LayerContainer
    {
        let d5 = info.previous_hidden * info.d3;

        d5 * -1.0
    }

    fn d8(info: D8Info) -> LayerContainer
    {
        info.update_gate * info.d3
    }

    // d10
    fn activation_gate_derivative(info: ActivationGateDerivativeInfo) -> LayerContainer
    {
        info.activation_gate.powi(2).one_minus_this() * info.d8
    }

    // d11
    fn update_gate_derivative(info: UpdateGateDerivativeInfo) -> LayerContainer
    {
        let d7 = info.activation_gate * info.d3;
        let d9 = d7 + info.d6;

        d9 * (info.update_gate.clone() * info.update_gate.one_minus_this())
    }

    fn d13(&self, info: D13Info) -> LayerContainer
    {
        self.hidden_activation_weights.mul_transposed(info.d10)
    }

    // d18
    fn reset_gate_derivative(info: ResetGateDerivativeInfo) -> LayerContainer
    {
        let d16 = info.previous_hidden * info.d13;

        (info.reset_gate.clone() * info.reset_gate.one_minus_this()) * &d16
    }

    // d23
    fn hidden_derivative(&self, info: HiddenDerivativeInfo) -> LayerContainer
    {
        let d15 = self.hidden_update_weights.mul_transposed(info.d11);

        let d17 = info.d13 * &info.reset_gate;

        let d19 = d17 + info.d4;

        let d21 = self.hidden_reset_weights.mul_transposed(info.d18);
        let d22 = d21 + d15;

        d19 + d22
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
    pub fn gradients(&self, input: InputOutputIter) -> GRUGradients
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let f_output = self.feedforward(input.iter().map(|x| *x));

        let mut output_gradients = WeightsContainer::new(
            self.output_weights.previous_size(), self.output_weights.this_size()
        );

        let mut input_update_gradients = WeightsContainer::new(
            self.input_update_weights.previous_size(), self.input_update_weights.this_size()
        );

        let mut input_reset_gradients = WeightsContainer::new(
            self.input_reset_weights.previous_size(), self.input_reset_weights.this_size()
        );

        let mut input_activation_gradients = WeightsContainer::new(
            self.input_activation_weights.previous_size(),
            self.input_activation_weights.this_size()
        );

        let mut hidden_update_gradients = WeightsContainer::new(
            self.hidden_update_weights.previous_size(), self.hidden_update_weights.this_size()
        );

        let mut hidden_reset_gradients = WeightsContainer::new(
            self.hidden_reset_weights.previous_size(), self.hidden_reset_weights.this_size()
        );

        let mut hidden_activation_gradients = WeightsContainer::new(
            self.hidden_activation_weights.previous_size(),
            self.hidden_activation_weights.this_size()
        );

        let mut update_bias_gradients = LayerContainer::new(HIDDEN_AMOUNT);
        let mut reset_bias_gradients = LayerContainer::new(HIDDEN_AMOUNT);
        let mut activation_bias_gradients = LayerContainer::new(HIDDEN_AMOUNT);

        let GRUOutput{
            outputs: f_outputs,
            hiddens,
            update,
            reset,
            activation
        } = f_output;

        for (t, predicted_output) in (0..output.len()).zip(f_outputs.into_iter()).rev()
        {
            let expected_output = unsafe{ *output.get_unchecked(t) };
            let hidden = unsafe{ hiddens.get_unchecked(t) };

            let expected_sum: f64 = expected_output.iter().sum();
            let diff = predicted_output.0 * expected_sum - expected_output;

            for y in 0..expected_output.len()
            {
                for x in 0..hidden.len()
                {
                    let weight = unsafe{ output_gradients.weight_unchecked_mut(x, y) };

                    *weight += unsafe{
                        diff.get_unchecked(y) * hidden.get_unchecked(x)
                    };
                }
            }

            let mut d3 = {
                let output_derivative = self.output_weights.mul_transposed(diff);
                let hidden_derivative = LayerContainer::new(HIDDEN_AMOUNT);

                Self::d3(D3Info{
                    output_derivative,
                    hidden_derivative: &hidden_derivative
                })
            };

            for b_t in (0..(t + 1)).rev()
            {
                let previous_hidden = if b_t == 0
                {
                    LayerContainer::new(HIDDEN_AMOUNT)
                } else
                {
                    unsafe{ hiddens.get_unchecked(b_t - 1) }.clone()
                };

                let d4 = Self::d4(D4Info{
                    update_gate: unsafe{ update.get_unchecked(b_t) }.clone(),
                    d3: &d3
                });

                let d6 = Self::d6(D6Info{
                    previous_hidden: previous_hidden.clone(),
                    d3: &d3
                });

                let d8 = Self::d8(D8Info{
                    update_gate: unsafe{ update.get_unchecked(b_t) }.clone(),
                    d3: &d3
                });

                // d10
                let activation_gate_derivative = Self::activation_gate_derivative(
                    ActivationGateDerivativeInfo{
                        activation_gate: unsafe{ activation.get_unchecked(b_t) }.clone(),
                        d8: &d8
                    }
                );

                // d11
                let update_gate_derivative = Self::update_gate_derivative(
                    UpdateGateDerivativeInfo{
                        activation_gate: unsafe{ activation.get_unchecked(b_t) }.clone(),
                        update_gate: unsafe{ update.get_unchecked(b_t) }.clone(),
                        d3: &d3,
                        d6: &d6
                    }
                );

                let d13 = self.d13(D13Info{
                    d10: &activation_gate_derivative
                });

                // d18
                let reset_gate_derivative = Self::reset_gate_derivative(ResetGateDerivativeInfo{
                    reset_gate: unsafe{ reset.get_unchecked(b_t) }.clone(),
                    previous_hidden: previous_hidden.clone(),
                    d13: &d13
                });

                hidden_update_gradients.add_outer_product(&update_gate_derivative, &previous_hidden);
                hidden_reset_gradients.add_outer_product(&reset_gate_derivative, &previous_hidden);
                
                {
                    let previous_hidden = previous_hidden * unsafe{ reset.get_unchecked(b_t) };
                    hidden_activation_gradients.add_outer_product(
                        &activation_gate_derivative,
                        previous_hidden
                    );
                }

                let this_input = unsafe{ *input.get_unchecked(b_t) };
                input_update_gradients.add_outer_product(&update_gate_derivative, this_input);
                input_reset_gradients.add_outer_product(&reset_gate_derivative, this_input);
                
                input_activation_gradients.add_outer_product(
                    &activation_gate_derivative,
                    this_input
                );

                update_bias_gradients += &update_gate_derivative;
                reset_bias_gradients += &reset_gate_derivative;
                activation_bias_gradients += activation_gate_derivative;

                d3 = self.hidden_derivative(HiddenDerivativeInfo{
                    reset_gate: unsafe{ reset.get_unchecked(b_t) }.clone(),
                    d4: &d4,
                    d11: &update_gate_derivative,
                    d13,
                    d18: &reset_gate_derivative
                });
            }
        }

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

    pub fn feedforward_single(
        &self,
        previous_hidden: &LayerContainer,
        input: &LayerContainer
    ) -> GRUSingleOutput
    {
        let mut update_gate =
            self.hidden_update_weights.mul(previous_hidden)
            + self.input_update_weights.mul(input)
            + &self.update_biases;

        update_gate.map(|x| 1.0 / (1.0 + f64::consts::E.powf(-x)));

        let mut reset_gate =
            self.hidden_reset_weights.mul(previous_hidden)
            + self.input_reset_weights.mul(input)
            + &self.reset_biases;

        reset_gate.map(|x| 1.0 / (1.0 + f64::consts::E.powf(-x)));

        let activation_v = reset_gate.clone() * previous_hidden;
        let mut activation_gate =
            self.hidden_activation_weights.mul(activation_v)
            + self.input_activation_weights.mul(input)
            + &self.activation_biases;

        activation_gate.map(f64::tanh);

        let this_activation = activation_gate.clone() * &update_gate;
        let hidden = update_gate.clone().one_minus_this() * previous_hidden + this_activation;

        let output = SoftmaxedLayer::new(self.output_weights.mul(&hidden));

        GRUSingleOutput{
            update: update_gate,
            reset: reset_gate,
            activation: activation_gate,
            hidden,
            output
        }
    }

    pub fn feedforward<L>(&self, input: impl Iterator<Item=L>) -> GRUOutput
    where
        L: Borrow<LayerContainer>
    {
        let first_hidden = LayerContainer::new(HIDDEN_AMOUNT);

        let (lower_bound, upper_bound) = input.size_hint();
        let time_total = upper_bound.unwrap_or(lower_bound);

        let mut update = Vec::with_capacity(time_total);
        let mut reset = Vec::with_capacity(time_total);
        let mut activation = Vec::with_capacity(time_total);
        let mut hiddens = Vec::with_capacity(time_total);
        let mut outputs = Vec::with_capacity(time_total);

        for (t, this_input) in input.enumerate()
        {
            let previous_hidden = if t == 0
            {
                &first_hidden
            } else
            {
                unsafe{ hiddens.get_unchecked(t - 1) }
            };

            let this_input = this_input.borrow();

            let GRUSingleOutput{
                update: this_update,
                reset: this_reset,
                activation: this_activation,
                hidden,
                output
            } = self.feedforward_single(previous_hidden, this_input);

            update.push(this_update);
            reset.push(this_reset);
            activation.push(this_activation);
            hiddens.push(hidden);
            outputs.push(output);
        }

        GRUOutput{
            update,
            reset,
            activation,
            hiddens,
            outputs
        }
    }
}

#[cfg(test)]
pub mod tests
{
    use std::iter;

    use super::*;
    use crate::neural_network::WeightsIterValue;

    fn close_enough(a: f64, b: f64, epsilon: f64) -> bool
    {
        if (a == b) || ((a.min(b) == -0.0) && (a.max(b) == 0.0))
        {
            return true;
        }

        ((a - b).abs() / (a.abs() + b.abs())) < epsilon
    }

    #[ignore]
    #[allow(dead_code)]
    // #[test]
    fn forwardprop()
    {
        assert_eq!(HIDDEN_AMOUNT, 2);

        let input_update_weights = WeightsContainer::from_raw(
            vec![4.63, -2.64, 4.76, 3.63].into_boxed_slice(),
            2,
            HIDDEN_AMOUNT
        );

        let input_reset_weights = WeightsContainer::from_raw(
            vec![-8.29, 9.96, -4.78, 2.24].into_boxed_slice(),
            2,
            HIDDEN_AMOUNT
        );

        let input_activation_weights = WeightsContainer::from_raw(
            vec![-5.09, 1.99, 1.15, 4.63].into_boxed_slice(),
            2,
            HIDDEN_AMOUNT
        );

        let hidden_update_weights = WeightsContainer::from_raw(
            vec![-0.48, 8.48, -6.14, 2.42].into_boxed_slice(),
            HIDDEN_AMOUNT,
            HIDDEN_AMOUNT
        );

        let hidden_reset_weights = WeightsContainer::from_raw(
            vec![-5.74, -2.66, -6.25, -9.21].into_boxed_slice(),
            HIDDEN_AMOUNT,
            HIDDEN_AMOUNT
        );

        let hidden_activation_weights = WeightsContainer::from_raw(
            vec![-3.95, -6.07, 6.36, -5.36].into_boxed_slice(),
            HIDDEN_AMOUNT,
            HIDDEN_AMOUNT
        );

        let update_biases = LayerContainer::from(vec![-2.00, -0.87]);
        let reset_biases = LayerContainer::from(vec![-8.36, -8.16]);
        let activation_biases = LayerContainer::from(vec![3.47, 3.52]);

        let output_weights = WeightsContainer::from_raw(
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

        let input = LayerContainer::from(vec![-5.90, 1.78]);
        let f_output = gru.feedforward(iter::once(input));

        let output = &f_output.outputs[0];

        let epsilon = 0.0000001;

        let z = vec![1.686366804463125e-15, 1.7044644055950665e-10];
        assert!(close_enough(z[0], f_output.update[0][0], epsilon));
        assert!(close_enough(z[1], f_output.update[0][1], epsilon));

        let r = vec![1.0, 0.9999999999633351];
        assert!(close_enough(r[0], f_output.reset[0][0], epsilon));
        assert!(close_enough(r[1], f_output.reset[0][1], epsilon));

        let h = vec![1.0, 0.9999048161632378];
        assert!(close_enough(h[0], f_output.activation[0][0], epsilon));
        assert!(close_enough(h[1], f_output.activation[0][1], epsilon));

        let hidden = vec![1.686366804463125e-15, 1.7043021681333174e-10];
        assert!(close_enough(hidden[0], f_output.hiddens[0][0], epsilon));
        assert!(close_enough(hidden[1], f_output.hiddens[0][1], epsilon));

        let o = vec![0.5000000000460197, 0.4999999999539802];
        let (correct, calculated) = (o[0], output.0[0]);
        assert!(
            close_enough(correct, calculated, epsilon),
            "correct: {correct}, calculated: {calculated}"
        );

        let (correct, calculated) = (o[1], output.0[1]);
        assert!(
            close_enough(correct, calculated, epsilon),
            "correct: {correct}, calculated: {calculated}"
        );
    }

    #[test]
    fn loss_correct()
    {
        let predicted = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.01, 0.01, 0.01, 0.96]
        ];

        let target = vec![
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0]
        ];

        let loss = GRU::cross_entropy(
            predicted.into_iter().map(|v| v.into_iter()),
            target.into_iter().map(|v| v.into_iter())
        );

        assert!(
            close_enough(loss, 0.71355817782, 0.000001),
            "loss: {loss}"
        );
    }

    pub fn input_update_gradients_check(
        network: &mut GRU,
        input: InputOutputIter 
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.input_update_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients(input);

        true_gradient.iter_pos().zip(calculated_gradient.input_update_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn input_reset_gradients_check(
        network: &mut GRU,
        input: InputOutputIter 
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.input_reset_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients(input);

        true_gradient.iter_pos().zip(calculated_gradient.input_reset_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn input_activation_gradients_check(
        network: &mut GRU,
        input: InputOutputIter 
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.input_activation_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients(input);

        true_gradient.iter_pos().zip(calculated_gradient.input_activation_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_update_gradients_check(
        network: &mut GRU,
        input: InputOutputIter 
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.hidden_update_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients(input);

        true_gradient.iter_pos().zip(calculated_gradient.hidden_update_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_reset_gradients_check(
        network: &mut GRU,
        input: InputOutputIter 
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.hidden_reset_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients(input);

        true_gradient.iter_pos().zip(calculated_gradient.hidden_reset_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_activation_gradients_check(
        network: &mut GRU,
        input: InputOutputIter 
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.hidden_activation_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients(input);

        true_gradient.iter_pos().zip(calculated_gradient.hidden_activation_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn update_bias_gradients_check(
        network: &mut GRU,
        input: InputOutputIter 
    )
    {
        let true_gradient = gradient_bias_check(
            network,
            |network| &mut network.update_biases,
            input.clone()
        );

        let calculated_gradient = network.gradients(input);

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.update_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn reset_bias_gradients_check(
        network: &mut GRU,
        input: InputOutputIter 
    )
    {
        let true_gradient = gradient_bias_check(
            network,
            |network| &mut network.reset_biases,
            input.clone()
        );

        let calculated_gradient = network.gradients(input);

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.reset_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn activation_bias_gradients_check(
        network: &mut GRU,
        input: InputOutputIter 
    )
    {
        let true_gradient = gradient_bias_check(
            network,
            |network| &mut network.activation_biases,
            input.clone()
        );

        let calculated_gradient = network.gradients(input);

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.activation_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn output_gradients_check(
        network: &mut GRU,
        input: InputOutputIter 
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.output_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients(input);

        true_gradient.iter_pos().zip(calculated_gradient.output_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    fn check_single_gradient(pair: (WeightsIterValue<f64>, WeightsIterValue<f64>))
    {
        let (true_gradient, calculated_gradient) = pair;
        let (previous, this) = (true_gradient.previous, true_gradient.this);

        let true_gradient = true_gradient.value;
        let calculated_gradient = calculated_gradient.value;

        assert!(
            close_enough(true_gradient, calculated_gradient, 0.1),
            "true_gradient: {true_gradient}, calculated_gradient: {calculated_gradient}, previous_index: {previous}, this_index: {this}"
        );
    }

    fn check_single_bias_gradient(pair: ((usize, &f64), (usize, &f64)))
    {
        let (true_gradient, calculated_gradient) = pair;
        let index = true_gradient.0;

        let true_gradient = true_gradient.1;
        let calculated_gradient = calculated_gradient.1;

        assert!(
            close_enough(*true_gradient, *calculated_gradient, 0.1),
            "true_gradient: {true_gradient}, calculated_gradient: {calculated_gradient}, index: {index}"
        );
    }

    pub fn gradient_bias_check(
        network: &mut GRU,
        mut bias_member: impl FnMut(&mut GRU) -> &mut LayerContainer,
        input: InputOutputIter
    ) -> LayerContainer
    {
        let biases = bias_member(network).iter().enumerate().map(|(index, bias)|
        {
            (index, *bias)
        }).collect::<Vec<_>>();

        let biases = biases.into_iter().map(|(index, bias)|
        {
            let epsilon = 0.0001;

            let mut set_this_weight = |network: &mut GRU, value|
            {
                bias_member(network)[index] = value;
            };

            set_this_weight(network, bias - epsilon);
            let under_loss = network.loss(input.clone());
            
            set_this_weight(network, bias + epsilon);
            let over_loss = network.loss(input.clone());

            set_this_weight(network, bias);

            (over_loss - under_loss) / (2.0 * epsilon)
        }).collect::<Vec<_>>();

        LayerContainer::from(biases)
    }

    pub fn gradient_check(
        network: &mut GRU,
        mut weights_member: impl FnMut(&mut GRU) -> &mut WeightsContainer,
        input: InputOutputIter
    ) -> WeightsContainer
    {
        let weights = weights_member(network).iter_pos().map(|weight|
        {
            weight.to_owned()
        }).collect::<Vec<WeightsIterValue<_>>>();

        let weights = weights.into_iter().map(|weight|
        {
            let WeightsIterValue{value: weight, previous, this} = weight;
            let epsilon = 0.0001;

            let mut set_this_weight = |network: &mut GRU, value|
            {
                *weights_member(network).weight_mut(previous, this) = value;
            };

            // ; ;
            set_this_weight(network, weight - epsilon);
            let under_loss = network.loss(input.clone());
            
            set_this_weight(network, weight + epsilon);
            let over_loss = network.loss(input.clone());

            set_this_weight(network, weight);

            (over_loss - under_loss) / (2.0 * epsilon)
        }).collect::<Box<[_]>>();

        WeightsContainer::from_raw(
            weights,
            weights_member(network).previous_size(),
            weights_member(network).this_size()
        )
    }
}
