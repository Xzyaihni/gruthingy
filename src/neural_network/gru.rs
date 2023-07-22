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
pub struct GRUGradients
{
    pub input_update_gradients: WeightsContainer,
    pub input_reset_gradients: WeightsContainer,
    pub input_activation_gradients: WeightsContainer,
    pub hidden_update_gradients: WeightsContainer,
    pub hidden_reset_gradients: WeightsContainer,
    pub hidden_activation_gradients: WeightsContainer,
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
				HIDDEN_AMOUNT + 1,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f64)
			),
        	hidden_reset_weights: WeightsContainer::new_with(
				HIDDEN_AMOUNT + 1,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f64)
			),
        	hidden_activation_weights: WeightsContainer::new_with(
				HIDDEN_AMOUNT + 1,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f64)
			),
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
        
        self.output_weights.iter_mut().zip(output_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });
    }

    pub fn average_loss(&self, input: InputOutputIter) -> f64
    {
        let len = input.len();
        self.loss(input) / len as f64
    }

    fn loss(&self, input: InputOutputIter) -> f64
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let f_output = self.feedforward(&input);

        let predicted = f_output.outputs;
        
        let s: f64 = predicted.into_iter().zip(output.into_iter()).map(|(p, o)|
        {
            let mut p = p.0;
            p.map(|v| v.ln());

            p.dot(o)
        }).sum();

        -s
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
        self.hidden_activation_weights.mul_transposed_skip_last(info.d10)
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
        let d15 = self.hidden_update_weights.mul_transposed_skip_last(info.d11);

        let d17 = info.d13 * &info.reset_gate;

        let d19 = d17 + info.d4;

        let d21 = self.hidden_reset_weights.mul_transposed_skip_last(info.d18);
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
        let f_output = self.feedforward(&input);

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

                hidden_update_gradients += update_gate_derivative.outer_product(&previous_hidden);
                hidden_reset_gradients += reset_gate_derivative.outer_product(&previous_hidden);
                
                {
                    let previous_hidden = previous_hidden * unsafe{ reset.get_unchecked(b_t) };
                    hidden_activation_gradients +=
                        activation_gate_derivative.outer_product(previous_hidden);
                }

                let this_input = unsafe{ *input.get_unchecked(b_t) };
                input_update_gradients += update_gate_derivative.outer_product(this_input);
                input_reset_gradients += reset_gate_derivative.outer_product(this_input);
                
                input_activation_gradients +=
                    activation_gate_derivative.outer_product(this_input);

                for y in 0..HIDDEN_AMOUNT
                {
                    unsafe{
                        *hidden_update_gradients.weight_unchecked_mut(HIDDEN_AMOUNT, y)
                            += update_gate_derivative.get_unchecked(y);
                        
                        *hidden_reset_gradients.weight_unchecked_mut(HIDDEN_AMOUNT, y)
                            += reset_gate_derivative.get_unchecked(y);
                        
                        *hidden_activation_gradients.weight_unchecked_mut(HIDDEN_AMOUNT, y)
                            += activation_gate_derivative.get_unchecked(y);
                    }
                }

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
            output_gradients
        }
    }

    pub fn feedforward<L>(&self, input: &[L]) -> GRUOutput
    where
        L: Borrow<LayerContainer>
    {
        let time_total = input.len();

        let first_hidden = LayerContainer::new(HIDDEN_AMOUNT);

        let mut update = Vec::with_capacity(time_total);
        let mut reset = Vec::with_capacity(time_total);
        let mut activation = Vec::with_capacity(time_total);
        let mut hiddens = Vec::with_capacity(time_total);
        let mut outputs = Vec::with_capacity(time_total);

        for t in 0..time_total
        {
            let previous_hidden = if t == 0
            {
                &first_hidden
            } else
            {
                unsafe{ hiddens.get_unchecked(t - 1) }
            };

            let this_input = unsafe{ input.get_unchecked(t).borrow() };

            let update_bias = unsafe{ self.hidden_update_weights.this_unchecked(HIDDEN_AMOUNT) };
            let update_bias: LayerContainer = update_bias.cloned().collect();
            let mut update_gate =
                self.hidden_update_weights.mul(previous_hidden)
                + self.input_update_weights.mul(this_input)
                + update_bias;

            update_gate.map(|x| 1.0 / (1.0 + f64::consts::E.powf(-x)));
            update.push(update_gate.clone());

            let reset_bias = unsafe{ self.hidden_reset_weights.this_unchecked(HIDDEN_AMOUNT) };
            let reset_bias: LayerContainer = reset_bias.cloned().collect();
            let mut reset_gate =
                self.hidden_reset_weights.mul(previous_hidden)
                + self.input_reset_weights.mul(this_input)
                + reset_bias;

            reset_gate.map(|x| 1.0 / (1.0 + f64::consts::E.powf(-x)));
            reset.push(reset_gate.clone());

            let activation_bias = unsafe{
                self.hidden_activation_weights.this_unchecked(HIDDEN_AMOUNT)
            };
            let activation_bias: LayerContainer = activation_bias.cloned().collect();

            let activation_v = reset_gate * previous_hidden;
            let mut activation_gate =
                self.hidden_activation_weights.mul(activation_v)
                + self.input_activation_weights.mul(this_input)
                + activation_bias;

            activation_gate.map(f64::tanh);
            activation.push(activation_gate.clone());

            let this_activation = activation_gate * &update_gate;
            let hidden = (update_gate * -1.0 + 1.0) * previous_hidden + this_activation;

            let output = SoftmaxedLayer::new(self.output_weights.mul(&hidden));

            outputs.push(output);

            hiddens.push(hidden);
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
            vec![-0.48, 8.48, -2.00, -6.14, 2.42, -0.87].into_boxed_slice(),
            HIDDEN_AMOUNT + 1,
            HIDDEN_AMOUNT
        );

        let hidden_reset_weights = WeightsContainer::from_raw(
            vec![-5.74, -2.66, -8.36, -6.25, -9.21, -8.16].into_boxed_slice(),
            HIDDEN_AMOUNT + 1,
            HIDDEN_AMOUNT
        );

        let hidden_activation_weights = WeightsContainer::from_raw(
            vec![-3.95, -6.07, 3.47, 6.36, -5.36, 3.52].into_boxed_slice(),
            HIDDEN_AMOUNT + 1,
            HIDDEN_AMOUNT
        );

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
            output_weights
        };

        let input = LayerContainer::from(vec![-5.90, 1.78]);
        let f_output = gru.feedforward(&[input]);

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
