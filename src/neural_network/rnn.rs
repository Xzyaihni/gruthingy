use serde::{Serialize, Deserialize};

use crate::neural_network::{
    LayerContainer,
    WeightsContainer,
    SoftmaxedLayer
};


const MAX_BPTT: usize = 10;
const HIDDEN_AMOUNT: usize = 5;

#[derive(Debug)]
pub struct RNNOutput
{
    pub hiddens: Vec<LayerContainer>,
    pub hiddens_ut: Vec<LayerContainer>,
    pub outputs: Vec<SoftmaxedLayer>
}

pub struct RNNGradients
{
    pub input_gradients: WeightsContainer,
    pub hidden_gradients: WeightsContainer,
    pub output_gradients: WeightsContainer
}

#[derive(Serialize, Deserialize)]
pub struct RNN
{
    input_weights: WeightsContainer,
    hidden_weights: WeightsContainer,
    output_weights: WeightsContainer
}

impl RNN
{
    pub fn new(word_vector_size: usize) -> Self
    {
        let weights_init = |previous: f64|
        {
            let v = 1.0 / previous.sqrt();

            (fastrand::f64() * 2.0 - 1.0) * v
        };

        Self{
            input_weights: WeightsContainer::new_with(
                word_vector_size,
                HIDDEN_AMOUNT,
                || weights_init(word_vector_size as f64)
            ),
            hidden_weights: WeightsContainer::new_with(
                HIDDEN_AMOUNT,
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

    pub fn adjust_weights(&mut self, gradients: RNNGradients)
    {
        let RNNGradients{output_gradients, hidden_gradients, input_gradients} = gradients;
        
        self.output_weights.iter_mut().zip(output_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });

        self.hidden_weights.iter_mut().zip(hidden_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });

        self.input_weights.iter_mut().zip(input_gradients.iter()).for_each(|(this, g)|
        {
            *this += g;
        });
    }

    #[inline(always)]
    fn activation_function(value: f64) -> f64
    {
        value.tanh()
        // this is a leaky relu, look at it
        /*if value > 0.0
        {
            value
        } else
        {
            0.01 * value
        }*/
    }

    #[inline(always)]
    fn activation_derivative(value: f64) -> f64
    {
        1.0 - value.tanh().powi(2)
        // if value > 0.0 {1.0} else {0.01}
    }

    pub fn average_loss(&self, input: &[Vec<LayerContainer>], output: &[Vec<LayerContainer>]) -> f64
    {
        let n: usize = input.iter().map(|i| i.len()).sum();

        self.loss(input, output) / n as f64
    }

    fn loss(&self, input: &[Vec<LayerContainer>], output: &[Vec<LayerContainer>]) -> f64
    {
        input.iter().zip(output.iter()).map(|(input, output)|
        {
            let f_output = self.feedforward(input);

            let predicted = f_output.outputs;
            
            let s: f64 = predicted.into_iter().zip(output.iter()).map(|(p, o)|
            {
                let mut p = p.0;
                p.map(|v| v.ln());

                p.dot(o.iter())
            }).sum();

            -s
        }).sum()
    }

    pub fn gradients(&self, input: &[LayerContainer], output: &[LayerContainer]) -> RNNGradients
    {
        let f_output = self.feedforward(input);

        let mut output_gradients = WeightsContainer::new(
            self.output_weights.previous_size(), self.output_weights.this_size()
        );

        let mut hidden_gradients = WeightsContainer::new(
            self.hidden_weights.previous_size(), self.hidden_weights.this_size()
        );

        let mut input_gradients = WeightsContainer::new(
            self.input_weights.previous_size(), self.input_weights.this_size()
        );

        let RNNOutput{outputs: f_outputs, hiddens, hiddens_ut} = f_output;

        // days spent trying to derive all this stuff, even when i did it its still very
        // confusing..
        for (i, predicted_output) in (0..output.len()).zip(f_outputs.into_iter()).rev()
        {
            let expected_output = unsafe{ output.get_unchecked(i) };
            let hidden = unsafe{ hiddens.get_unchecked(i) };
            let hidden_ut = unsafe{ hiddens_ut.get_unchecked(i) };

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

            let mut start_g: LayerContainer = (0..hidden.len()).map(|y|
            {
                let s = Self::activation_derivative(unsafe{ *hidden_ut.get_unchecked(y) });

                diff.dot(self.output_weights.this(y)) * s
            }).collect();

            let range_min = 0_i32.max(i as i32 - MAX_BPTT as i32) as usize;
            for b_i in (range_min..i+1).rev()
            {
                let this_input = &input[b_i];
                for y in 0..hidden.len()
                {
                    for x in 0..expected_output.len()
                    {
                        let weight = unsafe{ input_gradients.weight_unchecked_mut(x, y) };

                        *weight += unsafe{
                            start_g.get_unchecked(y) * this_input.get_unchecked(x)
                        };
                    }
                }

                if b_i != 0
                {
                    let previous_hiddens = unsafe{ hiddens.get_unchecked(b_i - 1) };
                    let previous_hiddens_ut = unsafe{ hiddens_ut.get_unchecked(b_i - 1) };

                    for y in 0..hidden.len()
                    {
                        for x in 0..hidden.len()
                        {
                            let weight = unsafe{ hidden_gradients.weight_unchecked_mut(x, y) };

                            *weight += unsafe{
                                start_g.get_unchecked(y) * previous_hiddens.get_unchecked(x)
                            };
                        }
                    }

                    start_g = (0..hidden.len()).map(|y|
                    {
                        let s = Self::activation_derivative(
                            unsafe{ *previous_hiddens_ut.get_unchecked(y) }
                        );

                        start_g.dot(unsafe{ self.hidden_weights.this_unchecked(y) }) * s
                    }).collect();
                } else
                {
                    start_g = (0..hidden.len()).map(|y|
                    {
                        start_g.dot(unsafe{ self.hidden_weights.this_unchecked(y) })
                    }).collect();
                }
            }
        }

        RNNGradients{
            input_gradients,
            hidden_gradients,
            output_gradients
        }
    }

    pub fn feedforward(&self, input: &[LayerContainer]) -> RNNOutput
    {
        let time_total = input.len();

        let first_hidden = LayerContainer::new(HIDDEN_AMOUNT);

        let mut hiddens_ut = Vec::with_capacity(time_total);
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

            let this_input = unsafe{ input.get_unchecked(t) };

            let mut hidden =
                self.input_weights.mul(this_input) + self.hidden_weights.mul(previous_hidden);

            hiddens_ut.push(hidden.clone());
            hidden.map(Self::activation_function);

            let output = SoftmaxedLayer::new(self.output_weights.mul(&hidden));

            outputs.push(output);

            hiddens.push(hidden);
        }

        RNNOutput{
            hiddens,
            hiddens_ut,
            outputs
        }
    }
}

#[cfg(test)]
pub mod tests
{
    use super::*;
    use crate::neural_network::WeightsIterValue;

    use std::slice;

    pub fn output_gradient_check(
        network: &mut RNN,
        input: &Vec<LayerContainer>,
        output: &Vec<LayerContainer>
    ) -> WeightsContainer
    {
        let output_weights = network.output_weights.iter_pos().map(|weight|
        {
            weight.to_owned()
        }).collect::<Vec<WeightsIterValue>>();

        let weights = output_weights.into_iter().map(|weight|
        {
            let WeightsIterValue{value: weight, previous, this} = weight;
            let epsilon = 0.001;

            let set_this_weight = |network: &mut RNN, value|
            {
                *network.output_weights.weight_mut(previous, this) = value;
            };

            // ; ;
            set_this_weight(network, weight - epsilon);
            let under_loss = network.loss(slice::from_ref(input), slice::from_ref(output));
            
            set_this_weight(network, weight + epsilon);
            let over_loss = network.loss(slice::from_ref(input), slice::from_ref(output));

            set_this_weight(network, weight);

            (over_loss - under_loss) / (2.0 * epsilon)
        }).collect::<Box<[_]>>();

        WeightsContainer::from_raw(
            weights,
            network.output_weights.previous_size(),
            network.output_weights.this_size()
        )
    }

    pub fn hidden_gradient_check(
        network: &mut RNN,
        input: &Vec<LayerContainer>,
        output: &Vec<LayerContainer>
    ) -> WeightsContainer
    {
        let hidden_weights = network.hidden_weights.iter_pos().map(|weight|
        {
            weight.to_owned()
        }).collect::<Vec<WeightsIterValue>>();

        let weights = hidden_weights.into_iter().map(|weight|
        {
            let WeightsIterValue{value: weight, previous, this} = weight;
            let epsilon = 0.001;

            let set_this_weight = |network: &mut RNN, value|
            {
                *network.hidden_weights.weight_mut(previous, this) = value;
            };

            // ; ;
            set_this_weight(network, weight - epsilon);
            let under_loss = network.loss(slice::from_ref(input), slice::from_ref(output));
            
            set_this_weight(network, weight + epsilon);
            let over_loss = network.loss(slice::from_ref(input), slice::from_ref(output));

            set_this_weight(network, weight);

            (over_loss - under_loss) / (2.0 * epsilon)
        }).collect::<Box<[_]>>();

        WeightsContainer::from_raw(
            weights,
            network.hidden_weights.previous_size(),
            network.hidden_weights.this_size()
        )
    }

    pub fn input_gradient_check(
        network: &mut RNN,
        input: &Vec<LayerContainer>,
        output: &Vec<LayerContainer>
    ) -> WeightsContainer
    {
        let input_weights = network.input_weights.iter_pos().map(|weight|
        {
            weight.to_owned()
        }).collect::<Vec<WeightsIterValue>>();

        let weights = input_weights.into_iter().map(|weight|
        {
            let WeightsIterValue{value: weight, previous, this} = weight;
            let epsilon = 0.001;

            let set_this_weight = |network: &mut RNN, value|
            {
                *network.input_weights.weight_mut(previous, this) = value;
            };

            // ; ;
            set_this_weight(network, weight - epsilon);
            let under_loss = network.loss(slice::from_ref(input), slice::from_ref(output));
            
            set_this_weight(network, weight + epsilon);
            let over_loss = network.loss(slice::from_ref(input), slice::from_ref(output));

            set_this_weight(network, weight);

            (over_loss - under_loss) / (2.0 * epsilon)
        }).collect::<Box<[_]>>();

        WeightsContainer::from_raw(
            weights,
            network.input_weights.previous_size(),
            network.input_weights.this_size()
        )
    }
}
