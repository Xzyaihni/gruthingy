use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        LayerType,
        ScalarType,
        LayerSizes,
        network::{NetworkOutput, LayerSize},
        network_unit::NetworkUnit
    }
};


pub type Gru<T> = WeightsContainer<T>;

create_weights_container!{
    (input_update, false, LayerSize::Hidden, LayerSize::Input),
    (input_reset, false, LayerSize::Hidden, LayerSize::Input),
    (input_activation, false, LayerSize::Hidden, LayerSize::Input),
    (hidden_update, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_reset, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_activation, true, LayerSize::Hidden, LayerSize::Hidden),
    (update_bias, false, LayerSize::Hidden, LayerSize::One),
    (reset_bias, false, LayerSize::Hidden, LayerSize::One),
    (activation_bias, false, LayerSize::Hidden, LayerSize::One),
    (output, false, LayerSize::Input, LayerSize::Hidden)
}

impl NetworkUnit for Gru<LayerType>
{
    type State = LayerType;

    fn new(sizes: LayerSizes) -> Self
    {
        WeightsContainer::new_randomized(sizes)
    }

    fn feedforward_unit(
        &mut self,
        previous_state: Option<&Self::State>,
        input: &LayerType
    ) -> NetworkOutput<Self::State, LayerType>
    {
        let mut update_gate = self.input_update.matmulv_add(input, &self.update_bias);
        let mut reset_gate = self.input_reset.matmulv_add(input, &self.reset_bias);
        let mut activation_gate = self.input_activation.matmulv_add(input, &self.activation_bias);

        if let Some(previous_state) = previous_state
        {
            update_gate += self.hidden_update.matmulv(previous_state);
            reset_gate += self.hidden_reset.matmulv(previous_state);
        }

        update_gate.sigmoid();
        reset_gate.sigmoid();

        if let Some(previous_state) = previous_state
        {
            let activation_v = &reset_gate * previous_state;
            activation_gate += self.hidden_activation.matmulv(activation_v);
        }

        activation_gate.tanh();

        let this_activation = &activation_gate * &update_gate;

        let state = if let Some(previous_state) = previous_state
        {
            ScalarType::new(1.0) - &update_gate * previous_state + this_activation
        } else
        {
            this_activation + ScalarType::new(1.0)
        };

        let output_untrans = self.output.matmulv(&state);

        NetworkOutput{
            state,
            output: output_untrans
        }
    }

    fn parameters_amount(&self, sizes: LayerSizes) -> u128
    {
        let i = sizes.input as u128;
        let h = sizes.hidden as u128;

        // i hope i calculated this right
        (4 * i * h) + (3 * h * h) + (3 * h)
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

    #[allow(dead_code)]
    pub fn close_enough_abs(a: f32, b: f32, epsilon: f32) -> bool
    {
        (a - b).abs() < epsilon
    }
}
