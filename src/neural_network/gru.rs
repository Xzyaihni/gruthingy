use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        LayerType,
        ScalarType,
        HIDDEN_AMOUNT,
        INPUT_SIZE,
        network::NetworkOutput,
        network_unit::NetworkUnit
    }
};


pub type Gru = WeightsContainer<LayerType>;

create_weights_container!{
    (input_update, false, HIDDEN_AMOUNT, INPUT_SIZE, Some(INPUT_SIZE)),
    (input_reset, false, HIDDEN_AMOUNT, INPUT_SIZE, Some(INPUT_SIZE)),
    (input_activation, false, HIDDEN_AMOUNT, INPUT_SIZE, Some(INPUT_SIZE)),
    (hidden_update, true, HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
    (hidden_reset, true, HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
    (hidden_activation, true, HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
    (update_bias, false, HIDDEN_AMOUNT, 1, None),
    (reset_bias, false, HIDDEN_AMOUNT, 1, None),
    (activation_bias, false, HIDDEN_AMOUNT, 1, None),
    (output, false, INPUT_SIZE, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT))
}

impl NetworkUnit for Gru
{
    type State = LayerType;
    type ThisWeightsContainer<T> = WeightsContainer<T>;

    fn new() -> Self
    {
        WeightsContainer::new_randomized()
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

    fn weights_named_info(&self) -> Self::ThisWeightsContainer<WeightsNamed<&LayerType>>
    {
        self.weights_named_info_inner()
    }

    fn for_each_weight<F: FnMut(&mut LayerType)>(&mut self, f: F)
    {
        self.for_each_weight_mut(f)
    }

    fn clone_weights_with_info<F>(&self, f: F) -> Self
    where
        F: FnMut(WeightsSize<&LayerType>) -> LayerType
    {
        self.clone_weights_with_info_inner(f)
    }

    fn map_weights_mut<F, U>(&mut self, f: F) -> Self::ThisWeightsContainer<U>
    where
        F: FnMut(&mut LayerType) -> U
    {
        self.map_weights_mut_inner(f)
    }

    fn parameters_amount(&self) -> u128
    {
        let i = INPUT_SIZE as u128;
        let h = HIDDEN_AMOUNT as u128;

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
