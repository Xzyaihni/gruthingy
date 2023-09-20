use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        LayerType,
        HIDDEN_AMOUNT,
        INPUT_SIZE,
        network::NetworkOutput,
        network_unit::NetworkUnit
    }
};


pub type LSTM = WeightsContainer<LayerType>;

create_weights_container!{
    (input_update, false, HIDDEN_AMOUNT, INPUT_SIZE, Some(INPUT_SIZE)),
    (input_forget, false, HIDDEN_AMOUNT, INPUT_SIZE, Some(INPUT_SIZE)),
    (input_output, false, HIDDEN_AMOUNT, INPUT_SIZE, Some(INPUT_SIZE)),
    (input_memory, false, HIDDEN_AMOUNT, INPUT_SIZE, Some(INPUT_SIZE)),
    (hidden_update, true, HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
    (hidden_forget, true, HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
    (hidden_output, true, HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
    (hidden_memory, true, HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
    (update_bias, false, HIDDEN_AMOUNT, 1, None),
    (forget_bias, false, HIDDEN_AMOUNT, 1, None),
    (output_bias, false, HIDDEN_AMOUNT, 1, None),
    (memory_bias, false, HIDDEN_AMOUNT, 1, None),
    (output, false, INPUT_SIZE, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT))
}

pub struct LSTMState
{
    hidden: LayerType,
    memory: LayerType
}

impl NetworkUnit for LSTM
{
    type State = LSTMState;
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
        let mut forget_gate = self.input_forget.matmul_add(input, &self.forget_bias);
        let mut update_gate = self.input_update.matmul_add(input, &self.update_bias);
        let mut output_gate = self.input_output.matmul_add(input, &self.output_bias);
        let mut memory_gate = self.input_memory.matmul_add(input, &self.memory_bias);

        if let Some(previous_state) = previous_state
        {
            forget_gate += self.hidden_forget.matmul(&previous_state.hidden);
            update_gate += self.hidden_update.matmul(&previous_state.hidden);
            output_gate += self.hidden_output.matmul(&previous_state.hidden);
            memory_gate += self.hidden_memory.matmul(&previous_state.hidden);
        }

        forget_gate.sigmoid();
        update_gate.sigmoid();
        output_gate.sigmoid();
        memory_gate.tanh();

        let this_memory_rhs = update_gate * memory_gate;

        let this_memory = if let Some(previous_state) = previous_state
        {
            forget_gate * &previous_state.memory + this_memory_rhs
        } else
        {
            this_memory_rhs
        };

        let hidden = {
            let mut memory = this_memory.clone_gradientable();
            memory.tanh();

            output_gate * memory
        };

        let output_untrans = self.output.matmul(&hidden);

        let state = LSTMState{
            hidden,
            memory: this_memory
        };

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

        (5 * i * h) + (4 * h * h) + (4 * h)
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    use crate::neural_network::LayerInnerType;
    
    fn close_enough(a: f32, b: f32, epsilon: f32) -> bool
    {
        if a == b
        {
            return true;
        }

        let relative_diff = (a - b).abs() / (a.abs() + b.abs());

        relative_diff < epsilon
    }

    fn assert_close_enough(a: f32, b: f32, epsilon: f32)
    {
        assert!(close_enough(a, b, epsilon), "a: {a}, b: {b}");
    }

    #[test]
    fn lstm_works()
    {
        let one_weight = |value: f32|
        {
            LayerType::new_diff(LayerInnerType::from_raw([value], 1, 1))
        };

        /*
        InputUpdate
        InputForget
        InputOutput
        InputMemory
        HiddenUpdate
        HiddenForget
        HiddenOutput
        HiddenMemory
        UpdateBias
        ForgetBias
        OutputBias
        MemoryBias
        Output
        */

        let mut lstm = WeightsContainer
        {
            input_update: one_weight(1.65),
            input_forget: one_weight(1.63),
            input_output: one_weight(-0.19),
            input_memory: one_weight(0.94),

            hidden_update: one_weight(2.00),
            hidden_forget: one_weight(2.70),
            hidden_output: one_weight(4.38),
            hidden_memory: one_weight(1.41),

            update_bias: one_weight(0.62),
            forget_bias: one_weight(1.62),
            output_bias: one_weight(0.59),
            memory_bias: one_weight(-0.32),

            output: one_weight(1.0)
        };

        let state = LSTMState{
            memory: one_weight(2.0),
            hidden: one_weight(1.0)
        };

        let input = one_weight(1.0);

        let output = lstm.feedforward_unit(Some(&state), &input);

        let epsilon = 0.0001;

        let single_value = |l: &LayerType|
        {
            l.as_vec()[0]
        };

        assert_close_enough(single_value(&output.state.memory), 2.947, epsilon);
        assert_close_enough(single_value(&output.state.hidden), 0.986229, epsilon);
    }
}
