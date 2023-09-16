use std::{
    f32,
    array,
    ops::{DivAssign, AddAssign}
};

use strum::EnumCount;
use strum_macros::{FromRepr, EnumCount};

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        LayerType,
        LayerInnerType,
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

    type Iter<'a, T> = std::slice::Iter<'a, T>
    where
        T: 'a;

    type IterMut<'a, T> = std::slice::IterMut<'a, T>
    where
        T: 'a;

    fn new() -> Self
    {
        /*let weights_init = |previous: f32|
        {
            let v = 1.0 / previous.sqrt();

            (fastrand::f32() * 2.0 - 1.0) * v
        };

        WEIGHTS_INFO.into_iter().map(|(previous, current, prev_layer)|
        {
            LayerType::new_diff(
                if let Some(prev_layer) = prev_layer
                {
                    LayerInnerType::new_with(previous, current, || weights_init(prev_layer as f32))
                } else
                {
                    LayerInnerType::new(previous, current)
                }
            )
        }).collect()*/todo!();
    }

    fn feedforward_unit(
        &mut self,
        previous_state: Option<&Self::State>,
        input: &LayerType
    ) -> NetworkOutput<Self::State, LayerType>
    {
        /*let mut forget_gate = self.weight(WeightIndex::InputForget)
            .matmul_add(input, self.weight(WeightIndex::ForgetBias));

        let mut update_gate = self.weight(WeightIndex::InputUpdate)
            .matmul_add(input, self.weight(WeightIndex::UpdateBias));

        let mut output_gate = self.weight(WeightIndex::InputOutput)
            .matmul_add(input, self.weight(WeightIndex::OutputBias));

        let mut memory_gate = self.weight(WeightIndex::InputMemory)
            .matmul_add(input, self.weight(WeightIndex::MemoryBias));

        if let Some(previous_state) = previous_state
        {
            forget_gate += self.weight(WeightIndex::HiddenForget).matmul(&previous_state.hidden);
            update_gate += self.weight(WeightIndex::HiddenUpdate).matmul(&previous_state.hidden);
            output_gate += self.weight(WeightIndex::HiddenOutput).matmul(&previous_state.hidden);
            memory_gate += self.weight(WeightIndex::HiddenMemory).matmul(&previous_state.hidden);
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

        let output_untrans = self.weight(WeightIndex::Output).matmul(&hidden);

        let state = LSTMState{
            hidden,
            memory: this_memory
        };

        NetworkOutput{
            state,
            output: output_untrans
        }*/todo!();
    }

    fn weights_size(&self) -> Vec<WeightsSize<&LayerType>>
    {
        self.inner_weights_size().collect()
    }

    fn weights_info(&self) -> Vec<WeightsNamed<&LayerType>>
    {
        self.inner_weights_info().collect()
    }

    fn parameters_amount(&self) -> u128
    {
        let i = INPUT_SIZE as u128;
        let h = HIDDEN_AMOUNT as u128;

        (5 * i * h) + (4 * h * h) + (4 * h)
    }

    fn iter(&self) -> Self::Iter<'_, LayerType>
    {
        todo!();
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_, LayerType>
    {
        todo!();
    }
}

#[cfg(test)]
mod tests
{
    use super::*;
    
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
