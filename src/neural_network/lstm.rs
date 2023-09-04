use std::{
    f32,
    array,
    ops::{DivAssign, AddAssign}
};

use strum::EnumCount;
use strum_macros::EnumCount;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        LayerType,
        LayerInnerType,
        HIDDEN_AMOUNT,
        network::{NetworkOutput, NewableLayer, WeightInfo},
        network_unit::NetworkUnit
    }
};


pub type LSTM = WeightsContainer<LayerType>;

#[repr(usize)]
#[derive(Debug, EnumCount)]
pub enum WeightIndex
{
    InputUpdate = 0,
    InputForget,
    InputOutput,
    InputMemory,
    HiddenUpdate,
    HiddenForget,
    HiddenOutput,
    HiddenMemory,
    UpdateBias,
    ForgetBias,
    OutputBias,
    MemoryBias,
    Output
}

const WEIGHTS_INFO: [(WeightInfo, WeightInfo, Option<WeightInfo>); 13] = [
    (WeightInfo::Hidden, WeightInfo::Input, Some(WeightInfo::Input)),
    (WeightInfo::Hidden, WeightInfo::Input, Some(WeightInfo::Input)),
    (WeightInfo::Hidden, WeightInfo::Input, Some(WeightInfo::Input)),
    (WeightInfo::Hidden, WeightInfo::Input, Some(WeightInfo::Input)),
    (WeightInfo::Hidden, WeightInfo::Hidden, Some(WeightInfo::Hidden)),
    (WeightInfo::Hidden, WeightInfo::Hidden, Some(WeightInfo::Hidden)),
    (WeightInfo::Hidden, WeightInfo::Hidden, Some(WeightInfo::Hidden)),
    (WeightInfo::Hidden, WeightInfo::Hidden, Some(WeightInfo::Hidden)),
    (WeightInfo::Hidden, WeightInfo::One, None),
    (WeightInfo::Hidden, WeightInfo::One, None),
    (WeightInfo::Hidden, WeightInfo::One, None),
    (WeightInfo::Hidden, WeightInfo::One, None),
    (WeightInfo::Input, WeightInfo::Hidden, Some(WeightInfo::Hidden)),
];

create_weights_container!{WeightIndex, WEIGHTS_INFO}

pub struct LSTMState
{
    hidden: LayerType,
    memory: LayerType
}

impl NetworkUnit for LSTM
{
    type State = LSTMState;
    type WeightsContainer<T> = WeightsContainer<T>;

    fn new(input_size: usize) -> Self
    {
        let weights_init = |previous: f32|
        {
            let v = 1.0 / previous.sqrt();

            (fastrand::f32() * 2.0 - 1.0) * v
        };

        WEIGHTS_INFO.into_iter().map(|(previous, current, prev_layer)|
        {
            let previous = previous.into_value(input_size);
            let current = current.into_value(input_size);
            let prev_layer = prev_layer.map(|prev_layer| prev_layer.into_value(input_size));

            LayerType::new_diff(
                if let Some(prev_layer) = prev_layer
                {
                    LayerInnerType::new_with(previous, current, || weights_init(prev_layer as f32))
                } else
                {
                    LayerInnerType::new(previous, current)
                }
            )
        }).collect()
    }

    fn feedforward_single_untrans(
        &mut self,
        previous_state: Option<&Self::State>,
        input: &LayerType
    ) -> NetworkOutput<Self::State, LayerType>
    {
        let mut forget_gate = self.weight(WeightIndex::InputForget).matmul(input)
            + self.weight(WeightIndex::ForgetBias);

        let mut update_gate = self.weight(WeightIndex::InputUpdate).matmul(input)
            + self.weight(WeightIndex::UpdateBias);

        let mut output_gate = self.weight(WeightIndex::InputOutput).matmul(input)
            + self.weight(WeightIndex::OutputBias);

        let mut memory_gate = self.weight(WeightIndex::InputMemory).matmul(input)
            + self.weight(WeightIndex::MemoryBias);

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
        }
    }

    fn parameters_amount(&self, i: u128) -> u128
    {
        let h = HIDDEN_AMOUNT as u128;

        (5 * i * h) + (4 * h * h) + (4 * h)
    }

    fn weights_mut(&mut self) -> &mut [LayerType]
    {
        &mut self.0
    }
}
