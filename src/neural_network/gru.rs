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
        ScalarType,
        LayerInnerType,
        HIDDEN_AMOUNT,
        network::{NetworkOutput, NewableLayer, WeightInfo},
        network_unit::NetworkUnit
    }
};


pub type GRU = WeightsContainer<LayerType>;

#[repr(usize)]
#[derive(Debug, EnumCount, FromRepr)]
pub enum WeightIndex
{
    InputUpdate = 0,
    InputReset,
    InputActivation,
    HiddenUpdate,
    HiddenReset,
    HiddenActivation,
    UpdateBias,
    ResetBias,
    ActivationBias,
    Output
}

impl WeightIndex
{
    pub fn is_hidden(self) -> bool
    {
        match self
        {
            Self::HiddenUpdate => true,
            Self::HiddenReset => true,
            Self::HiddenActivation => true,
            _ => false
        }
    }
}

const WEIGHTS_INFO: [(WeightInfo, WeightInfo, Option<WeightInfo>); 10] = [
    (WeightInfo::Hidden, WeightInfo::Input, Some(WeightInfo::Input)),
    (WeightInfo::Hidden, WeightInfo::Input, Some(WeightInfo::Input)),
    (WeightInfo::Hidden, WeightInfo::Input, Some(WeightInfo::Input)),
    (WeightInfo::Hidden, WeightInfo::Hidden, Some(WeightInfo::Hidden)),
    (WeightInfo::Hidden, WeightInfo::Hidden, Some(WeightInfo::Hidden)),
    (WeightInfo::Hidden, WeightInfo::Hidden, Some(WeightInfo::Hidden)),
    (WeightInfo::Hidden, WeightInfo::One, None),
    (WeightInfo::Hidden, WeightInfo::One, None),
    (WeightInfo::Hidden, WeightInfo::One, None),
    (WeightInfo::Input, WeightInfo::Hidden, Some(WeightInfo::Hidden)),
];

create_weights_container!{WeightIndex, WEIGHTS_INFO}

impl NetworkUnit for GRU
{
    type State = LayerType;
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

    fn feedforward_unit(
        &mut self,
        previous_state: Option<&Self::State>,
        input: &LayerType
    ) -> NetworkOutput<Self::State, LayerType>
    {
        let mut update_gate = self.weight(WeightIndex::InputUpdate)
            .matmul_add(input, self.weight(WeightIndex::UpdateBias));

        let mut reset_gate = self.weight(WeightIndex::InputReset)
            .matmul_add(input, self.weight(WeightIndex::ResetBias));

        let mut activation_gate = self.weight(WeightIndex::InputActivation)
            .matmul_add(input, self.weight(WeightIndex::ActivationBias));

        if let Some(previous_state) = previous_state
        {
            update_gate += self.weight(WeightIndex::HiddenUpdate).matmul(previous_state);
            reset_gate += self.weight(WeightIndex::HiddenReset).matmul(previous_state);
        }

        update_gate.sigmoid();
        reset_gate.sigmoid();

        if let Some(previous_state) = previous_state
        {
            let activation_v = &reset_gate * previous_state;
            activation_gate += self.weight(WeightIndex::HiddenActivation).matmul(activation_v);
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

        let output_untrans = self.weight(WeightIndex::Output).matmul(&state);

        NetworkOutput{
            state,
            output: output_untrans
        }
    }

    fn weights_size(&self, input_size: usize) -> Vec<WeightsSize<&LayerType>>
    {
        self.inner_weights_size(input_size).collect()
    }

    fn weights_info(&self, input_size: usize) -> Vec<WeightsNamed<&LayerType>>
    {
        self.inner_weights_info(input_size).collect()
    }

    fn parameters_amount(&self, i: u128) -> u128
    {
        let h = HIDDEN_AMOUNT as u128;

        // i hope i calculated this right
        (4 * i * h) + (3 * h * h) + (3 * h)
    }

    fn weights(&self) -> &[LayerType]
    {
        &self.0
    }

    fn weights_mut(&mut self) -> &mut [LayerType]
    {
        &mut self.0
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
