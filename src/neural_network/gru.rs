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
        INPUT_SIZE,
        network::{NetworkOutput, NewableLayer},
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

const WEIGHTS_INFO: [(usize, usize, Option<usize>); 10] = [
    (HIDDEN_AMOUNT, INPUT_SIZE, Some(INPUT_SIZE)),
    (HIDDEN_AMOUNT, INPUT_SIZE, Some(INPUT_SIZE)),
    (HIDDEN_AMOUNT, INPUT_SIZE, Some(INPUT_SIZE)),
    (HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
    (HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
    (HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
    (HIDDEN_AMOUNT, 1, None),
    (HIDDEN_AMOUNT, 1, None),
    (HIDDEN_AMOUNT, 1, None),
    (INPUT_SIZE, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
];

create_weights_container!{WeightIndex, WEIGHTS_INFO}

impl NetworkUnit for GRU
{
    type State = LayerType;
    type ThisWeightsContainer<T> = WeightsContainer<T>;

    type Iter<'a, T> = std::slice::Iter<'a, T>
    where
        T: 'a;

    type IterMut<'a, T> = std::slice::IterMut<'a, T>
    where
        T: 'a;

    fn new() -> Self
    {
        let weights_init = |previous: f32|
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

        // i hope i calculated this right
        (4 * i * h) + (3 * h * h) + (3 * h)
    }

    fn iter(&self) -> Self::Iter<'_, LayerType>
    {
        self.0.iter()
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_, LayerType>
    {
        self.0.iter_mut()
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
