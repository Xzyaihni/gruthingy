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
        ScalarType,
        LayerInnerType,
        HIDDEN_AMOUNT,
        network::{NetworkOutput, NewableLayer},
        network_unit::NetworkUnit
    }
};


pub type GRU = WeightsContainer<LayerType>;

#[repr(usize)]
#[derive(Debug, EnumCount)]
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

// create_weights_container!{WeightIndex}
#[derive(Debug, Serialize, Deserialize)]
pub struct WeightsContainer<T>([T; WeightIndex::COUNT]);

impl<T> IntoIterator for WeightsContainer<T>
{
    type Item = T;
    type IntoIter = array::IntoIter<T, { WeightIndex::COUNT }>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.0.into_iter()
    }
}

impl<T> FromIterator<T> for WeightsContainer<T>
{
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self
    {
        let inner = iter.into_iter().collect::<Vec<_>>().try_into().map_err(|_|
        {
            "iterator length doesnt match"
        }).unwrap();

        Self(inner)
    }
}

impl<T: NewableLayer> WeightsContainer<T>
{
    pub fn new_container(input_size: usize) -> Self
    {
        let weights = [
            (HIDDEN_AMOUNT, input_size, Some(input_size)),
            (HIDDEN_AMOUNT, input_size, Some(input_size)),
            (HIDDEN_AMOUNT, input_size, Some(input_size)),
            (HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
            (HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
            (HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
            (HIDDEN_AMOUNT, 1, None),
            (HIDDEN_AMOUNT, 1, None),
            (HIDDEN_AMOUNT, 1, None),
            (input_size, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
        ];

        weights.into_iter().map(|(previous, current, _prev_layer)|
        {
            T::new(previous, current)
        }).collect()
    }
}

impl<T: DivAssign<f32>> DivAssign<f32> for WeightsContainer<T>
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.0.iter_mut().for_each(|value|
        {
            *value /= rhs;
        });
    }
}

impl<T: AddAssign<T>> AddAssign for WeightsContainer<T>
{
    fn add_assign(&mut self, rhs: Self)
    {
        self.0.iter_mut().zip(rhs.0.into_iter()).for_each(|(value, rhs)|
        {
            *value += rhs;
        });
    }
}

impl<T> WeightsContainer<T>
{
    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut T>
    {
        self.0.iter_mut()
    }

    pub fn weight(&self, index: WeightIndex) -> &T
    {
        &self.0[index as usize]
    }
}

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

        let weights = [
            (HIDDEN_AMOUNT, input_size, Some(input_size)),
            (HIDDEN_AMOUNT, input_size, Some(input_size)),
            (HIDDEN_AMOUNT, input_size, Some(input_size)),
            (HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
            (HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
            (HIDDEN_AMOUNT, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
            (HIDDEN_AMOUNT, 1, None),
            (HIDDEN_AMOUNT, 1, None),
            (HIDDEN_AMOUNT, 1, None),
            (input_size, HIDDEN_AMOUNT, Some(HIDDEN_AMOUNT)),
        ];

        weights.into_iter().map(|(previous, current, prev_layer)|
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

    fn feedforward_single_untrans(
        &mut self,
        previous_state: Option<&LayerType>,
        input: &LayerType
    ) -> NetworkOutput<LayerType, LayerType>
    {
        let mut update_gate = self.weight(WeightIndex::InputUpdate).matmul(input)
            + self.weight(WeightIndex::UpdateBias);

        let mut reset_gate = self.weight(WeightIndex::InputReset).matmul(input)
            + self.weight(WeightIndex::ResetBias);

        let mut activation_gate = self.weight(WeightIndex::InputActivation).matmul(input)
            + self.weight(WeightIndex::ActivationBias);

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

    fn parameters_amount(&self, i: u128) -> u128
    {
        let h = HIDDEN_AMOUNT as u128;

        // i hope i calculated this right
        (4 * i * h) + (3 * h * h) + (3 * h)
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

    pub fn close_enough_loose(a: f32, b: f32, epsilon: f32) -> bool
    {
        if a == 0.0 || a == -0.0
        {
            return b.abs() < epsilon;
        }

        if b == 0.0 || b == -0.0
        {
            return a.abs() < epsilon;
        }

        ((a - b).abs() / (a.abs() + b.abs())) < epsilon
    }

    #[allow(dead_code)]
    pub fn close_enough_abs(a: f32, b: f32, epsilon: f32) -> bool
    {
        (a - b).abs() < epsilon
    }
}
