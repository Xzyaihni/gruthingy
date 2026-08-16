use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        OperationsRecorder,
        NetworkUnitStateable,
        TensorIndex,
        DiffTensor,
        DiffTensorPtr,
        DiffInputType,
        LayerSizes,
        WeightInfo,
        WeightInfoPtr,
        NetworkUnitNewable,
        network::{NetworkOutput, LayerSize},
        network_unit::{NetworkUnit, NetworkUnitParameterable}
    }
};


pub type Gru<T> = WeightsContainer<T>;

create_weights_container!{
    (input_update, false, LayerSize::Input, LayerSize::Hidden),
    (input_reset, false, LayerSize::Input, LayerSize::Hidden),
    (input_activation, false, LayerSize::Input, LayerSize::Hidden),
    (hidden_update, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_reset, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_activation, true, LayerSize::Hidden, LayerSize::Hidden),
    (update_bias, false, LayerSize::One, LayerSize::Hidden),
    (reset_bias, false, LayerSize::One, LayerSize::Hidden),
    (activation_bias, false, LayerSize::One, LayerSize::Hidden)
}

impl NetworkUnitStateable for DiffTensor
{
    fn set(&self, recorder: &mut OperationsRecorder, new: &Self)
    {
        let mut set_single = |old: TensorIndex, new: TensorIndex|
        {
            let new_tensor = recorder.get_tensor(new).clone();

            recorder.set_tensor(old, new_tensor);
        };

        set_single(self.as_value(), new.as_value());
        set_single(self.as_gradient().unwrap(), new.as_gradient().unwrap());
    }
}

impl NetworkUnitNewable for Gru<WeightInfoPtr>
{
    fn new(recorder: &mut OperationsRecorder, sizes: LayerSizes) -> Self
    {
        WeightsContainer::new_randomized(recorder, sizes)
    }
}

impl NetworkUnitParameterable for Gru<WeightInfo>
{
    fn parameters_amount(&self, sizes: LayerSizes) -> u128
    {
        let i = sizes.input as u128;
        let h = sizes.hidden as u128;

        // i hope i calculated this right
        (3 * i * h) + (3 * h * h) + (3 * h)
    }
}

impl NetworkUnit for Gru<WeightInfoPtr>
{
    type State<T> = T;

    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: Option<&Self::State<DiffTensorPtr>>,
        input: DiffInputType
    ) -> NetworkOutput<Self::State<DiffTensorPtr>, DiffTensorPtr>
    {
        let block = recorder.current_block();

        if previous_state.is_some()
        {
            let mut store_both = |weight: DiffTensorPtr|
            {
                recorder.store_tensor_until_end_in_block(block, weight.as_value());
                recorder.store_tensor_until_end_in_block(block, weight.as_gradient().unwrap());
            };

            store_both(self.hidden_update.weight_original);
            store_both(self.hidden_reset.weight_original);
            store_both(self.hidden_activation.weight_original);

            store_both(self.reset_bias.weight_original);

            store_both(self.input_reset.weight_original);
        }

        {
            let mut always_store = |weight: DiffTensorPtr|
            {
                recorder.store_tensor_until_end(weight.as_value());
                recorder.store_tensor_until_end(weight.as_gradient().unwrap());
            };

            always_store(self.update_bias.weight_original);
            always_store(self.activation_bias.weight_original);

            always_store(self.input_update.weight_original);
            always_store(self.input_activation.weight_original);
        }

        let block_index = block.into_index();

        let mut matmul_inputv_add = |weights: WeightInfoPtr, input, bias: WeightInfoPtr|
        {
            let weights = weights.weight_dropped[block_index];
            let bias = bias.weight_dropped[block_index];

            match input
            {
                DiffInputType::Normal(x) => recorder.matmulv_add(weights, x, bias),
                DiffInputType::OneHot(x) => recorder.matmul_onehotv_add(weights, x, bias)
            }
        };

        let mut update_gate = matmul_inputv_add(self.input_update, input, self.update_bias);
        let mut reset_gate = matmul_inputv_add(self.input_reset, input, self.reset_bias);
        let mut activation_gate = matmul_inputv_add(self.input_activation, input, self.activation_bias);

        if let Some(previous_state) = previous_state
        {
            let mut do_gate = |gate: &mut _, hidden: WeightInfoPtr|
            {
                let mm = recorder.matmulv(hidden.weight_dropped[block_index], *previous_state);
                *gate = recorder.add(*gate, mm);
            };

            do_gate(&mut update_gate, self.hidden_update);
            do_gate(&mut reset_gate, self.hidden_reset);
        }

        update_gate = recorder.sigmoid(update_gate);
        reset_gate = recorder.sigmoid(reset_gate);

        if let Some(previous_state) = previous_state
        {
            let activation_v = recorder.mul_componentwise(reset_gate, *previous_state);
            let mm = recorder.matmulv(self.hidden_activation.weight_dropped[block_index], activation_v);

            activation_gate = recorder.add(activation_gate, mm);
        }

        activation_gate = recorder.tanh(activation_gate);

        let this_activation = recorder.mul_componentwise(activation_gate, update_gate);

        let one = recorder.set_new_value(1.0);

        let state = if let Some(previous_state) = previous_state
        {
            let update = recorder.mul_componentwise(update_gate, *previous_state);

            let left = recorder.sub_from_scalar(one, update);

            recorder.add(left, this_activation)
        } else
        {
            recorder.add_scalar(this_activation, one)
        };

        NetworkOutput{
            state: state.clone(),
            output: state
        }
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
