use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        OperationsRecorder,
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


pub type Star<T> = WeightsContainer<T>;

create_weights_container!{
    (input_z, false, false, LayerSize::Input, LayerSize::Hidden),
    (input_x, false, false, LayerSize::Input, LayerSize::Hidden),
    (hidden, true, true, LayerSize::Hidden, LayerSize::Hidden),
    (z_bias, false, false, LayerSize::One, LayerSize::Hidden),
    (k_bias, false, false, LayerSize::One, LayerSize::Hidden)
}

impl NetworkUnitNewable for Star<WeightInfoPtr>
{
    fn new(recorder: &mut OperationsRecorder, sizes: LayerSizes) -> Self
    {
        WeightsContainer::new_randomized(recorder, sizes)
    }
}

impl NetworkUnitParameterable for Star<WeightInfo>
{
    fn parameters_amount(&self, sizes: LayerSizes) -> u128
    {
        let i = sizes.input as u128;
        let h = sizes.hidden as u128;

        // i hope i calculated this right
        (3 * i * h) + (3 * h * h) + (3 * h)
    }
}

impl NetworkUnit for Star<WeightInfoPtr>
{
    type State<T> = T;

    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: Option<&Self::State<DiffTensorPtr>>,
        input: DiffInputType
    ) -> NetworkOutput<Self::State<DiffTensorPtr>, DiffTensorPtr>
    {
        let matmul_inputv_add = |recorder: &mut OperationsRecorder, weights: WeightInfoPtr, input, bias: WeightInfoPtr|
        {
            let weights = weights.weight_dropped;

            debug_assert!(bias.weight_dropped.is_undefined());
            debug_assert!(bias.dropout.is_none());

            let bias = bias.weight_original;

            match input
            {
                DiffInputType::Normal(x) => recorder.matmulv_add(weights, x, bias),
                DiffInputType::OneHot(x) => recorder.matmul_onehotv_add(weights, x, bias)
            }
        };

        let z_pre = matmul_inputv_add(recorder, self.input_z, input, self.z_bias);
        recorder.name_diff_tensor(z_pre, "z_pre");

        let z = recorder.tanh(z_pre);
        recorder.name_diff_tensor(z, "z");

        let mut k = matmul_inputv_add(recorder, self.input_x, input, self.k_bias);
        recorder.name_diff_tensor(k, "k");

        if let Some(previous_state) = previous_state
        {
            k = recorder.matmulv_add(self.hidden.weight_dropped, *previous_state, k);
            recorder.name_diff_tensor(k, "k_inner");
        }

        k = recorder.sigmoid(k);
        recorder.name_diff_tensor(k, "k_activated");

        let mut h_pre = recorder.mul_componentwise(k, z);
        recorder.name_diff_tensor(h_pre, "h_pre");

        if let Some(previous_state) = previous_state
        {
            let one = recorder.set_new_value(1.0);

            let lhs = recorder.sub_from_scalar(one, k);
            recorder.name_diff_tensor(lhs, "lhs");

            let hidden_part = recorder.mul_componentwise(lhs, *previous_state);
            recorder.name_diff_tensor(hidden_part, "hidden_part");

            h_pre = recorder.add(hidden_part, h_pre);
            recorder.name_diff_tensor(h_pre, "h_pre_inner");
        }

        let h = recorder.tanh(h_pre);
        recorder.name_diff_tensor(h, "h");

        let h = recorder.copy(h);

        NetworkOutput{
            state: h.clone(),
            output: h
        }
    }
}
