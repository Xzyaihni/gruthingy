use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        OperationsRecorder,
        DiffTensorPtr,
        DiffInputType,
        LayerSizes,
        WeightInfoPtr,
        NetworkUnitNewable,
        network::{NetworkOutput, LayerSize},
        network_unit::{input_dropout, DropoutInfo, NetworkUnit}
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

impl NetworkUnit for Star<WeightInfoPtr>
{
    type State<T> = T;

    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: &Self::State<DiffTensorPtr>,
        input: DiffInputType,
        dropout: &mut DropoutInfo
    ) -> NetworkOutput<Self::State<DiffTensorPtr>, DiffTensorPtr>
    {
        let mut seed_index = 0;
        let mut matmul_inputv_add = |recorder: &mut OperationsRecorder, weights: WeightInfoPtr, input, bias: WeightInfoPtr|
        {
            let seed = dropout.seeds.get(recorder, seed_index);
            let dropped_input = input_dropout(recorder, input, dropout.input_probability, seed);

            seed_index += 1;

            match dropped_input
            {
                DiffInputType::Normal(x) => recorder.matmulv_add(weights.weight, x, bias.weight),
                DiffInputType::OneHot(x) => recorder.matmul_onehotv_add(weights.weight, x, bias.weight)
            }
        };

        let z_unactivated = matmul_inputv_add(recorder, self.input_z, input, self.z_bias);
        recorder.name_diff_tensor(z_unactivated, "z_unactivated");

        let z = recorder.tanh(z_unactivated);
        recorder.name_diff_tensor(z, "z");

        let k_unactivated_lhs = matmul_inputv_add(recorder, self.input_x, input, self.k_bias);
        recorder.name_diff_tensor(k_unactivated_lhs, "k_unactivated_lhs");

        let seed = dropout.seeds.get(recorder, seed_index);
        let previous_state_dropped = recorder.dropout_with_seed(*previous_state, dropout.hidden_probability, seed);

        let k_unactivated_rhs = recorder.matmulv(self.hidden.weight, previous_state_dropped);

        let k_unactivated = recorder.add(k_unactivated_lhs, k_unactivated_rhs);
        recorder.name_diff_tensor(k_unactivated, "k_unactivated");

        let k = recorder.sigmoid(k_unactivated);
        recorder.name_diff_tensor(k, "k");

        let h_unactivated_rhs = recorder.mul_componentwise(k, z);
        recorder.name_diff_tensor(h_unactivated_rhs, "h_unactivated_rhs");

        let k_m1 = recorder.sub_from_scalar(1.0, k);
        recorder.name_diff_tensor(k_m1, "k_m1");

        let h_unactivated_lhs = recorder.mul_componentwise(k_m1, *previous_state);
        recorder.name_diff_tensor(h_unactivated_lhs, "h_unactivated_lhs");

        let h_unactivated = recorder.add(h_unactivated_lhs, h_unactivated_rhs);
        recorder.name_diff_tensor(h_unactivated, "h_unactivated");

        let h = recorder.tanh(h_unactivated);
        recorder.name_diff_tensor(h, "h");

        let h = recorder.copy(h);

        NetworkOutput{
            state: h.clone(),
            output: h
        }
    }
}
