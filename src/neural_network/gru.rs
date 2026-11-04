use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        OperationsRecorder,
        PhiOtherSelectorRecordingIndex,
        NetworkStateSelectable,
        NetworkStateGettable,
        DiffTensorPtr,
        DiffInputType,
        LayerSizes,
        WeightInfoPtr,
        NetworkUnitNewable,
        network::{NetworkOutput, LayerSize},
        network_unit::{DropoutInfo, NetworkUnit}
    }
};


pub type Gru<T> = WeightsContainer<T>;

create_weights_container!{
    (input_update, false, false, LayerSize::Input, LayerSize::Hidden),
    (input_reset, false, true, LayerSize::Input, LayerSize::Hidden),
    (input_activation, false, false, LayerSize::Input, LayerSize::Hidden),
    (hidden_update, true, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_reset, true, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_activation, true, true, LayerSize::Hidden, LayerSize::Hidden),
    (update_bias, false, false, LayerSize::One, LayerSize::Hidden),
    (reset_bias, false, true, LayerSize::One, LayerSize::Hidden),
    (activation_bias, false, false, LayerSize::One, LayerSize::Hidden)
}

impl NetworkStateSelectable<PhiOtherSelectorRecordingIndex> for DiffTensorPtr
{
    fn phi_other_selector(&self, recorder: &mut OperationsRecorder) -> PhiOtherSelectorRecordingIndex
    {
        recorder.phi_other_selector(*self)
    }
}

impl NetworkStateGettable<DiffTensorPtr> for PhiOtherSelectorRecordingIndex
{
    fn select(&self, recorder: &mut OperationsRecorder) -> DiffTensorPtr
    {
        let state = recorder.select_tensor(*self);
        recorder.name_diff_tensor(state, "state");

        state
    }

    fn set_phi_other_selector(&self, recorder: &mut OperationsRecorder, other: DiffTensorPtr)
    {
        recorder.set_phi_other_selector(*self, other);
    }
}

impl NetworkUnitNewable for Gru<WeightInfoPtr>
{
    fn new(recorder: &mut OperationsRecorder, sizes: LayerSizes) -> Self
    {
        WeightsContainer::new_randomized(recorder, sizes)
    }
}

impl NetworkUnit for Gru<WeightInfoPtr>
{
    type State<T> = T;

    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: Option<&Self::State<DiffTensorPtr>>,
        input: DiffInputType,
        dropout: &mut DropoutInfo
    ) -> NetworkOutput<Self::State<DiffTensorPtr>, DiffTensorPtr>
    {
        let mut seed_index = 0;
        let mut matmul_inputv_add = |recorder: &mut OperationsRecorder, weights: WeightInfoPtr, input, bias: WeightInfoPtr|
        {
            let value = match input
            {
                DiffInputType::Normal(x) => recorder.matmulv_add(weights.weight, x, bias.weight),
                DiffInputType::OneHot(x) => recorder.matmul_onehotv_add(weights.weight, x, bias.weight)
            };

            let seed = dropout.seeds.get(recorder, seed_index);
            let value = recorder.dropout_with_seed(value, dropout.input_probability, seed);

            seed_index += 1;

            value
        };

        let mut update_gate = matmul_inputv_add(recorder, self.input_update, input, self.update_bias);
        let mut activation_gate = matmul_inputv_add(recorder, self.input_activation, input, self.activation_bias);

        recorder.name_diff_tensor(update_gate, "update_gate");
        recorder.name_diff_tensor(activation_gate, "activation_gate");

        let mut reset_gate = None;

        if let Some(previous_state) = previous_state
        {
            let mut reset_gate_inner = matmul_inputv_add(recorder, self.input_reset, input, self.reset_bias);
            recorder.name_diff_tensor(reset_gate_inner, "reset_gate");

            let mut do_gate = |gate: &mut _, hidden: WeightInfoPtr|
            {
                let value = recorder.matmulv_add(hidden.weight, *previous_state, *gate);

                let seed = dropout.seeds.get(recorder, seed_index);
                *gate = recorder.dropout_with_seed(value, dropout.hidden_probability, seed);

                seed_index += 1;
            };

            do_gate(&mut update_gate, self.hidden_update);
            do_gate(&mut reset_gate_inner, self.hidden_reset);

            let reset_gate_new = recorder.sigmoid(reset_gate_inner);
            recorder.name_diff_tensor(reset_gate_new, "reset_gate_activated");

            reset_gate = Some(reset_gate_new);
        }

        update_gate = recorder.sigmoid(update_gate);

        recorder.name_diff_tensor(update_gate, "update_gate_activated");

        if let Some(previous_state) = previous_state
        {
            let activation_v = recorder.mul_componentwise(reset_gate.unwrap(), *previous_state);

            let mm_pre = recorder.matmulv(self.hidden_activation.weight, activation_v);

            let seed = dropout.seeds.get(recorder, seed_index);
            let mm = recorder.dropout_with_seed(mm_pre, dropout.hidden_probability, seed);

            activation_gate = recorder.add(activation_gate, mm);
        }

        activation_gate = recorder.tanh(activation_gate);

        recorder.name_diff_tensor(activation_gate, "activation_gate_activated");

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
