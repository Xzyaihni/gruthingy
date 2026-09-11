use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        PhiOtherSelectorRecordingIndex,
        NetworkStateSelectable,
        NetworkStateGettable,
        DiffTensorPtr,
        DiffInputType,
        WeightInfo,
        WeightInfoPtr,
        LayerSizes,
        OperationsRecorder,
        NetworkUnitNewable,
        network::{NetworkOutput, LayerSize},
        network_unit::{NetworkUnit, NetworkUnitParameterable}
    }
};


pub type Lstm<T> = WeightsContainer<T>;

create_weights_container!{
    (input_update, false, false, LayerSize::Input, LayerSize::Hidden),
    (input_forget, false, true, LayerSize::Input, LayerSize::Hidden),
    (input_output, false, false, LayerSize::Input, LayerSize::Hidden),
    (input_memory, false, false, LayerSize::Input, LayerSize::Hidden),
    (hidden_update, true, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_forget, true, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_output, true, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_memory, true, true, LayerSize::Hidden, LayerSize::Hidden),
    (update_bias, false, false, LayerSize::One, LayerSize::Hidden),
    (forget_bias, false, true, LayerSize::One, LayerSize::Hidden),
    (output_bias, false, false, LayerSize::One, LayerSize::Hidden),
    (memory_bias, false, false, LayerSize::One, LayerSize::Hidden)
}

#[derive(Debug, Clone)]
pub struct LSTMState<T>
{
    hidden: T,
    memory: T
}

impl NetworkStateSelectable<LSTMState<PhiOtherSelectorRecordingIndex>> for LSTMState<DiffTensorPtr>
{
    fn phi_other_selector(&self, recorder: &mut OperationsRecorder) -> LSTMState<PhiOtherSelectorRecordingIndex>
    {
        LSTMState{
            hidden: recorder.phi_other_selector(self.hidden),
            memory: recorder.phi_other_selector(self.memory)
        }
    }
}

impl NetworkStateGettable<LSTMState<DiffTensorPtr>> for LSTMState<PhiOtherSelectorRecordingIndex>
{
    fn select(&self, recorder: &mut OperationsRecorder) -> LSTMState<DiffTensorPtr>
    {
        let hidden = recorder.select_tensor(self.hidden);
        recorder.name_diff_tensor(hidden, "hidden_selected");

        let memory = recorder.select_tensor(self.memory);
        recorder.name_diff_tensor(memory, "memory_selected");

        LSTMState{
            hidden,
            memory
        }
    }

    fn set_phi_other_selector(&self, recorder: &mut OperationsRecorder, other: LSTMState<DiffTensorPtr>)
    {
        recorder.set_phi_other_selector(self.hidden, other.hidden);
        recorder.set_phi_other_selector(self.memory, other.memory);
    }
}

impl NetworkUnitNewable for Lstm<WeightInfoPtr>
{
    fn new(recorder: &mut OperationsRecorder, sizes: LayerSizes) -> Self
    {
        WeightsContainer::new_randomized(recorder, sizes)
    }
}

impl NetworkUnitParameterable for Lstm<WeightInfo>
{
    fn parameters_amount(&self, sizes: LayerSizes) -> u128
    {
        let i = sizes.input as u128;
        let h = sizes.hidden as u128;

        (4 * i * h) + (4 * h * h) + (4 * h)
    }
}

impl NetworkUnit for Lstm<WeightInfoPtr>
{
    type State<T> = LSTMState<T>;

    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: Option<&Self::State<DiffTensorPtr>>,
        input: DiffInputType,
        store_gradient: bool
    ) -> NetworkOutput<Self::State<DiffTensorPtr>, DiffTensorPtr>
    {
        {
            let mut always_store = |weight: DiffTensorPtr|
            {
                let value = weight.as_value();
                recorder.store_tensor_until_end(value);

                if store_gradient
                {
                    let gradient = weight.as_gradient().unwrap();

                    recorder.store_tensor_until_end(gradient);
                }
            };

            always_store(self.hidden_update.weight_original);
            always_store(self.hidden_forget.weight_original);
            always_store(self.hidden_output.weight_original);
            always_store(self.hidden_memory.weight_original);

            always_store(self.update_bias.weight_original);
            always_store(self.forget_bias.weight_original);
            always_store(self.output_bias.weight_original);
            always_store(self.memory_bias.weight_original);

            always_store(self.input_update.weight_original);
            always_store(self.input_forget.weight_original);
            always_store(self.input_output.weight_original);
            always_store(self.input_memory.weight_original);
        }

        let matmul_inputv_add = |recorder: &mut OperationsRecorder, weights: WeightInfoPtr, input, bias: WeightInfoPtr|
        {
            let weights = weights.weight_dropped;
            let bias = bias.weight_dropped;

            match input
            {
                DiffInputType::Normal(x) => recorder.matmulv_add(weights, x, bias),
                DiffInputType::OneHot(x) => recorder.matmul_onehotv_add(weights, x, bias)
            }
        };

        let mut update_gate = matmul_inputv_add(recorder, self.input_update, input, self.update_bias);
        let mut output_gate = matmul_inputv_add(recorder, self.input_output, input, self.output_bias);
        let mut memory_gate = matmul_inputv_add(recorder, self.input_memory, input, self.memory_bias);

        recorder.name_diff_tensor(update_gate, "update_gate");
        recorder.name_diff_tensor(output_gate, "output_gate");
        recorder.name_diff_tensor(memory_gate, "memory_gate");

        let mut forget_gate = None;

        if let Some(previous_state) = previous_state
        {
            let mut forget_gate_inner = matmul_inputv_add(recorder, self.input_forget, input, self.forget_bias);

            recorder.name_diff_tensor(forget_gate_inner, "forget_gate");

            let mut do_gate = |gate: &mut _, hidden: WeightInfoPtr, previous_hidden|
            {
                *gate = recorder.matmulv_add(hidden.weight_dropped, previous_hidden, *gate);
            };

            do_gate(&mut forget_gate_inner, self.hidden_forget, previous_state.hidden);
            do_gate(&mut update_gate, self.hidden_update, previous_state.hidden);
            do_gate(&mut output_gate, self.hidden_output, previous_state.hidden);
            do_gate(&mut memory_gate, self.hidden_memory, previous_state.hidden);

            let forget_gate_new = recorder.sigmoid(forget_gate_inner);
            recorder.name_diff_tensor(forget_gate_new, "forget_gate_activated");

            forget_gate = Some(forget_gate_new);
        }

        update_gate = recorder.sigmoid(update_gate);
        output_gate = recorder.sigmoid(output_gate);
        memory_gate = recorder.tanh(memory_gate);

        recorder.name_diff_tensor(update_gate, "update_gate_activated");
        recorder.name_diff_tensor(output_gate, "output_gate_activated");
        recorder.name_diff_tensor(memory_gate, "memory_gate_activated");

        let this_memory_rhs = recorder.mul_componentwise(update_gate, memory_gate);

        recorder.name_diff_tensor(this_memory_rhs, "this_memory_rhs");

        let this_memory = if let Some(previous_state) = previous_state
        {
            let left = recorder.mul_componentwise(forget_gate.unwrap(), previous_state.memory);
            recorder.name_diff_tensor(left, "left");

            recorder.add(left, this_memory_rhs)
        } else
        {
            this_memory_rhs
        };

        recorder.name_diff_tensor(this_memory, "this_memory");

        let hidden = {
            let memory = recorder.tanh(this_memory);
            recorder.name_diff_tensor(memory, "memory_inner");

            recorder.mul_componentwise(output_gate, memory)
        };

        recorder.name_diff_tensor(hidden, "hidden");

        let hidden_copy = recorder.copy(hidden);
        recorder.name_diff_tensor(hidden, "hidden_copy");

        let state = LSTMState{
            hidden: hidden_copy,
            memory: this_memory
        };

        NetworkOutput{
            state,
            output: hidden_copy
        }
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    use crate::neural_network::{TensorIndex, LayerType, LayerSizes};

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
        let mut recorder = OperationsRecorder::new();

        let mut one_weight = |value: f32|
        {
            let w = recorder.set_new_tensor_gradientable(LayerType::from_raw([value], 1, 1).into());

            recorder.allow_discard(w.as_value());

            w
        };

        let mut one_weight_info = |value: f32|
        {
            let weight = one_weight(value);

            WeightInfoPtr{
                weight_dropped: weight.clone(),
                weight_original: weight,
                dropconnect_mask: None
            }
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

        let lstm: WeightsContainer<WeightInfoPtr> = WeightsContainer
        {
            sizes: LayerSizes{hidden: 1, input: 1, output: 1, layers: 1},

            input_update: one_weight_info(1.65),
            input_forget: one_weight_info(1.63),
            input_output: one_weight_info(-0.19),
            input_memory: one_weight_info(0.94),

            hidden_update: one_weight_info(2.00),
            hidden_forget: one_weight_info(2.70),
            hidden_output: one_weight_info(4.38),
            hidden_memory: one_weight_info(1.41),

            update_bias: one_weight_info(0.62),
            forget_bias: one_weight_info(1.62),
            output_bias: one_weight_info(0.59),
            memory_bias: one_weight_info(-0.32)
        };

        let state = LSTMState::<DiffTensorPtr>{
            memory: one_weight(2.0),
            hidden: one_weight(1.0)
        };

        let input = one_weight(1.0);

        let output = {
            let output = lstm.record_feedforward_unit(&mut recorder, Some(&state), DiffInputType::Normal(input), true);

            NetworkOutput{
                state: output.state,
                output: output.output
            }
        };

        let epsilon = 0.0001;

        recorder.finish();

        let memory = output.state.memory.as_value();
        let hidden = output.state.hidden.as_value();

        recorder.store_tensor_until_end(memory);
        recorder.store_tensor_until_end(hidden);

        recorder.gradient(output.output.into());

        recorder.resolve_memory();

        let memory = recorder.resolve_tensor_ptr(memory);
        let hidden = recorder.resolve_tensor_ptr(hidden);

        recorder.calculate();

        let single_value = |l: TensorIndex|
        {
            recorder.get_tensor(l).as_vec()[0]
        };

        assert_close_enough(single_value(memory), 2.947, epsilon);
        assert_close_enough(single_value(hidden), 0.986229, epsilon);
    }
}
