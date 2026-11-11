use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        DiffTensorPtr,
        DiffInputType,
        WeightInfoPtr,
        LayerSizes,
        OperationsRecorder,
        NetworkUnitNewable,
        network::{NetworkOutput, LayerSize},
        network_unit::{input_dropout, DropoutInfo, NetworkUnit}
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

impl NetworkUnitNewable for Lstm<WeightInfoPtr>
{
    fn new(recorder: &mut OperationsRecorder, sizes: LayerSizes) -> Self
    {
        WeightsContainer::new_randomized(recorder, sizes)
    }
}

impl NetworkUnit for Lstm<WeightInfoPtr>
{
    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: &[DiffTensorPtr],
        input: DiffInputType,
        dropout: &mut DropoutInfo
    ) -> NetworkOutput<Vec<DiffTensorPtr>, DiffTensorPtr>
    {
        let matmul_inputv_add = |recorder: &mut OperationsRecorder, weights: WeightInfoPtr, input, bias: WeightInfoPtr|
        {
            let dropped_input = input_dropout(recorder, input, dropout.input_probability);

            match dropped_input
            {
                DiffInputType::Normal(x) => recorder.matmulv_add(weights.weight, x, bias.weight),
                DiffInputType::OneHot(x) => recorder.matmul_onehotv_add(weights.weight, x, bias.weight)
            }
        };

        let update_gate_lhs = matmul_inputv_add(recorder, self.input_update, input, self.update_bias);
        let forget_gate_lhs = matmul_inputv_add(recorder, self.input_forget, input, self.forget_bias);
        let output_gate_lhs = matmul_inputv_add(recorder, self.input_output, input, self.output_bias);
        let memory_gate_lhs = matmul_inputv_add(recorder, self.input_memory, input, self.memory_bias);

        recorder.name_diff_tensor(update_gate_lhs, "update_gate_lhs");
        recorder.name_diff_tensor(forget_gate_lhs, "forget_gate_lhs");
        recorder.name_diff_tensor(output_gate_lhs, "output_gate_lhs");
        recorder.name_diff_tensor(memory_gate_lhs, "memory_gate_lhs");

        let mut do_gate = |gate, hidden: WeightInfoPtr, previous_hidden|
        {
            let state_dropped = recorder.dropout(previous_hidden, dropout.hidden_probability);

            let rhs = recorder.matmulv(hidden.weight, state_dropped);

            recorder.add(gate, rhs)
        };

        let update_gate_unactivated = do_gate(update_gate_lhs, self.hidden_update, previous_state[0]);
        let forget_gate_unactivated = do_gate(forget_gate_lhs, self.hidden_forget, previous_state[0]);
        let output_gate_unactivated = do_gate(output_gate_lhs, self.hidden_output, previous_state[0]);
        let memory_gate_unactivated = do_gate(memory_gate_lhs, self.hidden_memory, previous_state[0]);

        let update_gate = recorder.sigmoid(update_gate_unactivated);
        let forget_gate = recorder.sigmoid(forget_gate_unactivated);
        let output_gate = recorder.sigmoid(output_gate_unactivated);
        let memory_gate = recorder.tanh(memory_gate_unactivated);

        recorder.name_diff_tensor(update_gate, "update_gate");
        recorder.name_diff_tensor(forget_gate, "forget_gate");
        recorder.name_diff_tensor(output_gate, "output_gate");
        recorder.name_diff_tensor(memory_gate, "memory_gate");

        let memory_rhs = recorder.mul_componentwise(update_gate, memory_gate);
        recorder.name_diff_tensor(memory_rhs, "memory_rhs");

        let memory_lhs = recorder.mul_componentwise(forget_gate, previous_state[1]);
        recorder.name_diff_tensor(memory_lhs, "memory_lhs");

        let memory = recorder.add(memory_lhs, memory_rhs);
        recorder.name_diff_tensor(memory, "memory");

        let hidden = {
            let hidden_rhs = recorder.tanh(memory);
            recorder.name_diff_tensor(hidden_rhs, "hidden_rhs");

            recorder.mul_componentwise(output_gate, hidden_rhs)
        };

        recorder.name_diff_tensor(hidden, "hidden");

        let hidden_copy = recorder.copy(hidden);
        recorder.name_diff_tensor(hidden, "hidden_copy");

        let state = vec![hidden_copy, memory];

        NetworkOutput{
            state,
            output: hidden_copy
        }
    }

    fn hidden_states() -> usize { 2 }
}

#[cfg(test)]
mod tests
{
    use super::*;

    use crate::neural_network::{close_enough, ShapedTensorIndex, LayerType, LayerSizes};


    fn assert_close_enough(a: f32, b: f32, epsilon: f32)
    {
        assert!(close_enough(a, b, epsilon), "a: {a}, b: {b}");
    }

    #[test]
    fn lstm_works()
    {
        let mut recorder = OperationsRecorder::new();

        let one_weight = |recorder: &mut OperationsRecorder, value: f32|
        {
            let w = recorder.set_new_tensor_gradientable(LayerType::from_boxed([value].into(), 1, 1).into());

            recorder.allow_discard(w.as_value());

            w
        };

        let mut one_weight_info = |name: &'static str, value: f32|
        {
            let weight = one_weight(&mut recorder, value);

            let weight_info = WeightInfoPtr{weight};
            recorder.name_diff_tensor(weight_info.weight, name);

            weight_info
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
            sizes: LayerSizes{hidden: 1, initial_input: 1, input: 1, output: 1, final_output: 1, layers: 1, batch_size: 1},

            input_update: one_weight_info("input_update", 1.65),
            input_forget: one_weight_info("input_forget", 1.63),
            input_output: one_weight_info("input_output", -0.19),
            input_memory: one_weight_info("input_memory", 0.94),

            hidden_update: one_weight_info("hidden_update", 2.00),
            hidden_forget: one_weight_info("hidden_forget", 2.70),
            hidden_output: one_weight_info("hidden_output", 4.38),
            hidden_memory: one_weight_info("hidden_memory", 1.41),

            update_bias: one_weight_info("update_bias", 0.62),
            forget_bias: one_weight_info("forget_bias", 1.62),
            output_bias: one_weight_info("output_bias", 0.59),
            memory_bias: one_weight_info("memory_bias", -0.32)
        };

        let state = vec![
            one_weight(&mut recorder, 1.0),
            one_weight(&mut recorder, 2.0)
        ];

        let input = one_weight(&mut recorder, 1.0);

        recorder.name_diff_tensor(input, "input");

        let output = {
            let mut dropout_info = DropoutInfo{
                input_probability: 0.0,
                hidden_probability: 0.0
            };

            let output = lstm.record_feedforward_unit(&mut recorder, &state, DiffInputType::Normal(input), &mut dropout_info);

            NetworkOutput{
                state: output.state,
                output: output.output
            }
        };

        let epsilon = 0.001;

        recorder.finish();

        let memory = output.state[1].as_value();
        let hidden = output.state[0].as_value();

        recorder.store_tensor_as_output(memory);
        recorder.store_tensor_as_output(hidden);

        recorder.gradient(output.output.into());

        recorder.resolve_memory(true);

        let memory = recorder.resolve_tensor_ptr(memory);
        let hidden = recorder.resolve_tensor_ptr(hidden);

        recorder.calculate(0, true);

        let single_value = |l: ShapedTensorIndex|
        {
            recorder.get_tensor_blocking(l).as_vec()[0]
        };

        assert_close_enough(single_value(memory), 2.947, epsilon);
        assert_close_enough(single_value(hidden), 0.986229, epsilon);
    }
}
