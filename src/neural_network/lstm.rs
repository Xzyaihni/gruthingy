use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        DiffTensor,
        DiffTensorPtr,
        TensorIndex,
        DiffInputType,
        WeightInfo,
        WeightInfoPtr,
        LayerSizes,
        BlockIndex,
        OperationsRecorder,
        NetworkUnitStateable,
        NetworkUnitNewable,
        network::{NetworkOutput, LayerSize},
        network_unit::{NetworkUnit, NetworkUnitParameterable}
    }
};


pub type Lstm<T> = WeightsContainer<T>;

create_weights_container!{
    (input_update, false, LayerSize::Input, LayerSize::Hidden),
    (input_forget, false, LayerSize::Input, LayerSize::Hidden),
    (input_output, false, LayerSize::Input, LayerSize::Hidden),
    (input_memory, false, LayerSize::Input, LayerSize::Hidden),
    (hidden_update, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_forget, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_output, true, LayerSize::Hidden, LayerSize::Hidden),
    (hidden_memory, true, LayerSize::Hidden, LayerSize::Hidden),
    (update_bias, false, LayerSize::One, LayerSize::Hidden),
    (forget_bias, false, LayerSize::One, LayerSize::Hidden),
    (output_bias, false, LayerSize::One, LayerSize::Hidden),
    (memory_bias, false, LayerSize::One, LayerSize::Hidden)
}

#[derive(Clone)]
pub struct LSTMState<T>
{
    hidden: T,
    memory: T
}

impl NetworkUnitStateable for LSTMState<DiffTensor>
{
    fn set(&self, recorder: &mut OperationsRecorder, new: &Self)
    {
        let mut set_both = |old: &DiffTensor, new: &DiffTensor|
        {
            let mut set_single = |old: TensorIndex, new: TensorIndex|
            {
                let new_tensor = recorder.get_tensor(new).clone();

                recorder.set_tensor(old, new_tensor);
            };

            set_single(old.as_value(), new.as_value());
            set_single(old.as_gradient().unwrap(), new.as_gradient().unwrap());
        };

        set_both(&self.hidden, &new.hidden);
        set_both(&self.memory, &new.memory);
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
            store_both(self.hidden_forget.weight_original);
            store_both(self.hidden_output.weight_original);
            store_both(self.hidden_memory.weight_original);

            store_both(self.forget_bias.weight_original);

            store_both(self.input_forget.weight_original);
        }

        {
            let mut always_store = |weight: DiffTensorPtr|
            {
                recorder.store_tensor_until_end(weight.as_value());
                recorder.store_tensor_until_end(weight.as_gradient().unwrap());
            };

            always_store(self.update_bias.weight_original);
            always_store(self.output_bias.weight_original);
            always_store(self.memory_bias.weight_original);

            always_store(self.input_update.weight_original);
            always_store(self.input_output.weight_original);
            always_store(self.input_memory.weight_original);
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

        let mut forget_gate = matmul_inputv_add(self.input_forget, input, self.forget_bias);
        let mut update_gate = matmul_inputv_add(self.input_update, input, self.update_bias);
        let mut output_gate = matmul_inputv_add(self.input_output, input, self.output_bias);
        let mut memory_gate = matmul_inputv_add(self.input_memory, input, self.memory_bias);

        if let Some(previous_state) = previous_state
        {
            recorder.set_block_input(block, previous_state.hidden.as_value());
            recorder.set_block_input(block, previous_state.memory.as_value());

            let mut do_gate = |gate: &mut _, hidden: WeightInfoPtr, previous_hidden|
            {
                let mm = recorder.matmulv(hidden.weight_dropped[block_index], previous_hidden);
                *gate = recorder.add(*gate, mm);
            };

            do_gate(&mut forget_gate, self.hidden_forget, previous_state.hidden);
            do_gate(&mut update_gate, self.hidden_update, previous_state.hidden);
            do_gate(&mut output_gate, self.hidden_output, previous_state.hidden);
            do_gate(&mut memory_gate, self.hidden_memory, previous_state.hidden);
        }

        let sigmoid_inplace_use_that_here = ();
        forget_gate = recorder.sigmoid(forget_gate);
        update_gate = recorder.sigmoid(update_gate);
        output_gate = recorder.sigmoid(output_gate);
        memory_gate = recorder.tanh(memory_gate);

        Self::reuse_for_next_block(recorder, update_gate.as_value());
        Self::reuse_for_next_block(recorder, output_gate.as_value());
        Self::reuse_for_next_block(recorder, memory_gate.as_value());

        let this_memory_rhs = recorder.mul_componentwise(update_gate, memory_gate);

        let this_memory = if let Some(previous_state) = previous_state
        {
            let left = recorder.mul_componentwise(forget_gate, previous_state.memory);
            recorder.add(left, this_memory_rhs)
        } else
        {
            this_memory_rhs
        };

        let hidden = {
            let memory = recorder.tanh(this_memory);

            Self::reuse_for_next_block(recorder, memory.as_value());

            recorder.mul_componentwise(output_gate, memory)
        };

        let state = LSTMState{
            hidden: hidden.clone(),
            memory: this_memory
        };

        recorder.store_tensor_until_end_in_block(block, state.hidden.as_value());
        recorder.store_tensor_until_end_in_block(block, state.memory.as_value());

        NetworkOutput{
            state,
            output: hidden
        }
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    use crate::neural_network::{LayerType, LayerSizes};

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
            recorder.set_new_tensor_gradientable(LayerType::from_raw([value], 1, 1).into())
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
            let output = lstm.record_feedforward_unit(&mut recorder, Some(&state), DiffInputType::Normal(input));

            NetworkOutput{
                state: output.state,
                output: output.output
            }
        };

        let epsilon = 0.0001;

        recorder.finish();
        recorder.gradient_with_respect(vec![output.output.into()]);

        let memory = output.state.memory.as_value();
        let hidden = output.state.hidden.as_value();

        recorder.store_tensor_until_end(memory);
        recorder.store_tensor_until_end(hidden);

        recorder.resolve_memory();

        let memory = recorder.resolve_tensor_ptr(memory);
        let hidden = recorder.resolve_tensor_ptr(hidden);

        let current_block = recorder.current_block();
        recorder.calculate(current_block);

        let single_value = |l: TensorIndex|
        {
            recorder.get_tensor(l).as_vec()[0]
        };

        assert_close_enough(single_value(memory), 2.947, epsilon);
        assert_close_enough(single_value(hidden), 0.986229, epsilon);
    }
}
