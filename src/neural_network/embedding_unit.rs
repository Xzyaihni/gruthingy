use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        OperationsRecorder,
        DiffTensor,
        OneHotIndex,
        OneHotLayer,
        LayerSizes,
        InputType,
        WeightInfo,
        network::{NetworkOutput, LayerSize},
        network_unit::{NetworkUnit, Embeddingsable}
    }
};


pub type EmbeddingUnit<T> = WeightsContainer<T>;

create_weights_container!{
    (weights, false, LayerSize::Input, LayerSize::Hidden),
    (bias, false, LayerSize::One, LayerSize::Hidden)
}

impl Embeddingsable for EmbeddingUnit<WeightInfo>
{
    fn embeddings(&self, recorder: &mut OperationsRecorder, input: OneHotIndex) -> DiffTensor
    {
        recorder.matmul_onehotv_add(self.weights.weight_dropped, input, self.bias.weight_dropped)
    }

    fn embeddings_calculate(&self, recorder: &OperationsRecorder, input: &OneHotLayer) -> LayerType
    {
        let weights = recorder.get_tensor(self.weights.weight_dropped.as_value());
        let bias = recorder.get_tensor(self.bias.weight_dropped.as_value());

        weights.matmul_onehotv_add(input, bias)
    }
}

impl NetworkUnit for EmbeddingUnit<WeightInfo>
{
    type State = ();

    fn new(recorder: &mut OperationsRecorder, sizes: LayerSizes) -> Self
    {
        WeightsContainer::new_randomized(recorder, sizes)
    }

    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        _previous_state: Option<&Self::State>,
        input: InputType
    ) -> NetworkOutput<Self::State, DiffTensor>
    {
        let hidden = self.embeddings(recorder, input.into_one_hot());

        NetworkOutput{
            state: (),
            output: hidden
        }
    }

    fn parameters_amount(&self, sizes: LayerSizes) -> u128
    {
        let i = sizes.input as u128;
        let h = sizes.hidden as u128;

        i * h + h
    }
}
