use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        OperationsRecorder,
        DiffTensorPtr,
        OneHotIndex,
        OneHotLayer,
        LayerSizes,
        DiffInputType,
        WeightInfoPtr,
        NetworkUnitNewable,
        network::{NetworkOutput, LayerSize},
        network_unit::{DropoutInfo, NetworkUnit, Embeddingsable, EmbeddingsableOwned}
    }
};


pub type EmbeddingUnit<T> = WeightsContainer<T>;

create_weights_container!{
    (weights, false, false, LayerSize::Input, LayerSize::Hidden),
    (bias, false, false, LayerSize::One, LayerSize::Hidden)
}

impl Embeddingsable for EmbeddingUnit<WeightInfoPtr>
{
    fn embeddings(&self, recorder: &mut OperationsRecorder, input: OneHotIndex) -> DiffTensorPtr
    {
        recorder.matmul_onehotv_add(self.weights.weight, input, self.bias.weight)
    }
}

impl EmbeddingsableOwned for EmbeddingUnit<LayerType>
{
    fn embeddings_calculate(&self, input: &OneHotLayer) -> LayerType
    {
        self.weights.as_ref().matmul_onehotv_add(input, self.bias.as_ref().as_vector_ref())
    }
}

impl NetworkUnitNewable for EmbeddingUnit<WeightInfoPtr>
{
    fn new(recorder: &mut OperationsRecorder, sizes: LayerSizes) -> Self
    {
        WeightsContainer::new_randomized(recorder, sizes)
    }
}

impl NetworkUnit for EmbeddingUnit<WeightInfoPtr>
{
    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        _previous_state: &[DiffTensorPtr],
        input: DiffInputType,
        _dropout: &mut DropoutInfo
    ) -> NetworkOutput<Vec<DiffTensorPtr>, DiffTensorPtr>
    {
        let hidden = self.embeddings(recorder, input.into_one_hot());

        NetworkOutput{
            state: Vec::new(),
            output: hidden
        }
    }

    fn hidden_states() -> usize { 0 }
}
