use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        OperationsRecorder,
        DiffTensor,
        DiffTensorPtr,
        OneHotIndex,
        OneHotLayer,
        LayerSizes,
        DiffInputType,
        WeightInfo,
        WeightInfoPtr,
        NetworkUnitNewable,
        network::{NetworkOutput, LayerSize},
        network_unit::{NetworkUnit, Embeddingsable, EmbeddingsableOwned, NetworkUnitParameterable}
    }
};


pub type EmbeddingUnit<T> = WeightsContainer<T>;

create_weights_container!{
    (weights, false, LayerSize::Input, LayerSize::Hidden),
    (bias, false, LayerSize::One, LayerSize::Hidden)
}

impl Embeddingsable for EmbeddingUnit<WeightInfoPtr>
{
    fn embeddings(&self, recorder: &mut OperationsRecorder, input: OneHotIndex) -> DiffTensorPtr
    {
        recorder.matmul_onehotv_add(self.weights.weight_dropped, input, self.bias.weight_dropped)
    }
}

impl EmbeddingsableOwned for EmbeddingUnit<WeightInfo>
{
    fn embeddings_calculate(&self, recorder: &OperationsRecorder, input: &OneHotLayer) -> LayerType
    {
/*        let weights = recorder.get_tensor(self.weights.weight_dropped.as_value());
        let bias = recorder.get_tensor(self.bias.weight_dropped.as_value());

        weights.matmul_onehotv_add(input, bias)*/todo!()
    }
}

impl NetworkUnitNewable for EmbeddingUnit<WeightInfoPtr>
{
    fn new(recorder: &mut OperationsRecorder, sizes: LayerSizes) -> Self
    {
        WeightsContainer::new_randomized(recorder, sizes)
    }
}

impl NetworkUnitParameterable for EmbeddingUnit<WeightInfo>
{
    fn parameters_amount(&self, sizes: LayerSizes) -> u128
    {
        let i = sizes.input as u128;
        let h = sizes.hidden as u128;

        i * h + h
    }
}

impl NetworkUnit for EmbeddingUnit<WeightInfoPtr>
{
    type State<T> = ();

    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        _previous_state: Option<&Self::State<DiffTensorPtr>>,
        input: DiffInputType
    ) -> NetworkOutput<Self::State<DiffTensorPtr>, DiffTensorPtr>
    {
        let hidden = self.embeddings(recorder, input.into_one_hot());

        NetworkOutput{
            state: (),
            output: hidden
        }
    }
}
