use std::f32;

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
        let block_index = recorder.current_block().into_index();

        recorder.matmul_onehotv_add(self.weights.weight_dropped[block_index], input, self.bias.weight_dropped[block_index])
    }
}

impl EmbeddingsableOwned for EmbeddingUnit<WeightInfo>
{
    fn embeddings_calculate(&self, recorder: &OperationsRecorder, input: &OneHotLayer) -> LayerType
    {
        let weights = recorder.get_tensor(self.weights.weight.as_value());
        let bias = recorder.get_tensor(self.bias.weight.as_value());

        weights.matmul_onehotv_add(input, bias)
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
        let mut store_both = |weight: DiffTensorPtr|
        {
            recorder.store_tensor_until_end(weight.as_value());
            recorder.store_tensor_until_end(weight.as_gradient().unwrap());
        };

        store_both(self.weights.weight_original);
        store_both(self.bias.weight_original);

        let hidden = self.embeddings(recorder, input.into_one_hot());

        NetworkOutput{
            state: (),
            output: hidden
        }
    }
}
