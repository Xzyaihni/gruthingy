use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        NetworkStateSelectable,
        NetworkStateGettable,
        OperationsRecorder,
        DiffTensorPtr,
        OneHotIndex,
        OneHotLayer,
        LayerSizes,
        DiffInputType,
        WeightInfoPtr,
        NetworkUnitNewable,
        network::{NetworkOutput, LayerSize},
        network_unit::{NetworkUnit, Embeddingsable, EmbeddingsableOwned}
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
        recorder.matmul_onehotv_add(self.weights.weight_dropped, input, self.bias.weight_dropped)
    }
}

impl EmbeddingsableOwned for EmbeddingUnit<LayerType>
{
    fn embeddings_calculate(&self, input: &OneHotLayer) -> LayerType
    {
        self.weights.as_ref().matmul_onehotv_add(input, self.bias.as_ref().as_vector_ref())
    }
}

impl NetworkStateSelectable<()> for ()
{
    fn phi_other_selector(&self, _recorder: &mut OperationsRecorder) -> () {}
}

impl NetworkStateGettable<()> for ()
{
    fn select(&self, _recorder: &mut OperationsRecorder) -> () {}

    fn set_phi_other_selector(&self, _recorder: &mut OperationsRecorder, _other: ()) {}
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
