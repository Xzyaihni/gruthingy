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
        WeightInfo,
        WeightInfoPtr,
        NetworkUnitNewable,
        network::{NetworkOutput, LayerSize},
        network_unit::{NetworkUnit, Embeddingsable, EmbeddingsableOwned, NetworkUnitParameterable}
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

impl EmbeddingsableOwned for EmbeddingUnit<WeightInfoPtr>
{
    fn embeddings_calculate(&self, recorder: &OperationsRecorder, input: &OneHotLayer) -> LayerType
    {
        let weights = recorder.get_tensor_memory_value(self.weights.weight_original.as_value());
        let bias = recorder.get_tensor_memory_value(self.bias.weight_original.as_value());

        weights.matmul_onehotv_add(input, bias)
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

            always_store(self.bias.weight_original);
            always_store(self.weights.weight_original);
        }

        let hidden = self.embeddings(recorder, input.into_one_hot());

        NetworkOutput{
            state: (),
            output: hidden
        }
    }
}
