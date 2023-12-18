use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        DiffWrapper,
        OneHotLayer,
        LayerSizes,
        InputType,
        network::{NetworkOutput, LayerSize},
        network_unit::{NetworkUnit, Embeddingsable}
    }
};


pub type EmbeddingUnit<T> = WeightsContainer<T>;

create_weights_container!{
    (weights, false, LayerSize::Input, LayerSize::Hidden),
    (bias, false, LayerSize::One, LayerSize::Hidden)
}

impl Embeddingsable for EmbeddingUnit<DiffWrapper>
{
    fn embeddings(&self, input: &OneHotLayer) -> DiffWrapper
    {
        self.weights.matmul_onehotv_add(input, &self.bias)
    }
}

impl NetworkUnit for EmbeddingUnit<DiffWrapper>
{
    type State = ();

    fn new(sizes: LayerSizes) -> Self
    {
        WeightsContainer::new_randomized(sizes)
    }

    fn feedforward_unit(
        &self,
        _previous_state: Option<&Self::State>,
        input: &InputType
    ) -> NetworkOutput<Self::State, DiffWrapper>
    {
        let hidden = self.embeddings(input.as_one_hot());

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
