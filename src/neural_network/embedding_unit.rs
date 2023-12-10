use std::f32;

use serde::{Serialize, Deserialize};

use crate::{
    create_weights_container,
    neural_network::{
        DiffWrapper,
        LayerSizes,
        network::{NetworkOutput, LayerSize},
        network_unit::{NetworkUnit, Embeddingsable}
    }
};


pub type EmbeddingUnit<T> = WeightsContainer<T>;

create_weights_container!{
    (weights, false, LayerSize::Hidden, LayerSize::Input),
    (bias, false, LayerSize::Hidden, LayerSize::One),
    (output, false, LayerSize::Input, LayerSize::Hidden)
}

impl Embeddingsable for EmbeddingUnit<DiffWrapper>
{
    fn embeddings(&mut self, input: &DiffWrapper) -> DiffWrapper
    {
        self.weights.matmulv_add(input, &self.bias)
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
        &mut self,
        _previous_state: Option<&Self::State>,
        input: &DiffWrapper
    ) -> NetworkOutput<Self::State, DiffWrapper>
    {
        let hidden = self.embeddings(input);

        let output = self.output.matmulv(hidden);

        NetworkOutput{
            state: (),
            output
        }
    }

    fn parameters_amount(&self, sizes: LayerSizes) -> u128
    {
        let i = sizes.input as u128;
        let h = sizes.hidden as u128;

        i * h
    }
}
