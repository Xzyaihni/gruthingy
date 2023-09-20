use crate::neural_network::{
    LAYER_ACTIVATION,
    AFType,
    ScalarType,
    LayerType,
    LayerInnerType,
    network::{WeightsSize, WeightsNamed, NetworkOutput}
};


pub trait NetworkUnit
where
    Self: Sized
{
    type State;
    // i could probably rewrite stuff to remove this but im lazy
    type ThisWeightsContainer<T>;

    fn new() -> Self;

    fn feedforward_unit(
        &mut self,
        previous_state: Option<&Self::State>,
        input: &LayerType
    ) -> NetworkOutput<Self::State, LayerType>;

    fn feedforward_unit_last(
        &mut self,
        previous_state: Option<&Self::State>,
        input: &LayerType,
        targets: LayerInnerType
    ) -> NetworkOutput<Self::State, ScalarType>
    {
        let NetworkOutput{
            state,
            output
        } = self.feedforward_unit(previous_state, input);

        NetworkOutput{
            state,
            output: output.softmax_cross_entropy(targets)
        }
    }

    fn feedforward_unit_nonlast(
        &mut self,
        previous_state: Option<&Self::State>,
        dropout_mask: &LayerType,
        input: &LayerType
    ) -> NetworkOutput<Self::State, LayerType>
    {
        let mut output = self.feedforward_unit(previous_state, input);

        match LAYER_ACTIVATION
        {
            AFType::LeakyRelu =>
            {
                output.output.leaky_relu();
            },
            AFType::Tanh =>
            {
                output.output.tanh();
            }
        }

        output.output *= dropout_mask;

        output
    }

    fn parameters_amount(&self) -> u128;

    fn for_each_weight<F: FnMut(&mut LayerType)>(&mut self, f: F);

    fn clone_weights_with_info<F>(&self, f: F) -> Self
    where
        F: FnMut(WeightsSize<&LayerType>) -> LayerType;

    fn map_weights_mut<F, U>(&mut self, f: F) -> Self::ThisWeightsContainer<U>
    where
        F: FnMut(&mut LayerType) -> U;

    fn clear(&mut self)
    {
        self.for_each_weight(|v| v.clear());
    }

    fn enable_gradients(&mut self)
    {
        self.for_each_weight(|v| v.enable_gradients());
    }

    fn disable_gradients(&mut self)
    {
        self.for_each_weight(|v| v.disable_gradients());
    }
}
