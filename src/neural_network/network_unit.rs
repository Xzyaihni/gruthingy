use crate::neural_network::{
    LAYER_ACTIVATION,
    AFType,
    ScalarType,
    LayerType,
    LayerInnerType,
    network::{WeightsSize, WeightsNamed, NetworkOutput}
};


pub trait NetworkUnit
{
    type State;
    type WeightsContainer<T>: IntoIterator<Item=T> + FromIterator<T>;

    fn new(input_size: usize) -> Self;

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

    fn weights_size(&self, input_size: usize) -> Vec<WeightsSize<&LayerType>>;
    fn weights_info(&self, input_size: usize) -> Vec<WeightsNamed<&LayerType>>;

    fn parameters_amount(&self, inputs_amount: u128) -> u128;

    fn weights(&self) -> &[LayerType];
    fn weights_mut(&mut self) -> &mut [LayerType];

    fn clear(&mut self)
    {
        self.weights_mut().iter_mut().for_each(|v| v.clear());
    }

    fn enable_gradients(&mut self)
    {
        self.weights_mut().iter_mut().for_each(|v| v.enable_gradients());
    }

    fn disable_gradients(&mut self)
    {
        self.weights_mut().iter_mut().for_each(|v| v.disable_gradients());
    }
}
