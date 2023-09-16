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
    Self: Sized + FromIterator<LayerType>
{
    type State;
    // i could probably rewrite stuff to remove this but im lazy
    type ThisWeightsContainer<T>: FromIterator<T>;

    type Iter<'a, T>: Iterator<Item=&'a T>
    where
        Self: 'a,
        T: 'a;

    type IterMut<'a, T>: Iterator<Item=&'a mut T>
    where
        Self: 'a,
        T: 'a;

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

    fn iter<'a>(&'a self) -> Self::Iter<'a, LayerType>;
    fn iter_mut<'a>(&'a mut self) -> Self::IterMut<'a, LayerType>;

    fn for_each_weight<F: FnMut(&mut LayerType)>(&mut self, f: F)
    {
        self.iter_mut().for_each(f);
    }

    fn clone_weights_with_info<F>(&self, mut f: F, input_size: usize) -> Self
    where
        F: FnMut(&LayerType, WeightsSize<&LayerType>) -> LayerType
    {
        self.iter().zip(self.weights_size(input_size).into_iter()).map(|(layer, info)|
        {
            f(layer, info)
        }).collect()
    }

    fn map_weights_mut<F, U>(&mut self, f: F) -> Self::ThisWeightsContainer<U>
    where
        F: FnMut(&mut LayerType) -> U
    {
        self.iter_mut().map(f).collect()
    }

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
