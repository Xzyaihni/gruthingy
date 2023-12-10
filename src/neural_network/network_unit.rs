use crate::neural_network::{
    LAYER_ACTIVATION,
    AFType,
    DiffWrapper,
    LayerInnerType,
    LayerSizes,
    WeightsNamed,
    Optimizer,
    network::{WeightsSize, NetworkOutput}
};

use serde::{Serialize, de::DeserializeOwned};


pub trait UnitFactory
{
    type Unit<T>;
}

// i hate rust generics i hate rust generics i hate rust generics
pub trait Embeddingsable
{
    fn embeddings(&mut self, input: &DiffWrapper) -> DiffWrapper;
}

pub trait GenericUnit<T>
{
    type Unit<U>;

    fn dropconnectable() -> bool;

    fn map<U, F>(self, f: F) -> Self::Unit<U>
    where
        F: FnMut(T) -> U;

    fn map_mut<U, F>(&mut self, f: F) -> Self::Unit<U>
    where
        F: FnMut(&mut T) -> U;

    fn clone_weights_with_info<F>(&self, f: F) -> Self
    where
        F: FnMut(WeightsSize<&T>) -> T;

    fn weights_named_info(&self) -> Self::Unit<WeightsNamed<&T>>;

    fn for_each_weight<F: FnMut(T)>(self, f: F);
    fn for_each_weight_ref<F: FnMut(&T)>(&self, f: F);
    fn for_each_weight_mut<F: FnMut(&mut T)>(&mut self, f: F);
}

pub trait NewableLayer
{
    fn new(previous: usize, current: usize) -> Self;
}

pub trait OptimizerUnit<T>: GenericUnit<T> + Serialize + DeserializeOwned
{
    fn new_zeroed(sizes: LayerSizes) -> Self;

    fn gradients_to_change<O>(
        &mut self,
        gradients: Self::Unit<LayerInnerType>,
        optimizer: &O
    ) -> Self::Unit<DiffWrapper>
    where
        O: Optimizer<WeightParam=T>;
}

pub trait NetworkUnit: GenericUnit<DiffWrapper> + Serialize + DeserializeOwned + Clone
where
    Self: Sized
{
    type State;

    fn new(sizes: LayerSizes) -> Self;

    fn feedforward_unit(
        &mut self,
        previous_state: Option<&Self::State>,
        input: &DiffWrapper
    ) -> NetworkOutput<Self::State, DiffWrapper>;

    fn feedforward_unit_last(
        &mut self,
        previous_state: Option<&Self::State>,
        input: &DiffWrapper,
        targets: LayerInnerType
    ) -> NetworkOutput<Self::State, DiffWrapper>
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
        dropout_mask: &DiffWrapper,
        input: &DiffWrapper
    ) -> NetworkOutput<Self::State, DiffWrapper>
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

    fn parameters_amount(&self, sizes: LayerSizes) -> u128;

    fn enable_gradients(&mut self)
    {
        self.for_each_weight_mut(|v| v.enable_gradients());
    }

    fn disable_gradients(&mut self)
    {
        self.for_each_weight_mut(|v| v.disable_gradients());
    }
}
