use crate::neural_network::{
    LAYER_ACTIVATION,
    AFType,
    OperationsRecorder,
    DiffTensor,
    LayerType,
    TensorIndex,
    InputType,
    WeightInfo,
    OneHotLayer,
    OneHotIndex,
    LayerSizes,
    WeightsNamed,
    network::{WeightsSize, NetworkOutput}
};

use serde::{Serialize, de::DeserializeOwned};


pub trait UnitFactory
{
    type Unit<T>;
}

pub trait Embeddingsable
{
    fn embeddings(&self, recorder: &mut OperationsRecorder, input: OneHotIndex) -> DiffTensor;
    fn embeddings_calculate(&self, recorder: &OperationsRecorder, input: &OneHotLayer) -> LayerType;
}

pub trait GenericUnit<T>
{
    type Unit<U>;

    fn dropconnectable() -> bool;

    fn map<U, F>(self, f: F) -> Self::Unit<U>
    where
        F: FnMut(T) -> U;

    fn map_with_info<U, F>(self, f: F) -> Self::Unit<U>
    where
        F: FnMut(WeightsSize<T>) -> U;

    fn map_ref<U, F>(&self, f: F) -> Self::Unit<U>
    where
        F: FnMut(&T) -> U;

    fn clone_weights_with_info<F>(&self, f: F) -> Self
    where
        F: FnMut(WeightsSize<&T>) -> T;

    fn weights_named_info(&self, layer: usize) -> Self::Unit<WeightsNamed<&T>>;

    fn for_each_weight<F: FnMut(T)>(self, f: F);
    fn for_each_weight_ref<F: FnMut(&T)>(&self, f: F);
    fn for_each_weight_mut<F: FnMut(&mut T)>(&mut self, f: F);
}

pub trait OptimizerUnit<T>: GenericUnit<T> + Clone
{
    fn new_zeroed(sizes: LayerSizes) -> Self;
}

pub trait NetworkUnit: GenericUnit<WeightInfo> + Clone
where
    Self: Sized
{
    type State;

    fn new(recorder: &mut OperationsRecorder, sizes: LayerSizes) -> Self;

    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: Option<&Self::State>,
        input: InputType
    ) -> NetworkOutput<Self::State, DiffTensor>;

    fn record_feedforward_unit_nonlast(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: Option<&Self::State>,
        dropout_mask: TensorIndex,
        input: InputType
    ) -> NetworkOutput<Self::State, DiffTensor>
    {
        let mut output = self.record_feedforward_unit(recorder, previous_state, input);

        let new_output = match LAYER_ACTIVATION
        {
            AFType::LeakyRelu =>
            {
                recorder.leaky_relu(output.output)
            },
            AFType::Tanh =>
            {
                recorder.tanh(output.output)
            }
        };

        output.output = recorder.mul_componentwise(new_output, DiffTensor::no_gradient(dropout_mask));

        output
    }

    fn parameters_amount(&self, sizes: LayerSizes) -> u128;
}
