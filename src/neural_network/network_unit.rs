use crate::neural_network::{
    OperationsRecorder,
    DiffTensorPtr,
    LayerType,
    DiffInputType,
    WeightInfoPtr,
    OneHotLayer,
    OneHotIndex,
    LayerSizes,
    network::{WeightsFullInfo, NetworkOutput}
};


pub trait UnitFactory
{
    type Unit<T>;
}

pub trait Embeddingsable
{
    fn embeddings(&self, recorder: &mut OperationsRecorder, input: OneHotIndex) -> DiffTensorPtr;
}

pub trait EmbeddingsableOwned
{
    fn embeddings_calculate(&self, input: &OneHotLayer) -> LayerType;
}

pub trait GenericUnit<T>
{
    type Unit<U>;

    fn sizes(&self) -> &LayerSizes;

    fn iter_with_info<'a>(&'a self) -> impl Iterator<Item=WeightsFullInfo<&'a T>>
    where
        T: 'a;

    fn map<U, F>(self, f: F) -> Self::Unit<U>
    where
        F: FnMut(T) -> U;

    fn map_with_info<U, F>(self, f: F) -> Self::Unit<U>
    where
        F: FnMut(WeightsFullInfo<T>) -> U;

    fn map_ref<U, F>(&self, f: F) -> Self::Unit<U>
    where
        F: FnMut(&T) -> U;

    fn map_ref_with_info<U, F>(&self, f: F) -> Self::Unit<U>
    where
        F: FnMut(WeightsFullInfo<&T>) -> U;
}

pub trait NetworkUnitNewable
{
    fn new(recorder: &mut OperationsRecorder, sizes: LayerSizes) -> Self;
}

pub struct DropoutInfo
{
    pub input_probability: f32,
    pub hidden_probability: f32
}

pub fn input_dropout(
    recorder: &mut OperationsRecorder,
    input: DiffInputType,
    input_probability: f32
) -> DiffInputType
{
    match input
    {
        DiffInputType::Normal(x) => DiffInputType::Normal(recorder.dropout(x, input_probability)),
        DiffInputType::OneHot(x) => DiffInputType::OneHot(x)
    }
}

pub trait NetworkUnit: GenericUnit<WeightInfoPtr> + Clone
where
    Self: Sized
{
    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: &[DiffTensorPtr],
        input: DiffInputType,
        dropout: &mut DropoutInfo
    ) -> NetworkOutput<Vec<DiffTensorPtr>, DiffTensorPtr>;

    fn record_feedforward_unit_nonlast(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: &[DiffTensorPtr],
        input: DiffInputType,
        dropout: &mut DropoutInfo
    ) -> NetworkOutput<Vec<DiffTensorPtr>, DiffTensorPtr>
    {
        // my backprop thingy struggles too much if i dont do a copy lol
        self.record_feedforward_unit(recorder, previous_state, input, dropout).map(|x| recorder.copy(x))
    }

    fn hidden_states() -> usize;
}
