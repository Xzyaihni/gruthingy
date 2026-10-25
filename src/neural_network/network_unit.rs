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

    fn map_inplace_with_info<F>(&mut self, f: F)
    where
        F: FnMut(WeightsFullInfo<&mut T>);

    fn map_with_info<U, F>(self, f: F) -> Self::Unit<U>
    where
        F: FnMut(WeightsFullInfo<T>) -> U;

    fn map_ref<U, F>(&self, f: F) -> Self::Unit<U>
    where
        F: FnMut(&T) -> U;

    fn map_ref_with_info<U, F>(&self, f: F) -> Self::Unit<U>
    where
        F: FnMut(WeightsFullInfo<&T>) -> U;

    fn for_each_weight<F: FnMut(T)>(self, f: F);
    fn for_each_weight_ref<F: FnMut(&T)>(&self, f: F);
    fn for_each_weight_mut<F: FnMut(&mut T)>(&mut self, f: F);
}

pub trait NetworkStateSelectable<T>
{
    fn phi_other_selector(&self, recorder: &mut OperationsRecorder) -> T;
}

pub trait NetworkStateGettable<T>
{
    fn select(&self, recorder: &mut OperationsRecorder) -> T;

    fn set_phi_other_selector(&self, recorder: &mut OperationsRecorder, other: T);
}

pub trait NetworkUnitNewable
{
    fn new(recorder: &mut OperationsRecorder, sizes: LayerSizes) -> Self;
}

pub trait NetworkUnit: GenericUnit<WeightInfoPtr> + Clone
where
    Self: Sized
{
    type State<T>;

    fn record_feedforward_unit(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: Option<&Self::State<DiffTensorPtr>>,
        input: DiffInputType
    ) -> NetworkOutput<Self::State<DiffTensorPtr>, DiffTensorPtr>;

    fn record_feedforward_unit_nonlast(
        &self,
        recorder: &mut OperationsRecorder,
        previous_state: Option<&Self::State<DiffTensorPtr>>,
        input: DiffInputType
    ) -> NetworkOutput<Self::State<DiffTensorPtr>, DiffTensorPtr>
    {
        // my backprop thingy struggles too much if i dont do a copy lol
        self.record_feedforward_unit(recorder, previous_state, input).map(|x| recorder.copy(x))
    }
}
