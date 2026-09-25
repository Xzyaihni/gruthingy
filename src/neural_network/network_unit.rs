use crate::neural_network::{
    OperationsRecorder,
    DiffTensorPtr,
    LayerType,
    DiffInputType,
    WeightInfoPtr,
    OneHotLayer,
    OneHotIndex,
    LayerSizes,
    WeightsNamed,
    network::{WeightsSize, NetworkOutput}
};


#[cfg(not(debug_assertions))]
pub struct DebugUnitInfo;

#[cfg(debug_assertions)]
pub struct DebugUnitInfo
{
    pub name: &'static str
}

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

    fn map<U, F>(self, f: F) -> Self::Unit<U>
    where
        F: FnMut(T) -> U;

    fn map_inplace_with_info<F>(&mut self, f: F)
    where
        F: FnMut(WeightsSize<&mut T>, DebugUnitInfo);

    fn map_with_info<U, F>(self, f: F) -> Self::Unit<U>
    where
        F: FnMut(WeightsSize<T>) -> U;

    fn map_ref<U, F>(&self, f: F) -> Self::Unit<U>
    where
        F: FnMut(&T) -> U;

    fn map_ref_with_info<U, F>(&self, f: F) -> Self::Unit<U>
    where
        F: FnMut(WeightsSize<&T>) -> U;

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
