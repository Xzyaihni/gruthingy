use crate::neural_network::{
    LAYER_ACTIVATION,
    AFType,
    ScalarType,
    LayerType,
    LayerInnerType,
    LayerSizes,
    Optimizer,
    network::{WeightsSize, WeightsNamed, NetworkOutput}
};


pub trait NewableLayer
{
    fn new(previous: usize, current: usize) -> Self;
}

pub trait UnitContainer: IntoIterator
{
    type UnitContainer<U>: UnitContainer<Item=U>;

    type IntoInfoIter: Iterator<Item=WeightsSize<Self::Item>>;

    // nice in theory, hell to implement with this generics salad in practice
    /*fn zip<V>(self, other: V) -> Self::UnitContainer<(Self::Item, V::Item)>
    where
        V: UnitContainer;*/

    fn into_iter_with_info(self) -> Self::IntoInfoIter;

    fn map<U, F>(self, f: F) -> Self::UnitContainer<U>
    where
        F: FnMut(Self::Item) -> U;
}

pub trait NewableUnitContainer: UnitContainer
{
    fn new_zeroed(sizes: LayerSizes) -> Self;
}

pub trait GradientableUnitContainer: UnitContainer
{
    // im not returning anything from this cuz its generics hell and i cant find an exit
    fn apply_change<'a, Lhs, G, O>(
        &mut self,
        lhs: &'a mut Lhs,
        gradients: G,
        optimizer: &O
    )
    where
        &'a mut Lhs: UnitContainer<Item=&'a mut LayerType>,
        G: UnitContainer<Item=LayerInnerType>,
        O: Optimizer<WeightParam=Self::Item>;
}

pub trait NetworkUnit: UnitContainer<Item=LayerType>
where
    Self: Sized
{
    type State;

    fn new(sizes: LayerSizes) -> Self;

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

    fn parameters_amount(&self, sizes: LayerSizes) -> u128;

    fn weights_named_info(&self) -> Self::UnitContainer<WeightsNamed<&LayerType>>;

    fn for_each_weight<F: FnMut(&mut LayerType)>(&mut self, f: F);

    fn clone_weights_with_info<F>(&self, f: F) -> Self
    where
        F: FnMut(WeightsSize<&LayerType>) -> LayerType;

    fn map_weights_mut<F, U>(&mut self, f: F) -> Self::UnitContainer<U>
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
