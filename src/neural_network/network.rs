use std::{
    f32,
    vec,
    iter,
    convert,
    cmp::Ordering,
    borrow::Borrow
};

use serde::{Serialize, Deserialize};

use crate::{
    EmbeddingsUnitFactory,
    neural_network::{
        OperationsRecorder,
        Softmaxer,
        NetworkUnitStateable,
        NetworkUnitNewable,
        BlockIndex,
        DiffTensor,
        DiffTensorPtr,
        DiffScalar,
        TensorIndex,
        TensorPtr,
        OneHotLayer,
        OneHotIndex,
        InputType,
        InputTypePtr,
        DiffInputType,
        OwnedInputType,
        LayerType,
        NetworkUnit,
        NewableLayer,
        GenericUnit,
        Optimizer,
        OptimizerUnit,
        UnitFactory,
        DROPCONNECT_PROBABILITY,
        network_unit::{Embeddingsable, EmbeddingsableOwned, NetworkUnitParameterable}
    }
};


pub struct WeightsSize<T>
{
    pub weights: T,
    pub previous_size: usize,
    pub this_size: usize,
    pub is_hidden: bool
}

impl<T> WeightsSize<T>
{
    fn map<F, U>(self, f: F) -> WeightsSize<U>
    where
        F: FnOnce(T) -> U
    {
        WeightsSize{
            previous_size: self.previous_size,
            this_size: self.this_size,
            is_hidden: self.is_hidden,
            weights: f(self.weights)
        }
    }
}

pub struct WeightsNamed<T>
{
    pub name: String,
    pub layer: usize,
    pub weights_size: WeightsSize<T>
}

impl<T> WeightsNamed<T>
{
    fn map<F, U>(self, f: F) -> WeightsNamed<U>
    where
        F: FnOnce(T) -> U
    {
        WeightsNamed{
            name: self.name,
            layer: self.layer,
            weights_size: self.weights_size.map(f)
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LayerSizes
{
    pub input: usize,
    pub output: usize,
    pub hidden: usize,
    pub layers: usize
}

pub enum LayerSize
{
    Input,
    Hidden,
    One
}

impl LayerSize
{
    pub fn into_number(self, sizes: LayerSizes) -> usize
    {
        match self
        {
            Self::Input => sizes.input,
            Self::Hidden => sizes.hidden,
            Self::One => 1
        }
    }
}

pub fn dropconnected_weights(
    recorder: &mut OperationsRecorder,
    weights: DiffTensorPtr,
    dropconnect_mask: DiffTensorPtr
) -> [DiffTensorPtr; 2]
{
    let initial_block = recorder.current_block();

    let new_weights = if recorder.blocks_count() == 1
    {
        let new_weights = recorder.mul_componentwise(weights, dropconnect_mask);

        [new_weights, new_weights]
    } else
    {
        recorder.blocks_iter().map(|block|
        {
            recorder.set_current_block(block);

            recorder.mul_componentwise(weights, dropconnect_mask)
        }).collect::<Vec<_>>().try_into().unwrap()
    };

    recorder.set_current_block(initial_block);

    new_weights
}

#[macro_export]
macro_rules! create_weights_container
{
    ($(($name:ident, $is_hidden:expr, $previous_size:expr, $this_size:expr)),+) =>
    {
        use std::ops::{SubAssign, AddAssign, DivAssign};

        use $crate::neural_network::{
            LayerType,
            NewableLayer,
            GenericUnit,
            OptimizerUnit,
            network::{WeightsNamed, WeightsSize}
        };


        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct WeightsContainer<T>
        {
            sizes: $crate::neural_network::LayerSizes,
            $(
                $name: T,
            )+
        }

        impl<T: DivAssign<f32>> DivAssign<f32> for WeightsContainer<T>
        {
            fn div_assign(&mut self, rhs: f32)
            {
                $(
                    self.$name /= rhs;
                )+
            }
        }

        impl<T: SubAssign<T>> SubAssign for WeightsContainer<T>
        {
            fn sub_assign(&mut self, rhs: Self)
            {
                $(
                    self.$name -= rhs.$name;
                )+
            }
        }

        impl<T: AddAssign<T>> AddAssign for WeightsContainer<T>
        {
            fn add_assign(&mut self, rhs: Self)
            {
                $(
                    self.$name += rhs.$name;
                )+
            }
        }

        impl<T> WeightsContainer<T>
        {
            pub const fn len() -> usize
            {
                [$(
                    stringify!($name),
                )+].len()
            }

            pub fn iter_mut_with_info(&mut self) -> impl Iterator<Item=WeightsSize<&mut T>>
            {
                [
                    $(
                        WeightsSize{
                            weights: &mut self.$name,
                            this_size: $this_size.into_number(self.sizes),
                            previous_size: $previous_size.into_number(self.sizes),
                            is_hidden: $is_hidden
                        },
                    )+
                ].into_iter()
            }

            pub fn as_mut(&mut self) -> WeightsContainer<&mut T>
            {
                WeightsContainer{
                    sizes: self.sizes,
                    $(
                        $name: &mut self.$name,
                    )+
                }
            }

            pub fn zip<U>(self, other: WeightsContainer<U>) -> WeightsContainer<(T, U)>
            {
                WeightsContainer{
                    sizes: self.sizes,
                    $(
                        $name: (self.$name, other.$name),
                    )+
                }
            }
        }

        impl WeightsContainer<$crate::neural_network::WeightInfoPtr>
        {
            pub fn new_randomized(recorder: &mut OperationsRecorder, sizes: $crate::neural_network::LayerSizes) -> Self
            {
                use $crate::neural_network::network::LayerSize;

                Self{sizes, $(
                    $name: {
                        let this_size = $this_size.into_number(sizes);
                        let previous_size = $previous_size.into_number(sizes);

                        let weights = match $previous_size
                        {
                            LayerSize::One =>
                            {
                                recorder.new_tensor(this_size, previous_size)
                            },
                            x =>
                            {
                                let previous_layer = x.into_number(sizes);

                                recorder.set_new_tensor_gradientable(LayerType::new_with(this_size, previous_size, ||
                                {
                                    let v = 1.0 / (previous_layer as f32).sqrt();

                                    (fastrand::f32() * 2.0 - 1.0) * v
                                }))
                            }
                        };

                        let weight_original = weights;

                        if $is_hidden
                        {
                            let dropconnect_mask = recorder.new_tensor_no_gradient(this_size, previous_size);

                            WeightInfoPtr{
                                weight_dropped: [DiffTensorPtr::undefined(), DiffTensorPtr::undefined()],
                                weight_original,
                                dropconnect_mask: Some(dropconnect_mask.as_value())
                            }
                        } else
                        {
                            WeightInfoPtr{
                                weight_dropped: [weight_original, weight_original],
                                weight_original,
                                dropconnect_mask: None
                            }
                        }
                    },
                )+}
            }
        }

        impl<T> OptimizerUnit<T> for WeightsContainer<T>
        where
            T: Clone + NewableLayer + Serialize + serde::de::DeserializeOwned
        {
            fn new_zeroed(sizes: $crate::neural_network::LayerSizes) -> Self
            {
                Self{
                    sizes,
                    $(
                        $name: T::new(
                            $previous_size.into_number(sizes),
                            $this_size.into_number(sizes)
                        ),
                    )+
                }
            }
        }

        impl<T> GenericUnit<T> for WeightsContainer<T>
        {
            type Unit<U> = WeightsContainer<U>;

            fn dropconnectable() -> bool
            {
                false $(|| $is_hidden)+
            }

            fn map<U, F>(self, mut f: F) -> WeightsContainer<U>
            where
                F: FnMut(T) -> U
            {
                WeightsContainer{
                    sizes: self.sizes,
                    $(
                        $name: f(self.$name),
                    )+
                }
            }

            fn map_inplace_with_info<F>(&mut self, mut f: F)
            where
                F: FnMut(WeightsSize<&mut T>)
            {
                $(
                    f(WeightsSize{
                        weights: &mut self.$name,
                        this_size: $this_size.into_number(self.sizes),
                        previous_size: $previous_size.into_number(self.sizes),
                        is_hidden: $is_hidden
                    });
                )+
            }

            fn map_with_info<U, F>(self, mut f: F) -> WeightsContainer<U>
            where
                F: FnMut(WeightsSize<T>) -> U
            {
                WeightsContainer{
                    sizes: self.sizes,
                    $(
                        $name: f(WeightsSize{
                            weights: self.$name,
                            this_size: $this_size.into_number(self.sizes),
                            previous_size: $previous_size.into_number(self.sizes),
                            is_hidden: $is_hidden
                        }),
                    )+
                }
            }

            fn map_ref<U, F>(&self, mut f: F) -> WeightsContainer<U>
            where
                F: FnMut(&T) -> U
            {
                WeightsContainer{
                    sizes: self.sizes,
                    $(
                        $name: f(&self.$name),
                    )+
                }
            }

            fn clone_weights_with_info<F>(&self, mut f: F) -> Self
            where
                F: FnMut(WeightsSize<&T>) -> T
            {
                Self{
                    sizes: self.sizes,
                    $(
                        $name: f(
                            WeightsSize{
                                weights: &self.$name,
                                this_size: $this_size.into_number(self.sizes),
                                previous_size: $previous_size.into_number(self.sizes),
                                is_hidden: $is_hidden
                            }
                        ),
                    )+
                }
            }

            fn weights_named_info(&self, layer: usize) -> Self::Unit<WeightsNamed<&T>>
            {
                WeightsContainer{
                    sizes: self.sizes,
                    $(
                        $name: WeightsNamed{
                            name: stringify!($name).to_owned(),
                            layer,
                            weights_size: WeightsSize{
                                weights: &self.$name,
                                this_size: $this_size.into_number(self.sizes),
                                previous_size: $previous_size.into_number(self.sizes),
                                is_hidden: $is_hidden
                            }
                        },
                    )+
                }
            }

            fn for_each_weight<F: FnMut(T)>(self, mut f: F)
            {
                let Self{
                    sizes: _,
                    $(
                        $name,
                    )+
                } = self;

                $(
                    f($name);
                )+
            }

            fn for_each_weight_ref<F: FnMut(&T)>(&self, mut f: F)
            {
                $(
                    f(&self.$name);
                )+
            }

            fn for_each_weight_mut<F: FnMut(&mut T)>(&mut self, mut f: F)
            {
                $(
                    f(&mut self.$name);
                )+
            }
        }

        impl<T> IntoIterator for WeightsContainer<T>
        {
            type Item = T;
            type IntoIter = std::array::IntoIter<Self::Item, { WeightsContainer::<()>::len() }>;

            fn into_iter(self) -> Self::IntoIter
            {
                [
                    $(
                        self.$name,
                    )+
                ].into_iter()
            }
        }

        impl<'a, T> IntoIterator for &'a WeightsContainer<T>
        {
            type Item = &'a T;
            type IntoIter = std::array::IntoIter<Self::Item, { WeightsContainer::<()>::len() }>;

            fn into_iter(self) -> Self::IntoIter
            {
                [
                    $(
                        &self.$name,
                    )+
                ].into_iter()
            }
        }

        impl<'a, T> IntoIterator for &'a mut WeightsContainer<T>
        {
            type Item = &'a mut T;
            type IntoIter = std::array::IntoIter<Self::Item, { WeightsContainer::<()>::len() }>;

            fn into_iter(self) -> Self::IntoIter
            {
                [
                    $(
                        &mut self.$name,
                    )+
                ].into_iter()
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NetworkOutput<State, Output>
{
    pub state: State,
    pub output: Output
}

impl<State, Output> NetworkOutput<State, Output>
{
    fn map<F, NewOutput>(self, f: F) -> NetworkOutput<State, NewOutput>
    where
        F: FnOnce(Output) -> NewOutput
    {
        NetworkOutput{state: self.state, output: f(self.output)}
    }
}

pub type UnitState<N, T> = <<N as UnitFactory>::Unit<WeightInfoPtr> as NetworkUnit>::State<T>;

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize, N::Unit<T>: Serialize", deserialize = "T: Deserialize<'de>, N::Unit<T>: Deserialize<'de>"))]
pub struct WeightsFullContainer<N: UnitFactory, T>
{
    layers: Vec<N::Unit<T>>,
    output: T
}

impl<N: UnitFactory, T> Clone for WeightsFullContainer<N, T>
where
    T: Clone,
    N::Unit<T>: Clone
{
    fn clone(&self) -> Self
    {
        Self{
            layers: self.layers.clone(),
            output: self.output.clone()
        }
    }
}

impl<N: UnitFactory, T> IntoIterator for WeightsFullContainer<N, T>
where
    N::Unit<T>: IntoIterator<Item=T>
{
    type Item = T;
    type IntoIter = iter::Chain<iter::Flatten<vec::IntoIter<N::Unit<T>>>, iter::Once<T>>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.layers.into_iter().flatten().chain(iter::once(self.output))
    }
}

impl<N: UnitFactory, T> WeightsFullContainer<N, T>
{
    pub fn new(
        sizes: LayerSizes,
        unit_f: impl FnMut(LayerSizes) -> N::Unit<T>,
        output: T
    ) -> Self
    {
        Self{
            layers: (0..sizes.layers).map(|index|
            {
                if index == 0
                {
                    sizes
                } else
                {
                    LayerSizes{
                        input: sizes.hidden,
                        ..sizes
                    }
                }
            }).map(unit_f).collect(),
            output
        }
    }

    pub fn map<F, U>(self, mut f: F) -> WeightsFullContainer<N, U>
    where
        N::Unit<T>: GenericUnit<T, Unit<U>=N::Unit<U>>,
        F: FnMut(T) -> U
    {
        WeightsFullContainer{
            output: f(self.output),
            layers: self.layers.into_iter().map(|layer| layer.map(&mut f)).collect()
        }
    }

    pub fn map_ref<F, U>(&self, mut f: F) -> WeightsFullContainer<N, U>
    where
        N::Unit<T>: GenericUnit<T, Unit<U>=N::Unit<U>>,
        F: FnMut(&T) -> U
    {
        WeightsFullContainer{
            output: f(&self.output),
            layers: self.layers.iter().map(|layer| layer.map_ref(&mut f)).collect()
        }
    }

    pub fn iter(&self) -> impl Iterator<Item=&T>
    where
        for<'a> &'a N::Unit<T>: IntoIterator<Item=&'a T>
    {
        self.layers.iter().flatten().chain(iter::once(&self.output))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut T>
    where
        for<'a> &'a mut N::Unit<T>: IntoIterator<Item=&'a mut T>
    {
        self.layers.iter_mut().flatten().chain(iter::once(&mut self.output))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WeightInfoGeneric<D, I>
{
    pub weight_dropped: [D; 2],
    pub weight_original: D,
    pub dropconnect_mask: Option<I>
}

pub type WeightInfoPtr = WeightInfoGeneric<DiffTensorPtr, TensorPtr>;

#[derive(Debug, Clone, Copy)]
pub struct WeightInfo
{
    pub weight: DiffTensor,
    pub dropconnect_mask: Option<TensorIndex>
}

pub type SaveWeightType = LayerType;

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "O: Serialize, N::Unit<O>: Serialize, N::Unit<SaveWeightType>: Serialize", deserialize = "O: Deserialize<'de>, N::Unit<O>: Deserialize<'de>, N::Unit<SaveWeightType>: Deserialize<'de>"))]
pub struct SaveNetwork<N: UnitFactory, O>
{
    sizes: LayerSizes,
    dropout_probability: f32,
    optimizer_info: Option<WeightsFullContainer<N, O>>,
    weights: WeightsFullContainer<N, SaveWeightType>
}

impl<N: UnitFactory, O> From<Network<N, O>> for SaveNetwork<N, O>
where
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo, Unit<SaveWeightType>=N::Unit<SaveWeightType>>
{
    fn from(x: Network<N, O>) -> Self
    {
        Self{
            sizes: x.sizes,
            dropout_probability: x.dropout_probability,
            optimizer_info: x.optimizer_info,
            weights: x.weights.unwrap().map(|weight_info|
            {
                x.recorder.get_tensor(weight_info.weight.as_value()).clone()
            })
        }
    }
}

struct BlockInfo<N: UnitFactory>
where
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
{
    output_ptr: Option<DiffTensorPtr>,
    output: DiffTensor,
    loss: DiffScalar,
    next_state_ptr: Vec<UnitState<N, DiffTensorPtr>>,
    next_state: Vec<UnitState<N, DiffTensor>>,
    index: BlockIndex,
}

impl<N: UnitFactory> BlockInfo<N>
where
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
{
    fn undefined() -> Self
    {
        Self{
            output_ptr: None,
            output: DiffTensor::undefined(),
            loss: DiffScalar::undefined(),
            next_state_ptr: Vec::new(),
            next_state: Vec::new(),
            index: BlockIndex::undefined()
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(from = "SaveNetwork<N, O>")]
#[serde(into = "SaveNetwork<N, O>")]
#[serde(bound(serialize = "O: Serialize + Clone, N::Unit<O>: Serialize + Clone, N::Unit<SaveWeightType>: Serialize, N::Unit<WeightInfo>: Clone + GenericUnit<WeightInfo, Unit<SaveWeightType>=N::Unit<SaveWeightType>>", deserialize = "O: Deserialize<'de>, N::Unit<O>: Deserialize<'de>, N::Unit<SaveWeightType>: Deserialize<'de> + GenericUnit<SaveWeightType, Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>"))]
pub struct Network<N: UnitFactory, O>
where
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
{
    recorder: OperationsRecorder,
    sizes: LayerSizes,
    dropout_probability: f32,
    dropout_masks_ptrs: Vec<TensorPtr>,
    dropout_masks: Vec<TensorIndex>,
    input_ptr: Option<InputTypePtr>,
    input_target: (InputType, OneHotIndex),
    no_state: BlockInfo<N>,
    with_state: BlockInfo<N>,
    optimizer_info: Option<WeightsFullContainer<N, O>>,
    weights_ptr: Option<WeightsFullContainer<N, WeightInfoPtr>>,
    weights: Option<WeightsFullContainer<N, WeightInfo>>
}

// this clone is ONLY used for serialization, dont use for ANYTHING else
impl<N: UnitFactory, O> Clone for Network<N, O>
where
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfo>: Clone,
    O: Clone,
    N::Unit<O>: Clone
{
    fn clone(&self) -> Self
    {
        Self{
            recorder: self.recorder.clone(),
            sizes: self.sizes,
            dropout_probability: self.dropout_probability,
            dropout_masks_ptrs: Vec::new(),
            dropout_masks: Vec::new(),
            input_ptr: None,
            input_target: (InputType::undefined(), OneHotIndex::undefined()),
            no_state: BlockInfo::undefined(),
            with_state: BlockInfo::undefined(),
            optimizer_info: self.optimizer_info.clone(),
            weights_ptr: None,
            weights: self.weights.clone()
        }
    }
}

impl<N: UnitFactory, O> From<SaveNetwork<N, O>> for Network<N, O>
where
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<SaveWeightType>: GenericUnit<SaveWeightType, Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>
{
    fn from(x: SaveNetwork<N, O>) -> Self
    {
        let mut recorder = OperationsRecorder::new();

        // no optimizer info means im not going to train this network
        let discard_gradients = x.optimizer_info.is_none();

        let weight_info_from = |recorder: &mut OperationsRecorder, value: SaveWeightType| -> WeightInfoPtr
        {
            let weights = if discard_gradients
            {
                recorder.set_new_tensor(value)
            } else
            {
                recorder.set_new_tensor_gradientable(value)
            };

            WeightInfoPtr{
                weight_dropped: [weights, weights],
                weight_original: weights,
                dropconnect_mask: None
            }
        };

        Self{
            sizes: x.sizes,
            dropout_probability: x.dropout_probability,
            optimizer_info: x.optimizer_info,
            weights_ptr: Some(WeightsFullContainer{
                output: weight_info_from(&mut recorder, x.weights.output),
                layers: x.weights.layers.into_iter().map(|x| x.map_with_info(|WeightsSize{weights: value, this_size, previous_size, is_hidden}|
                {
                    let info = weight_info_from(&mut recorder, value);

                    if is_hidden
                    {
                        let dropconnect_mask = recorder.new_tensor_no_gradient(this_size, previous_size);

                        WeightInfoPtr{
                            weight_dropped: [DiffTensorPtr::undefined(), DiffTensorPtr::undefined()],
                            weight_original: info.weight_original,
                            dropconnect_mask: Some(dropconnect_mask.as_value())
                        }
                    } else
                    {
                        info
                    }
                })).collect()
            }),
            weights: None,
            dropout_masks_ptrs: Vec::new(),
            dropout_masks: Vec::new(),
            input_ptr: None,
            input_target: (InputType::undefined(), OneHotIndex::undefined()),
            no_state: BlockInfo::undefined(),
            with_state: BlockInfo::undefined(),
            recorder
        }
    }
}

impl<N: UnitFactory, O> Network<N, O>
where
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>
{
    pub fn sizes(&self) -> &LayerSizes
    {
        &self.sizes
    }
}

impl<N: UnitFactory, O> Network<N, O>
where
    N::Unit<O>: OptimizerUnit<O>,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    UnitState<N, DiffTensorPtr>: Clone,
    for<'a> &'a N::Unit<DiffTensor>: IntoIterator<Item=&'a DiffTensor>,
    for<'a> &'a mut N::Unit<DiffTensor>: IntoIterator<Item=&'a mut DiffTensor>
{
    pub fn new(
        sizes: LayerSizes,
        dropout_probability: f32,
        is_multistep: bool,
        is_input_one_hot: bool
    ) -> Self
    where
        O: NewableLayer
    {
        let mut recorder = OperationsRecorder::new();

        let optimizer_info: Option<_> =
            Some(WeightsFullContainer::new(sizes, |size|
            {
                N::Unit::new_zeroed(size)
            }, {
                O::new(sizes.hidden, sizes.output)
            }));

        let output_weights_ptr_tensor = {
            let weights = recorder.set_new_tensor_gradientable(LayerType::new_with(sizes.output, sizes.hidden, ||
            {
                let v = 1.0 / (sizes.hidden as f32).sqrt();

                (fastrand::f32() * 2.0 - 1.0) * v
            }));

            WeightInfoPtr{
                weight_dropped: [weights, weights],
                weight_original: weights,
                dropconnect_mask: None
            }
        };

        let weights_ptr = WeightsFullContainer::new(sizes, |size|
        {
            N::Unit::new(&mut recorder, size)
        }, output_weights_ptr_tensor);

        let weights = None;

        let mut this = Self{
            recorder,
            sizes,
            dropout_masks_ptrs: Vec::new(),
            dropout_masks: Vec::new(),
            input_ptr: None,
            input_target: (InputType::undefined(), OneHotIndex::undefined()),
            no_state: BlockInfo::undefined(),
            with_state: BlockInfo::undefined(),
            optimizer_info,
            weights_ptr: Some(weights_ptr),
            weights,
            dropout_probability
        };

        this.initialize(is_multistep, is_input_one_hot);

        this
    }

    pub fn initialize(&mut self, is_multistep: bool, is_input_one_hot: bool)
    {
        if self.optimizer_info.is_none()
        {
            // doesnt need any initialization
            return;
        }

        self.no_state.index = self.recorder.current_block();

        if is_multistep
        {
            self.with_state.index = self.recorder.new_block();
        }

        self.initialize_dropped_weights();

        self.record_feedforward(is_multistep, is_input_one_hot);

        self.recorder.finish();
    }

    pub fn initialize_dropped_weights(&mut self)
    {
        self.weights_ptr.as_mut().unwrap().layers.iter_mut().for_each(|layer|
        {
            layer.map_inplace_with_info(|WeightsSize{weights: value, is_hidden, ..}|
            {
                if is_hidden
                {
                    value.weight_dropped = dropconnected_weights(
                        &mut self.recorder,
                        value.weight_original,
                        DiffTensorPtr::no_gradient(value.dropconnect_mask.unwrap())
                    );
                }
            });
        });
    }

    pub fn calculate_gradients(&mut self)
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>
    {
        if !self.recorder.is_ready()
        {
            debug_assert_ne!(self.no_state.loss, DiffScalar::undefined());

            let is_multiblock = self.recorder.blocks_count() > 1;

            let respect = if !is_multiblock
            {
                vec![self.no_state.loss.into()]
            } else
            {
                debug_assert_ne!(self.with_state.loss, DiffScalar::undefined());

                vec![self.no_state.loss.into(), self.with_state.loss.into()]
            };

            self.recorder.gradient_with_respect(respect);

            {
                let weight = self.weights_ptr.as_ref().unwrap().output.weight_original;

                self.recorder.store_tensor_until_end(weight.as_value());
                self.recorder.store_tensor_until_end(weight.as_gradient().unwrap());
            }

            let mut prepare_state_block = |state_block: &mut BlockInfo<_>|
            {
                self.recorder.store_value_until_end(state_block.loss.as_value());
            };

            prepare_state_block(&mut self.no_state);

            if is_multiblock
            {
                prepare_state_block(&mut self.with_state);
            }

            self.recorder.resolve_memory();

            let weights = self.weights_ptr.take().unwrap().map(|weight_info|
            {
                WeightInfo{
                    weight: self.recorder.resolve_diff_tensor_ptr(weight_info.weight_original),
                    dropconnect_mask: weight_info.dropconnect_mask.map(|x| self.recorder.resolve_tensor_ptr(x))
                }
            });

            self.weights = Some(weights);

            self.input_target.0 = match self.input_ptr.unwrap()
            {
                InputTypePtr::Normal(x) => InputType::Normal(self.recorder.resolve_tensor_ptr(x)),
                InputTypePtr::OneHot(x) => InputType::OneHot(x)
            };
        }
    }

    fn record_feedforward(&mut self, is_multistep: bool, is_input_one_hot: bool)
    {
        let dropout_masks_ptrs: Vec<_> = self.weights_ptr.as_ref().unwrap().layers.iter().skip(1).map(|_|
        {
            self.recorder.set_new_tensor(LayerType::repeat(self.sizes.hidden, 1, 0.0)).as_value()
        }).collect();

        let this_input: InputTypePtr = if is_input_one_hot
        {
            InputTypePtr::OneHot(self.recorder.new_one_hot())
        } else
        {
            InputTypePtr::Normal(self.recorder.new_tensor_no_gradient(self.sizes.input, 1).as_value())
        };

        let this_target = self.recorder.new_one_hot();

        self.input_ptr = Some(this_input);
        self.input_target.1 = this_target;

        let set_from_output = |NetworkOutput{
            state: next_state_ptr,
            output: (output, loss)
        }: NetworkOutput<Vec<UnitState<N, DiffTensorPtr>>, _>, block: &mut BlockInfo<_>| -> Vec<UnitState<N, DiffTensorPtr>>
        {
            block.next_state_ptr = next_state_ptr.clone();
            block.output_ptr = Some(output);
            block.loss = loss;

            next_state_ptr
        };

        let no_state_next_state = set_from_output(
            self.record_feedforward_single_input(None, &dropout_masks_ptrs, this_input, this_target),
            &mut self.no_state
        );

        if is_multistep
        {
            self.recorder.set_current_block(self.with_state.index);

            set_from_output(
                self.record_feedforward_single_input(Some(no_state_next_state), &dropout_masks_ptrs, this_input, this_target),
                &mut self.with_state
            );
        }

        self.dropout_masks_ptrs = dropout_masks_ptrs;
    }

    fn record_feedforward_single_input(
        &mut self,
        previous_states: Option<Vec<UnitState<N, DiffTensorPtr>>>,
        dropout_masks: &[TensorPtr],
        input: InputTypePtr,
        targets: OneHotIndex
    ) -> NetworkOutput<Vec<UnitState<N, DiffTensorPtr>>, (DiffTensorPtr, DiffScalar)>
    {
        self.record_feedforward_single_input_with_activation(|this, layer_index, previous_state, input|
        {
            this.record_feedforward_unit_last(
                layer_index,
                previous_state,
                input
            ).map(|output| (output, this.recorder.softmax_cross_entropy(output, targets).1))
        }, previous_states, dropout_masks, input)
    }

    fn record_feedforward_single_input_with_activation<F, T>(
        &mut self,
        last_f: F,
        previous_states: Option<Vec<UnitState<N, DiffTensorPtr>>>,
        dropout_masks: &[TensorPtr],
        input: InputTypePtr
    ) -> NetworkOutput<Vec<UnitState<N, DiffTensorPtr>>, T>
    where
        F: FnOnce(&mut Self, usize, Option<&UnitState<N, DiffTensorPtr>>, DiffInputType) -> NetworkOutput<UnitState<N, DiffTensorPtr>, T>
    {
        let mut output: Option<T> = None;
        let mut last_output: Option<DiffInputType> = None;

        let mut states = Vec::with_capacity(self.sizes.layers);

        #[allow(clippy::needless_range_loop)]
        for l_i in 0..self.sizes.layers
        {
            let input = last_output.unwrap_or_else(||
            {
                match input
                {
                    InputTypePtr::Normal(x) => DiffInputType::Normal(DiffTensorPtr::no_gradient(x)),
                    InputTypePtr::OneHot(x) => DiffInputType::OneHot(x)
                }
            });

            let layer = &self.weights_ptr.as_ref().unwrap().layers[l_i];

            let previous_state = previous_states.as_ref().map(|x| &x[l_i]);

            if l_i == (self.sizes.layers - 1)
            {
                let NetworkOutput{
                    state,
                    output: this_output
                } = last_f(self, l_i, previous_state, input);

                output = Some(this_output);

                states.push(state);

                break;
            } else
            {
                let NetworkOutput{
                    state,
                    output: this_output
                } = layer.record_feedforward_unit_nonlast(
                    &mut self.recorder,
                    previous_state,
                    dropout_masks[l_i],
                    input
                );

                last_output = Some(DiffInputType::Normal(this_output));

                states.push(state);
            }
        }

        NetworkOutput{
            state: states,
            output: output.unwrap()
        }
    }

    fn record_feedforward_unit_last(
        &mut self,
        layer_index: usize,
        previous_state: Option<&UnitState<N, DiffTensorPtr>>,
        input: DiffInputType
    ) -> NetworkOutput<UnitState<N, DiffTensorPtr>, DiffTensorPtr>
    {
        self.weights_ptr.as_ref().unwrap().layers[layer_index]
            .record_feedforward_unit(&mut self.recorder, previous_state, input)
            .map(|output|
            {
                self.recorder.matmulv(
                    self.weights_ptr.as_ref().unwrap().output.weight_dropped[self.recorder.current_block().into_index()],
                    output
                )
            })
    }

    pub fn apply_gradients<OP>(
        &mut self,
        gradients: WeightsFullContainer<N, LayerType>,
        optimizer: &mut OP,
        gradient_clip: Option<f32>
    )
    where
        OP: Optimizer<WeightParam=O>,
        N::Unit<LayerType>: IntoIterator<Item=LayerType>,
        for<'b> &'b mut N::Unit<WeightInfo>: IntoIterator<Item=&'b mut WeightInfo>,
        for<'b> &'b mut N::Unit<O>: IntoIterator<Item=&'b mut O>
    {
        gradients.into_iter()
            .zip(self.weights.as_mut().unwrap().iter_mut().zip(self.optimizer_info.as_mut().unwrap().iter_mut()))
            .for_each(|(mut gradient, (network_weights, optimizer_info))|
            {
                if let Some(gradient_clip) = gradient_clip
                {
                    gradient = gradient.cap_magnitude(gradient_clip);
                }

                let change = optimizer.gradient_to_change(optimizer_info, gradient);

                let maybe_optimize_this = ();
                *self.recorder.get_tensor_mut(network_weights.weight.as_value()) -= change;
            });

        optimizer.advance_time();
    }

    pub fn gradients(
        &mut self,
        input: impl ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>
    ) -> (f32, WeightsFullContainer<N, LayerType>)
    where
        N::Unit<WeightInfo>: GenericUnit<WeightInfo, Unit<LayerType>=N::Unit<LayerType>>,
        N::Unit<LayerType>: IntoIterator<Item=LayerType>,
        UnitState<N, DiffTensor>: NetworkUnitStateable,
        for<'b> &'b mut N::Unit<LayerType>: IntoIterator<Item=&'b mut LayerType>
    {
        let mut gradients: Option<WeightsFullContainer<N, LayerType>> = None;

        let total_loss = self.feedforward_with(OperationsRecorder::calculate, |this, is_with_state|
        {
            let this_gradients = this.weights.as_ref().unwrap().map_ref(|weight|
            {
                this.recorder.get_tensor(weight.weight.as_gradient().unwrap()).clone()
            });

            if is_with_state
            {
                gradients.as_mut().unwrap().iter_mut().zip(this_gradients.into_iter()).for_each(|(gradients, this_gradients)|
                {
                    *gradients += this_gradients;
                });
            } else
            {
                debug_assert!(gradients.is_none());

                gradients = Some(this_gradients);
            }
        }, input);

        (total_loss, gradients.expect("input must not be empty"))
    }

    fn feedforward_with(
        &mut self,
        calculate_method: fn(&mut OperationsRecorder, BlockIndex),
        f: impl FnMut(&mut Self, bool),
        input: impl ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>
    ) -> f32
    where
        UnitState<N, DiffTensor>: NetworkUnitStateable
    {
        self.feedforward_setup_dropout();

        self.feedforward_like(calculate_method, |this, this_target|
        {
            this.recorder.set_one_hot(this.input_target.1, this_target);
        }, convert::identity, f, input)
    }

    fn feedforward_like<T, U>(
        &mut self,
        calculate_method: fn(&mut OperationsRecorder, BlockIndex),
        mut setup: impl FnMut(&mut Self, U),
        get_input: impl Fn(T) -> (OwnedInputType, U),
        mut f: impl FnMut(&mut Self, bool),
        input: impl ExactSizeIterator<Item=T>
    ) -> f32
    where
        UnitState<N, DiffTensor>: NetworkUnitStateable
    {
        let is_multiblock = self.recorder.blocks_count() > 1;

        let mut total_loss = 0.0;

        input.enumerate().for_each(|(index, input)|
        {
            let (this_input, rest_input) = get_input(input);

            let is_with_state = index != 0 && is_multiblock;

            match self.input_target.0
            {
                InputType::Normal(x) => self.recorder.set_tensor(x, this_input.into_normal()),
                InputType::OneHot(x) => self.recorder.set_one_hot(x, this_input.into_one_hot())
            }

            setup(self, rest_input);

            let this_info = if is_with_state { &self.with_state } else { &self.no_state };

            if is_with_state && index != 1
            {
                self.no_state.next_state.iter()
                    .zip(this_info.next_state.iter())
                    .for_each(|(no_state, with_state)| no_state.set(&mut self.recorder, with_state));
            }

            calculate_method(&mut self.recorder, this_info.index);

            total_loss += self.recorder.get_value(this_info.loss.as_value());

            f(self, is_with_state);
        });

        total_loss
    }

    pub fn weights_info<'b, 'c>(
        &'b self
    ) -> Vec<WeightsNamed<&'b LayerType>>
    where
        for<'a> N::Unit<WeightInfo>: GenericUnit<WeightInfo, Unit<WeightsNamed<&'a WeightInfo>>=N::Unit<WeightsNamed<&'a WeightInfo>>>,
        N::Unit<WeightsNamed<&'b WeightInfo>>: IntoIterator<Item=WeightsNamed<&'b WeightInfo>>
    {
        self.weights.as_ref().unwrap().layers.iter().enumerate()
            .flat_map(|(layer_index, layer)|
            {
                layer.weights_named_info(layer_index).into_iter()
            })
            .chain(iter::once(WeightsNamed{
                name: "output".to_owned(),
                layer: self.sizes.layers.saturating_sub(1),
                weights_size: WeightsSize{
                    weights: &self.weights.as_ref().unwrap().output,
                    this_size: self.sizes.output,
                    previous_size: self.sizes.hidden,
                    is_hidden: false
                }
            }))
            .map(|x| x.map(|x| self.recorder.get_tensor(x.weight.as_value())))
            .collect::<Vec<_>>()
    }

    #[allow(dead_code)]
    pub fn parameters_amount(&self) -> u128
    where
        N::Unit<WeightInfo>: NetworkUnitParameterable
    {
        let layers_sum: u128 = self.weights.as_ref().unwrap().layers.iter().map(|layer|
        {
            layer.parameters_amount(self.sizes)
        }).sum();

        layers_sum + self.sizes.input as u128 * self.sizes.hidden as u128
    }

    fn with_predict<T, F>(
        &mut self,
        input: impl Iterator<Item=(OwnedInputType, OneHotLayer)>,
        f: F
    ) -> impl Iterator<Item=(usize, T)>
    where
        UnitState<N, DiffTensor>: NetworkUnitStateable,
        F: Fn(&LayerType, usize, usize) -> T
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();

        self.predict(input.into_iter()).into_iter().zip(output).map(move |(predicted, target)|
        {
            let positions = &target.positions;
            assert_eq!(positions.len(), 1);

            let target_index = positions[0];

            let predicted = predicted.borrow();
            let highest_index = predicted.highest_index();

            (highest_index, f(predicted, highest_index, target_index))
        })
    }

    #[allow(dead_code)]
    pub fn top_guesses(
        &mut self,
        input: impl Iterator<Item=(OwnedInputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, u32)>
    where
        UnitState<N, DiffTensor>: NetworkUnitStateable
    {
        self.with_predict(input, |predicted, _highest_index, target_index|
        {
            let mut predicted: Vec<(usize, f32)> = predicted.as_vec().into_iter().enumerate().collect();
            predicted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            predicted.into_iter()
                .enumerate()
                .find(|(_, (index, _))| *index == target_index)
                .expect("target index must be within bounds")
                .0 as u32
        })
    }

    #[allow(dead_code)]
    pub fn certainty_guesses(
        &mut self,
        input: impl Iterator<Item=(OwnedInputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, f32)>
    where
        UnitState<N, DiffTensor>: NetworkUnitStateable
    {
        self.with_predict(input, |predicted, _highest_index, target_index|
        {
            *predicted.iter().nth(target_index).unwrap()
        })
    }

    #[allow(dead_code)]
    pub fn correct_guesses(
        &mut self,
        input: impl Iterator<Item=(OwnedInputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, bool)>
    where
        UnitState<N, DiffTensor>: NetworkUnitStateable
    {
        self.with_predict(input, |_predicted, highest_index, target_index|
        {
            highest_index == target_index
        })
    }

    #[allow(dead_code)]
    pub fn accuracy(
        &mut self,
        input: impl Iterator<Item=(OwnedInputType, OneHotLayer)>
    ) -> f32
    where
        UnitState<N, DiffTensor>: NetworkUnitStateable
    {
        let mut total = 0;
        let correct_amount = self.correct_guesses(input).filter(|(_, x)|
        {
            total += 1;

            *x
        }).count();

        correct_amount as f32 / total as f32
    }

    pub fn feedforward_setup_dropout(&mut self)
    {
        let weights = self.weights.as_mut().unwrap();

        if N::Unit::<WeightInfo>::dropconnectable()
        {
            weights.layers.iter().for_each(|layer|
            {
                layer.for_each_weight_ref(|weight_info|
                {
                    if let Some(dropconnect_mask) = weight_info.dropconnect_mask
                    {
                        Self::set_dropout_mask(self.recorder.get_tensor_mut(dropconnect_mask), DROPCONNECT_PROBABILITY);
                    }
                });
            });
        }

        self.dropout_masks.iter().for_each(|dropout_mask|
        {
            Self::set_dropout_mask(self.recorder.get_tensor_mut(*dropout_mask), self.dropout_probability);
        });
    }

    pub fn feedforward_no_gradient(
        &mut self,
        input: impl ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>
    ) -> f32
    where
        UnitState<N, DiffTensor>: NetworkUnitStateable
    {
        self.feedforward_with(OperationsRecorder::calculate_feedforward, |_this, _is_with_state| {}, input)
    }

    fn predict(
        &mut self,
        input: impl Iterator<Item=OwnedInputType> + ExactSizeIterator
    ) -> Vec<LayerType>
    where
        UnitState<N, DiffTensor>: NetworkUnitStateable
    {
        self.predict_temperature(1.0, input)
    }

    pub fn predict_temperature(
        &mut self,
        temperature: f32,
        input: impl Iterator<Item=OwnedInputType> + ExactSizeIterator
    ) -> Vec<LayerType>
    where
        UnitState<N, DiffTensor>: NetworkUnitStateable
    {
        let weights = self.weights.as_mut().unwrap();
        let mut outputs: Vec<LayerType> = Vec::with_capacity(input.len());

        if N::Unit::<WeightInfo>::dropconnectable()
        {
            weights.layers.iter().for_each(|layer|
            {
                layer.for_each_weight_ref(|weight_info|
                {
                    if let Some(dropconnect_mask) = weight_info.dropconnect_mask
                    {
                        Self::set_dropout_mask(self.recorder.get_tensor_mut(dropconnect_mask), 0.0);
                    }
                });
            });
        }

        self.dropout_masks.iter().for_each(|dropout_mask|
        {
            Self::set_dropout_mask(self.recorder.get_tensor_mut(*dropout_mask), 0.0);
        });

        self.feedforward_like(
            OperationsRecorder::calculate_feedforward,
            |_, _| {},
            |x| (x, ()),
            |this, is_with_state|
            {
                let output = if is_with_state { this.with_state.output } else { this.no_state.output };
                let mut output = this.recorder.get_tensor(output.as_value()).clone();

                Softmaxer::softmax_temperature(&mut output, temperature);

                outputs.push(output);
            },
            input
        );

        outputs
    }

    fn set_dropout_mask(
        target: &mut LayerType,
        probability: f32
    )
    {
        let scaled_value = (1.0 - probability).recip();

        if probability == 0.0
        {
            target.fill(1.0);
        } else
        {
            target.fill_with(||
            {
                let roll = fastrand::f32();

                if roll >= probability
                {
                    scaled_value
                } else
                {
                    0.0
                }
            });
        }
    }
}

impl<O> Network<EmbeddingsUnitFactory, O>
where
    EmbeddingsUnitFactory: UnitFactory
{
    pub fn without_optimizer(self) -> Network<EmbeddingsUnitFactory, ()>
    {
        Network{
            recorder: self.recorder,
            sizes: self.sizes,
            dropout_probability: self.dropout_probability,
            optimizer_info: None,
            weights_ptr: self.weights_ptr,
            weights: self.weights,
            dropout_masks_ptrs: self.dropout_masks_ptrs,
            dropout_masks: self.dropout_masks,
            input_ptr: self.input_ptr,
            input_target: self.input_target,
            no_state: self.no_state,
            with_state: self.with_state
        }
    }

    pub fn embeddings(&self, input: &OneHotLayer) -> LayerType
    {
        let weights = self.weights.as_ref().unwrap();
        debug_assert_eq!(weights.layers.len(), 1);

        weights.layers[0].embeddings_calculate(&self.recorder, input)
    }
}
