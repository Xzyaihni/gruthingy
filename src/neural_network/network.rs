use std::{
    f32,
    vec,
    mem,
    iter,
    fmt::{self, Debug},
    cmp::Ordering,
    borrow::Borrow
};

use serde::{Serialize, Deserialize};

use crate::{
    EmbeddingsUnitFactory,
    neural_network::{
        OperationsRecorder,
        Softmaxer,
        PhiOtherSelectorRecordingIndex,
        NetworkStateSelectable,
        NetworkStateGettable,
        NetworkUnitNewable,
        DiffTensor,
        DiffTensorPtr,
        DiffScalar,
        LoopIndex,
        TensorIndex,
        TensorPtr,
        OneHotLayer,
        OneHotIndex,
        InputType,
        InputTypePtr,
        DiffInputType,
        OwnedInputType,
        LayerType,
        LayerTypeRef,
        LayerTypeMut,
        NetworkUnit,
        NewableLayer,
        GenericUnit,
        Optimizer,
        OptimizerUnit,
        UnitFactory,
        DROPCONNECT_PROBABILITY,
        network_unit::{EmbeddingsableOwned, NetworkUnitParameterable}
    }
};


#[derive(Debug, PartialEq)]
pub struct WeightsSize<T>
{
    pub weights: T,
    pub previous_size: usize,
    pub this_size: usize,
    pub is_hidden: bool,
    pub is_state_reliant: bool
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
            is_state_reliant: self.is_state_reliant,
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

#[macro_export]
macro_rules! create_weights_container
{
    ($(($name:ident, $is_hidden:expr, $is_state_reliant:expr, $previous_size:expr, $this_size:expr)),+) =>
    {
        use std::ops::{SubAssign, AddAssign, DivAssign};

        use $crate::neural_network::{
            DebugUnitInfo,
            LayerType,
            NewableLayer,
            GenericUnit,
            OptimizerUnit,
            network::{WeightsNamed, WeightsSize}
        };


        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
                            is_hidden: $is_hidden,
                            is_state_reliant: $is_state_reliant
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
                                let bias = recorder.new_tensor(this_size, previous_size);

                                recorder.set_tensor_ptr_zeroed(bias.as_value());

                                bias
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

                        recorder.name_diff_tensor(weights, stringify!($name));

                        let weight_original = weights;

                        if $is_hidden
                        {
                            let dropconnect_mask = recorder.new_tensor_no_gradient(this_size, previous_size);
                            recorder.name_diff_tensor(dropconnect_mask, stringify!($name).to_owned() + "_dropconnect_mask");

                            WeightInfoPtr{
                                weight_dropped: DiffTensorPtr::undefined(),
                                weight_original,
                                dropconnect_mask: Some(dropconnect_mask.as_value())
                            }
                        } else
                        {
                            WeightInfoPtr{
                                weight_dropped: weight_original,
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
                F: FnMut(WeightsSize<&mut T>, DebugUnitInfo)
            {
                $(
                    let debug_unit_info;

                    #[cfg(debug_assertions)]
                    {
                        debug_unit_info = DebugUnitInfo{
                            name: stringify!($name)
                        };
                    }

                    #[cfg(not(debug_assertions))]
                    {
                        debug_unit_info = DebugUnitInfo;
                    }

                    f(WeightsSize{
                        weights: &mut self.$name,
                        this_size: $this_size.into_number(self.sizes),
                        previous_size: $previous_size.into_number(self.sizes),
                        is_hidden: $is_hidden,
                        is_state_reliant: $is_state_reliant
                    }, debug_unit_info);
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
                            is_hidden: $is_hidden,
                            is_state_reliant: $is_state_reliant
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

            fn map_ref_with_info<U, F>(&self, mut f: F) -> WeightsContainer<U>
            where
                F: FnMut(WeightsSize<&T>) -> U
            {
                WeightsContainer{
                    sizes: self.sizes,
                    $(
                        $name: f(WeightsSize{
                            weights: &self.$name,
                            this_size: $this_size.into_number(self.sizes),
                            previous_size: $previous_size.into_number(self.sizes),
                            is_hidden: $is_hidden,
                            is_state_reliant: $is_state_reliant
                        }),
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
                                is_hidden: $is_hidden,
                                is_state_reliant: $is_state_reliant
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
                                is_hidden: $is_hidden,
                                is_state_reliant: $is_state_reliant
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

impl<N: UnitFactory, T> PartialEq for WeightsFullContainer<N, T>
where
    T: PartialEq,
    N::Unit<T>: PartialEq
{
    fn eq(&self, other: &Self) -> bool
    {
        self.layers == other.layers && self.output == other.output
    }
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

impl<N: UnitFactory, T> Debug for WeightsFullContainer<N, T>
where
    T: Debug,
    N::Unit<T>: Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        f.debug_struct("WeightsFullContainer")
            .field("layers", &self.layers)
            .field("output", &self.output)
            .finish()
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
    pub weight_dropped: D,
    pub weight_original: D,
    pub dropconnect_mask: Option<I>
}

pub type WeightInfoPtr = WeightInfoGeneric<DiffTensorPtr, TensorPtr>;

#[derive(Debug, Clone, Copy, PartialEq)]
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
    N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<SaveWeightType>=N::Unit<SaveWeightType>>,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo, Unit<SaveWeightType>=N::Unit<SaveWeightType>>
{
    fn from(x: Network<N, O>) -> Self
    {
        let weights = if let Some(w) = x.weights
        {
            w.map(|weight_info|
            {
                x.recorder.get_tensor(weight_info.weight.as_value()).clone_owned()
            })
        } else
        {
            x.weights_ptr.unwrap().map(|weight_info|
            {
                x.recorder.get_tensor_memory_value(weight_info.weight_original.as_value()).clone_owned()
            })
        };

        Self{
            sizes: x.sizes,
            dropout_probability: x.dropouts.dropout_probability,
            optimizer_info: x.optimizer_info,
            weights
        }
    }
}

#[derive(Clone)]
struct NetworkDropoutData
{
    dropout_probability: f32,
    dropout_masks_ptrs: Vec<TensorPtr>,
    dropout_masks: Vec<TensorIndex>
}

impl NetworkDropoutData
{
    fn new(dropout_probability: f32) -> Self
    {
        Self{
            dropout_masks_ptrs: Vec::new(),
            dropout_masks: Vec::new(),
            dropout_probability
        }
    }
}

#[derive(Clone)]
struct NetworkInputsData
{
    steps_loop: Option<LoopIndex>,
    input_ptr: Option<InputTypePtr>,
    initial_input: InputType,
    initial_target: InputType
}

impl Default for NetworkInputsData
{
    fn default() -> Self
    {
        Self{
            steps_loop: None,
            input_ptr: None,
            initial_input: InputType::undefined(),
            initial_target: InputType::undefined()
        }
    }
}

#[derive(Clone)]
struct NetworkOutputsData
{
    output_ptr: Option<DiffTensorPtr>,
    output: DiffTensor,
    loss: DiffScalar
}

impl Default for NetworkOutputsData
{
    fn default() -> Self
    {
        Self{
            output_ptr: None,
            output: DiffTensor::undefined(),
            loss: DiffScalar::undefined()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum NetworkMode
{
    Predict,
    Train
}

#[derive(Serialize, Deserialize)]
#[serde(from = "SaveNetwork<N, O>")]
#[serde(into = "SaveNetwork<N, O>")]
#[serde(bound(serialize = "O: Serialize + Clone, N::Unit<O>: Serialize + Clone, N::Unit<SaveWeightType>: Serialize, N::Unit<WeightInfo>: Clone + GenericUnit<WeightInfo, Unit<SaveWeightType>=N::Unit<SaveWeightType>>, N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<SaveWeightType>=N::Unit<SaveWeightType>>", deserialize = "O: Deserialize<'de>, N::Unit<O>: Deserialize<'de>, N::Unit<SaveWeightType>: Deserialize<'de> + GenericUnit<SaveWeightType, Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>, N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>, for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>"))]
pub struct Network<N: UnitFactory, O>
where
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
{
    recorder: OperationsRecorder,
    network_mode: Option<NetworkMode>,
    sizes: LayerSizes,
    is_multistep: Option<bool>,
    is_input_one_hot: Option<bool>,
    dropouts: NetworkDropoutData,
    inputs: NetworkInputsData,
    outputs: NetworkOutputsData,
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
            network_mode: self.network_mode,
            sizes: self.sizes,
            is_multistep: self.is_multistep,
            is_input_one_hot: self.is_input_one_hot,
            dropouts: self.dropouts.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            optimizer_info: self.optimizer_info.clone(),
            weights_ptr: self.weights_ptr.clone(),
            weights: self.weights.clone()
        }
    }
}

impl<N: UnitFactory, O> From<SaveNetwork<N, O>> for Network<N, O>
where
    N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<SaveWeightType>: GenericUnit<SaveWeightType, Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
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
                weight_dropped: weights,
                weight_original: weights,
                dropconnect_mask: None
            }
        };

        let mut this = Self{
            sizes: x.sizes,
            network_mode: None,
            is_multistep: None,
            is_input_one_hot: None,
            optimizer_info: x.optimizer_info,
            weights_ptr: Some(WeightsFullContainer{
                output: weight_info_from(&mut recorder, x.weights.output),
                layers: x.weights.layers.into_iter().map(|x| x.map_with_info(|WeightsSize{weights: value, this_size, previous_size, is_hidden, ..}|
                {
                    let info = weight_info_from(&mut recorder, value);

                    if is_hidden
                    {
                        let dropconnect_mask = recorder.new_tensor_no_gradient(this_size, previous_size);

                        WeightInfoPtr{
                            weight_dropped: DiffTensorPtr::undefined(),
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
            dropouts: NetworkDropoutData::new(x.dropout_probability),
            inputs: NetworkInputsData::default(),
            outputs: NetworkOutputsData::default(),
            recorder
        };

        if discard_gradients
        {
            this.initialize_no_gradient();
        }

        this
    }
}

impl<N: UnitFactory, O> Network<N, O>
where
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>
{
    fn initialize_no_gradient(&mut self)
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        self.recorder.finish();

        self.recorder.no_gradient();

        self.recorder.resolve_memory();
    }

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
    UnitState<N, DiffTensorPtr>: Clone + NetworkStateSelectable<UnitState<N, PhiOtherSelectorRecordingIndex>>,
    UnitState<N, PhiOtherSelectorRecordingIndex>: NetworkStateGettable<UnitState<N, DiffTensorPtr>>,
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
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
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

            recorder.name_diff_tensor(weights, "output_weights");

            WeightInfoPtr{
                weight_dropped: weights,
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
            network_mode: None,
            sizes,
            dropouts: NetworkDropoutData::new(dropout_probability),
            inputs: NetworkInputsData::default(),
            outputs: NetworkOutputsData::default(),
            optimizer_info,
            weights_ptr: Some(weights_ptr),
            weights,
            is_multistep: Some(is_multistep),
            is_input_one_hot: Some(is_input_one_hot)
        };

        this.initialize();

        this
    }

    pub fn set_train_mode(&mut self)
    {
        debug_assert!(self.network_mode.is_none());

        self.network_mode = Some(NetworkMode::Train);
    }

    pub fn set_predict_mode(&mut self)
    {
        debug_assert!(self.network_mode.is_none());

        self.network_mode = Some(NetworkMode::Predict);
    }

    pub fn initialize_with_params(&mut self, is_multistep: bool, is_input_one_hot: bool)
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        self.is_multistep = Some(is_multistep);
        self.is_input_one_hot = Some(is_input_one_hot);

        self.initialize();
    }

    pub fn initialize(&mut self)
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        if self.optimizer_info.is_none()
        {
            return;
        }

        self.initialize_dropped_weights();
    }

    pub fn initialize_dropped_weights(&mut self)
    {
        if let Some(weights_ptr) = self.weights_ptr.as_mut()
        {
            weights_ptr.layers.iter_mut().for_each(|layer|
            {
                layer.map_inplace_with_info(|weights_size, _debug_info|
                {
                    if weights_size.is_hidden
                    {
                        let weight_dropped = self.recorder.mul_componentwise(
                            weights_size.weights.weight_original,
                            DiffTensorPtr::no_gradient(weights_size.weights.dropconnect_mask.unwrap())
                        );

                        #[cfg(debug_assertions)]
                        {
                            self.recorder.name_diff_tensor(weight_dropped, _debug_info.name.to_owned() + "_dropped");

                            self.recorder.allow_discard(weights_size.weights.dropconnect_mask.unwrap());
                        }

                        weights_size.weights.weight_dropped = weight_dropped;
                    }
                });
            });
        }
    }

    pub fn prepare(&mut self, store_gradient: bool)
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        if !self.recorder.is_ready()
        {
            self.record_feedforward(store_gradient);

            self.prepare_setup_shared();

            if store_gradient
            {
                self.weights_ptr.as_ref().unwrap().iter().for_each(|weight|
                {
                    self.recorder.store_tensor_until_end(weight.weight_original.as_value());
                    self.recorder.store_tensor_until_end(weight.weight_original.as_gradient().unwrap());
                });

                self.recorder.gradient(self.outputs.loss.into());
            } else
            {
                self.recorder.store_tensor_until_end(self.outputs.output_ptr.unwrap().as_value());

                self.recorder.no_gradient();
            }

            self.prepare_shared(store_gradient);

            if !store_gradient
            {
                let output_ptr = DiffTensorPtr::no_gradient(self.outputs.output_ptr.unwrap().as_value());

                self.outputs.output = self.recorder.resolve_diff_tensor_ptr(output_ptr);
            }
        }
    }

    fn prepare_setup_shared(&mut self)
    {
        self.recorder.finish();

        self.recorder.store_tensor_until_end(self.weights_ptr.as_ref().unwrap().output.weight_original.as_value());

        if self.network_mode == Some(NetworkMode::Train)
        {
            debug_assert_ne!(self.outputs.loss, DiffScalar::undefined());

            self.recorder.store_value_until_end(self.outputs.loss.as_value());
        }
    }

    fn prepare_shared(&mut self, store_gradient: bool)
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>
    {
        self.recorder.resolve_memory();

        {
            let input_ptr = self.inputs.input_ptr.take().expect("input ptr must be set");

            self.inputs.initial_input = match input_ptr
            {
                InputTypePtr::Normal(x) => InputType::Normal(self.recorder.resolve_tensor_ptr(x)),
                InputTypePtr::OneHot(x) => InputType::OneHot(x)
            };
        }

        let weights = self.weights_ptr.take().unwrap().map(|mut weight_info|
        {
            if !store_gradient { weight_info.weight_original.clear_gradient(); }

            WeightInfo{
                weight: self.recorder.resolve_diff_tensor_ptr(weight_info.weight_original),
                dropconnect_mask: weight_info.dropconnect_mask.map(|x| self.recorder.resolve_tensor_ptr(x))
            }
        });

        self.weights = Some(weights);

        self.resolve_dropout_masks();
    }

    fn resolve_dropout_masks(&mut self)
    {
        self.dropouts.dropout_masks = mem::take(&mut self.dropouts.dropout_masks_ptrs).into_iter().map(|dropout_mask|
        {
            self.recorder.resolve_tensor_ptr(dropout_mask)
        }).collect();
    }

    fn create_dropout_masks_ptrs(&mut self) -> Vec<TensorPtr>
    {
        let ptrs: Vec<TensorPtr> = self.weights_ptr.as_ref().unwrap().layers.iter().skip(1).map(|_|
        {
            let ptr = self.recorder.new_tensor_no_gradient(self.sizes.hidden, 1).as_value();
            self.recorder.name_tensor(ptr, "dropout_mask");

            self.recorder.allow_discard(ptr);

            ptr
        }).collect();

        self.dropouts.dropout_masks_ptrs = ptrs.clone();

        ptrs
    }

    fn record_feedforward(&mut self, store_gradient: bool)
    {
        debug_assert!(self.network_mode.is_some());

        let dropout_masks_ptrs = self.create_dropout_masks_ptrs();

        let create_input = {
            let is_input_one_hot = self.is_input_one_hot.unwrap();
            let input_size = self.sizes.input;

            move |recorder: &mut OperationsRecorder| -> InputTypePtr
            {
                if is_input_one_hot
                {
                    InputTypePtr::OneHot(recorder.new_one_hot())
                } else
                {
                    InputTypePtr::Normal(recorder.new_tensor_no_gradient(input_size, 1).as_value())
                }
            }
        };

        let this_input_first = create_input(&mut self.recorder);

        self.recorder.name_input(this_input_first, "input_first");

        if let InputTypePtr::Normal(tensor) = this_input_first
        {
            self.recorder.allow_discard(tensor);
        }

        self.inputs.input_ptr = Some(this_input_first);

        let has_target = self.network_mode == Some(NetworkMode::Train);

        let this_target_first = if has_target
        {
            let this_target_first = self.recorder.new_one_hot();

            self.recorder.name_one_hot(this_target_first, "target_first");

            self.inputs.initial_target = this_target_first.into();

            Some(this_target_first)
        } else
        {
            None
        };

        let no_state_output = self.record_feedforward_single_input(
            None,
            &dropout_masks_ptrs,
            this_input_first,
            this_target_first,
            store_gradient
        );

        self.recorder.name_diff_tensor(no_state_output.output.0, "no_state_output");

        let no_state_loss = no_state_output.output.1;

        if let Some(no_state_loss) = no_state_loss
        {
            self.recorder.name_diff_scalar(no_state_loss, "no_state_loss");
        }

        let (final_output, final_loss) = if self.is_multistep.unwrap()
        {
            let this_input_loop = create_input(&mut self.recorder);

            let this_target_loop = has_target.then(|| self.recorder.new_one_hot());

            let final_loss_selector = no_state_loss.map(|no_state_loss| self.recorder.phi_other_selector(no_state_loss));

            let state_selectors: Vec<_> = no_state_output.state.iter().map(|state| state.phi_other_selector(&mut self.recorder)).collect();

            let loop_index = {
                let inputs = if let Some(this_target_loop) = this_target_loop
                {
                    vec![this_input_loop, this_target_loop.into()]
                } else
                {
                    vec![this_input_loop]
                };

                self.recorder.begin_loop(inputs)
            };

            let previous_state_selected: Vec<_> = state_selectors.iter().map(|selector| selector.select(&mut self.recorder)).collect();

            let final_output = self.record_feedforward_single_input(
                Some(previous_state_selected),
                &dropout_masks_ptrs,
                this_input_loop,
                this_target_loop,
                store_gradient
            );

            self.recorder.name_diff_tensor(final_output.output.0, "final_output");

            let final_output_loss = final_output.output.1;

            if let Some(final_output_loss) = final_output_loss
            {
                self.recorder.name_diff_scalar(final_output_loss, "final_output_loss");
            }

            let new_combined = final_output_loss.map(|final_output_loss|
            {
                let final_loss_selector = final_loss_selector.expect("must be set");

                let final_loss_selected = self.recorder.select_value(final_loss_selector);

                let new_combined = self.recorder.add_scalars(final_loss_selected, final_output_loss);

                self.recorder.set_phi_other_selector(final_loss_selector, new_combined);

                new_combined
            });

            state_selectors.iter().zip(final_output.state).for_each(|(selector, final_state)|
            {
                selector.set_phi_other_selector(&mut self.recorder, final_state)
            });

            self.recorder.end_loop(loop_index);

            self.inputs.steps_loop = Some(loop_index);

            (final_output.output.0, new_combined)
        } else
        {
            no_state_output.output
        };

        self.outputs.output_ptr = Some(final_output);

        if let Some(final_loss) = final_loss
        {
            self.outputs.loss = final_loss;
        }
    }

    fn record_feedforward_single_input(
        &mut self,
        previous_states: Option<Vec<UnitState<N, DiffTensorPtr>>>,
        dropout_masks: &[TensorPtr],
        input: InputTypePtr,
        targets: Option<OneHotIndex>,
        store_gradient: bool
    ) -> NetworkOutput<Vec<UnitState<N, DiffTensorPtr>>, (DiffTensorPtr, Option<DiffScalar>)>
    {
        self.record_feedforward_single_input_with_activation(|this, layer_index, previous_state, input|
        {
            this.record_feedforward_unit_last(
                layer_index,
                previous_state,
                input,
                store_gradient
            ).map(|output|
            {
                (output, targets.map(|targets| this.recorder.softmax_cross_entropy(output, targets).1))
            })
        }, previous_states, dropout_masks, input, store_gradient)
    }

    fn record_feedforward_single_input_with_activation<F, T>(
        &mut self,
        last_f: F,
        previous_states: Option<Vec<UnitState<N, DiffTensorPtr>>>,
        dropout_masks: &[TensorPtr],
        input: InputTypePtr,
        store_gradient: bool
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
                    input,
                    store_gradient
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
        input: DiffInputType,
        store_gradient: bool
    ) -> NetworkOutput<UnitState<N, DiffTensorPtr>, DiffTensorPtr>
    {
        self.weights_ptr.as_ref().unwrap().layers[layer_index]
            .record_feedforward_unit(&mut self.recorder, previous_state, input, store_gradient)
            .map(|output|
            {
                self.recorder.matmulv(self.weights_ptr.as_ref().unwrap().output.weight_dropped, output)
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

                self.recorder.get_tensor_mut::<true>(network_weights.weight.as_value()).sub_inplace(LayerTypeRef::from(&change));
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
        for<'b> &'b mut N::Unit<LayerType>: IntoIterator<Item=&'b mut LayerType>
    {
        let inputs_count = input.len();

        let total_loss = self.feedforward_with(OperationsRecorder::calculate, input);

        let f = |weight: &WeightInfo|
        {
            self.recorder.get_tensor(weight.weight.as_gradient().unwrap()).clone_owned()
        };

        let gradients = if inputs_count == 1
        {
            let weights = self.weights.as_ref().unwrap();
            WeightsFullContainer{
                output: f(&weights.output),
                layers: weights.layers.iter().map(|x|
                {
                    x.map_ref_with_info(|WeightsSize{weights: weight, this_size, previous_size, is_state_reliant, ..}|
                    {
                        if is_state_reliant
                        {
                            LayerType::new(this_size, previous_size)
                        } else
                        {
                            f(weight)
                        }
                    })
                }).collect()
            }
        } else
        {
            self.weights.as_ref().unwrap().map_ref(f)
        };

        (total_loss, gradients)
    }

    fn feedforward_with(
        &mut self,
        calculate_function: fn(&mut OperationsRecorder),
        input: impl ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>
    ) -> f32
    {
        debug_assert_eq!(self.network_mode, Some(NetworkMode::Train));

        self.feedforward_setup_dropout();

        let inputs_count = input.len();
        let mut inputs = input.flat_map(|(input, target)| [input, OwnedInputType::OneHot(target)]);

        debug_assert!(inputs_count > 0, "inputs must not be empty");

        self.recorder.set_input(self.inputs.initial_input, inputs.next().unwrap());
        self.recorder.set_input(self.inputs.initial_target, inputs.next().unwrap());

        if inputs_count > 1
        {
            let steps_loop = self.inputs.steps_loop.unwrap();

            self.recorder.set_loop_inputs(steps_loop, inputs.collect());

            self.recorder.set_loop_times(steps_loop, inputs_count - 1);
        }

        calculate_function(&mut self.recorder);

        self.recorder.get_value(self.outputs.loss.as_value())
    }

    pub fn weights_info<'b, 'c>(
        &'b self
    ) -> Vec<WeightsNamed<LayerTypeRef<'b>>>
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
                    is_hidden: false,
                    is_state_reliant: false
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
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
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
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
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
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
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
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
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
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
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
                        Self::set_dropout_mask(self.recorder.get_tensor_mut::<false>(dropconnect_mask), DROPCONNECT_PROBABILITY);
                    }
                });
            });
        }

        self.dropouts.dropout_masks.iter().for_each(|dropout_mask|
        {
            Self::set_dropout_mask(self.recorder.get_tensor_mut::<false>(*dropout_mask), self.dropouts.dropout_probability);
        });
    }

    pub fn feedforward_no_gradient(
        &mut self,
        input: impl ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>
    ) -> f32
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        self.prepare(false);

        self.feedforward_with(OperationsRecorder::calculate_feedforward, input)
    }

    fn predict(
        &mut self,
        input: impl Iterator<Item=OwnedInputType> + ExactSizeIterator
    ) -> Vec<LayerType>
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        let mut outputs: Vec<LayerType> = Vec::with_capacity(input.len());

        self.predict_temperature(1.0, input, |output| outputs.push(output));

        outputs
    }

    pub fn predict_temperature(
        &mut self,
        temperature: f32,
        mut input: impl Iterator<Item=OwnedInputType> + ExactSizeIterator,
        mut f_output: impl FnMut(LayerType)
    )
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        debug_assert_eq!(self.network_mode, Some(NetworkMode::Predict));

        self.prepare(false);

        let weights = self.weights.as_mut().unwrap();

        if N::Unit::<WeightInfo>::dropconnectable()
        {
            weights.layers.iter().for_each(|layer|
            {
                layer.for_each_weight_ref(|weight_info|
                {
                    if let Some(dropconnect_mask) = weight_info.dropconnect_mask
                    {
                        Self::set_dropout_mask(self.recorder.get_tensor_mut::<false>(dropconnect_mask), 0.0);
                    }
                });
            });
        }

        self.dropouts.dropout_masks.iter().for_each(|dropout_mask|
        {
            Self::set_dropout_mask(self.recorder.get_tensor_mut::<false>(*dropout_mask), 0.0);
        });

        let inputs_count = input.len();

        debug_assert!(inputs_count > 0, "inputs must not be empty");

        self.recorder.set_input(self.inputs.initial_input, input.next().unwrap());

        if inputs_count > 1
        {
            let steps_loop = self.inputs.steps_loop.unwrap();

            self.recorder.set_loop_inputs(steps_loop, input.collect());

            self.recorder.set_loop_times(steps_loop, inputs_count - 1);
        }

        self.recorder.calculate_feedforward();

        let mut output = self.recorder.get_tensor(self.outputs.output.as_value()).clone_owned();

        Softmaxer::softmax_temperature(&mut output, temperature);

        f_output(output);
    }

    fn set_dropout_mask(
        target: LayerTypeMut,
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
            network_mode: self.network_mode,
            sizes: self.sizes,
            is_multistep: self.is_multistep,
            is_input_one_hot: self.is_input_one_hot,
            optimizer_info: None,
            weights_ptr: self.weights_ptr,
            weights: self.weights,
            dropouts: self.dropouts,
            inputs: self.inputs,
            outputs: self.outputs
        }
    }

    pub fn embeddings(&self, input: &OneHotLayer) -> LayerType
    {
        let weights = self.weights_ptr.as_ref().unwrap();
        debug_assert_eq!(weights.layers.len(), 1);

        weights.layers[0].embeddings_calculate(&self.recorder, input)
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[allow(unused_imports)]
    use crate::neural_network::{EmbeddingUnit, Lstm, Gru};


    const SEED: u64 = 123;

    const DROPOUT_PROBABILITY: f32 = 0.35;

    const IS_INPUT_ONE_HOT: bool = true;

    const SIZES: LayerSizes = LayerSizes{
        hidden: 2,
        input: 2,
        layers: 2,
        output: 2
    };

    #[allow(dead_code)]
    #[derive(Debug)]
    struct LstmUnitFactory;

    impl UnitFactory for LstmUnitFactory
    {
        type Unit<T> = Lstm<T>;
    }

    #[allow(dead_code)]
    #[derive(Debug)]
    struct GruUnitFactory;

    impl UnitFactory for GruUnitFactory
    {
        type Unit<T> = Gru<T>;
    }

    #[allow(dead_code)]
    #[derive(Debug)]
    struct EmbeddingUnitFactory;

    impl UnitFactory for EmbeddingUnitFactory
    {
        type Unit<T> = EmbeddingUnit<T>;
    }

    type ThisFactory = LstmUnitFactory;

    type NetworkType = Network<ThisFactory, ()>;

    fn inputs_outputs() -> (Vec<OwnedInputType>, Vec<OneHotLayer>)
    {
        let inputs = vec![
            OwnedInputType::OneHot(OneHotLayer::new([0], 2)),
            OwnedInputType::OneHot(OneHotLayer::new([1], 2)),
            OwnedInputType::OneHot(OneHotLayer::new([0], 2)),
            OwnedInputType::OneHot(OneHotLayer::new([1], 2)),
            OwnedInputType::OneHot(OneHotLayer::new([0], 2))
        ];

        let outputs = vec![
            OneHotLayer::new([1], 2),
            OneHotLayer::new([0], 2),
            OneHotLayer::new([0], 2),
            OneHotLayer::new([1], 2),
            OneHotLayer::new([1], 2)
        ];

        assert_eq!(inputs.len(), outputs.len());

        (inputs, outputs)
    }

    fn run_unrolled() -> (NetworkType, (f32, WeightsFullContainer<ThisFactory, LayerType>))
    {
        fastrand::seed(SEED);

        let (inputs, outputs) = inputs_outputs();

        let input_outputs = inputs.iter().cloned().zip(outputs);

        let mut at_once: NetworkType = Network::new(SIZES, DROPOUT_PROBABILITY, false, IS_INPUT_ONE_HOT);
        at_once.set_train_mode();

        let dropout_masks_ptrs = at_once.create_dropout_masks_ptrs();

        let input_outputs_ptrs: Vec<(InputTypePtr, OneHotIndex)> = (0..input_outputs.len()).map(|_|
        {
            assert!(IS_INPUT_ONE_HOT);

            (InputTypePtr::OneHot(at_once.recorder.new_one_hot()), at_once.recorder.new_one_hot())
        }).collect();

        let mut previous_state = None;
        let mut output = None;

        input_outputs_ptrs.iter().for_each(|(this_input, this_target)|
        {
            let NetworkOutput{
                state: next_state_ptr,
                output: (this_output, loss)
            } = at_once.record_feedforward_single_input(previous_state.take(), &dropout_masks_ptrs, *this_input, Some(*this_target), true);

            at_once.recorder.name_diff_tensor(this_output, "output");
            at_once.recorder.name_diff_scalar(loss.unwrap(), "loss");

            previous_state = Some(next_state_ptr);

            if let Some(output) = output.as_mut()
            {
                *output = at_once.recorder.add_scalars(*output, loss.unwrap());

                at_once.recorder.name_diff_scalar(*output, "loss_combined");
            } else
            {
                output = Some(loss.unwrap());
            }
        });

        let input_outputs_indices: Vec<_> = input_outputs_ptrs.into_iter().map(|(input_ptr, target_index)|
        {
            (if let InputTypePtr::OneHot(x) = input_ptr { x } else { unreachable!() }, target_index)
        }).collect();

        input_outputs_indices.into_iter().zip(input_outputs).for_each(|((input_index, target_index), (input_layer, target_layer))|
        {
            at_once.recorder.set_one_hot(input_index, input_layer.into_one_hot());
            at_once.recorder.set_one_hot(target_index, target_layer);
        });

        at_once.recorder.finish();

        at_once.recorder.store_value_until_end(output.unwrap().as_value());

        at_once.weights_ptr.as_ref().unwrap().iter().for_each(|x|
        {
            at_once.recorder.store_tensor_until_end(x.weight_original.as_value());
            at_once.recorder.store_tensor_until_end(x.weight_original.as_gradient().unwrap());

            if let Some(dropconnect_mask) = x.dropconnect_mask
            {
                at_once.recorder.store_tensor_until_end(dropconnect_mask);
            }
        });

        at_once.recorder.gradient(output.unwrap().into());

        at_once.recorder.resolve_memory();

        at_once.weights = Some(at_once.weights_ptr.take().unwrap().map(|x|
        {
            WeightInfo{
                weight: at_once.recorder.resolve_diff_tensor_ptr(x.weight_original),
                dropconnect_mask: x.dropconnect_mask.map(|x| at_once.recorder.resolve_tensor_ptr(x))
            }
        }));

        at_once.resolve_dropout_masks();

        at_once.feedforward_setup_dropout();

        at_once.recorder.calculate();

        let loss = at_once.recorder.get_value(output.unwrap().as_value());

        let gradients = {
            let weights = at_once.weights.clone().unwrap();

            let f = |x: WeightInfo|
            {
                let gradient = x.weight.as_gradient().unwrap();

                if !at_once.recorder.is_undefined_location(gradient)
                {
                    Some(at_once.recorder.get_tensor(gradient).clone_owned())
                } else
                {
                    None
                }
            };

            let g = |x: WeightsSize<_>|
            {
                f(x.weights).unwrap_or_else(|| LayerType::new(x.this_size, x.previous_size))
            };

            WeightsFullContainer{
                output: f(weights.output).unwrap(),
                layers: weights.layers.into_iter().map(|x| x.map_with_info(g)).collect()
            }
        };

        (at_once, (loss, gradients))
    }

    #[test]
    fn nonzero_gradients()
    {
        let (_network, (loss, gradients)) = run_unrolled();

        assert!(loss > 0.0);

        assert!(gradients.iter().all(|x| !x.as_vec().into_iter().all(|x| x == 0.0)), "gradients have unused fields: {gradients:?}");
    }

    #[test]
    fn steps_equivalent()
    {
        fastrand::seed(SEED);

        let (inputs, outputs) = inputs_outputs();

        let is_multistep = inputs.len() > 1;

        let input_outputs = inputs.iter().cloned().zip(outputs);

        let mut with_steps: NetworkType = Network::new(SIZES, DROPOUT_PROBABILITY, is_multistep, IS_INPUT_ONE_HOT);
        with_steps.set_train_mode();

        with_steps.prepare(true);
        let with_steps_gradient = with_steps.gradients(input_outputs.clone());

        let (mut at_once, at_once_gradient) = run_unrolled();

        let map_weights = |r: &mut OperationsRecorder, w: WeightInfo| -> LayerType
        {
            r.get_tensor(w.weight.as_value()).clone_owned()
        };

        assert_eq!(
            with_steps.weights.unwrap().map(|x| map_weights(&mut with_steps.recorder, x)),
            at_once.weights.unwrap().map(|x| map_weights(&mut at_once.recorder, x))
        );

        eprintln!("{with_steps_gradient:?}");

        assert_eq!(with_steps_gradient, at_once_gradient);
    }
}
