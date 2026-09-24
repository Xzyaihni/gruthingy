use std::{
    f32,
    vec,
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
        OperationsRecorderMemory,
        TensorShape,
        Softmaxer,
        PhiOtherSelectorRecordingIndex,
        NetworkStateSelectable,
        NetworkStateGettable,
        NetworkUnitNewable,
        DiffTensor,
        DiffTensorPtr,
        LoopIndex,
        LoopInputs,
        ShapedTensorIndex,
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
        USE_EMBEDDING_LAYER,
        network_unit::{EmbeddingsableOwned, NetworkUnitParameterable}
    }
};


pub trait DropoutRoll: Debug
{
    fn roll(&mut self) -> f32;
}

impl DropoutRoll for ()
{
    fn roll(&mut self) -> f32 { 0.0 }
}

impl DropoutRoll for fastrand::Rng
{
    fn roll(&mut self) -> f32 { self.f32() }
}

#[derive(Debug, Clone)]
pub struct PrecomputedRng
{
    pub index: usize,
    pub values: Vec<f32>
}

impl DropoutRoll for PrecomputedRng
{
    fn roll(&mut self) -> f32
    {
        let value = self.values[self.index];

        self.index += 1;

        value
    }
}

pub fn initialize_orthogonal(m: usize, k: usize) -> LayerType
{
    let x = if m <= k
    {
        LayerType::new_with(m, k, fastrand::f32)
    } else
    {
        LayerType::new_with(k, m, fastrand::f32)
    };

    let output = x.as_ref().gemm_tr(x.as_ref()).map(|x| x.sqrt().recip()).as_ref().gemm(x.as_ref());

    if m <= k
    {
        output
    } else
    {
        output.as_ref().transpose().mul_scalar((m as f32 / k as f32).sqrt())
    }
}

pub fn maybe_dropout_weights(
    recorder: &mut OperationsRecorder,
    is_skip: bool,
    weights: DiffTensorPtr,
    shape: TensorShape
) -> WeightInfoPtr
{
    if is_skip
    {
        WeightInfoPtr{
            weight_dropped: DiffTensorPtr::undefined(),
            weight_original: weights,
            dropout: None
        }
    } else
    {
        dropout_weights(recorder, weights, shape)
    }
}

pub fn dropout_weights(
    recorder: &mut OperationsRecorder,
    weight_original: DiffTensorPtr,
    shape: TensorShape
) -> WeightInfoPtr
{
    let dropout = recorder.new_tensor_batched_no_gradient(shape);

    WeightInfoPtr{
        weight_dropped: DiffTensorPtr::undefined(),
        weight_original,
        dropout: Some(dropout.as_value())
    }
}

#[derive(Debug, PartialEq)]
pub struct WeightsSize<T>
{
    pub weights: T,
    pub previous_size: usize,
    pub this_size: usize,
    pub is_hidden: bool,
    pub is_bias: bool,
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
            is_bias: self.is_bias,
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
    pub initial_input: usize,
    pub input: usize,
    pub output: usize,
    pub final_output: usize,
    pub hidden: usize,
    pub layers: usize,
    pub batch_size: usize
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

        #[allow(unused_imports)]
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
                            is_bias: matches!($previous_size, LayerSize::One),
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
                use $crate::neural_network::{TensorShape, network::{initialize_orthogonal, maybe_dropout_weights, LayerSize}};

                Self{sizes, $(
                    $name: {
                        let this_size = $this_size.into_number(sizes);
                        let previous_size = $previous_size.into_number(sizes);

                        let weights = match $previous_size
                        {
                            LayerSize::One =>
                            {
                                let bias = recorder.new_tensor(this_size, previous_size, sizes.batch_size);

                                recorder.set_tensor_ptr_zeroed(bias.as_value());

                                bias
                            },
                            _x =>
                            {
                                let weights = initialize_orthogonal(this_size, previous_size);

                                recorder.set_new_tensor_gradientable(weights, sizes.batch_size)
                            }
                        };

                        recorder.name_diff_tensor(weights, stringify!($name));

                        let is_bias = matches!($previous_size, LayerSize::One);
                        let weights = maybe_dropout_weights(recorder, is_bias, weights, TensorShape{
                            rows: this_size,
                            columns: previous_size,
                            batch_size: sizes.batch_size
                        });

                        if let Some(dropout) = weights.dropout
                        {
                            recorder.name_diff_tensor(DiffTensorPtr::no_gradient(dropout), stringify!($name).to_owned() + "_dropout");
                        }

                        weights
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
                        is_bias: matches!($previous_size, LayerSize::One),
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
                            is_bias: matches!($previous_size, LayerSize::One),
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
                            is_bias: matches!($previous_size, LayerSize::One),
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
                                is_bias: matches!($previous_size, LayerSize::One),
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
                                is_bias: matches!($previous_size, LayerSize::One),
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
    pub fn map<F, NewOutput>(self, f: F) -> NetworkOutput<State, NewOutput>
    where
        F: FnOnce(Output) -> NewOutput
    {
        NetworkOutput{state: self.state, output: f(self.output)}
    }
}

pub type UnitState<N, T> = <<N as UnitFactory>::Unit<WeightInfoPtr> as NetworkUnit>::State<T>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingsLayers<T>
{
    input: T,
    output: T
}

impl<T> EmbeddingsLayers<T>
{
    pub fn map<F: FnMut(T) -> U, U>(self, mut f: F) -> EmbeddingsLayers<U>
    {
        EmbeddingsLayers{
            input: f(self.input),
            output: f(self.output)
        }
    }

    pub fn map_ref<F: FnMut(&T) -> U, U>(&self, mut f: F) -> EmbeddingsLayers<U>
    {
        EmbeddingsLayers{
            input: f(&self.input),
            output: f(&self.output)
        }
    }

    pub fn iter(&self) -> impl Iterator<Item=&T>
    {
        iter::once(&self.input).chain(iter::once(&self.output))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut T>
    {
        iter::once(&mut self.input).chain(iter::once(&mut self.output))
    }

    pub fn into_iter(self) -> impl Iterator<Item=T>
    {
        iter::once(self.input).chain(iter::once(self.output))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize, N::Unit<T>: Serialize", deserialize = "T: Deserialize<'de>, N::Unit<T>: Deserialize<'de>"))]
pub struct WeightsFullContainer<N: UnitFactory, T>
{
    layers: Vec<N::Unit<T>>,
    embeddings: Option<EmbeddingsLayers<T>>,
    output: T
}

impl<N: UnitFactory, T> PartialEq for WeightsFullContainer<N, T>
where
    T: PartialEq,
    N::Unit<T>: PartialEq
{
    fn eq(&self, other: &Self) -> bool
    {
        self.layers == other.layers && self.embeddings == other.embeddings && self.output == other.output
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
            embeddings: self.embeddings.clone(),
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
            .field("embeddings", &self.embeddings)
            .field("output", &self.output)
            .finish()
    }
}

impl<N: UnitFactory, T> WeightsFullContainer<N, T>
{
    pub fn new(
        sizes: LayerSizes,
        unit_f: impl FnMut(LayerSizes) -> N::Unit<T>,
        embeddings: Option<EmbeddingsLayers<T>>,
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
            embeddings,
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
            embeddings: self.embeddings.map(|x| x.map(&mut f)),
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
            embeddings: self.embeddings.as_ref().map(|x| x.map_ref(&mut f)),
            layers: self.layers.iter().map(|layer| layer.map_ref(&mut f)).collect()
        }
    }

    pub fn iter(&self) -> impl Iterator<Item=&T>
    where
        for<'a> &'a N::Unit<T>: IntoIterator<Item=&'a T>
    {
        self.layers.iter().flatten().chain(self.embeddings.iter().map(|x| x.iter()).flatten()).chain(iter::once(&self.output))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut T>
    where
        for<'a> &'a mut N::Unit<T>: IntoIterator<Item=&'a mut T>
    {
        self.layers.iter_mut().flatten().chain(self.embeddings.iter_mut().map(|x| x.iter_mut()).flatten()).chain(iter::once(&mut self.output))
    }

    pub fn into_iter(self) -> impl Iterator<Item=T>
    where
        N::Unit<T>: IntoIterator<Item=T>
    {
        self.layers.into_iter().flatten().chain(self.embeddings.into_iter().map(|x| x.into_iter()).flatten()).chain(iter::once(self.output))
    }
}

impl<N: UnitFactory> WeightsFullContainer<N, LayerType>
{
    pub fn average_batch(mut self) -> Self
    where
        for<'a> &'a mut N::Unit<LayerType>: IntoIterator<Item=&'a mut LayerType>
    {
        self.iter_mut().for_each(|gradient|
        {
            let batch_size = gradient.shape().batch_size;

            let mut new_gradient = gradient.as_ref().batch_slice_ref(0).clone_owned();

            for batch_index in 1..batch_size
            {
                new_gradient.add_inplace(gradient.as_ref().batch_slice_ref(batch_index));
            }

            new_gradient.mul_scalar_inplace((batch_size as f32).recip());

            *gradient = new_gradient;
        });

        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WeightInfoGeneric<D, I>
{
    pub weight_dropped: D,
    pub weight_original: D,
    pub dropout: Option<I>
}

pub type WeightInfoPtr = WeightInfoGeneric<DiffTensorPtr, TensorPtr>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightInfo
{
    pub weight: DiffTensor,
    pub dropout: Option<ShapedTensorIndex>
}

pub type SaveWeightType = LayerType;

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "O: Serialize, N::Unit<O>: Serialize, N::Unit<SaveWeightType>: Serialize", deserialize = "O: Deserialize<'de>, N::Unit<O>: Deserialize<'de>, N::Unit<SaveWeightType>: Deserialize<'de>"))]
pub struct SaveNetwork<N: UnitFactory, O>
{
    sizes: LayerSizes,
    input_dropout_probability: f32,
    dropout_probability: f32,
    optimizer_info: Option<WeightsFullContainer<N, O>>,
    weights: WeightsFullContainer<N, SaveWeightType>
}

impl<N: UnitFactory, O> Clone for SaveNetwork<N, O>
where
    O: Clone,
    N::Unit<O>: Clone,
    N::Unit<SaveWeightType>: Clone
{
    fn clone(&self) -> Self
    {
        Self{
            sizes: self.sizes.clone(),
            input_dropout_probability: self.input_dropout_probability,
            dropout_probability: self.dropout_probability,
            optimizer_info: self.optimizer_info.clone(),
            weights: self.weights.clone()
        }
    }
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
            input_dropout_probability: x.dropouts.input_dropout_probability,
            dropout_probability: x.dropouts.dropout_probability,
            optimizer_info: x.optimizer_info,
            weights
        }
    }
}

impl<N: UnitFactory, O> SaveNetwork<N, O>
{
    pub fn sizes(&self) -> &LayerSizes
    {
        &self.sizes
    }
}

#[derive(Clone)]
struct NetworkDropoutData
{
    input_dropout_probability: f32,
    dropout_probability: f32,
    input_dropouts: Vec<Vec<usize>>
}

impl NetworkDropoutData
{
    fn new(input_dropout_probability: f32, dropout_probability: f32) -> Self
    {
        Self{
            input_dropouts: Vec::new(),
            input_dropout_probability,
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
    first_output_value_ptr: Option<TensorPtr>,
    first_output_value: ShapedTensorIndex,
    loss_ptr: Option<DiffTensorPtr>,
    loss: DiffTensor
}

impl Default for NetworkOutputsData
{
    fn default() -> Self
    {
        Self{
            output_ptr: None,
            output: DiffTensor::undefined(),
            first_output_value_ptr: None,
            first_output_value: ShapedTensorIndex::undefined(),
            loss_ptr: None,
            loss: DiffTensor::undefined()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum NetworkMode
{
    Predict,
    Train
}

#[derive(Debug, Clone, PartialEq)]
pub struct NetworkConfigInfo
{
    pub print_optional_info: bool,
    pub is_multistep: bool,
    pub is_input_one_hot: bool
}

#[derive(Serialize)]
#[serde(into = "SaveNetwork<N, O>")]
#[serde(bound(serialize = "O: Serialize + Clone, N::Unit<O>: Serialize + Clone, N::Unit<SaveWeightType>: Serialize, N::Unit<WeightInfo>: Clone + GenericUnit<WeightInfo, Unit<SaveWeightType>=N::Unit<SaveWeightType>>, N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<SaveWeightType>=N::Unit<SaveWeightType>>", deserialize = "O: Deserialize<'de>, N::Unit<O>: Deserialize<'de>, N::Unit<SaveWeightType>: Deserialize<'de> + GenericUnit<SaveWeightType, Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>, N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>, for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>"))]
pub struct Network<N: UnitFactory, O>
where
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
{
    recorder: OperationsRecorder,
    network_mode: Option<NetworkMode>,
    sizes: LayerSizes,
    config: Option<NetworkConfigInfo>,
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
            config: self.config.clone(),
            dropouts: self.dropouts.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            optimizer_info: self.optimizer_info.clone(),
            weights_ptr: self.weights_ptr.clone(),
            weights: self.weights.clone()
        }
    }
}

impl<N: UnitFactory, O> Network<N, O>
where
    N::Unit<O>: OptimizerUnit<O>,
    N::Unit<WeightInfo>: GenericUnit<WeightInfo>,
    N::Unit<WeightInfoPtr>: NetworkUnitNewable,
    UnitState<N, DiffTensorPtr>: Clone + NetworkStateSelectable<UnitState<N, PhiOtherSelectorRecordingIndex>>,
    UnitState<N, PhiOtherSelectorRecordingIndex>: NetworkStateGettable<UnitState<N, DiffTensorPtr>>,
    for<'a> &'a N::Unit<WeightInfo>: IntoIterator<Item=&'a WeightInfo>,
    for<'a> &'a N::Unit<DiffTensor>: IntoIterator<Item=&'a DiffTensor>,
    for<'a> &'a mut N::Unit<DiffTensor>: IntoIterator<Item=&'a mut DiffTensor>,
    N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
    N::Unit<WeightInfoPtr>: NetworkUnit<Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    N::Unit<SaveWeightType>: GenericUnit<SaveWeightType, Unit<WeightInfoPtr>=N::Unit<WeightInfoPtr>>,
    for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
{
    pub fn load(
        mut x: SaveNetwork<N, O>,
        config: NetworkConfigInfo,
        batch_size: usize
    ) -> Self
    {
        if !USE_EMBEDDING_LAYER
        {
            assert_eq!(x.sizes.initial_input, x.sizes.input);
            assert_eq!(x.sizes.output, x.sizes.final_output);
        }

        x.sizes.batch_size = batch_size;

        let mut recorder = OperationsRecorder::new();

        // no optimizer info means im not going to train this network
        let discard_gradients = x.optimizer_info.is_none();

        let weight_info_from = |recorder: &mut OperationsRecorder, is_skip: bool, value: SaveWeightType| -> WeightInfoPtr
        {
            let shape = value.shape();

            let weights = if discard_gradients
            {
                recorder.set_new_tensor(value)
            } else
            {
                recorder.set_new_tensor_gradientable(value, batch_size)
            };

            maybe_dropout_weights(recorder, is_skip, weights, TensorShape{
                batch_size,
                ..shape
            })
        };

        let weights_ptr = WeightsFullContainer{
            output: weight_info_from(&mut recorder, false, x.weights.output),
            embeddings: x.weights.embeddings.map(|embeddings|
            {
                EmbeddingsLayers{
                    input: weight_info_from(&mut recorder, true, embeddings.input),
                    output: weight_info_from(&mut recorder, false, embeddings.output)
                }
            }),
            layers: x.weights.layers.into_iter().map(|x| x.map_with_info(|WeightsSize{weights: value, is_bias, ..}|
            {
                weight_info_from(&mut recorder, is_bias, value)
            })).collect()
        };

        let mut this = Self{
            sizes: x.sizes,
            network_mode: None,
            config: None,
            optimizer_info: x.optimizer_info,
            weights_ptr: Some(weights_ptr),
            weights: None,
            dropouts: NetworkDropoutData::new(x.input_dropout_probability, x.dropout_probability),
            inputs: NetworkInputsData::default(),
            outputs: NetworkOutputsData::default(),
            recorder
        };

        if discard_gradients
        {
            this.initialize_no_gradient();
        }

        this.initialize_with_params(config);

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

        self.recorder.resolve_memory(false);
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
    for<'a> &'a N::Unit<WeightInfo>: IntoIterator<Item=&'a WeightInfo>,
    for<'a> &'a N::Unit<DiffTensor>: IntoIterator<Item=&'a DiffTensor>,
    for<'a> &'a mut N::Unit<DiffTensor>: IntoIterator<Item=&'a mut DiffTensor>
{
    pub fn new(
        sizes: LayerSizes,
        input_dropout_probability: f32,
        dropout_probability: f32,
        config: NetworkConfigInfo
    ) -> Self
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        O: NewableLayer
    {
        let mut this = Self::new_no_init(sizes, input_dropout_probability, dropout_probability, config);

        this.initialize_with_sizes(sizes);

        this
    }

    pub fn initialize_with_sizes(&mut self, sizes: LayerSizes)
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        let mut create_weights = |name: &'static str, is_skip: bool, previous: usize, current: usize| -> WeightInfoPtr
        {
            let weights = self.recorder.set_new_tensor_gradientable(initialize_orthogonal(current, previous), sizes.batch_size);

            self.recorder.name_diff_tensor(weights, name);

            maybe_dropout_weights(&mut self.recorder, is_skip, weights, TensorShape{
                rows: current,
                columns: previous,
                batch_size: sizes.batch_size
            })
        };

        let output_weights_ptr = create_weights("output_weights", false, sizes.hidden, sizes.output);
        let embeddings_weights_ptr = USE_EMBEDDING_LAYER.then(||
        {
            EmbeddingsLayers{
                input: create_weights("embeddings_input", true, sizes.initial_input, sizes.input),
                output: create_weights("embeddings_output", false, sizes.output, sizes.final_output)
            }
        });

        let weights_ptr = WeightsFullContainer::new(sizes, |size|
        {
            N::Unit::new(&mut self.recorder, size)
        }, embeddings_weights_ptr, output_weights_ptr);

        self.weights_ptr = Some(weights_ptr);

        self.initialize();
    }

    pub fn new_no_init(
        sizes: LayerSizes,
        input_dropout_probability: f32,
        dropout_probability: f32,
        config: NetworkConfigInfo
    ) -> Self
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>,
        O: NewableLayer
    {
        let recorder = OperationsRecorder::new();

        let optimizer_info: Option<_> = Some({
            let embeddings = USE_EMBEDDING_LAYER.then(|| EmbeddingsLayers{
                input: O::new(sizes.initial_input, sizes.input),
                output: O::new(sizes.output, sizes.final_output)
            });

            let output = O::new(sizes.hidden, sizes.output);

            WeightsFullContainer::new(sizes, N::Unit::new_zeroed, embeddings, output)
        });

        Self{
            recorder,
            network_mode: None,
            sizes,
            dropouts: NetworkDropoutData::new(input_dropout_probability, dropout_probability),
            inputs: NetworkInputsData::default(),
            outputs: NetworkOutputsData::default(),
            optimizer_info,
            weights_ptr: None,
            weights: None,
            config: Some(config)
        }
    }

    pub fn set_train_mode(&mut self)
    {
        assert!(self.network_mode.is_none());

        self.network_mode = Some(NetworkMode::Train);
    }

    pub fn set_predict_mode(&mut self)
    {
        assert!(self.network_mode.is_none());

        self.network_mode = Some(NetworkMode::Predict);
    }

    pub fn initialize_with_params(&mut self, config: NetworkConfigInfo)
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        self.config = Some(config);

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
        let initialize_dropped_weight = |recorder: &mut OperationsRecorder, weights: &mut WeightInfoPtr|
        {
            if let Some(dropout) = weights.dropout
            {
                let weight_dropped = recorder.mul_componentwise(
                    weights.weight_original,
                    DiffTensorPtr::no_gradient(dropout)
                );

                recorder.store_tensor_until_end(dropout);

                weights.weight_dropped = weight_dropped;
            }
        };

        if let Some(weights_ptr) = self.weights_ptr.as_mut()
        {
            weights_ptr.embeddings.as_mut().map(|embeddings|
            {
                embeddings.iter_mut().for_each(|x|
                {
                    initialize_dropped_weight(&mut self.recorder, x);
                });
            });

            initialize_dropped_weight(&mut self.recorder, &mut weights_ptr.output);

            weights_ptr.layers.iter_mut().for_each(|layer|
            {
                layer.map_inplace_with_info(|mut weights_size, _debug_info|
                {
                    initialize_dropped_weight(&mut self.recorder, &mut weights_size.weights);

                    #[cfg(debug_assertions)]
                    {
                        if weights_size.weights.dropout.is_some()
                        {
                            self.recorder.name_diff_tensor(weights_size.weights.weight_dropped, _debug_info.name.to_owned() + "_dropped");
                        }
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
            self.record_feedforward();

            self.prepare_setup_shared();

            self.weights_ptr.as_ref().unwrap().iter().for_each(|weight|
            {
                self.recorder.store_tensor_until_end(weight.weight_original.as_value());
            });

            if store_gradient
            {
                self.weights_ptr.as_ref().unwrap().iter().for_each(|weight|
                {
                    self.recorder.store_tensor_until_end(weight.weight_original.as_gradient().unwrap());
                });

                self.recorder.gradient(self.outputs.loss_ptr.expect("must be initialized").into());
            } else
            {
                self.recorder.store_tensor_until_end(self.outputs.first_output_value_ptr.unwrap());
                self.recorder.store_tensor_until_end(self.outputs.output_ptr.unwrap().as_value());

                self.recorder.no_gradient();
            }

            self.prepare_shared(store_gradient);

            if let Some(loss_ptr) = self.outputs.loss_ptr
            {
                self.outputs.loss = self.recorder.resolve_diff_tensor_ptr(DiffTensorPtr::no_gradient(loss_ptr.as_value()));
            }

            if !store_gradient
            {
                let output_ptr = DiffTensorPtr::no_gradient(self.outputs.output_ptr.unwrap().as_value());
                self.outputs.output = self.recorder.resolve_diff_tensor_ptr(output_ptr);

                self.outputs.first_output_value = self.recorder.resolve_tensor_ptr(self.outputs.first_output_value_ptr.unwrap());
            }
        }
    }

    fn prepare_setup_shared(&mut self)
    {
        self.recorder.finish();

        {
            let weights_ptr = self.weights_ptr.as_ref().unwrap();

            let mut store_tensor = |weight: &WeightInfoGeneric<DiffTensorPtr, TensorPtr>|
            {
                self.recorder.store_tensor_until_end(weight.weight_original.as_value());
            };

            store_tensor(&weights_ptr.output);
            weights_ptr.embeddings.as_ref().map(|embeddings| embeddings.map_ref(|x| store_tensor(x)));
        }

        if self.network_mode == Some(NetworkMode::Train)
        {
            self.recorder.store_tensor_until_end(self.outputs.loss_ptr.expect("must be initialized").as_value());
        }
    }

    fn prepare_shared(&mut self, store_gradient: bool)
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>
    {
        self.recorder.resolve_memory(self.config.as_ref().unwrap().print_optional_info);

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
                dropout: weight_info.dropout.map(|x| self.recorder.resolve_tensor_ptr(x))
            }
        });

        self.weights = Some(weights);
    }

    fn record_feedforward(&mut self)
    {
        assert!(self.network_mode.is_some());

        let config = self.config.clone().unwrap();

        let batch_size = self.sizes.batch_size;

        let create_input = {
            let is_input_one_hot = config.is_input_one_hot;
            let input_size = self.sizes.input;

            move |recorder: &mut OperationsRecorder| -> InputTypePtr
            {
                if is_input_one_hot
                {
                    InputTypePtr::OneHot(recorder.new_one_hot_batched(batch_size))
                } else
                {
                    InputTypePtr::Normal(recorder.new_tensor_batched_no_gradient(TensorShape{rows: input_size, columns: 1, batch_size}).as_value())
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
            let this_target_first = self.recorder.new_one_hot_batched(batch_size);

            self.recorder.name_one_hot(this_target_first, "target_first");

            self.inputs.initial_target = this_target_first.into();

            Some(this_target_first)
        } else
        {
            None
        };

        let no_state_output = self.record_feedforward_single_input(
            None,
            this_input_first,
            this_target_first
        );

        self.recorder.name_diff_tensor(no_state_output.output.0, "no_state_output");

        let no_state_loss = no_state_output.output.1;

        if let Some(no_state_loss) = no_state_loss
        {
            self.recorder.name_diff_tensor(no_state_loss, "no_state_loss");
        }

        self.outputs.first_output_value_ptr = Some(no_state_output.output.0.as_value());

        let (final_output, final_loss) = if config.is_multistep
        {
            let this_input_loop = create_input(&mut self.recorder);

            let this_target_loop = has_target.then(|| self.recorder.new_one_hot_batched(batch_size));

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
                this_input_loop,
                this_target_loop
            );

            self.recorder.name_diff_tensor(final_output.output.0, "final_output");

            let final_output_loss = final_output.output.1;

            if let Some(final_output_loss) = final_output_loss
            {
                self.recorder.name_diff_tensor(final_output_loss, "final_output_loss");
            }

            let new_combined = final_output_loss.map(|final_output_loss|
            {
                let final_loss_selector = final_loss_selector.expect("must be set");

                let final_loss_selected = self.recorder.select_tensor(final_loss_selector);

                let new_combined = self.recorder.add(final_loss_selected, final_output_loss);

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
        self.outputs.loss_ptr = final_loss;
    }

    fn record_feedforward_single_input(
        &mut self,
        previous_states: Option<Vec<UnitState<N, DiffTensorPtr>>>,
        input: InputTypePtr,
        targets: Option<OneHotIndex>
    ) -> NetworkOutput<Vec<UnitState<N, DiffTensorPtr>>, (DiffTensorPtr, Option<DiffTensorPtr>)>
    {
        self.record_feedforward_single_input_with_activation(|this, layer_index, previous_state, input|
        {
            this.record_feedforward_unit_last(
                layer_index,
                previous_state,
                input
            ).map(|output|
            {
                let output = this.weights_ptr.as_ref().unwrap().embeddings.as_ref().map(|embeddings|
                {
                    this.recorder.matmulv(embeddings.output.weight_dropped, output)
                }).unwrap_or(output);

                (output, targets.map(|targets| this.recorder.softmax_cross_entropy(output, targets).1))
            })
        }, previous_states, input)
    }

    fn record_feedforward_single_input_with_activation<F, T>(
        &mut self,
        last_f: F,
        previous_states: Option<Vec<UnitState<N, DiffTensorPtr>>>,
        input: InputTypePtr
    ) -> NetworkOutput<Vec<UnitState<N, DiffTensorPtr>>, T>
    where
        F: FnOnce(&mut Self, usize, Option<&UnitState<N, DiffTensorPtr>>, DiffInputType) -> NetworkOutput<UnitState<N, DiffTensorPtr>, T>
    {
        let mut output: Option<T> = None;
        let mut last_output: Option<DiffInputType> = None;

        let weights_ptr = self.weights_ptr.as_ref().unwrap();

        let mut states = Vec::with_capacity(self.sizes.layers);

        #[allow(clippy::needless_range_loop)]
        for l_i in 0..self.sizes.layers
        {
            let input = last_output.unwrap_or_else(||
            {
                if let Some(embeddings) = weights_ptr.embeddings.as_ref()
                {
                    debug_assert!(embeddings.input.weight_dropped.is_undefined());
                    debug_assert!(embeddings.input.dropout.is_none());

                    return DiffInputType::Normal(self.recorder.matmul_onehotv(embeddings.input.weight_original, input.into_one_hot()));
                }

                match input
                {
                    InputTypePtr::Normal(x) => DiffInputType::Normal(DiffTensorPtr::no_gradient(x)),
                    InputTypePtr::OneHot(x) => DiffInputType::OneHot(x)
                }
            });

            let layer = &weights_ptr.layers[l_i];

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
        let weights = self.weights_ptr.as_ref().unwrap();

        weights.layers[layer_index]
            .record_feedforward_unit(&mut self.recorder, previous_state, input)
            .map(|output|
            {
                self.recorder.matmulv(weights.output.weight_dropped, output)
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
                    gradient.cap_magnitude_inplace(gradient_clip);
                }

                let change = optimizer.gradient_to_change(optimizer_info, gradient);

                self.recorder.get_tensor_mut::<true>(network_weights.weight.as_value()).sub_inplace(LayerTypeRef::from(&change));
            });

        optimizer.advance_time();
    }

    pub fn gradients<I: ExactSizeIterator<Item=(OwnedInputType, OneHotLayer)>, R: DropoutRoll>(
        &mut self,
        mut rng: R,
        input: I
    ) -> (f32, WeightsFullContainer<N, LayerType>)
    where
        N::Unit<WeightInfo>: GenericUnit<WeightInfo, Unit<LayerType>=N::Unit<LayerType>>,
        N::Unit<LayerType>: IntoIterator<Item=LayerType>,
        for<'b> &'b mut N::Unit<LayerType>: IntoIterator<Item=&'b mut LayerType>
    {
        let inputs_count = input.len();

        self.feedforward_setup_dropout(&mut rng);

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
                embeddings: weights.embeddings.as_ref().map(|embeddings| embeddings.map_ref(f)),
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
        assert_eq!(self.network_mode, Some(NetworkMode::Train));

        let inputs_count = input.len();
        let mut inputs = input.flat_map(|(input, target)|
        {
            let dropped_input = match input
            {
                OwnedInputType::Normal(x) => OwnedInputType::Normal(x),
                OwnedInputType::OneHot(mut x) =>
                {
                    x.positions.iter_mut().enumerate().for_each(|(batch_index, batch_positions)|
                    {
                        let dropouts = &self.dropouts.input_dropouts[batch_index];

                        *batch_positions = batch_positions.iter().copied().filter(|x| !dropouts.contains(x)).collect();
                    });

                    OwnedInputType::OneHot(x)
                }
            };

            [dropped_input, OwnedInputType::OneHot(target)]
        });

        assert!(inputs_count > 0, "inputs must not be empty");

        self.recorder.set_input(self.inputs.initial_input, inputs.next().unwrap());
        self.recorder.set_input(self.inputs.initial_target, inputs.next().unwrap());

        if inputs_count > 1
        {
            let steps_loop = self.inputs.steps_loop.unwrap();

            self.recorder.set_loop_inputs(steps_loop, inputs.collect::<Vec<_>>());

            self.recorder.set_loop_times(steps_loop, inputs_count - 1);
        }

        calculate_function(&mut self.recorder);

        self.recorder.get_tensor(self.outputs.loss.as_value()).average()
    }

    pub fn weights_info<'b, 'c>(
        &'b self
    ) -> Vec<WeightsNamed<LayerTypeRef<'b>>>
    where
        for<'a> N::Unit<WeightInfo>: GenericUnit<WeightInfo, Unit<WeightsNamed<&'a WeightInfo>>=N::Unit<WeightsNamed<&'a WeightInfo>>>,
        N::Unit<WeightsNamed<&'b WeightInfo>>: IntoIterator<Item=WeightsNamed<&'b WeightInfo>>
    {
        let weights = self.weights.as_ref().unwrap();

        weights.layers.iter().enumerate()
            .flat_map(|(layer_index, layer)|
            {
                layer.weights_named_info(layer_index).into_iter()
            })
            .chain(iter::once(WeightsNamed{
                name: "output".to_owned(),
                layer: self.sizes.layers.saturating_sub(1),
                weights_size: WeightsSize{
                    weights: &weights.output,
                    this_size: self.sizes.output,
                    previous_size: self.sizes.hidden,
                    is_hidden: false,
                    is_bias: false,
                    is_state_reliant: false
                }
            }))
            .chain(weights.embeddings.as_ref().map(|embeddings|
            {
                [WeightsNamed{
                    name: "embeddings_input".to_owned(),
                    layer: 0,
                    weights_size: WeightsSize{
                        weights: &embeddings.input,
                        this_size: self.sizes.input,
                        previous_size: self.sizes.initial_input,
                        is_hidden: false,
                        is_bias: false,
                        is_state_reliant: false
                    }
                },
                WeightsNamed{
                    name: "embeddings_output".to_owned(),
                    layer: self.sizes.layers.saturating_sub(1),
                    weights_size: WeightsSize{
                        weights: &embeddings.output,
                        this_size: self.sizes.final_output,
                        previous_size: self.sizes.output,
                        is_hidden: false,
                        is_bias: false,
                        is_state_reliant: false
                    }
                }]
            }).into_iter().flatten())
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
        debug_assert_eq!(self.network_mode, Some(NetworkMode::Predict));

        let (input, output): (Vec<_>, Vec<_>) = input.unzip();

        self.predict(input.into_iter()).into_iter().zip(output).map(move |(predicted, target)|
        {
            assert_eq!(target.batch_size(), 1);

            let positions = &target.positions[0];
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

    pub fn feedforward_setup_dropout<R: DropoutRoll>(&mut self, rng: &mut R)
    {
        self.setup_dropout_common(rng, self.dropouts.input_dropout_probability, self.dropouts.dropout_probability)
    }

    fn setup_dropout_common<R: DropoutRoll>(&mut self, rng: &mut R, input_dropout_probability: f32, dropout_probability: f32)
    {
        let weights = self.weights.as_mut().unwrap();

        weights.iter().for_each(|weight_info|
        {
            if let Some(dropout) = weight_info.dropout
            {
                let dropout = self.recorder.get_tensor_mut::<false>(dropout);

                debug_assert_eq!(dropout.shape().batch_size, self.sizes.batch_size);

                Self::set_dropout_mask(rng, dropout, dropout_probability);
            }
        });

        self.dropouts.input_dropouts = (0..self.sizes.batch_size).map(|_|
        {
            (0..self.sizes.initial_input).filter(|_| rng.roll() < input_dropout_probability).collect::<Vec<usize>>()
        }).collect();
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

        self.feedforward_setup_dropout(&mut fastrand::Rng::new());

        let batch_size = self.sizes.batch_size;
        let input = input.map(move |(a, b)| (a.batch_replicate(batch_size), b.batch_replicate(batch_size)));

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
        input: impl Iterator<Item=OwnedInputType> + ExactSizeIterator,
        mut f_output: impl FnMut(LayerType)
    )
    where
        N::Unit<WeightInfoPtr>: GenericUnit<WeightInfoPtr, Unit<WeightInfo>=N::Unit<WeightInfo>>,
        for<'b> &'b N::Unit<WeightInfoPtr>: IntoIterator<Item=&'b WeightInfoPtr>
    {
        assert!(self.network_mode == Some(NetworkMode::Predict));

        self.prepare(false);

        self.feedforward_setup_dropout(&mut fastrand::Rng::new());

        let inputs_count = input.len();

        let mut input = input.map(|x| x.batch_replicate(self.sizes.batch_size));

        assert!(inputs_count > 0, "inputs must not be empty");

        self.recorder.set_input(self.inputs.initial_input, input.next().unwrap());

        let output_value = self.outputs.output.as_value();

        if inputs_count > 1
        {
            let steps_loop = self.inputs.steps_loop.unwrap();

            self.recorder.set_loop_inputs(steps_loop, LoopInputs::Dependent(0));

            self.recorder.set_loop_times(steps_loop, inputs_count - 1);
        }

        let mut consume_output = |mut output: LayerType| -> LayerType
        {
            (0..output.shape().batch_size).for_each(|batch_index|
            {
                Softmaxer::softmax_temperature(output.as_mut().batch_slice_mut(batch_index), temperature)
            });

            f_output(output.as_ref().average_tensor());

            output
        };

        {
            let first_output_value = self.outputs.first_output_value;

            let mut is_first_input = true;
            self.recorder.calculate_feedforward_with_dependent(|_, memory: &OperationsRecorderMemory|
            {
                let output = if is_first_input
                {
                    is_first_input = false;

                    memory.get_tensor(first_output_value)
                } else
                {
                    memory.get_tensor(output_value)
                };

                consume_output(output.clone_owned());

                input.next().expect("must be called the same amount of times as there are inputs")
            });
        }

        consume_output(self.recorder.get_tensor(output_value).clone_owned());
    }

    fn set_dropout_mask<R: DropoutRoll>(
        rng: &mut R,
        target: LayerTypeMut,
        probability: f32
    )
    {
        if probability == 0.0
        {
            target.fill(1.0);
        } else
        {
            let mut set_value = 0.0;
            let mut column = 0;

            let columns = target.columns();

            target.fill_with(move ||
            {
                if column == 0
                {
                    set_value = if rng.roll() >= probability
                    {
                        1.0
                    } else
                    {
                        0.0
                    };
                }

                column += 1;

                if column == columns
                {
                    column = 0;
                }

                set_value
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
            config: self.config,
            optimizer_info: None,
            weights_ptr: self.weights_ptr,
            weights: self.weights,
            dropouts: self.dropouts,
            inputs: self.inputs,
            outputs: self.outputs
        }
    }
}

impl SaveNetwork<EmbeddingsUnitFactory, ()>
{
    pub fn embeddings(&self, input: &OneHotLayer) -> LayerType
    {
        let weights = &self.weights;
        debug_assert_eq!(weights.layers.len(), 1);

        weights.layers[0].embeddings_calculate(input)
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[allow(unused_imports)]
    use crate::neural_network::{EmbeddingUnit, Lstm, Gru, Star};


    const SEED: u64 = 123;

    // i dont wanna do this for the unrolled version too
    const INPUT_DROPOUT_PROBABILITY: f32 = 0.0;

    const DROPOUT_PROBABILITY: f32 = 0.3;

    const IS_INPUT_ONE_HOT: bool = true;

    const SIZES: LayerSizes = LayerSizes{
        hidden: 10,
        initial_input: 3,
        input: 3,
        layers: 2,
        output: 3,
        final_output: 3,
        batch_size: 1
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
    struct StarUnitFactory;

    impl UnitFactory for StarUnitFactory
    {
        type Unit<T> = Star<T>;
    }

    #[allow(dead_code)]
    #[derive(Debug)]
    struct EmbeddingUnitFactory;

    impl UnitFactory for EmbeddingUnitFactory
    {
        type Unit<T> = EmbeddingUnit<T>;
    }

    type ThisFactory = StarUnitFactory;

    type NetworkType = Network<ThisFactory, ()>;

    fn inputs_outputs() -> (Vec<OwnedInputType>, Vec<OneHotLayer>)
    {
        let oh = |v: usize| OneHotLayer::new([[v].into()].into(), SIZES.initial_input, 1);

        let inputs = vec![
            OwnedInputType::OneHot(oh(0)),
            OwnedInputType::OneHot(oh(1)),
            OwnedInputType::OneHot(oh(0)),
            OwnedInputType::OneHot(oh(2)),
            OwnedInputType::OneHot(oh(2))
        ];

        let outputs = vec![
            oh(1),
            oh(0),
            oh(2),
            oh(1),
            oh(2)
        ];

        assert_eq!(inputs.len(), outputs.len());

        (inputs, outputs)
    }

    fn run_unrolled() -> (NetworkType, (f32, WeightsFullContainer<ThisFactory, LayerType>))
    {
        fastrand::seed(SEED);

        let (inputs, outputs) = inputs_outputs();

        let input_outputs = inputs.iter().cloned().zip(outputs);

        let network_config = NetworkConfigInfo{
            is_multistep: false,
            is_input_one_hot: IS_INPUT_ONE_HOT,
            print_optional_info: false
        };

        let mut at_once: NetworkType = Network::new(SIZES, INPUT_DROPOUT_PROBABILITY, DROPOUT_PROBABILITY, network_config);
        at_once.set_train_mode();

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
            } = at_once.record_feedforward_single_input(previous_state.take(), *this_input, Some(*this_target));

            at_once.recorder.name_diff_tensor(this_output, "output");
            at_once.recorder.name_diff_tensor(loss.unwrap(), "loss");

            previous_state = Some(next_state_ptr);

            if let Some(output) = output.as_mut()
            {
                *output = at_once.recorder.add(*output, loss.unwrap());

                at_once.recorder.name_diff_tensor(*output, "loss_combined");
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

        at_once.recorder.store_tensor_until_end(output.unwrap().as_value());

        at_once.weights_ptr.as_ref().unwrap().iter().for_each(|x|
        {
            at_once.recorder.store_tensor_until_end(x.weight_original.as_value());
            at_once.recorder.store_tensor_until_end(x.weight_original.as_gradient().unwrap());

            x.dropout.map(|x| at_once.recorder.store_tensor_until_end(x));
        });

        at_once.recorder.gradient(output.unwrap().into());

        dbg!(&at_once.recorder);
        at_once.recorder.resolve_memory(false);

        at_once.weights = Some(at_once.weights_ptr.take().unwrap().map(|x|
        {
            WeightInfo{
                weight: at_once.recorder.resolve_diff_tensor_ptr(x.weight_original),
                dropout: x.dropout.map(|x| at_once.recorder.resolve_tensor_ptr(x))
            }
        }));

        let output_value = at_once.recorder.resolve_tensor_ptr(output.unwrap().as_value());

        at_once.feedforward_setup_dropout(&mut fastrand::Rng::with_seed(SEED));

        at_once.recorder.calculate();

        let loss_batch = at_once.recorder.get_tensor(output_value);
        let loss = loss_batch.average();

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
                embeddings: weights.embeddings.map(|embeddings| embeddings.map(|x| f(x).unwrap())),
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

        assert!(gradients.iter().any(|x| !x.as_vec().into_iter().all(|x| x == 0.0)), "gradients are all 0: {gradients:?}");
    }

    #[test]
    fn steps_equivalent()
    {
        fastrand::seed(SEED);

        let (inputs, outputs) = inputs_outputs();

        let is_multistep = inputs.len() > 1;

        let input_outputs = inputs.iter().cloned().zip(outputs);

        let network_config = NetworkConfigInfo{
            is_multistep,
            is_input_one_hot: IS_INPUT_ONE_HOT,
            print_optional_info: true
        };

        let mut with_steps: NetworkType = Network::new(SIZES, INPUT_DROPOUT_PROBABILITY, DROPOUT_PROBABILITY, network_config);
        with_steps.set_train_mode();

        with_steps.prepare(true);

        let with_steps_gradient = with_steps.gradients(fastrand::Rng::with_seed(SEED), input_outputs.clone());

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
