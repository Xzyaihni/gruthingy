use std::{
    f32,
    fmt,
    borrow::Borrow
};

use serde::{Serialize, Deserialize};

use crate::neural_network::{
    Softmaxer,
    DiffWrapper,
    LayerInnerType,
    NetworkUnit,
    NewableLayer,
    GenericUnit,
    OptimizerUnit,
    UnitFactory,
    DROPOUT_PROBABILITY,
    DROPCONNECT_PROBABILITY
};


pub struct WeightsSize<T>
{
    pub weights: T,
    pub previous_size: usize,
    pub current_size: usize,
    pub is_hidden: bool
}

pub struct WeightsNamed<T>
{
    pub name: String,
    pub weights_size: WeightsSize<T>
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LayerSizes
{
    pub input: usize,
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
    ($(($name:ident, $is_hidden:expr, $previous_size:expr, $current_size:expr)),+) =>
    {
        use std::ops::{SubAssign, AddAssign, DivAssign};

        use $crate::neural_network::{
            LayerInnerType,
            GRADIENT_CLIP,
            NewableLayer,
            GenericUnit,
            OptimizerUnit,
            Optimizer,
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
                            current_size: $current_size.into_number(self.sizes),
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

        impl WeightsContainer<DiffWrapper>
        {
            pub fn new_randomized(sizes: $crate::neural_network::LayerSizes) -> Self
            {
                use $crate::neural_network::network::LayerSize;

                Self{sizes, $(
                    $name: DiffWrapper::new_diff({
                        let previous_size = $previous_size.into_number(sizes);
                        let current_size = $current_size.into_number(sizes);

                        match $current_size
                        {
                            LayerSize::One =>
                            {
                                LayerInnerType::new(previous_size, current_size)
                            },
                            x =>
                            {
                                let previous_layer = x.into_number(sizes);

                                LayerInnerType::new_with(previous_size, current_size, ||
                                {
                                    let v = 1.0 / (previous_layer as f32).sqrt();

                                    (fastrand::f32() * 2.0 - 1.0) * v
                                })
                            }
                        }.into()
                    }),
                )+}
            }
        }

        impl<T> OptimizerUnit<T> for WeightsContainer<T>
        where
            T: NewableLayer + Serialize + serde::de::DeserializeOwned
        {
            fn new_zeroed(sizes: $crate::neural_network::LayerSizes) -> Self
            {
                Self{
                    sizes,
                    $(
                        $name: T::new(
                            $previous_size.into_number(sizes),
                            $current_size.into_number(sizes)
                        ),
                    )+
                }
            }

            fn gradients_to_change<O>(
                &mut self,
                gradients: Self::Unit<LayerInnerType>,
                optimizer: &O
            ) -> Self::Unit<DiffWrapper>
            where
                O: Optimizer<WeightParam=T>
            {
                gradients.zip(self.as_mut()).map(|(gradient, this)|
                {
                    let gradient = gradient.cap_magnitude(GRADIENT_CLIP);

                    let change = optimizer.gradient_to_change(this, gradient);

                    DiffWrapper::new_undiff(change.into())
                })
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

            fn map_mut<U, F>(&mut self, mut f: F) -> WeightsContainer<U>
            where
                F: FnMut(&mut T) -> U
            {
                WeightsContainer{
                    sizes: self.sizes,
                    $(
                        $name: f(&mut self.$name),
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
                                current_size: $current_size.into_number(self.sizes),
                                previous_size: $previous_size.into_number(self.sizes),
                                is_hidden: $is_hidden
                            }
                        ),
                    )+
                }
            }

            fn weights_named_info(&self) -> Self::Unit<WeightsNamed<&T>>
            {
                WeightsContainer{
                    sizes: self.sizes,
                    $(
                        $name: WeightsNamed{
                            name: stringify!($name).to_owned(),
                            weights_size: WeightsSize{
                                weights: &self.$name,
                                current_size: $current_size.into_number(self.sizes),
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

pub struct NetworkOutput<State, Output>
{
    pub state: State,
    pub output: Output
}

pub struct NetworkDropped<N, O>(Network<N, O>)
where
    N: UnitFactory,
    N::Unit<O>: OptimizerUnit<O>,
    N::Unit<DiffWrapper>: NetworkUnit;

impl<N, O> NetworkDropped<N, O>
where
    N: UnitFactory,
    N::Unit<O>: OptimizerUnit<O>,
    N::Unit<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=N::Unit<DiffWrapper>>
{
    pub fn gradients(
        &mut self,
        input: impl Iterator<Item=(LayerInnerType, LayerInnerType)>
    ) -> (f32, Vec<N::Unit<LayerInnerType>>)
    where
        // i am going to go on a rampage, this is insane, this shouldnt be a thing, why is rust
        // like this??????????/
        N::Unit<DiffWrapper>: NetworkUnit<Unit<LayerInnerType>=N::Unit<LayerInnerType>> + fmt::Debug
    {
        let loss = self.0.feedforward(input);

        let loss_value = *loss.scalar();

        loss.calculate_gradients();

        let gradients = self.0.layers.iter_mut().map(|layer|
        {
            layer.map_mut(|weight|
            {
                debug_assert!(weight.parent().is_none());

                weight.take_gradient_tensor()
            })
        }).collect::<Vec<_>>();

        (loss_value, gradients)
    }
}

type UnitState<N> = <<N as UnitFactory>::Unit<DiffWrapper> as NetworkUnit>::State;

#[derive(Debug, Serialize, Deserialize)]
pub struct Network<N: UnitFactory, O>
where
    N::Unit<O>: OptimizerUnit<O>,
    N::Unit<DiffWrapper>: NetworkUnit
{
    sizes: LayerSizes,
    optimizer_info: Option<Vec<N::Unit<O>>>,
    layers: Vec<N::Unit<DiffWrapper>>
}

impl<N, O> Network<N, O>
where
    N::Unit<O>: OptimizerUnit<O>,
    N::Unit<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
    N: UnitFactory
{
    pub fn new(sizes: LayerSizes) -> Self
    where
        O: NewableLayer
    {
        let optimizer_info: Option<Vec<_>> = 
            Some((0..sizes.layers).map(|_| N::Unit::new_zeroed(sizes)).collect());

        let layers: Vec<_> =
            (0..sizes.layers).map(|_| N::Unit::new(sizes)).collect();

        Self{
            sizes,
            optimizer_info,
            layers
        }
    }

    pub fn dropconnected(&self) -> NetworkDropped<N, O>
    {
        let layers = if N::Unit::<DiffWrapper>::dropconnectable()
        {
            self.layers.iter().map(|layer|
            {
                layer.clone_weights_with_info(|info|
                {
                    if info.is_hidden
                    {
                        let dropconnect_mask = Self::create_dropout_mask(
                            info.previous_size,
                            info.current_size,
                            DROPCONNECT_PROBABILITY
                        );

                        info.weights * dropconnect_mask
                    } else
                    {
                        info.weights.clone()
                    }
                })
            }).collect()
        } else
        {
            self.layers.clone()
        };

        NetworkDropped(Self{
            sizes: self.sizes,
            optimizer_info: None,
            layers
        })
    }

    // oh my god wut am i even doing at this point its so over
    pub fn enable_gradients(&mut self)
    {
        self.layers.iter_mut().for_each(|layer|
        {
            layer.enable_gradients();
        });
    }

    pub fn disable_gradients(&mut self)
    {
        self.layers.iter_mut().for_each(|layer|
        {
            layer.disable_gradients();
        });
    }

    pub fn gradients_info(
        &mut self
    ) -> impl Iterator<Item=(&'_ mut N::Unit<DiffWrapper>, &'_ mut N::Unit<O>)>
    {
        self.layers.iter_mut().zip(self.optimizer_info.as_mut().unwrap().iter_mut())
    }

    pub fn weights_info(
        &self
    ) -> Vec<N::Unit<WeightsNamed<&DiffWrapper>>>
    where
        // WORKING LANGUAGE BY THE WAY ITS WORKING JUST FINE HAHAHAHHAHAHAHAHAHHA
        for<'a> N::Unit<DiffWrapper>: NetworkUnit<Unit<WeightsNamed<&'a DiffWrapper>>=N::Unit<WeightsNamed<&'a DiffWrapper>>>
    {
        self.layers.iter().map(|layer|
        {
            layer.weights_named_info()
        }).collect::<Vec<_>>()
    }

    pub fn assert_empty(&self)
    {
        self.layers.iter().for_each(|layer|
        {
            layer.for_each_weight_ref(|weight| assert!(weight.parent().is_none()));
        });
    }

    #[allow(dead_code)]
    pub fn parameters_amount(&self) -> u128
    {
        self.layers.iter().map(|layer|
        {
            layer.parameters_amount(self.sizes)
        }).sum()
    }

    #[allow(dead_code)]
    pub fn accuracy(
        &mut self,
        input: impl Iterator<Item=(LayerInnerType, LayerInnerType)>
    ) -> f32
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let amount = input.len();

        let f_output = self.predict(input.into_iter());

        Self::correct_guesses(
            f_output.into_iter(),
            output.into_iter()
        ) as f32 / amount as f32
    }

    fn correct_guesses<P, T>(
        predicted: impl Iterator<Item=P>,
        target: impl Iterator<Item=T>
    ) -> usize
    where
        P: Borrow<LayerInnerType>,
        T: Borrow<LayerInnerType>
    {
        predicted.zip(target).map(|(predicted, target)|
        {
            let target_index = target.borrow().highest_index();
            if predicted.borrow().highest_index() == target_index
            {
                1
            } else
            {
                0
            }
        }).sum()
    }

    fn feedforward_single_input_with_activation<F, T>(
        &mut self,
        last_f: F,
        previous_states: Option<Vec<UnitState<N>>>,
        dropout_masks: &[DiffWrapper],
        input: &DiffWrapper
    ) -> NetworkOutput<Vec<UnitState<N>>, T>
    where
        F: FnOnce(&mut N::Unit<DiffWrapper>, Option<&UnitState<N>>, &DiffWrapper) -> NetworkOutput<UnitState<N>, T>
    {
        let mut output: Option<T> = None;
        let mut last_output: Option<DiffWrapper> = None;

        let mut states = Vec::with_capacity(self.sizes.layers);

        // stfu clippy this is more readable
        #[allow(clippy::needless_range_loop)]
        for l_i in 0..self.sizes.layers
        {
            let input = last_output.as_ref().unwrap_or(input);

            debug_assert!(l_i < self.layers.len());
            let layer = unsafe{ self.layers.get_unchecked_mut(l_i) };

            let previous_state = unsafe{
                previous_states.as_ref().map(|previous_state|
                {
                    previous_state.get_unchecked(l_i)
                })
            };

            if l_i == (self.sizes.layers - 1)
            {
                // last layer
                let NetworkOutput{
                    state,
                    output: this_output
                } = last_f(layer, previous_state, input);

                output = Some(this_output);

                states.push(state);

                // i like how rust cant figure out that the last index is the last iteration
                // without this
                break;
            } else
            {
                let dropout_mask = &dropout_masks[l_i];

                let NetworkOutput{
                    state,
                    output: this_output
                } = layer.feedforward_unit_nonlast(
                    previous_state,
                    dropout_mask,
                    input
                );

                last_output = Some(this_output);

                states.push(state);
            }
        }

        NetworkOutput{
            state: states,
            output: output.unwrap()
        }
    }

    fn feedforward_single_input(
        &mut self,
        previous_states: Option<Vec<UnitState<N>>>,
        dropout_masks: &[DiffWrapper],
        input: &DiffWrapper,
        targets: LayerInnerType
    ) -> NetworkOutput<Vec<UnitState<N>>, DiffWrapper>
    {
        self.feedforward_single_input_with_activation(|layer, previous_state, input|
        {
            layer.feedforward_unit_last(
                previous_state,
                input,
                targets
            )
        }, previous_states, dropout_masks, input)
    }

    #[allow(dead_code)]
    pub fn feedforward(
        &mut self,
        input: impl Iterator<Item=(LayerInnerType, LayerInnerType)>
    ) -> DiffWrapper
    {
        let mut output: Option<DiffWrapper> = None;
        let mut previous_states: Option<Vec<UnitState<N>>> = None;

        let dropout_masks = self.create_dropout_masks(self.sizes.input, DROPOUT_PROBABILITY);

        for (this_input, this_output) in input
        {
            let this_input = DiffWrapper::new_undiff(this_input.into());

            let NetworkOutput{
                state,
                output: this_output
            } = self.feedforward_single_input(
                previous_states.take(),
                &dropout_masks,
                &this_input,
                this_output
            );

            if let Some(output) = output.as_mut()
            {
                *output += this_output;
            } else
            {
                output = Some(this_output)
            }

            previous_states = Some(state);
        }

        output.unwrap()
    }

    pub fn predict_single_input(
        &mut self,
        previous_states: Option<Vec<UnitState<N>>>,
        dropout_masks: &[DiffWrapper],
        input: &DiffWrapper,
        temperature: f32
    ) -> NetworkOutput<Vec<UnitState<N>>, LayerInnerType>
    {
        self.feedforward_single_input_with_activation(|layer, previous_state, input|
        {
            let NetworkOutput{
                state,
                output
            } = layer.feedforward_unit(
                previous_state,
                input
            );

            let mut output = output.tensor().clone();

            Softmaxer::softmax_temperature(&mut output, temperature);

            NetworkOutput{
                state,
                output
            }
        }, previous_states, dropout_masks, input)
    }

    fn predict(
        &mut self,
        input: impl Iterator<Item=LayerInnerType> + ExactSizeIterator
    ) -> Vec<LayerInnerType>
    {
        let mut outputs: Vec<LayerInnerType> = Vec::with_capacity(input.len());
        let mut previous_state: Option<Vec<_>> = None;

        let dropout_masks = self.create_dropout_masks(self.sizes.input, 0.0);

        for this_input in input
        {
            let this_input = DiffWrapper::new_undiff(this_input.into());

            let NetworkOutput{
                state,
                output
            } = self.predict_single_input(
                previous_state.take(),
                &dropout_masks,
                &this_input,
                1.0
            );

            outputs.push(output);
            previous_state = Some(state);
        }

        outputs
    }

    pub fn create_dropout_masks(&self, input_size: usize, probability: f32) -> Vec<DiffWrapper>
    {
        // lmao iterating over the layers just to get the layer amount thats stored in a literal
        // constant
        self.layers.iter().map(|_|
        {
            Self::create_dropout_mask(input_size, 1, probability)
        }).collect()
    }

    // i love my inconsistent naming of current/this size thing
    fn create_dropout_mask(
        previous_size: usize,
        this_size: usize,
        probability: f32
    ) -> DiffWrapper
    {
        let scaled_value = (1.0 - probability).recip();

        DiffWrapper::new_undiff(LayerInnerType::new_with(previous_size, this_size, ||
        {
            let roll = fastrand::f32();
            
            if roll >= probability
            {
                scaled_value
            } else
            {
                0.0
            }
        }).into())
    }
}
