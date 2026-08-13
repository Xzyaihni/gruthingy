use std::{
    f32,
    fmt,
    vec,
    iter,
    cmp::Ordering,
    borrow::Borrow,
    ops::SubAssign
};

use serde::{Serialize, Deserialize, de::DeserializeOwned};

use crate::{
//    EmbeddingsUnitFactory,
    neural_network::{
        OperationsRecorder,
        Softmaxer,
        DiffTensor,
        DiffScalar,
        TensorIndex,
        OneHotLayer,
        OneHotIndex,
        InputType,
        LayerType,
        NetworkUnit,
        NewableLayer,
//        GenericUnit,
//        Optimizer,
        OptimizerUnit,
        UnitFactory,
//        EN,
        DROPCONNECT_PROBABILITY,
//        network_unit::Embeddingsable
    }
};


pub struct WeightsSize<T>
{
    pub weights: T,
    pub previous_size: usize,
    pub this_size: usize,
    pub is_hidden: bool
}

pub struct WeightsNamed<T>
{
    pub name: String,
    pub layer: usize,
    pub weights_size: WeightsSize<T>
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

        impl WeightsContainer<DiffTensor>
        {
            pub fn new(recorder: &mut OperationsRecorder, sizes: $crate::neural_network::LayerSizes) -> Self
            {
                use $crate::neural_network::network::LayerSize;

                Self{sizes, $(
                    $name: recorder.new_tensor($this_size.into_number(sizes), $previous_size.into_number(sizes)),
                )+}
            }
            /*pub fn new_randomized(sizes: $crate::neural_network::LayerSizes) -> Self
            {
                use $crate::neural_network::network::LayerSize;

                Self{sizes, $(
                    $name: DiffWrapper::new_diff({
                        let previous_size = $previous_size.into_number(sizes);
                        let this_size = $this_size.into_number(sizes);

                        match $previous_size
                        {
                            LayerSize::One =>
                            {
                                LayerType::new(this_size, previous_size)
                            },
                            x =>
                            {
                                let previous_layer = x.into_number(sizes);

                                LayerType::new_with(this_size, previous_size, ||
                                {
                                    let v = 1.0 / (previous_layer as f32).sqrt();

                                    (fastrand::f32() * 2.0 - 1.0) * v
                                })
                            }
                        }.into()
                    }),
                )+}
            }*/
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

pub struct NetworkOutput<State, Output>
{
    pub state: State,
    pub output: Output
}

type UnitState<N> = <<N as UnitFactory>::Unit<DiffTensor> as NetworkUnit>::State;

#[derive(Serialize, Deserialize)]
pub struct WeightsFullContainer<U, T>
{
    layers: Vec<U>,
    output: T
}

/*impl<N: UnitFactory, T> Clone for WeightsFullContainer<N, T>
where
    T: Clone,
    N::Unit<T>: Clone + Serialize + DeserializeOwned
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
    N::Unit<T>: IntoIterator<Item=T> + Serialize + DeserializeOwned
{
    type Item = T;
    type IntoIter = iter::Chain<iter::Flatten<vec::IntoIter<N::Unit<T>>>, iter::Once<T>>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.layers.into_iter().flatten().chain(iter::once(self.output))
    }
}*/

impl<U, T> WeightsFullContainer<U, T>
{
    pub fn new(
        sizes: LayerSizes,
        unit_f: impl FnMut(LayerSizes) -> U,
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

    /*pub fn map_mut<F, U>(&mut self, mut f: F) -> WeightsFullContainer<N, U>
    where
        N::Unit<T>: GenericUnit<T, Unit<U>=N::Unit<U>>,
        N::Unit<U>: Serialize + DeserializeOwned,
        F: FnMut(&mut T) -> U
    {
        WeightsFullContainer{
            output: f(&mut self.output),
            layers: self.layers.iter_mut().map(|layer| layer.map_mut(&mut f)).collect()
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
    }*/
}

pub struct Network<N: UnitFactory, O>
{
    recorder: OperationsRecorder,
    sizes: LayerSizes,
    dropout_probability: f32,
    inputs_targets: Vec<(TensorIndex, OneHotIndex)>,
    loss: DiffScalar,
    optimizer_info: Option<WeightsFullContainer<N::Unit<O>, O>>,
    weights: WeightsFullContainer<N::Unit<DiffTensor>, DiffTensor>
}

impl<N: UnitFactory, O> Network<N, O>
{
    pub fn sizes(&self) -> &LayerSizes
    {
        &self.sizes
    }
}

impl<N: UnitFactory, O> Network<N, O>
where
    N::Unit<O>: OptimizerUnit<O>,
    N::Unit<DiffTensor>: NetworkUnit<Unit<DiffTensor>=N::Unit<DiffTensor>>,
//    for<'a> &'a N::Unit<DiffWrapper>: IntoIterator<Item=&'a DiffWrapper>,
//    for<'a> &'a mut N::Unit<DiffWrapper>: IntoIterator<Item=&'a mut DiffWrapper>
{
    pub fn new(
        sizes: LayerSizes,
        inputs_count: usize,
        dropout_probability: f32
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

        let output_weights_tensor = {
            let weights = recorder.new_tensor(sizes.output, sizes.hidden);
            recorder.set_tensor(weights.as_value(), LayerType::new_with(sizes.output, sizes.hidden, ||
            {
                let v = 1.0 / (sizes.hidden as f32).sqrt();

                (fastrand::f32() * 2.0 - 1.0) * v
            }));

            weights
        };

        let weights = WeightsFullContainer::new(sizes, |size|
        {
            N::Unit::new(&mut recorder, size)
        }, output_weights_tensor);

        let mut this = Self{
            recorder,
            sizes,
            inputs_targets: Vec::new(),
            loss: DiffScalar::undefined(),
            optimizer_info,
            weights,
            dropout_probability
        };

        this.record_feedforward(inputs_count);

        this.recorder.finish();

        this
    }

    fn record_feedforward(&mut self, inputs_count: usize)
    {
        let mut loss: Option<DiffScalar> = None;
        let mut previous_states: Option<Vec<UnitState<N>>> = None;

        let dropout_masks: Vec<_> = self.weights.layers.iter().skip(1).map(|_|
        {
            self.recorder.set_new_tensor(LayerType::repeat(self.sizes.hidden, 1, 0.0)).as_value()
        }).collect();

        for _ in 0..inputs_count
        {
            let this_input = self.recorder.new_tensor(self.sizes.input, 1).as_value();
            let this_target = self.recorder.new_one_hot();

            self.inputs_targets.push((this_input, this_target));

            let NetworkOutput{
                state,
                output: this_loss
            } = self.record_feedforward_single_input(
                previous_states.take(),
                &dropout_masks,
                this_input.into(),
                this_target
            );

            // this might cause some networks to not converge at 0 loss (since sometimes its impossible to predict the next word with 0 context)
            // but it makes it more robust i think?? i forgot why i did it this way but it IS important and it is NOT a bug
            if let Some(loss) = loss.as_mut()
            {
                *loss = self.recorder.add_scalars(*loss, this_loss);
            } else
            {
                loss = Some(this_loss)
            }

            previous_states = Some(state);
        }

        self.loss = loss.unwrap();
    }

    fn record_feedforward_single_input(
        &mut self,
        previous_states: Option<Vec<UnitState<N>>>,
        dropout_masks: &[TensorIndex],
        input: InputType,
        targets: OneHotIndex
    ) -> NetworkOutput<Vec<UnitState<N>>, DiffScalar>
    {
        self.record_feedforward_single_input_with_activation(|this, layer_index, previous_state, input|
        {
            let output = this.record_feedforward_unit_last(
                layer_index,
                previous_state,
                input
            );

            NetworkOutput{
                state: output.state,
                output: this.recorder.softmax_cross_entropy(output.output, targets)
            }
        }, previous_states, dropout_masks, input)
    }

    fn record_feedforward_single_input_with_activation<F, T>(
        &mut self,
        last_f: F,
        previous_states: Option<Vec<UnitState<N>>>,
        dropout_masks: &[TensorIndex],
        input: InputType
    ) -> NetworkOutput<Vec<UnitState<N>>, T>
    where
        F: FnOnce(&mut Self, usize, Option<&UnitState<N>>, InputType) -> NetworkOutput<UnitState<N>, T>
    {
        let mut output: Option<T> = None;
        let mut last_output: Option<InputType> = None;

        let mut states = Vec::with_capacity(self.sizes.layers);

        #[allow(clippy::needless_range_loop)]
        for l_i in 0..self.sizes.layers
        {
            let input = last_output.unwrap_or(input);

            let layer = &self.weights.layers[l_i];

            let previous_state = previous_states.as_ref().map(|previous_state|
            {
                &previous_state[l_i]
            });

            if l_i == (self.sizes.layers - 1)
            {
                // last layer
                let NetworkOutput{
                    state,
                    output: this_output
                } = last_f(self, l_i, previous_state, input);

                output = Some(this_output);

                states.push(state);

                break;
            } else
            {
                let dropout_mask = dropout_masks[l_i];

                let NetworkOutput{
                    state,
                    output: this_output
                } = layer.feedforward_unit_nonlast(
                    &mut self.recorder,
                    previous_state,
                    dropout_mask,
                    input
                );

                last_output = Some(this_output.as_value().into());

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
        previous_state: Option<&UnitState<N>>,
        input: InputType
    ) -> NetworkOutput<UnitState<N>, DiffTensor>
    {
        let mut output = self.weights.layers[layer_index].feedforward_unit(&mut self.recorder, previous_state, input);

        output.output = self.recorder.matmulv(self.weights.output, output.output);

        output
    }

/*    pub fn apply_gradients<OP>(
        &mut self,
        gradients: WeightsFullContainer<N, LayerType>,
        optimizer: &mut OP,
        gradient_clip: Option<f32>
    )
    where
        OP: Optimizer<WeightParam=O>,
        N::Unit<DiffWrapper>: SubAssign,
        N::Unit<LayerType>: Serialize + DeserializeOwned + IntoIterator<Item=LayerType>,
        for<'b> &'b mut N::Unit<O>: IntoIterator<Item=&'b mut O>,
        N::Unit<O>: OptimizerUnit<O, Unit<DiffWrapper>=N::Unit<DiffWrapper>>,
        N::Unit<O>: OptimizerUnit<O, Unit<LayerType>=N::Unit<LayerType>>
    {
        self.disable_gradients();

        gradients.into_iter()
            .zip(self.weights.iter_mut().zip(self.optimizer_info.as_mut().unwrap().iter_mut()))
            .for_each(|(mut gradient, (network_weights, optimizer_info))|
            {
                if let Some(gradient_clip) = gradient_clip
                {
                    gradient = gradient.cap_magnitude(gradient_clip);
                }

                let change = optimizer.gradient_to_change(optimizer_info, gradient);

                *network_weights -= DiffWrapper::new_undiff(change.into());
            });

        optimizer.advance_time();

        self.enable_gradients();
    }

    fn dropconnected(&self) -> Self
    {
        let weights = if N::Unit::<DiffWrapper>::dropconnectable()
        {
            WeightsFullContainer{
                layers: self.weights.layers.iter().map(|layer|
                {
                    layer.clone_weights_with_info(|info|
                    {
                        if info.is_hidden
                        {
                            let dropconnect_mask = Self::create_dropout_mask(
                                info.previous_size,
                                info.this_size,
                                DROPCONNECT_PROBABILITY
                            );

                            info.weights * dropconnect_mask
                        } else
                        {
                            info.weights.clone()
                        }
                    })
                }).collect(),
                output: self.weights.output.clone()
            }
        } else
        {
            self.weights.clone()
        };

        Self{
            weights,
            optimizer_info: None,
            sizes: self.sizes,
            dropout_probability: self.dropout_probability,
        }
    }

    pub fn gradients(
        &mut self,
        input: impl Iterator<Item=(InputType, OneHotLayer)>
    ) -> (f32, WeightsFullContainer<N, LayerType>)
    where
        // i am going to go on a rampage, this is insane, this shouldnt be a thing, why is rust
        // like this??????????/
        N::Unit<LayerType>: Serialize + DeserializeOwned,
        N::Unit<DiffWrapper>: NetworkUnit<Unit<LayerType>=N::Unit<LayerType>> + fmt::Debug
    {
        let loss = {
            let mut dropconnected = self.dropconnected();

            dropconnected.feedforward(input)
        };

        let loss_value = *loss.scalar();

        loss.calculate_gradients();

        let gradients = self.weights.map_mut(|weight|
        {
            debug_assert!(weight.parent().is_none());

            weight.take_gradient_tensor()
        });

        (loss_value, gradients)
    }

    // oh my god wut am i even doing at this point its so over
    pub fn enable_gradients(&mut self)
    {
        self.weights.iter_mut().for_each(|weight|
        {
            weight.enable_gradients();
        });
    }

    pub fn disable_gradients(&mut self)
    {
        self.weights.iter_mut().for_each(|weight|
        {
            weight.disable_gradients();
        });
    }

    pub fn weights_info<'b, 'c>(
        &'b self
    ) -> Vec<WeightsNamed<&'b DiffWrapper>>
    where
        // WORKING LANGUAGE BY THE WAY ITS WORKING JUST FINE HAHAHAHHAHAHAHAHAHHA
        for<'a> N::Unit<DiffWrapper>: NetworkUnit<Unit<WeightsNamed<&'a DiffWrapper>>=N::Unit<WeightsNamed<&'a DiffWrapper>>>,
        N::Unit<WeightsNamed<&'b DiffWrapper>>: IntoIterator<Item=WeightsNamed<&'b DiffWrapper>>
    {
        self.weights.layers.iter().enumerate()
            .flat_map(|(layer_index, layer)|
            {
                layer.weights_named_info(layer_index).into_iter()
            })
            .chain(iter::once(WeightsNamed{
                name: "output".to_owned(),
                layer: self.sizes.layers.saturating_sub(1),
                weights_size: WeightsSize{
                    weights: &self.weights.output,
                    this_size: self.sizes.output,
                    previous_size: self.sizes.hidden,
                    is_hidden: false
                }
            }))
            .collect::<Vec<_>>()
    }

    pub fn assert_empty(&self)
    {
        self.weights.iter().for_each(|weight|
        {
            assert!(weight.parent().is_none());
        });
    }

    #[allow(dead_code)]
    pub fn parameters_amount(&self) -> u128
    {
        let layers_sum: u128 = self.weights.layers.iter().map(|layer|
        {
            layer.parameters_amount(self.sizes)
        }).sum();

        layers_sum + self.sizes.input as u128 * self.sizes.hidden as u128
    }

    fn with_predict<T, F>(
        &mut self,
        input: impl Iterator<Item=(InputType, OneHotLayer)>,
        f: F
    ) -> impl Iterator<Item=(usize, T)>
    where
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
        input: impl Iterator<Item=(InputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, u32)>
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
        input: impl Iterator<Item=(InputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, f32)>
    {
        self.with_predict(input, |predicted, _highest_index, target_index|
        {
            *predicted.iter().nth(target_index).unwrap()
        })
    }

    #[allow(dead_code)]
    pub fn correct_guesses(
        &mut self,
        input: impl Iterator<Item=(InputType, OneHotLayer)>
    ) -> impl Iterator<Item=(usize, bool)>
    {
        self.with_predict(input, |_predicted, highest_index, target_index|
        {
            highest_index == target_index
        })
    }

    #[allow(dead_code)]
    pub fn accuracy(
        &mut self,
        input: impl Iterator<Item=(InputType, OneHotLayer)>
    ) -> f32
    {
        let mut total = 0;
        let correct_amount = self.correct_guesses(input).filter(|(_, x)|
        {
            total += 1;

            *x
        }).count();

        correct_amount as f32 / total as f32
    }*/

    #[allow(dead_code)]
    pub fn feedforward(
        &mut self,
        input: impl Iterator<Item=(InputType, OneHotLayer)>
    )
    {
todo!()
/*        */
    }

/*    pub fn predict_single_input(
        &mut self,
        previous_states: Option<Vec<UnitState<N>>>,
        dropout_masks: &[DiffWrapper],
        input: &InputType,
        temperature: f32
    ) -> NetworkOutput<Vec<UnitState<N>>, LayerType>
    {
        self.feedforward_single_input_with_activation(|layer, previous_state, input|
        {
            let NetworkOutput{
                state,
                output
            } = self.feedforward_unit_last(
                layer,
                previous_state,
                input
            );

            let mut output = NetworkOutput{
                state,
                output: output.tensor().clone()
            };

            Softmaxer::softmax_temperature(&mut output.output, temperature);

            output
        }, previous_states, dropout_masks, input)
    }

    fn predict(
        &mut self,
        input: impl Iterator<Item=InputType> + ExactSizeIterator
    ) -> Vec<LayerType>
    {
        let mut outputs: Vec<LayerType> = Vec::with_capacity(input.len());
        let mut previous_state: Option<Vec<_>> = None;

        let dropout_masks = self.create_dropout_masks(self.sizes.hidden, 0.0);

        for this_input in input
        {
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
        self.weights.layers.iter().skip(1).map(|_|
        {
            Self::create_dropout_mask(1, input_size, probability)
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

        let inner = if probability == 0.0
        {
            LayerType::repeat(previous_size, this_size, 1.0)
        } else
        {
            LayerType::new_with(previous_size, this_size, ||
            {
                let roll = fastrand::f32();

                if roll >= probability
                {
                    scaled_value
                } else
                {
                    0.0
                }
            })
        };

        DiffWrapper::new_undiff(inner.into())
    }*/
}

/*impl<O> Network<EmbeddingsUnitFactory, O>
where
    EN<O>: OptimizerUnit<O>,
    EN<DiffWrapper>: NetworkUnit<Unit<DiffWrapper>=EN<DiffWrapper>> + Embeddingsable,
    EmbeddingsUnitFactory: UnitFactory
{
    pub fn without_optimizer(self) -> Network<EmbeddingsUnitFactory, ()>
    {
        Network{
            sizes: self.sizes,
            dropout_probability: self.dropout_probability,
            optimizer_info: None,
            weights: self.weights
        }
    }

    pub fn embeddings(&self, input: OneHotLayer) -> DiffWrapper
    {
        debug_assert!(self.weights.layers.len() == 1);

        self.weights.layers[0].embeddings(&input)
    }
}
*/
