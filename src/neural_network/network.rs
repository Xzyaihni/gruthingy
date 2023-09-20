use std::{
    f32,
    borrow::Borrow
};

use serde::{Serialize, Deserialize};

use crate::neural_network::{
    Softmaxer,
    LayerType,
    ScalarType,
    LayerInnerType,
    LAYERS_AMOUNT,
    INPUT_SIZE,
    DROPOUT_PROBABILITY,
    DROPCONNECT_PROBABILITY,
    network_unit::NetworkUnit
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

#[macro_export]
macro_rules! create_weights_container
{
    ($(($name:ident, $is_hidden:expr, $previous_size:expr, $current_size:expr, $previous_layer:expr)),+) =>
    {
        use std::ops::{SubAssign, AddAssign, DivAssign};

        use crate::neural_network::{
            Optimizer,
            CurrentOptimizer,
            LayerInnerType,
            network::{NewableLayer, WeightsNamed, WeightsSize}
        };


        #[derive(Debug, Serialize, Deserialize)]
        pub struct WeightsContainer<T>
        {
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
            pub fn for_each_weight<F: FnMut(T)>(self, mut f: F)
            {
                let Self{
                    $(
                        $name,
                    )+
                } = self;

                $(
                    f($name);
                )+
            }

            pub fn for_each_weight_mut<F: FnMut(&mut T)>(&mut self, mut f: F)
            {
                $(
                    f(&mut self.$name);
                )+
            }
        }

        impl WeightsContainer<LayerType>
        {
            pub fn new_randomized() -> Self
            {
                Self{
                    $(
                        $name: LayerType::new_diff(
                            if let Some(previous_layer) = $previous_layer
                            {
                                // i dont know how to do this another way
                                let previous_layer: usize = previous_layer;

                                LayerInnerType::new_with($previous_size, $current_size, ||
                                {
                                    let v = 1.0 / (previous_layer as f32).sqrt();

                                    (fastrand::f32() * 2.0 - 1.0) * v
                                })
                            } else
                            {
                                LayerInnerType::new($previous_size, $current_size)
                            }
                        ),
                    )+
                }
            }

            fn weights_named_info_inner(&self) -> WeightsContainer<WeightsNamed<&LayerType>>
            {
                WeightsContainer{
                    $(
                        $name: WeightsNamed{
                            name: "$name".to_owned(),
                            weights_size: WeightsSize{
                                weights: &self.$name,
                                current_size: $current_size,
                                previous_size: $previous_size,
                                is_hidden: $is_hidden
                            }
                        },
                    )+
                }
            }

            fn clone_weights_with_info_inner<F>(&self, mut f: F) -> Self
            where
                F: FnMut(WeightsSize<&LayerType>) -> LayerType
            {
                Self{
                    $(
                        $name: f(
                            WeightsSize{
                                weights: &self.$name,
                                current_size: $current_size,
                                previous_size: $previous_size,
                                is_hidden: $is_hidden
                            }
                        ),
                    )+
                }
            }

            fn map_weights_mut_inner<F, U>(&mut self, mut f: F) -> WeightsContainer<U>
            where
                F: FnMut(&mut LayerType) -> U
            {
                WeightsContainer{
                    $(
                        $name: f(&mut self.$name),
                    )+
                }
            }
        }

        impl<T: NewableLayer> WeightsContainer<T>
        {
            #[allow(dead_code)]
            pub fn new_container() -> Self
            {
                Self{
                    $(
                        $name: T::new($previous_size, $current_size),
                    )+
                }
            }
        }

        impl WeightsContainer<<CurrentOptimizer as Optimizer>::WeightParam>
        {
            pub fn gradients_to_change(
                &mut self,
                gradients: WeightsContainer<LayerInnerType>,
                hyper: &mut <CurrentOptimizer as Optimizer>::HyperParams
            ) -> WeightsContainer<LayerType>
            {
                let WeightsContainer{
                    $(
                        $name,
                    )+
                } = gradients;

                let change = WeightsContainer{
                    $(
                        $name: {
                            let gradient_info = &mut self.$name;
                            let gradient = CurrentOptimizer::gradient_clipped($name);

                            let change = CurrentOptimizer::gradient_to_change(
                                gradient_info,
                                gradient,
                                hyper
                            );

                            LayerType::new_undiff(change)
                        },
                    )+
                };

                change
            }
        }
    }
}

pub trait NewableLayer
{
    fn new(previous: usize, current: usize) -> Self;
}

pub struct NetworkOutput<State, Output>
{
    pub state: State,
    pub output: Output
}

pub struct NetworkDropped<Layer>(Network<Layer>);

impl<Layer: NetworkUnit> NetworkDropped<Layer>
{
    pub fn gradients(
        &mut self,
        input: impl Iterator<Item=(LayerType, LayerType)> + ExactSizeIterator
    ) -> (f32, Vec<Layer::ThisWeightsContainer<LayerInnerType>>)
    {
        self.0.clear();

        let loss = self.0.feedforward(input);

        let loss_value = loss.value_clone();

        loss.calculate_gradients();

        let gradients = self.0.layers.iter_mut().map(|layer|
        {
            layer.map_weights_mut(|weight| weight.take_gradient())
        }).collect::<Vec<_>>();

        (loss_value, gradients)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Network<Layer>
{
    layers: Vec<Layer>
}

impl<Layer: NetworkUnit> Network<Layer>
{
    pub fn dropconnected(&self) -> NetworkDropped<Layer>
    {
        let layers = self.layers.iter().map(|layer|
        {
            let layer: Layer = layer.clone_weights_with_info(|info|
            {
                if info.is_hidden
                {
                    let dropconnect_mask = self.create_dropout_mask(
                        info.previous_size,
                        info.current_size,
                        DROPCONNECT_PROBABILITY
                    );

                    info.weights * dropconnect_mask
                } else
                {
                    info.weights.recreate()
                }
            });

            layer
        }).collect::<Vec<Layer>>();

        NetworkDropped(Self{layers})
    }

    pub fn clear(&mut self)
    {
        self.layers.iter_mut().for_each(|layer| layer.clear());
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

    pub fn layers_mut(&mut self) -> &mut [Layer]
    {
        &mut self.layers
    }

    pub fn weights_info(
        &self
    ) -> Vec<<Layer as NetworkUnit>::ThisWeightsContainer<WeightsNamed<&LayerType>>>
    {
        self.layers.iter().map(|layer|
        {
            layer.weights_named_info()
        }).collect::<Vec<_>>()
    }

    #[allow(dead_code)]
    pub fn parameters_amount(&self) -> u128
    {
        self.layers.iter().map(|layer|
        {
            layer.parameters_amount()
        }).sum()
    }

    #[allow(dead_code)]
    pub fn accuracy(
        &mut self,
        input: impl Iterator<Item=(LayerType, LayerType)>
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
        T: Borrow<LayerType>
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
        previous_states: Option<Vec<Layer::State>>,
        dropout_masks: &[LayerType],
        input: &LayerType
    ) -> NetworkOutput<Vec<Layer::State>, T>
    where
        F: FnOnce(&mut Layer, Option<&Layer::State>, &LayerType) -> NetworkOutput<Layer::State, T>
    {
        let mut output: Option<T> = None;
        let mut last_output: Option<LayerType> = None;

        let mut states = Vec::with_capacity(LAYERS_AMOUNT);

        for l_i in 0..LAYERS_AMOUNT
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

            if l_i == (LAYERS_AMOUNT - 1)
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
        previous_states: Option<Vec<Layer::State>>,
        dropout_masks: &[LayerType],
        input: &LayerType,
        targets: LayerInnerType
    ) -> NetworkOutput<Vec<Layer::State>, ScalarType>
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
        input: impl Iterator<Item=(LayerType, LayerType)> + ExactSizeIterator
    ) -> ScalarType
    {
        let mut output: Option<ScalarType> = None;
        let mut previous_states: Option<Vec<Layer::State>> = None;

        let dropout_masks = self.create_dropout_masks(INPUT_SIZE, DROPOUT_PROBABILITY);

        for (this_input, mut this_output) in input
        {
            let NetworkOutput{
                state,
                output: this_output
            } = self.feedforward_single_input(
                previous_states.take(),
                &dropout_masks,
                &this_input,
                this_output.value_take()
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

    pub fn gradients(
        &mut self,
        input: impl Iterator<Item=(LayerType, LayerType)> + ExactSizeIterator
    ) -> (f32, Vec<Layer::ThisWeightsContainer<LayerInnerType>>)
    {
        let mut dropped = self.dropconnected();

        dropped.gradients(input)
    }

    pub fn predict_single_input(
        &mut self,
        previous_states: Option<Vec<Layer::State>>,
        dropout_masks: &[LayerType],
        input: &LayerType,
        temperature: f32
    ) -> NetworkOutput<Vec<Layer::State>, LayerInnerType>
    {
        self.feedforward_single_input_with_activation(|layer, previous_state, input|
        {
            let NetworkOutput{
                state,
                mut output
            } = layer.feedforward_unit(
                previous_state,
                input
            );

            let mut output = output.value_take();

            Softmaxer::softmax_temperature(&mut output, temperature);

            NetworkOutput{
                state,
                output
            }
        }, previous_states, dropout_masks, input)
    }

    fn predict(
        &mut self,
        input: impl Iterator<Item=LayerType> + ExactSizeIterator
    ) -> Vec<LayerInnerType>
    {
        let mut outputs: Vec<LayerInnerType> = Vec::with_capacity(input.len());
        let mut previous_state: Option<Vec<_>> = None;

        let dropout_masks = self.create_dropout_masks(INPUT_SIZE, 0.0);

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

    pub fn create_dropout_masks(&self, input_size: usize, probability: f32) -> Vec<LayerType>
    {
        // lmao iterating over the layers just to get the layer amount thats stored in a literal
        // constant
        self.layers.iter().map(|_|
        {
            self.create_dropout_mask(input_size, 1, probability)
        }).collect()
    }

    // i love my inconsistent naming of current/this size thing
    fn create_dropout_mask(
        &self,
        previous_size: usize,
        this_size: usize,
        probability: f32
    ) -> LayerType
    {
        let scaled_value = (1.0 - probability).recip();

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
    }
}

impl<Layer: NetworkUnit> Network<Layer>
{
    pub fn new() -> Self
    {
        let layers: Vec<_> =
            (0..LAYERS_AMOUNT).map(|_| Layer::new()).collect();

        Self{
            layers
        }
    }
}

