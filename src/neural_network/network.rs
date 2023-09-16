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
    HIDDEN_AMOUNT,
    DROPOUT_PROBABILITY,
    DROPCONNECT_PROBABILITY,
    network_unit::NetworkUnit
};


pub enum WeightInfo
{
    Hidden,
    Input,
    One
}

impl WeightInfo
{
    pub fn into_value(self, input_size: usize) -> usize
    {
        match self
        {
            Self::Hidden => HIDDEN_AMOUNT,
            Self::Input => input_size,
            Self::One => 1
        }
    }
}

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
    ($index_enum:ident, $weights_info:expr) =>
    {
        use crate::neural_network::network::{WeightsNamed, WeightsSize};

        #[derive(Debug, Serialize, Deserialize)]
        pub struct WeightsContainer<T>([T; $index_enum::COUNT]);

        impl<T> IntoIterator for WeightsContainer<T>
        {
            type Item = T;
            type IntoIter = array::IntoIter<T, { $index_enum::COUNT }>;

            fn into_iter(self) -> Self::IntoIter
            {
                self.0.into_iter()
            }
        }

        impl<T> FromIterator<T> for WeightsContainer<T>
        {
            fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self
            {
                let v = iter.into_iter().collect::<Vec<_>>();

                let v_len = v.len();
                let inner = v.try_into().map_err(|_|
                {
                    format!(
                        "iterator length doesnt match (expected {}, got {v_len})",
                        $index_enum::COUNT
                    )
                }).unwrap();

                Self(inner)
            }
        }

        impl<T: DivAssign<f32>> DivAssign<f32> for WeightsContainer<T>
        {
            fn div_assign(&mut self, rhs: f32)
            {
                self.0.iter_mut().for_each(|value|
                {
                    *value /= rhs;
                });
            }
        }

        impl<T: AddAssign<T>> AddAssign for WeightsContainer<T>
        {
            fn add_assign(&mut self, rhs: Self)
            {
                self.0.iter_mut().zip(rhs.0.into_iter()).for_each(|(value, rhs)|
                {
                    *value += rhs;
                });
            }
        }

        impl<T: NewableLayer> WeightsContainer<T>
        {
            #[allow(dead_code)]
            pub fn new_container(input_size: usize) -> Self
            {
                $weights_info.into_iter().map(|(previous, current, _prev_layer)|
                {
                    let previous = previous.into_value(input_size);
                    let current = current.into_value(input_size);

                    T::new(previous, current)
                }).collect()
            }
        }

        impl<T> WeightsContainer<T>
        {
            pub fn inner_weights_size(
                &self,
                input_size: usize
            ) -> impl Iterator<Item=WeightsSize<&T>> + '_
            {
                self.0.iter().zip($weights_info.into_iter()).enumerate()
                    .map(move |(index, (weights, (previous, current, _prev_layer)))|
                    {
                        let previous = previous.into_value(input_size);
                        let current = current.into_value(input_size);

                        let this_enum = $index_enum::from_repr(index).unwrap();

                        WeightsSize{
                            weights,
                            previous_size: previous,
                            current_size: current,
                            is_hidden: this_enum.is_hidden()
                        }
                    })
            }

            pub fn inner_weights_info(
                &self,
                input_size: usize
            ) -> impl Iterator<Item=WeightsNamed<&T>> + '_
            {
                self.inner_weights_size(input_size).enumerate()
                    .map(move |(index, weights_size)|
                    {
                        let this_enum = $index_enum::from_repr(index).unwrap();
                        let name = format!("{this_enum:?}");

                        WeightsNamed{
                            name,
                            weights_size
                        }
                    })
            }

            #[allow(dead_code)]
            pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut T>
            {
                self.0.iter_mut()
            }

            pub fn weight(&self, index: WeightIndex) -> &T
            {
                self.raw_index(index as usize)
            }

            pub fn raw_index(&self, index: usize) -> &T
            {
                &self.0[index]
            }

            #[allow(dead_code)]
            pub fn raw_index_mut(&mut self, index: usize) -> &mut T
            {
                &mut self.0[index]
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

impl<Layer: NetworkUnit + FromIterator<LayerType>> NetworkDropped<Layer>
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
    layers: Vec<Layer>,
    word_vector_size: usize
}

impl<Layer: NetworkUnit + FromIterator<LayerType>> Network<Layer>
{
    pub fn dropconnected(&self) -> NetworkDropped<Layer>
    {
        let layers = self.layers.iter().map(|layer|
        {
            let layer: Layer = layer.clone_weights_with_info(|weights, info|
            {
                if info.is_hidden
                {
                    let dropconnect_mask = self.create_dropout_mask(
                        info.previous_size,
                        info.current_size,
                        DROPCONNECT_PROBABILITY
                    );

                    weights * dropconnect_mask
                } else
                {
                    weights.recreate()
                }
            }, self.word_vector_size);

            layer
        }).collect::<Vec<Layer>>();

        NetworkDropped(Self{
            layers,
            word_vector_size: self.word_vector_size
        })
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

    pub fn weights_info(&self) -> Vec<Vec<WeightsNamed<&LayerType>>>
    {
        self.layers.iter().map(|layer|
        {
            layer.weights_info(self.word_vector_size)
        }).collect()
    }

    #[allow(dead_code)]
    pub fn parameters_amount(&self) -> u128
    {
        self.layers.iter().map(|layer|
        {
            layer.parameters_amount(self.word_vector_size as u128)
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

        let dropout_masks = self.create_dropout_masks(self.word_vector_size, DROPOUT_PROBABILITY);

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

        let dropout_masks = self.create_dropout_masks(self.word_vector_size, 0.0);

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
    pub fn new(word_vector_size: usize) -> Self
    {
        let layers: Vec<_> =
            (0..LAYERS_AMOUNT).map(|_| Layer::new(word_vector_size)).collect();

        Self{
            layers,
            word_vector_size
        }
    }
}

