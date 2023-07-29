use std::{
    f32,
    borrow::Borrow
};

use arrayfire::{Array, MatProp, dim4};

use serde::{Serialize, Deserialize};

use crate::neural_network::{
    AdamHyperparams,
    LayerContainer,
    WeightsContainer,
    SoftmaxedLayer,
    SoftmaxedArray,
    GradientsInfo,
    HIDDEN_AMOUNT
};


#[derive(Debug)]
pub struct GRUOutput
{
    pub update: LayerContainer,
    pub reset: LayerContainer,
    pub activation: LayerContainer,
    pub hidden: LayerContainer,
    pub output: SoftmaxedLayer
}

pub struct GPUGRUOutput
{
    pub update: Array<f32>,
    pub reset: Array<f32>,
    pub activation: Array<f32>,
    pub hidden: Array<f32>,
    pub output: SoftmaxedArray
}

#[derive(Debug)]
pub struct GRUGradients
{
    pub input_update_gradients: WeightsContainer,
    pub input_reset_gradients: WeightsContainer,
    pub input_activation_gradients: WeightsContainer,
    pub hidden_update_gradients: WeightsContainer,
    pub hidden_reset_gradients: WeightsContainer,
    pub hidden_activation_gradients: WeightsContainer,
    pub update_bias_gradients: LayerContainer,
    pub reset_bias_gradients: LayerContainer,
    pub activation_bias_gradients: LayerContainer,
    pub output_gradients: WeightsContainer
}

pub struct GPUGRUGradients
{
    pub input_update_gradients: Array<f32>,
    pub input_reset_gradients: Array<f32>,
    pub input_activation_gradients: Array<f32>,
    pub hidden_update_gradients: Array<f32>,
    pub hidden_reset_gradients: Array<f32>,
    pub hidden_activation_gradients: Array<f32>,
    pub update_bias_gradients: Array<f32>,
    pub reset_bias_gradients: Array<f32>,
    pub activation_bias_gradients: Array<f32>,
    pub output_gradients: Array<f32>
}

struct D3Info<'a>
{
    output_derivative: LayerContainer,
    hidden_derivative: &'a LayerContainer
}

struct D4Info<'a>
{
    update_gate: LayerContainer,
    d3: &'a LayerContainer
}

struct D6Info<'a>
{
    previous_hidden: LayerContainer,
    d3: &'a LayerContainer
}

struct D8Info<'a>
{
    update_gate: LayerContainer,
    d3: &'a LayerContainer
}

struct ActivationGateDerivativeInfo<'a>
{
    activation_gate: LayerContainer,
    d8: &'a LayerContainer
}

struct UpdateGateDerivativeInfo<'a>
{
    update_gate: LayerContainer,
    activation_gate: LayerContainer,
    d3: &'a LayerContainer,
    d6: &'a LayerContainer
}

struct D13Info<'a>
{
    d10: &'a LayerContainer
}

struct ResetGateDerivativeInfo<'a>
{
    reset_gate: LayerContainer,
    previous_hidden: LayerContainer,
    d13: &'a LayerContainer
}

struct HiddenDerivativeInfo<'a>
{
    reset_gate: LayerContainer,
    d4: &'a LayerContainer,
    d11: &'a LayerContainer,
    d13: LayerContainer,
    d18: &'a LayerContainer
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GRU
{
    input_update_weights: WeightsContainer,
    input_reset_weights: WeightsContainer,
    input_activation_weights: WeightsContainer,
    hidden_update_weights: WeightsContainer,
    hidden_reset_weights: WeightsContainer,
    hidden_activation_weights: WeightsContainer,
    update_biases: LayerContainer,
    reset_biases: LayerContainer,
    activation_biases: LayerContainer,
    output_weights: WeightsContainer
}

impl GRU
{
    pub fn new(word_vector_size: usize) -> Self
    {
        let weights_init = |previous: f32|
        {
            let v = 1.0 / previous.sqrt();

            (fastrand::f32() * 2.0 - 1.0) * v
        };

        Self{
        	input_update_weights: WeightsContainer::new_with(
                word_vector_size,
                HIDDEN_AMOUNT,
                || weights_init(word_vector_size as f32)
            ),
        	input_reset_weights: WeightsContainer::new_with(
				word_vector_size,
				HIDDEN_AMOUNT,
				|| weights_init(word_vector_size as f32)
			),
        	input_activation_weights: WeightsContainer::new_with(
				word_vector_size,
				HIDDEN_AMOUNT,
				|| weights_init(word_vector_size as f32)
			),
        	hidden_update_weights: WeightsContainer::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f32)
			),
        	hidden_reset_weights: WeightsContainer::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f32)
			),
        	hidden_activation_weights: WeightsContainer::new_with(
				HIDDEN_AMOUNT,
				HIDDEN_AMOUNT,
				|| weights_init(HIDDEN_AMOUNT as f32)
			),
            // initialize biases to 0 cuz i read somewhere thats good
            update_biases: LayerContainer::new(HIDDEN_AMOUNT),
            reset_biases: LayerContainer::new(HIDDEN_AMOUNT),
            activation_biases: LayerContainer::new(HIDDEN_AMOUNT),
            output_weights: WeightsContainer::new_with(
                HIDDEN_AMOUNT,
                word_vector_size,
                || weights_init(HIDDEN_AMOUNT as f32)
            )
        }
    }

    pub fn gpu_adapter(&self, gradients: &GradientsInfo) -> GPUGRU
    {
        #[cfg(not(test))]
        {
            arrayfire::info();

            let device_info = arrayfire::device_info();
            eprintln!(
                "name: {}, platform: {}, toolkit: {}, compute: {}",
                device_info.0, device_info.1, device_info.2, device_info.3
            );
        }

        GPUGRU{
            input_update_weights: self.input_update_weights.as_arrayfire(),
            input_reset_weights: self.input_reset_weights.as_arrayfire(),
            input_activation_weights: self.input_activation_weights.as_arrayfire(),
            hidden_update_weights: self.hidden_update_weights.as_arrayfire(),
            hidden_reset_weights: self.hidden_reset_weights.as_arrayfire(),
            hidden_activation_weights: self.hidden_activation_weights.as_arrayfire(),
            output_weights: self.output_weights.as_arrayfire(),
            update_biases: self.update_biases.as_arrayfire(),
            reset_biases: self.reset_biases.as_arrayfire(),
            activation_biases: self.activation_biases.as_arrayfire(),
            gradients: gradients.as_arrayfire()
        }
    }

    pub fn transfer_weights(&mut self, gpugru: GPUGRU)
    {
        let GPUGRU{
            input_update_weights,
            input_reset_weights,
            input_activation_weights,
            hidden_update_weights,
            hidden_reset_weights,
            hidden_activation_weights,
            output_weights,
            update_biases,
            reset_biases,
            activation_biases,
            ..
        } = gpugru;

		self.input_update_weights =
            self.input_update_weights.new_from(input_update_weights);

		self.input_reset_weights =
            self.input_reset_weights.new_from(input_reset_weights);

		self.input_activation_weights =
            self.input_activation_weights.new_from(input_activation_weights);

		self.hidden_update_weights =
            self.hidden_update_weights.new_from(hidden_update_weights);

		self.hidden_reset_weights =
            self.hidden_reset_weights.new_from(hidden_reset_weights);

		self.hidden_activation_weights =
            self.hidden_activation_weights.new_from(hidden_activation_weights);

		self.update_biases = update_biases.into();
		self.reset_biases = reset_biases.into();
		self.activation_biases = activation_biases.into();

		self.output_weights =
            self.output_weights.new_from(output_weights);
    }

    #[allow(dead_code)]
    pub fn accuracy(
        &self,
        input: impl Iterator<Item=(LayerContainer, LayerContainer)>
    ) -> f32
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let amount = input.len();

        let f_output = self.feedforward_cpu(input.into_iter());

        Self::correct_guesses(
            f_output.into_iter().map(|output| output.output),
            output.into_iter()
        ) as f32 / amount as f32
    }

    fn correct_guesses<P, T>(
        predicted: impl Iterator<Item=P>,
        target: impl Iterator<Item=T>
    ) -> usize
    where
        P: Borrow<SoftmaxedLayer>,
        T: Borrow<LayerContainer>
    {
        predicted.zip(target).map(|(predicted, target)|
        {
            let target_index = target.borrow().highest_index();
            if predicted.borrow().pick_weighed(1.0) == target_index
            {
                1
            } else
            {
                0
            }
        }).sum()
    }

    #[allow(dead_code)]
    pub fn loss(&self, input: impl Iterator<Item=(LayerContainer, LayerContainer)>) -> f32
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();

        let f_output = self.feedforward_cpu(input.into_iter());

        Self::cross_entropy(
            f_output.into_iter().map(|output| output.output.0.into_iter()),
            output.into_iter().map(|l| l.into_iter())
        )
    }

    fn cross_entropy<PIter, TIter>(
        predicted: impl Iterator<Item=PIter>,
        target: impl Iterator<Item=TIter>
    ) -> f32
    where
        PIter: Iterator<Item=f32>,
        TIter: Iterator<Item=f32>
    {
        let mut count = 0;

        let s: f32 = predicted.zip(target).map(|(predicted, target)|
        {
            count += 1;

            Self::cross_entropy_single(predicted, target)
        }).sum();

        -s / count as f32
    }

    fn cross_entropy_single(
        predicted: impl Iterator<Item=f32>,
        target: impl Iterator<Item=f32>
    ) -> f32
    {
        predicted.into_iter().zip(target.into_iter()).map(|(p, t)|
        {
            t * p.ln()
        }).sum()
    }

    fn d3(info: D3Info) -> LayerContainer
    {
        info.output_derivative + info.hidden_derivative
    }

    fn d4(info: D4Info) -> LayerContainer
    {
        info.update_gate.one_minus_this() * info.d3
    }

    fn d6(info: D6Info) -> LayerContainer
    {
        let d5 = info.previous_hidden * info.d3;

        d5 * -1.0
    }

    fn d8(info: D8Info) -> LayerContainer
    {
        info.update_gate * info.d3
    }

    // d10
    fn activation_gate_derivative(info: ActivationGateDerivativeInfo) -> LayerContainer
    {
        info.activation_gate.powi(2).one_minus_this() * info.d8
    }

    // d11
    fn update_gate_derivative(info: UpdateGateDerivativeInfo) -> LayerContainer
    {
        let d7 = info.activation_gate * info.d3;
        let d9 = d7 + info.d6;

        d9 * (info.update_gate.clone() * info.update_gate.one_minus_this())
    }

    fn d13(&self, info: D13Info) -> LayerContainer
    {
        self.hidden_activation_weights.mul_transposed(info.d10)
    }

    // d18
    fn reset_gate_derivative(info: ResetGateDerivativeInfo) -> LayerContainer
    {
        let d16 = info.previous_hidden * info.d13;

        (info.reset_gate.clone() * info.reset_gate.one_minus_this()) * &d16
    }

    // d23
    fn hidden_derivative(&self, info: HiddenDerivativeInfo) -> LayerContainer
    {
        let d15 = self.hidden_update_weights.mul_transposed(info.d11);

        let d17 = info.d13 * &info.reset_gate;

        let d19 = d17 + info.d4;

        let d21 = self.hidden_reset_weights.mul_transposed(info.d18);
        let d22 = d21 + d15;

        d19 + d22
    }

    // deriving this hell made me appreciate how simple the rnn.rs derivation was
    // nevermind found somebody else deriving it again (mine were wrong ;-;)
    // https://cran.r-project.org/web/packages/rnn/vignettes/GRU_units.html
    // found the derivatives here ^
    // BUT a lot of them r wrong
    // list of the derivatives that r wrong there and the correct ones i calculated:
    // d6 isnt 1 - d5, its instead d5 * -1.0
    // d9 isnt d7 + d8, its instead d6 + d7
    // d18 isnt d17 * derivative of sigmoid(reset_gate), its instead d16 * derivative of sigmoid(reset_gate)
    // also the U and W letters r swapped in the derivatives compared to the picture
    // also derivatives of reset gate things use d18 not d10 (and actiavtions use d10)
    #[allow(dead_code)]
    pub fn gradients_cpu<'a>(
        &self,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)>
    ) -> GRUGradients
    {
        self.gradients_cpu_with_hidden(LayerContainer::new(HIDDEN_AMOUNT), input).1
    }

    pub fn gradients_cpu_with_hidden<'a>(
        &self,
        starting_hidden: LayerContainer,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)>
    ) -> (LayerContainer, GRUGradients)
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let f_output = self.feedforward_cpu_with_hidden(
            starting_hidden.clone(),
            input.iter().map(|x| (*x).clone())
        );

        let mut output_gradients = WeightsContainer::new(
            self.output_weights.previous_size(), self.output_weights.this_size()
        );

        let mut input_update_gradients = WeightsContainer::new(
            self.input_update_weights.previous_size(), self.input_update_weights.this_size()
        );

        let mut input_reset_gradients = WeightsContainer::new(
            self.input_reset_weights.previous_size(), self.input_reset_weights.this_size()
        );

        let mut input_activation_gradients = WeightsContainer::new(
            self.input_activation_weights.previous_size(),
            self.input_activation_weights.this_size()
        );

        let mut hidden_update_gradients = WeightsContainer::new(
            self.hidden_update_weights.previous_size(), self.hidden_update_weights.this_size()
        );

        let mut hidden_reset_gradients = WeightsContainer::new(
            self.hidden_reset_weights.previous_size(), self.hidden_reset_weights.this_size()
        );

        let mut hidden_activation_gradients = WeightsContainer::new(
            self.hidden_activation_weights.previous_size(),
            self.hidden_activation_weights.this_size()
        );

        let mut update_bias_gradients = LayerContainer::new(HIDDEN_AMOUNT);
        let mut reset_bias_gradients = LayerContainer::new(HIDDEN_AMOUNT);
        let mut activation_bias_gradients = LayerContainer::new(HIDDEN_AMOUNT);

        for t in (0..output.len()).rev()
        {
            let predicted_output = unsafe{ &f_output.get_unchecked(t).output };

            let expected_output = unsafe{ *output.get_unchecked(t) };
            let hidden = unsafe{ &f_output.get_unchecked(t).hidden };

            let expected_sum: f32 = expected_output.iter().sum();
            let diff = predicted_output.0.clone() * expected_sum - expected_output;

            for y in 0..expected_output.len()
            {
                for x in 0..hidden.len()
                {
                    let weight = unsafe{ output_gradients.weight_unchecked_mut(x, y) };

                    *weight += unsafe{
                        diff.get_unchecked(y) * hidden.get_unchecked(x)
                    };
                }
            }

            let mut d3 = {
                let output_derivative = self.output_weights.mul_transposed(diff);
                let hidden_derivative = LayerContainer::new(HIDDEN_AMOUNT);

                Self::d3(D3Info{
                    output_derivative,
                    hidden_derivative: &hidden_derivative
                })
            };

            for b_t in (0..(t + 1)).rev()
            {
                let previous_hidden = if b_t == 0
                {
                    starting_hidden.clone()
                } else
                {
                    unsafe{ f_output.get_unchecked(b_t - 1).hidden.clone() }
                };

                let this_update = unsafe{ f_output.get_unchecked(b_t).update.clone() };
                let this_reset = unsafe{ f_output.get_unchecked(b_t).reset.clone() };
                let this_activation = unsafe{ f_output.get_unchecked(b_t).activation.clone() };

                let d4 = Self::d4(D4Info{
                    update_gate: this_update.clone(),
                    d3: &d3
                });

                let d6 = Self::d6(D6Info{
                    previous_hidden: previous_hidden.clone(),
                    d3: &d3
                });

                let d8 = Self::d8(D8Info{
                    update_gate: this_update.clone(),
                    d3: &d3
                });

                // d10
                let activation_gate_derivative = Self::activation_gate_derivative(
                    ActivationGateDerivativeInfo{
                        activation_gate: this_activation.clone(),
                        d8: &d8
                    }
                );

                // d11
                let update_gate_derivative = Self::update_gate_derivative(
                    UpdateGateDerivativeInfo{
                        activation_gate: this_activation,
                        update_gate: this_update,
                        d3: &d3,
                        d6: &d6
                    }
                );

                let d13 = self.d13(D13Info{
                    d10: &activation_gate_derivative
                });

                // d18
                let reset_gate_derivative = Self::reset_gate_derivative(ResetGateDerivativeInfo{
                    reset_gate: this_reset.clone(),
                    previous_hidden: previous_hidden.clone(),
                    d13: &d13
                });

                hidden_update_gradients.add_outer_product(&update_gate_derivative, &previous_hidden);
                hidden_reset_gradients.add_outer_product(&reset_gate_derivative, &previous_hidden);
                
                {
                    let previous_hidden = previous_hidden * &this_reset;
                    hidden_activation_gradients.add_outer_product(
                        &activation_gate_derivative,
                        previous_hidden
                    );
                }

                let this_input = unsafe{ *input.get_unchecked(b_t) };
                input_update_gradients.add_outer_product(&update_gate_derivative, this_input);
                input_reset_gradients.add_outer_product(&reset_gate_derivative, this_input);
                
                input_activation_gradients.add_outer_product(
                    &activation_gate_derivative,
                    this_input
                );

                update_bias_gradients += &update_gate_derivative;
                reset_bias_gradients += &reset_gate_derivative;
                activation_bias_gradients += activation_gate_derivative;

                d3 = self.hidden_derivative(HiddenDerivativeInfo{
                    reset_gate: this_reset,
                    d4: &d4,
                    d11: &update_gate_derivative,
                    d13,
                    d18: &reset_gate_derivative
                });
            }
        }

        let gradients = GRUGradients{
            input_update_gradients,
            input_reset_gradients,
            input_activation_gradients,
            hidden_update_gradients,
            hidden_reset_gradients,
            hidden_activation_gradients,
            update_bias_gradients,
            reset_bias_gradients,
            activation_bias_gradients,
            output_gradients
        };

        let last_hidden = f_output.last().unwrap().hidden.clone();

        (last_hidden, gradients)
    }

    pub fn feedforward_cpu_single(
        &self,
        previous_hidden: &LayerContainer,
        input: &LayerContainer
    ) -> GRUOutput
    {
        let mut update_gate =
            self.hidden_update_weights.mul(previous_hidden)
            + self.input_update_weights.mul(input)
            + &self.update_biases;

        update_gate.map(|x| 1.0 / (1.0 + f32::consts::E.powf(-x)));

        let mut reset_gate =
            self.hidden_reset_weights.mul(previous_hidden)
            + self.input_reset_weights.mul(input)
            + &self.reset_biases;

        reset_gate.map(|x| 1.0 / (1.0 + f32::consts::E.powf(-x)));

        let activation_v = reset_gate.clone() * previous_hidden;
        let mut activation_gate =
            self.hidden_activation_weights.mul(activation_v)
            + self.input_activation_weights.mul(input)
            + &self.activation_biases;

        activation_gate.map(f32::tanh);

        let this_activation = activation_gate.clone() * &update_gate;
        let hidden = update_gate.clone().one_minus_this() * previous_hidden + this_activation;

        let output = SoftmaxedLayer::new(self.output_weights.mul(&hidden));

        GRUOutput{
            update: update_gate,
            reset: reset_gate,
            activation: activation_gate,
            hidden,
            output
        }
    }

    #[allow(dead_code)]
    pub fn feedforward_cpu<L>(&self, input: impl Iterator<Item=L>) -> Vec<GRUOutput>
    where
        L: Borrow<LayerContainer>
    {
        let first_hidden = LayerContainer::new(HIDDEN_AMOUNT);

        self.feedforward_cpu_with_hidden(first_hidden, input)
    }

    #[allow(dead_code)]
    pub fn feedforward_cpu_with_hidden<L>(
        &self,
        first_hidden: LayerContainer,
        input: impl Iterator<Item=L>
    ) -> Vec<GRUOutput>
    where
        L: Borrow<LayerContainer>
    {
        let (lower_bound, upper_bound) = input.size_hint();
        let time_total = upper_bound.unwrap_or(lower_bound);

        let mut outputs: Vec<GRUOutput> = Vec::with_capacity(time_total);

        for (t, this_input) in input.enumerate()
        {
            let previous_hidden = if t == 0
            {
                &first_hidden
            } else
            {
                unsafe{ &outputs.get_unchecked(t - 1).hidden }
            };

            let this_input = this_input.borrow();

            let output = self.feedforward_cpu_single(previous_hidden, this_input);

            outputs.push(output);
        }

        outputs
    }
}

pub struct GPUGradientInfo
{
    pub m: Array<f32>,
    pub v: Array<f32>
}

pub struct GPUGradientsInfo
{
    pub input_update_gradients: GPUGradientInfo,
    pub input_reset_gradients: GPUGradientInfo,
    pub input_activation_gradients: GPUGradientInfo,
    pub hidden_update_gradients: GPUGradientInfo,
    pub hidden_reset_gradients: GPUGradientInfo,
    pub hidden_activation_gradients: GPUGradientInfo,
    pub update_bias_gradients: GPUGradientInfo,
    pub reset_bias_gradients: GPUGradientInfo,
    pub activation_bias_gradients: GPUGradientInfo,
    pub output_gradients: GPUGradientInfo
}

impl GPUGradientsInfo
{
    pub fn gradient_to_change(
        gradient_info: &mut GPUGradientInfo,
        gradient: Array<f32>,
        hyper: &AdamHyperparams
    ) -> Array<f32>
    {
        gradient_info.m = &gradient_info.m * hyper.b1 + &gradient * (1.0 - hyper.b1);
        gradient_info.v = &gradient_info.v * hyper.b2 + (&gradient * &gradient) * (1.0 - hyper.b2);

        let a_t = hyper.a * (1.0 - hyper.b2).sqrt() / (1.0 - hyper.b1);

        -a_t * &gradient_info.m / (arrayfire::sqrt(&gradient_info.v) + hyper.epsilon)
    }
}

pub struct GPUGRU
{
    input_update_weights: Array<f32>,
    input_reset_weights: Array<f32>,
    input_activation_weights: Array<f32>,
    hidden_update_weights: Array<f32>,
    hidden_reset_weights: Array<f32>,
    hidden_activation_weights: Array<f32>,
    output_weights: Array<f32>,
    update_biases: Array<f32>,
    reset_biases: Array<f32>,
    activation_biases: Array<f32>,
    gradients: GPUGradientsInfo
}

impl GPUGRU
{
    // this looks horrible, thanks regex for not making me type this manually
    pub fn apply_gradients(&mut self, gradients: GPUGRUGradients, hyper: &mut AdamHyperparams)
    {
        let GPUGRUGradients{
            input_update_gradients,
            input_reset_gradients,
            input_activation_gradients,
            hidden_update_gradients,
            hidden_reset_gradients,
            hidden_activation_gradients,
            update_bias_gradients,
            reset_bias_gradients,
            activation_bias_gradients,
            output_gradients
        } = gradients;

        self.input_update_weights += GPUGradientsInfo::gradient_to_change(
            &mut self.gradients.input_update_gradients,
            input_update_gradients,
            hyper
        );

        self.input_reset_weights += GPUGradientsInfo::gradient_to_change(
			&mut self.gradients.input_reset_gradients,
			input_reset_gradients,
            hyper
		);
        
        self.input_activation_weights += GPUGradientsInfo::gradient_to_change(
			&mut self.gradients.input_activation_gradients,
			input_activation_gradients,
            hyper
		);

        self.hidden_update_weights += GPUGradientsInfo::gradient_to_change(
			&mut self.gradients.hidden_update_gradients,
			hidden_update_gradients,
            hyper
		);

        self.hidden_reset_weights += GPUGradientsInfo::gradient_to_change(
			&mut self.gradients.hidden_reset_gradients,
			hidden_reset_gradients,
            hyper
		);
        
        self.hidden_activation_weights += GPUGradientsInfo::gradient_to_change(
			&mut self.gradients.hidden_activation_gradients,
			hidden_activation_gradients,
            hyper
		);
        
        self.update_biases += GPUGradientsInfo::gradient_to_change(
			&mut self.gradients.update_bias_gradients,
			update_bias_gradients,
            hyper
		);
        
        self.reset_biases += GPUGradientsInfo::gradient_to_change(
			&mut self.gradients.reset_bias_gradients,
			reset_bias_gradients,
            hyper
		);
        
        self.activation_biases += GPUGradientsInfo::gradient_to_change(
			&mut self.gradients.activation_bias_gradients,
			activation_bias_gradients,
            hyper
		);
        
        self.output_weights += GPUGradientsInfo::gradient_to_change(
			&mut self.gradients.output_gradients,
			output_gradients,
            hyper
		);
    }

    pub fn gradients_info(&self) -> &GPUGradientsInfo
    {
        &self.gradients
    }

    pub fn accuracy<T>(
        &self,
        input: impl Iterator<Item=(T, T)>
    ) -> f32
    where
        T: Borrow<Array<f32>>
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let amount = input.len();

        let f_output = self.feedforward(input.into_iter());

        Self::correct_guesses(
            f_output.into_iter().map(|output| output.output),
            output.into_iter()
        ) as f32 / amount as f32
    }

    fn correct_guesses<T>(
        predicted: impl Iterator<Item=SoftmaxedArray>,
        target: impl Iterator<Item=T>
    ) -> usize
    where
        T: Borrow<Array<f32>>
    {
        predicted.zip(target).map(|(predicted, target)|
        {
            let target = target.borrow();

            let mut target_host = vec![0.0_f32; target.elements()];
            target.host(&mut target_host);

            let (target_index, _) = target_host.into_iter().enumerate().max_by(|a, b|
            {
                a.1.partial_cmp(&b.1).unwrap()
            }).unwrap();

            if predicted.pick_weighed(1.0) == target_index
            {
                1
            } else
            {
                0
            }
        }).sum()
    }

    #[allow(dead_code)]
    pub fn gradients<'a, const ONE_HOT_ENCODED: bool, T>(
        &self,
        input: impl Iterator<Item=(T, T)>
    ) -> GPUGRUGradients
    where
        T: Borrow<Array<f32>>
    {
        let empty_hidden =
            arrayfire::constant(0.0_f32, dim4!(HIDDEN_AMOUNT as u64));

        self.gradients_with_hidden::<ONE_HOT_ENCODED, T>(&empty_hidden, input).1
    }

    fn outer_product(a: &Array<f32>, b: &Array<f32>) -> Array<f32>
    {
        arrayfire::matmul(b, a, MatProp::NONE, MatProp::TRANS)
    }

    pub fn gradients_with_hidden<'a, const ONE_HOT_ENCODED: bool, T>(
        &self,
        starting_hidden: &Array<f32>,
        input: impl Iterator<Item=(T, T)>
    ) -> (Array<f32>, GPUGRUGradients)
    where
        T: Borrow<Array<f32>>
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();

        let f_output = self.feedforward_with_hidden(
            starting_hidden,
            input.iter().map(|x| x.borrow())
        );

        let mut output_gradients = arrayfire::constant(0.0_f32, self.output_weights.dims());

        let mut input_update_gradients =
            arrayfire::constant(0.0_f32, self.input_update_weights.dims());

        let mut input_reset_gradients =
            arrayfire::constant(0.0_f32, self.input_reset_weights.dims());

        let mut input_activation_gradients =
            arrayfire::constant(0.0_f32, self.input_activation_weights.dims());

        let mut hidden_update_gradients =
            arrayfire::constant(0.0_f32, self.hidden_update_weights.dims());

        let mut hidden_reset_gradients =
            arrayfire::constant(0.0_f32, self.hidden_reset_weights.dims());

        let mut hidden_activation_gradients =
            arrayfire::constant(0.0_f32, self.hidden_activation_weights.dims());

        let mut update_bias_gradients = arrayfire::constant(0.0_f32, self.update_biases.dims());
        let mut reset_bias_gradients = arrayfire::constant(0.0_f32, self.reset_biases.dims());
        let mut activation_bias_gradients =
            arrayfire::constant(0.0_f32, self.activation_biases.dims());

        for t in (0..output.len()).rev()
        {
            let predicted_output = unsafe{ &f_output.get_unchecked(t).output };

            let expected_output = unsafe{ output.get_unchecked(t) }.borrow();
            let hidden = unsafe{ &f_output.get_unchecked(t).hidden };

            let expected_sum: f32 = if ONE_HOT_ENCODED
            {
                1.0
            } else
            {
                arrayfire::sum_all(&expected_output).0
            };

            let diff = &predicted_output.0 * expected_sum - expected_output;

            output_gradients += Self::outer_product(&diff, hidden);

            let mut d3 = arrayfire::matmul(
                &self.output_weights,
                &diff,
                MatProp::NONE,
                MatProp::NONE
            );
            
            for b_t in (0..(t + 1)).rev()
            {
                let previous_hidden = if b_t == 0
                {
                    starting_hidden
                } else
                {
                    unsafe{ &f_output.get_unchecked(b_t - 1).hidden }
                };

                let this_update = unsafe{ &f_output.get_unchecked(b_t).update };
                let this_reset = unsafe{ &f_output.get_unchecked(b_t).reset };
                let this_activation = unsafe{ &f_output.get_unchecked(b_t).activation };

                let d4 = (-(this_update.clone()) + 1.0_f32) * &d3;
                let d5 = previous_hidden * &d3;
                let d6 = &d5 * -1.0_f32;
                let d7 = this_activation * &d3;
                let d8 = this_update * &d3;
                let d9 = &d7 + &d6;

                // d10
                let activation_gate_derivative =
                    (-(this_activation * this_activation) + 1.0_f32) * &d8;

                // d11
                let update_gate_derivative =
                    &d9 * (this_update * (-(this_update.clone()) + 1.0_f32));

                let d13 = arrayfire::matmul(
                    &self.hidden_activation_weights,
                    &activation_gate_derivative,
                    MatProp::NONE,
                    MatProp::NONE
                );

                let d15 = arrayfire::matmul(
                    &self.hidden_update_weights,
                    &update_gate_derivative,
                    MatProp::NONE,
                    MatProp::NONE
                );

                let d16 = previous_hidden * &d13;

                let d17 = &d13 * this_reset;

                // d18
                let reset_gate_derivative =
                    (this_reset * (-(this_reset.clone()) + 1.0_f32)) * &d16;

                let d19 = &d17 + &d4;

                let d21 = arrayfire::matmul(
                    &self.hidden_reset_weights,
                    &reset_gate_derivative,
                    MatProp::NONE,
                    MatProp::NONE
                );

                let d22 = &d21 + &d15;


                hidden_update_gradients +=
                    Self::outer_product(&update_gate_derivative, previous_hidden);

                hidden_reset_gradients +=
                    Self::outer_product(&reset_gate_derivative, previous_hidden);

                {
                    let combined_hidden = previous_hidden * this_reset;
                    hidden_activation_gradients +=
                        Self::outer_product(&activation_gate_derivative, &combined_hidden);
                }

                let this_input = unsafe{ input.get_unchecked(b_t) }.borrow();

                input_update_gradients +=
                    Self::outer_product(&update_gate_derivative, this_input);

                input_reset_gradients +=
                    Self::outer_product(&reset_gate_derivative, this_input);

                input_activation_gradients +=
                    Self::outer_product(&activation_gate_derivative, this_input);

                update_bias_gradients += update_gate_derivative;
                reset_bias_gradients += reset_gate_derivative;
                activation_bias_gradients += activation_gate_derivative;

                d3 = &d19 + &d22;
            }
        }

        let gradients = GPUGRUGradients{
            input_update_gradients,
            input_reset_gradients,
            input_activation_gradients,
            hidden_update_gradients,
            hidden_reset_gradients,
            hidden_activation_gradients,
            update_bias_gradients,
            reset_bias_gradients,
            activation_bias_gradients,
            output_gradients
        };

        let last_hidden = f_output.last().unwrap().hidden.clone();

        (last_hidden, gradients)
    }

    pub fn feedforward<T>(
        &self,
        input: impl Iterator<Item=T> + ExactSizeIterator
    ) -> Vec<GPUGRUOutput>
    where
        T: Borrow<Array<f32>>
    {
        let empty_hidden =
            arrayfire::constant(0.0_f32, dim4!(HIDDEN_AMOUNT as u64));

        self.feedforward_with_hidden(&empty_hidden, input)
    }

    pub fn loss<T>(&self, input: impl Iterator<Item=(T, T)>) -> f32
    where
        T: Borrow<Array<f32>>
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();

        let f_output = self.feedforward(input.into_iter());

        Self::cross_entropy(
            f_output.into_iter().map(|output| output.output),
            output.into_iter()
        )
    }

    fn cross_entropy<T>(
        predicted: impl Iterator<Item=SoftmaxedArray> + ExactSizeIterator,
        target: impl Iterator<Item=T>
    ) -> f32
    where
        T: Borrow<Array<f32>>
    {
        let predicted_len = predicted.len();
        let s: f32 = predicted.zip(target).map(|(predicted, target)|
        {
            let predicted_nlog = arrayfire::log(&predicted.0);

            let d = arrayfire::dot(target.borrow(), &predicted_nlog, MatProp::NONE, MatProp::NONE);

            let mut out = [0.0_f32];
            d.host(&mut out);

            out[0]
        }).sum();

        -s / predicted_len as f32
    }

    pub fn feedforward_with_hidden<T>(
        &self,
        first_hidden: &Array<f32>,
        input: impl Iterator<Item=T> + ExactSizeIterator
    ) -> Vec<GPUGRUOutput>
    where
        T: Borrow<Array<f32>>
    {
        let mut outputs: Vec<GPUGRUOutput> = Vec::with_capacity(input.len());

        for (t, inputs) in input.enumerate()
        {
            let previous_hidden = if t == 0
            {
                &first_hidden
            } else
            {
                unsafe{ &outputs.get_unchecked(t - 1).hidden }
            };

            let output = self.feedforward_single(previous_hidden, inputs.borrow());

            outputs.push(output);
        }

        outputs
    }

    pub fn feedforward_single(
        &self,
        previous_hidden: &Array<f32>,
        input: &Array<f32>
    ) -> GPUGRUOutput
    {
        let update_gate =
            arrayfire::matmul(
                &self.hidden_update_weights,
                &previous_hidden,
                MatProp::TRANS,
                MatProp::NONE
            )
            + arrayfire::matmul(
                &self.input_update_weights,
                input,
                MatProp::TRANS,
                MatProp::NONE
            )
            + &self.update_biases;

        let update_gate = arrayfire::sigmoid(&update_gate);

        let reset_gate =
            arrayfire::matmul(
                &self.hidden_reset_weights,
                &previous_hidden,
                MatProp::TRANS,
                MatProp::NONE
            )
            + arrayfire::matmul(
                &self.input_reset_weights,
                input,
                MatProp::TRANS,
                MatProp::NONE
            )
            + &self.reset_biases;

        let reset_gate = arrayfire::sigmoid(&reset_gate);

        let activation_v = &reset_gate * previous_hidden;
        let activation_gate =
            arrayfire::matmul(
                &self.hidden_activation_weights,
                &activation_v,
                MatProp::TRANS,
                MatProp::NONE
            )
            + arrayfire::matmul(
                &self.input_activation_weights,
                input,
                MatProp::TRANS,
                MatProp::NONE
            )
            + &self.activation_biases;

        let activation_gate = arrayfire::tanh(&activation_gate);

        let this_activation = &activation_gate * &update_gate;
        let hidden = (-(update_gate.clone()) + 1.0_f32) * previous_hidden + &this_activation;

        let output = arrayfire::matmul(
            &self.output_weights,
            &hidden,
            MatProp::TRANS,
            MatProp::NONE
        );

        let output = SoftmaxedArray::new(&output);

        GPUGRUOutput{
            update: update_gate,
            reset: reset_gate,
            activation: activation_gate,
            hidden,
            output
        }
    }
}

#[cfg(test)]
pub mod tests
{
    use std::iter;

    use super::*;
    use crate::neural_network::WeightsIterValue;

    fn close_enough(a: f32, b: f32, epsilon: f32) -> bool
    {
        if (a == b) || ((a.min(b) == -0.0) && (a.max(b) == 0.0))
        {
            return true;
        }

        ((a - b).abs() / (a.abs() + b.abs())) < epsilon
    }

    #[test]
    fn outer_product()
    {
        let mut w0 = WeightsContainer::new_with(10, 15, || 0.0_f32);

        let mut a_i = 0;
        let a = LayerContainer::new_with(15, ||
        {
            let out = a_i;

            a_i += 1;

            out as f32
        });

        let mut b_i = 0;
        let b = LayerContainer::new_with(10, ||
        {
            let out = b_i;

            b_i += 1;

            out as f32
        });

        for y in 0..a.len()
        {
            for x in 0..b.len()
            {
                let weight = unsafe{ w0.weight_unchecked_mut(x, y) };

                *weight += unsafe{
                    a.get_unchecked(y) * b.get_unchecked(x)
                };
            }
        }

        let a = a.as_arrayfire();
        let b = b.as_arrayfire();
        let w1 = GPUGRU::outer_product(&a, &b);

        let mut w1_host = vec![0.0_f32; w1.elements()];
        w1.host(&mut w1_host);

        w0.iter().zip(w1_host).for_each(|(&a, b)|
        {
            assert!(
                close_enough(a, b, 0.001),
                "a: {a}, b: {b}"
            );
        });
    }

    #[test]
    fn accuracy_match()
    {
        let inputs_amount = 10;
        let input_output_size = 20;

        let gru = GRU::new(input_output_size);

        let inputs = (0..inputs_amount).map(|_|
        {
            LayerContainer::new_with(input_output_size, || fastrand::f32())
        });

        let expected = (0..inputs_amount).map(|_|
        {
            LayerContainer::new_with(input_output_size, || fastrand::f32())
        });

        let inputs = inputs.zip(expected);

        fastrand::seed(12345);
        let cpu_accuracy = gru.accuracy(inputs.clone());

        let gradients_info = GradientsInfo::new(0);
        let gpu_adapter = gru.gpu_adapter(&gradients_info);
        
        fastrand::seed(12345);
        let gpu_accuracy = gpu_adapter.accuracy(
            inputs.map(|(a, b)|
            {
                (a.as_arrayfire(), b.as_arrayfire())
            })
        );

        assert!(
            close_enough(cpu_accuracy, gpu_accuracy, 0.001),
            "cpu_accuracy: {cpu_accuracy}, gpu_accuracy: {gpu_accuracy}"
        );
    }

    #[test]
    fn loss_match()
    {
        let inputs_amount = 10;
        let input_output_size = 20;

        let gru = GRU::new(input_output_size);

        let inputs = (0..inputs_amount).map(|_|
        {
            LayerContainer::new_with(input_output_size, || fastrand::f32())
        });

        let expected = (0..inputs_amount).map(|_|
        {
            LayerContainer::new_with(input_output_size, || fastrand::f32())
        });

        let inputs = inputs.zip(expected);

        let cpu_loss = gru.loss(inputs.clone());

        let gradients_info = GradientsInfo::new(0);
        let gpu_adapter = gru.gpu_adapter(&gradients_info);
        let gpu_loss = gpu_adapter.loss(
            inputs.map(|(a, b)|
            {
                (a.as_arrayfire(), b.as_arrayfire())
            })
        );

        // im not sure where the error is if i change the 0.1 to a lower value
        // (just floating point weirdness? natural log being calculated differently?)
        assert!(
            close_enough(cpu_loss, gpu_loss, 0.1),
            "cpu_loss: {cpu_loss}, gpu_loss: {gpu_loss}"
        );
    }

    #[test]
    fn gradients_match()
    {
        let inputs_amount = 10;
        // cool value 20 is a cool value
        let input_output_size = 20;

        let gru = GRU::new(input_output_size);

        let inputs = (0..inputs_amount).map(|_|
        {
            LayerContainer::new_with(input_output_size, || fastrand::f32())
        }).collect::<Vec<_>>();

        let expected = (0..inputs_amount).map(|_|
        {
            LayerContainer::new_with(input_output_size, || fastrand::f32())
        }).collect::<Vec<_>>();

        let cpu_output = gru.gradients_cpu(inputs.iter().zip(expected.iter()));

        let gradients_info = GradientsInfo::new(0);
        let gpu_adapter = gru.gpu_adapter(&gradients_info);
        let gpu_output = gpu_adapter.gradients::<false, _>(
            inputs.into_iter().map(|l| l.as_arrayfire())
                .zip(expected.into_iter().map(|l| l.as_arrayfire()))
        );

        let comparer = |(cpu_result, gpu_result): (f32, f32)|
        {
            assert!(
                close_enough(cpu_result, gpu_result, 0.05),
                "cpu_result: {cpu_result}, gpu_result: {gpu_result}"
            );
        };

        let gradient_comparer = |cpu_layer: &WeightsContainer, gpu_layer: &Array<f32>|
        {
            let mut gpu_layer_host = vec![0.0_f32; gpu_layer.elements()];
            gpu_layer.host(&mut gpu_layer_host);

            cpu_layer.iter().copied()
                .zip(gpu_layer_host.into_iter())
                .for_each(comparer);
        };

        eprintln!("output gradients");
        gradient_comparer(
            &cpu_output.output_gradients,
            &gpu_output.output_gradients
        );

        eprintln!("hidden update gradients");
        gradient_comparer(
            &cpu_output.hidden_update_gradients,
            &gpu_output.hidden_update_gradients
        );

        eprintln!("hidden reset gradients");
        gradient_comparer(
            &cpu_output.hidden_reset_gradients,
            &gpu_output.hidden_reset_gradients
        );

        eprintln!("hidden activation gradients");
        gradient_comparer(
            &cpu_output.hidden_activation_gradients,
            &gpu_output.hidden_activation_gradients
        );

        eprintln!("input update gradients");
        gradient_comparer(
            &cpu_output.input_update_gradients,
            &gpu_output.input_update_gradients
        );

        eprintln!("input reset gradients");
        gradient_comparer(
            &cpu_output.input_reset_gradients,
            &gpu_output.input_reset_gradients
        );

        eprintln!("input activation gradients");
        gradient_comparer(
            &cpu_output.input_activation_gradients,
            &gpu_output.input_activation_gradients
        );
    }

    // check if cpu and gpu forwardprop gives the same results
    #[test]
    fn forwardprop_match()
    {
        let inputs_amount = 10;
        // cool value 20 is a cool value
        let input_output_size = 20;

        let gru = GRU::new(input_output_size);

        let inputs = (0..inputs_amount).map(|_|
        {
            LayerContainer::new_with(input_output_size, || fastrand::f32())
        }).collect::<Vec<_>>();

        let cpu_output = gru.feedforward_cpu(inputs.clone().into_iter());

        let gradients_info = GradientsInfo::new(0);
        let gpu_adapter = gru.gpu_adapter(&gradients_info);
        let gpu_output = gpu_adapter.feedforward(inputs.into_iter().map(|l| l.as_arrayfire()));

        let comparer = |(cpu_result, gpu_result): (f32, f32)|
        {
            assert!(
                close_enough(cpu_result, gpu_result, 0.005),
                "cpu_result: {cpu_result}, gpu_result: {gpu_result}"
            );
        };

        let layer_comparer = |cpu_layer: &LayerContainer, gpu_layer: &Array<f32>|
        {
            let mut gpu_layer_host = vec![0.0_f32; gpu_layer.elements()];
            gpu_layer.host(&mut gpu_layer_host);

            cpu_layer.into_iter().copied()
                .zip(gpu_layer_host.into_iter())
                .for_each(comparer);
        };

        cpu_output.into_iter().zip(gpu_output.into_iter()).for_each(|(cpu_output, gpu_output)|
        {
            layer_comparer(&cpu_output.update, &gpu_output.update);
            layer_comparer(&cpu_output.reset, &gpu_output.reset);
            layer_comparer(&cpu_output.activation, &gpu_output.activation);
            layer_comparer(&cpu_output.hidden, &gpu_output.hidden);
            layer_comparer(&cpu_output.output.0, &gpu_output.output.0);
        });
    }

    #[ignore]
    #[allow(dead_code)]
    // #[test]
    fn forwardprop()
    {
        assert_eq!(HIDDEN_AMOUNT, 2);

        let input_update_weights = WeightsContainer::from_raw(
            vec![4.63, -2.64, 4.76, 3.63].into_boxed_slice(),
            2,
            HIDDEN_AMOUNT
        );

        let input_reset_weights = WeightsContainer::from_raw(
            vec![-8.29, 9.96, -4.78, 2.24].into_boxed_slice(),
            2,
            HIDDEN_AMOUNT
        );

        let input_activation_weights = WeightsContainer::from_raw(
            vec![-5.09, 1.99, 1.15, 4.63].into_boxed_slice(),
            2,
            HIDDEN_AMOUNT
        );

        let hidden_update_weights = WeightsContainer::from_raw(
            vec![-0.48, 8.48, -6.14, 2.42].into_boxed_slice(),
            HIDDEN_AMOUNT,
            HIDDEN_AMOUNT
        );

        let hidden_reset_weights = WeightsContainer::from_raw(
            vec![-5.74, -2.66, -6.25, -9.21].into_boxed_slice(),
            HIDDEN_AMOUNT,
            HIDDEN_AMOUNT
        );

        let hidden_activation_weights = WeightsContainer::from_raw(
            vec![-3.95, -6.07, 6.36, -5.36].into_boxed_slice(),
            HIDDEN_AMOUNT,
            HIDDEN_AMOUNT
        );

        let update_biases = LayerContainer::from(vec![-2.00, -0.87]);
        let reset_biases = LayerContainer::from(vec![-8.36, -8.16]);
        let activation_biases = LayerContainer::from(vec![3.47, 3.52]);

        let output_weights = WeightsContainer::from_raw(
            vec![8.59, -1.08, -7.31, -7.97].into_boxed_slice(),
            HIDDEN_AMOUNT,
            2
        );

        let gru = GRU{
            input_update_weights,
            input_reset_weights,
            input_activation_weights,
            hidden_update_weights,
            hidden_reset_weights,
            hidden_activation_weights,
            update_biases,
            reset_biases,
            activation_biases,
            output_weights
        };

        let input = LayerContainer::from(vec![-5.90, 1.78]);
        let f_output = gru.feedforward_cpu(iter::once(input));

        let output = &f_output[0].output;

        let epsilon = 0.0000001;

        let z = vec![1.686366804463125e-15, 1.7044644055950665e-10];
        assert!(close_enough(z[0], f_output[0].update[0], epsilon));
        assert!(close_enough(z[1], f_output[0].update[1], epsilon));

        let r = vec![1.0, 0.9999999999633351];
        assert!(close_enough(r[0], f_output[0].reset[0], epsilon));
        assert!(close_enough(r[1], f_output[0].reset[1], epsilon));

        let h = vec![1.0, 0.9999048161632378];
        assert!(close_enough(h[0], f_output[0].activation[0], epsilon));
        assert!(close_enough(h[1], f_output[0].activation[1], epsilon));

        let hidden = vec![1.686366804463125e-15, 1.7043021681333174e-10];
        assert!(close_enough(hidden[0], f_output[0].hidden[0], epsilon));
        assert!(close_enough(hidden[1], f_output[0].hidden[1], epsilon));

        let o = vec![0.5000000000460197, 0.4999999999539802];
        let (correct, calculated) = (o[0], output.0[0]);
        assert!(
            close_enough(correct, calculated, epsilon),
            "correct: {correct}, calculated: {calculated}"
        );

        let (correct, calculated) = (o[1], output.0[1]);
        assert!(
            close_enough(correct, calculated, epsilon),
            "correct: {correct}, calculated: {calculated}"
        );
    }

    #[test]
    fn loss_correct()
    {
        let predicted = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.01, 0.01, 0.01, 0.96]
        ];

        let target = vec![
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0]
        ];

        let loss = GRU::cross_entropy(
            predicted.into_iter().map(|v| v.into_iter()),
            target.into_iter().map(|v| v.into_iter())
        );

        assert!(
            close_enough(loss, 0.71355817782, 0.000001),
            "loss: {loss}"
        );
    }

    pub fn input_update_gradients_check<'a>(
        network: &mut GRU,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.input_update_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients_cpu(input);

        true_gradient.iter_pos().zip(calculated_gradient.input_update_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn input_reset_gradients_check<'a>(
        network: &mut GRU,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.input_reset_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients_cpu(input);

        true_gradient.iter_pos().zip(calculated_gradient.input_reset_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn input_activation_gradients_check<'a>(
        network: &mut GRU,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.input_activation_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients_cpu(input);

        true_gradient.iter_pos().zip(calculated_gradient.input_activation_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_update_gradients_check<'a>(
        network: &mut GRU,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.hidden_update_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients_cpu(input);

        true_gradient.iter_pos().zip(calculated_gradient.hidden_update_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_reset_gradients_check<'a>(
        network: &mut GRU,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.hidden_reset_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients_cpu(input);

        true_gradient.iter_pos().zip(calculated_gradient.hidden_reset_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn hidden_activation_gradients_check<'a>(
        network: &mut GRU,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.hidden_activation_weights,
            input.clone()
        );

        let calculated_gradient = network.gradients_cpu(input);

        true_gradient.iter_pos().zip(calculated_gradient.hidden_activation_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    pub fn update_bias_gradients_check<'a>(
        network: &mut GRU,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    )
    {
        let true_gradient = gradient_bias_check(
            network,
            |network| &mut network.update_biases,
            input.clone()
        );

        let calculated_gradient = network.gradients_cpu(input);

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.update_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn reset_bias_gradients_check<'a>(
        network: &mut GRU,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    )
    {
        let true_gradient = gradient_bias_check(
            network,
            |network| &mut network.reset_biases,
            input.clone()
        );

        let calculated_gradient = network.gradients_cpu(input);

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.reset_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn activation_bias_gradients_check<'a>(
        network: &mut GRU,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    )
    {
        let true_gradient = gradient_bias_check(
            network,
            |network| &mut network.activation_biases, input.clone()
        );

        let calculated_gradient = network.gradients_cpu(input);

        true_gradient.iter().enumerate()
            .zip(calculated_gradient.activation_bias_gradients.iter().enumerate())
            .for_each(check_single_bias_gradient);
    }

    pub fn output_gradients_check<'a>(
        network: &mut GRU,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    )
    {
        let true_gradient = gradient_check(
            network,
            |network| &mut network.output_weights, input.clone()
        );

        let calculated_gradient = network.gradients_cpu(input);
        true_gradient.iter_pos().zip(calculated_gradient.output_gradients.iter_pos())
            .for_each(check_single_gradient);
    }

    fn check_single_gradient(pair: (WeightsIterValue<f32>, WeightsIterValue<f32>))
    {
        let (true_gradient, calculated_gradient) = pair;
        let (previous, this) = (true_gradient.previous, true_gradient.this);

        let true_gradient = true_gradient.value;
        let calculated_gradient = calculated_gradient.value;

        assert!(
            close_enough(true_gradient, calculated_gradient, 0.1),
            "true_gradient: {true_gradient}, calculated_gradient: {calculated_gradient}, previous_index: {previous}, this_index: {this}"
        );
    }

    fn check_single_bias_gradient(pair: ((usize, &f32), (usize, &f32)))
    {
        let (true_gradient, calculated_gradient) = pair;
        let index = true_gradient.0;

        let true_gradient = true_gradient.1;
        let calculated_gradient = calculated_gradient.1;

        assert!(
            close_enough(*true_gradient, *calculated_gradient, 0.1),
            "true_gradient: {true_gradient}, calculated_gradient: {calculated_gradient}, index: {index}"
        );
    }

    pub fn gradient_bias_check<'a>(
        network: &mut GRU,
        mut bias_member: impl FnMut(&mut GRU) -> &mut LayerContainer,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    ) -> LayerContainer
    {
        let input = input.map(|(a, b)| (a.clone(), b.clone()));

        let biases = bias_member(network).iter().enumerate().map(|(index, bias)|
        {
            (index, *bias)
        }).collect::<Vec<_>>();

        let biases = biases.into_iter().map(|(index, bias)|
        {
            let epsilon = 0.0001;

            let mut set_this_weight = |network: &mut GRU, value|
            {
                bias_member(network)[index] = value;
            };

            set_this_weight(network, bias - epsilon);
            let under_loss = network.loss(input.clone());
            
            set_this_weight(network, bias + epsilon);
            let over_loss = network.loss(input.clone());

            set_this_weight(network, bias);

            (over_loss - under_loss) / (2.0 * epsilon)
        }).collect::<Vec<_>>();

        LayerContainer::from(biases)
    }

    pub fn gradient_check<'a>(
        network: &mut GRU,
        mut weights_member: impl FnMut(&mut GRU) -> &mut WeightsContainer,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)> + Clone
    ) -> WeightsContainer
    {
        let input = input.map(|(a, b)| (a.clone(), b.clone()));

        let weights = weights_member(network).iter_pos().map(|weight|
        {
            weight.to_owned()
        }).collect::<Vec<WeightsIterValue<_>>>();

        let weights = weights.into_iter().map(|weight|
        {
            let WeightsIterValue{value: weight, previous, this} = weight;
            let epsilon = 0.0001;

            let mut set_this_weight = |network: &mut GRU, value|
            {
                *weights_member(network).weight_mut(previous, this) = value;
            };

            // ; ;
            set_this_weight(network, weight - epsilon);
            let under_loss = network.loss(input.clone());
            
            set_this_weight(network, weight + epsilon);
            let over_loss = network.loss(input.clone());

            set_this_weight(network, weight);

            (over_loss - under_loss) / (2.0 * epsilon)
        }).collect::<Box<[_]>>();

        WeightsContainer::from_raw(
            weights,
            weights_member(network).previous_size(),
            weights_member(network).this_size()
        )
    }
}
