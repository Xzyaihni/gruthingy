use std::{
    f32,
    borrow::Borrow,
    ops::{Div, DivAssign, AddAssign}
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
    pub output: LayerContainer
}

pub struct GPUGRUOutput
{
    pub update: Array<f32>,
    pub reset: Array<f32>,
    pub activation: Array<f32>,
    pub hidden: Array<f32>,
    pub output: Array<f32>
}

impl GPUGRUOutput
{
    pub fn join(&mut self, value: GPUGRUOutput)
    {
		self.update = arrayfire::join(1, &self.update, &value.update);
		self.reset = arrayfire::join(1, &self.reset, &value.reset);
		self.activation = arrayfire::join(1, &self.activation, &value.activation);
		self.hidden = arrayfire::join(1, &self.hidden, &value.hidden);
		self.output = arrayfire::join(1, &self.output, &value.output);
    }
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

impl DivAssign<f32> for GRUGradients
{
    fn div_assign(&mut self, rhs: f32)
    {
		self.input_update_gradients /= rhs;
		self.input_reset_gradients /= rhs;
		self.input_activation_gradients /= rhs;
		self.hidden_update_gradients /= rhs;
		self.hidden_reset_gradients /= rhs;
		self.hidden_activation_gradients /= rhs;
		self.update_bias_gradients /= rhs;
		self.reset_bias_gradients /= rhs;
		self.activation_bias_gradients /= rhs;
		self.output_gradients /= rhs;
    }
}

impl AddAssign for GRUGradients
{
    fn add_assign(&mut self, rhs: Self)
    {
		self.input_update_gradients += rhs.input_update_gradients;
		self.input_reset_gradients += rhs.input_reset_gradients;
		self.input_activation_gradients += rhs.input_activation_gradients;
		self.hidden_update_gradients += rhs.hidden_update_gradients;
		self.hidden_reset_gradients += rhs.hidden_reset_gradients;
		self.hidden_activation_gradients += rhs.hidden_activation_gradients;
		self.update_bias_gradients += rhs.update_bias_gradients;
		self.reset_bias_gradients += rhs.reset_bias_gradients;
		self.activation_bias_gradients += rhs.activation_bias_gradients;
		self.output_gradients += rhs.output_gradients;
    }
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

impl Div<f32> for GPUGRUGradients
{
    type Output = Self;

    fn div(self, rhs: f32) -> Self
    {
        Self{
            input_update_gradients: self.input_update_gradients / rhs,
            input_reset_gradients: self.input_reset_gradients / rhs,
            input_activation_gradients: self.input_activation_gradients / rhs,
            hidden_update_gradients: self.hidden_update_gradients / rhs,
            hidden_reset_gradients: self.hidden_reset_gradients / rhs,
            hidden_activation_gradients: self.hidden_activation_gradients / rhs,
            update_bias_gradients: self.update_bias_gradients / rhs,
            reset_bias_gradients: self.reset_bias_gradients / rhs,
            activation_bias_gradients: self.activation_bias_gradients / rhs,
            output_gradients: self.output_gradients / rhs
        }
    }
}

impl AddAssign for GPUGRUGradients
{
    fn add_assign(&mut self, rhs: Self)
    {
		self.input_update_gradients += rhs.input_update_gradients;
		self.input_reset_gradients += rhs.input_reset_gradients;
		self.input_activation_gradients += rhs.input_activation_gradients;
		self.hidden_update_gradients += rhs.hidden_update_gradients;
		self.hidden_reset_gradients += rhs.hidden_reset_gradients;
		self.hidden_activation_gradients += rhs.hidden_activation_gradients;
		self.update_bias_gradients += rhs.update_bias_gradients;
		self.reset_bias_gradients += rhs.reset_bias_gradients;
		self.activation_bias_gradients += rhs.activation_bias_gradients;
		self.output_gradients += rhs.output_gradients;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GRU
{
	pub input_update_weights: WeightsContainer,
	pub input_reset_weights: WeightsContainer,
	pub input_activation_weights: WeightsContainer,
	pub hidden_update_weights: WeightsContainer,
	pub hidden_reset_weights: WeightsContainer,
	pub hidden_activation_weights: WeightsContainer,
	pub update_biases: LayerContainer,
	pub reset_biases: LayerContainer,
	pub activation_biases: LayerContainer,
	pub output_weights: WeightsContainer
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
        arrayfire::set_device(0);
        arrayfire::set_cublas_mode(arrayfire::CublasMathMode::TENSOR_OP);

        #[cfg(not(test))]
        {
            arrayfire::info();

            let (name, platform, toolkit, compute) = arrayfire::device_info();
            eprintln!(
                "name: {}, platform: {}, toolkit: {}, compute: {}",
                name, platform, toolkit, compute
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
            self.input_update_weights.new_from(&input_update_weights);

		self.input_reset_weights =
            self.input_reset_weights.new_from(&input_reset_weights);

		self.input_activation_weights =
            self.input_activation_weights.new_from(&input_activation_weights);

		self.hidden_update_weights =
            self.hidden_update_weights.new_from(&hidden_update_weights);

		self.hidden_reset_weights =
            self.hidden_reset_weights.new_from(&hidden_reset_weights);

		self.hidden_activation_weights =
            self.hidden_activation_weights.new_from(&hidden_activation_weights);

		self.update_biases = update_biases.into();
		self.reset_biases = reset_biases.into();
		self.activation_biases = activation_biases.into();

		self.output_weights =
            self.output_weights.new_from(&output_weights);
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
        P: Borrow<LayerContainer>,
        T: Borrow<LayerContainer>
    {
        predicted.zip(target).map(|(predicted, target)|
        {
            let target_index = target.borrow().highest_index();
            if SoftmaxedLayer::pick_weighed_associated(predicted.borrow(), 1.0) == target_index
            {
                1
            } else
            {
                0
            }
        }).sum()
    }


    pub fn loss(
        &self,
        input: impl Iterator<Item=(LayerContainer, LayerContainer)> + ExactSizeIterator
    ) -> f32
    {
        let amount = input.len();

        self.loss_unscaled(input) / amount as f32
    }

    pub fn loss_unscaled(&self, input: impl Iterator<Item=(LayerContainer, LayerContainer)>) -> f32
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();

        let f_output = self.feedforward_cpu(input.into_iter());

        Self::cross_entropy(
            f_output.into_iter().map(|output| output.output.into_iter()),
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
        let s: f32 = predicted.zip(target).map(|(predicted, target)|
        {
            Self::cross_entropy_single(predicted, target)
        }).sum();

        -s
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

    #[inline(always)]
    pub fn zeroed_gradients(&self) -> GRUGradients
    {
        let output_gradients = WeightsContainer::new(
            self.output_weights.previous_size(), self.output_weights.this_size()
        );

        let input_update_gradients = WeightsContainer::new(
            self.input_update_weights.previous_size(), self.input_update_weights.this_size()
        );

        let input_reset_gradients = WeightsContainer::new(
            self.input_reset_weights.previous_size(), self.input_reset_weights.this_size()
        );

        let input_activation_gradients = WeightsContainer::new(
            self.input_activation_weights.previous_size(),
            self.input_activation_weights.this_size()
        );

        let hidden_update_gradients = WeightsContainer::new(
            self.hidden_update_weights.previous_size(), self.hidden_update_weights.this_size()
        );

        let hidden_reset_gradients = WeightsContainer::new(
            self.hidden_reset_weights.previous_size(), self.hidden_reset_weights.this_size()
        );

        let hidden_activation_gradients = WeightsContainer::new(
            self.hidden_activation_weights.previous_size(),
            self.hidden_activation_weights.this_size()
        );

        let update_bias_gradients = LayerContainer::new(HIDDEN_AMOUNT);
        let reset_bias_gradients = LayerContainer::new(HIDDEN_AMOUNT);
        let activation_bias_gradients = LayerContainer::new(HIDDEN_AMOUNT);

        GRUGradients{
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
        }
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
    pub fn gradients_cpu<'a, const ONE_HOT_ENCODED: bool>(
        &self,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)>
    ) -> GRUGradients
    {
        self.gradients_cpu_with_hidden::<ONE_HOT_ENCODED>(
            LayerContainer::new(HIDDEN_AMOUNT),
            input
        )
    }

    pub fn gradients_cpu_with_hidden<'a, const ONE_HOT_ENCODED: bool>(
        &self,
        starting_hidden: LayerContainer,
        input: impl Iterator<Item=(&'a LayerContainer, &'a LayerContainer)>
    ) -> GRUGradients
    {
        let (input, output): (Vec<_>, Vec<_>) = input.unzip();
        let f_output = self.feedforward_cpu_with_hidden(
            &starting_hidden,
            input.iter().map(|v| *v)
        );

        let mut gradients = self.zeroed_gradients();

        for t in (0..output.len()).rev()
        {
            let predicted_output = unsafe{ &f_output.get_unchecked(t).output };

            let expected_output = unsafe{ *output.get_unchecked(t) };
            let hidden = unsafe{ &f_output.get_unchecked(t).hidden };

            let expected_sum: f32 = if ONE_HOT_ENCODED
            {
                1.0
            } else
            {
                expected_output.iter().sum()
            };

            let diff = predicted_output * expected_sum - expected_output;

            gradients.output_gradients.add_outer_product(&diff, hidden);

            let mut d3 = self.output_weights.mul_transposed(diff);

            for b_t in (0..(t + 1)).rev()
            {
                let previous_hidden = if b_t == 0
                {
                    &starting_hidden
                } else
                {
                    unsafe{ &f_output.get_unchecked(b_t - 1).hidden }
                };

                let this_update = unsafe{ &f_output.get_unchecked(b_t).update };
                let this_reset = unsafe{ &f_output.get_unchecked(b_t).reset };
                let this_activation = unsafe{ &f_output.get_unchecked(b_t).activation };

                let d4 = this_update.clone().one_minus_this() * &d3;

                let d5 = previous_hidden * &d3;
                let d6 = d5 * -1.0;
                let d7 = this_activation * &d3;
                let d8 = this_update * &d3;
                let d9 = d7 + d6;

                // d10
                let activation_gate_derivative =
                    this_activation.clone().powi(2).one_minus_this() * d8;

                // d11
                let update_gate_derivative =
                    &d9 * (this_update * this_update.clone().one_minus_this());

                let d13 =
                    self.hidden_activation_weights.mul_transposed(&activation_gate_derivative);

                let d15 = self.hidden_update_weights.mul_transposed(&update_gate_derivative);
                let d16 = previous_hidden * &d13;
                let d17 = d13 * this_reset;

                // d18
                let reset_gate_derivative =
                    (this_reset * this_reset.clone().one_minus_this()) * &d16;

                let d19 = d17 + d4;

                let d21 = self.hidden_reset_weights.mul_transposed(&reset_gate_derivative);
                let d22 = d21 + d15;

                gradients.hidden_update_gradients
                    .add_outer_product(&update_gate_derivative, previous_hidden);

                gradients.hidden_reset_gradients
                    .add_outer_product(&reset_gate_derivative, previous_hidden);
                
                {
                    let previous_hidden = previous_hidden * this_reset;
                    gradients.hidden_activation_gradients
                        .add_outer_product(&activation_gate_derivative, previous_hidden);
                }

                let this_input = unsafe{ *input.get_unchecked(b_t) };

                gradients.input_update_gradients
                    .add_outer_product(&update_gate_derivative, this_input);

                gradients.input_reset_gradients
                    .add_outer_product(&reset_gate_derivative, this_input);
                
                gradients.input_activation_gradients
                    .add_outer_product(&activation_gate_derivative, this_input);

                gradients.update_bias_gradients += &update_gate_derivative;
                gradients.reset_bias_gradients += &reset_gate_derivative;
                gradients.activation_bias_gradients += activation_gate_derivative;

                let d23 = d19 + d22;

                d3 = d23;
            }
        }

        gradients
    }

    #[inline(always)]
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

        let activation_v = &reset_gate * previous_hidden;
        let mut activation_gate =
            self.hidden_activation_weights.mul(activation_v)
            + self.input_activation_weights.mul(input)
            + &self.activation_biases;

        activation_gate.map(f32::tanh);

        let this_activation = &activation_gate * &update_gate;
        let hidden = update_gate.clone().one_minus_this() * previous_hidden + this_activation;

        let output = SoftmaxedLayer::softmax(self.output_weights.mul(&hidden));

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

        self.feedforward_cpu_with_hidden(&first_hidden, input)
    }

    #[allow(dead_code)]
    pub fn feedforward_cpu_with_hidden<L>(
        &self,
        first_hidden: &LayerContainer,
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
                first_hidden
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

        let a_t = hyper.a * hyper.one_minus_b2_t.sqrt() / hyper.one_minus_b1_t;

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

        hyper.advance_time();
    }

    pub fn gradients_info(&self) -> &GPUGradientsInfo
    {
        &self.gradients
    }

    #[inline(always)]
    pub fn zeroed_gradients(&self) -> GPUGRUGradients
    {
        let zeroed = |dims|
        {
            arrayfire::constant(0.0_f32, dims)
        };

        GPUGRUGradients{
            output_gradients: zeroed(self.output_weights.dims()),
            input_update_gradients: zeroed(self.input_update_weights.dims()),
            input_reset_gradients: zeroed(self.input_reset_weights.dims()),
            input_activation_gradients: zeroed(self.input_activation_weights.dims()),
            hidden_update_gradients: zeroed(self.hidden_update_weights.dims()),
            hidden_reset_gradients: zeroed(self.hidden_reset_weights.dims()),
            hidden_activation_gradients: zeroed(self.hidden_activation_weights.dims()),
            update_bias_gradients: zeroed(self.update_biases.dims()),
            reset_bias_gradients: zeroed(self.reset_biases.dims()),
            activation_bias_gradients: zeroed(self.activation_biases.dims())
        }
    }

    // i could write this properly but im too lezy to rewrite code so its gonna be slow
    pub fn accuracy(
        &self,
        input: (Array<f32>, Array<f32>)
    ) -> f32
    {
        let (input, output) = input;
        let amount = input.dims()[1] as f32;

        let f_output = self.feedforward(&input);

        Self::correct_guesses(
            f_output.output,
            output
        ) as f32 / amount
    }

    fn correct_guesses(
        predicted: Array<f32>,
        target: Array<f32>
    ) -> usize
    {
        let amount = target.dims()[1] as i64;

        (0..amount).map(|i| arrayfire::col(&predicted, i))
            .zip((0..amount).map(|i| arrayfire::col(&target, i)))
            .map(|(predicted, target)|
        {
            let target = target.borrow();

            let mut target_host = vec![0.0_f32; target.elements()];
            target.host(&mut target_host);

            let (target_index, _) = target_host.into_iter().enumerate().max_by(|a, b|
            {
                a.1.partial_cmp(&b.1).unwrap()
            }).unwrap();

            if SoftmaxedArray::pick_weighed_associated(&predicted, 1.0) == target_index
            {
                1
            } else
            {
                0
            }
        }).sum()
    }

    #[allow(dead_code)]
    pub fn gradients<const ONE_HOT_ENCODED: bool>(
        &self,
        input: (Array<f32>, Array<f32>)
    ) -> GPUGRUGradients
    {
        let empty_hidden =
            arrayfire::constant(0.0_f32, dim4!(HIDDEN_AMOUNT as u64));

        self.gradients_with_hidden::<ONE_HOT_ENCODED>(&empty_hidden, input)
    }

    #[inline(always)]
    fn outer_product(a: &Array<f32>, b: &Array<f32>) -> Array<f32>
    {
        arrayfire::matmul(b, a, MatProp::NONE, MatProp::TRANS)
    }

    pub fn gradients_with_hidden<const ONE_HOT_ENCODED: bool>(
        &self,
        starting_hidden: &Array<f32>,
        input: (Array<f32>, Array<f32>)
    ) -> GPUGRUGradients
    {
        let (input, output) = input;

        let f_output = self.feedforward_with_hidden(
            starting_hidden.clone(),
            &input
        );

        let mut gradients = self.zeroed_gradients();

        for t in (0..(output.dims()[1] as i64)).rev()
        {
            let predicted_output = arrayfire::col(&f_output.output, t);

            let expected_output = arrayfire::col(&output, t);
            let hidden = arrayfire::col(&f_output.hidden, t);

            let expected_sum: f32 = if ONE_HOT_ENCODED
            {
                1.0
            } else
            {
                arrayfire::sum_all(&expected_output).0
            };

            let diff = &predicted_output * expected_sum - expected_output;

            gradients.output_gradients += Self::outer_product(&diff, &hidden);

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
                    starting_hidden.clone()
                } else
                {
                    arrayfire::col(&f_output.hidden, b_t - 1)
                };

                let this_update = arrayfire::col(&f_output.update, b_t);
                let this_reset = arrayfire::col(&f_output.reset, b_t);
                let this_activation = arrayfire::col(&f_output.activation, b_t);
                let this_input = arrayfire::col(&input, b_t);

                let one_minus_this_update = -(this_update.clone()) + 1.0_f32;

                let d4 = &one_minus_this_update * &d3;
                let d5 = &previous_hidden * &d3;
                let d6 = d5 * -1.0_f32;
                let d7 = &this_activation * &d3;
                let d8 = &this_update * d3;
                let d9 = d7 + d6;

                // d10
                let activation_gate_derivative =
                    (-(&this_activation * &this_activation) + 1.0_f32) * &d8;

                // d11
                let update_gate_derivative =
                    d9 * (this_update * one_minus_this_update);

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

                let d16 = &previous_hidden * &d13;

                let d17 = d13 * &this_reset;

                // d18
                let reset_gate_derivative =
                    (&this_reset * (-(this_reset.clone()) + 1.0_f32)) * &d16;

                let d19 = d17 + d4;

                let d21 = arrayfire::matmul(
                    &self.hidden_reset_weights,
                    &reset_gate_derivative,
                    MatProp::NONE,
                    MatProp::NONE
                );

                let d22 = d21 + d15;


                gradients.hidden_update_gradients +=
                    Self::outer_product(&update_gate_derivative, &previous_hidden);

                gradients.hidden_reset_gradients +=
                    Self::outer_product(&reset_gate_derivative, &previous_hidden);

                {
                    let combined_hidden = previous_hidden * this_reset;
                    gradients.hidden_activation_gradients +=
                        Self::outer_product(&activation_gate_derivative, &combined_hidden);
                }

                gradients.input_update_gradients +=
                    Self::outer_product(&update_gate_derivative, &this_input);

                gradients.input_reset_gradients +=
                    Self::outer_product(&reset_gate_derivative, &this_input);

                gradients.input_activation_gradients +=
                    Self::outer_product(&activation_gate_derivative, &this_input);

                gradients.update_bias_gradients += update_gate_derivative;
                gradients.reset_bias_gradients += reset_gate_derivative;
                gradients.activation_bias_gradients += activation_gate_derivative;

                d3 = d19 + d22;
            }
        }

        gradients
    }

    pub fn feedforward(
        &self,
        input: &Array<f32>
    ) -> GPUGRUOutput
    {
        let empty_hidden =
            arrayfire::constant(0.0_f32, dim4!(HIDDEN_AMOUNT as u64));

        self.feedforward_with_hidden(empty_hidden, input)
    }

    pub fn loss(&self, input: (Array<f32>, Array<f32>)) -> f32
    {
        let (input, output) = input;

        let f_output = self.feedforward(&input);

        Self::cross_entropy(
            f_output.output,
            output
        )
    }

    fn cross_entropy(
        predicted: Array<f32>,
        target: Array<f32>
    ) -> f32
    {
        let predicted_nlog = arrayfire::log(&predicted);

        let amount = target.dims()[1] as i64;
        let s: f32 = (0..amount).map(|i|
        {
            let d = arrayfire::dot(
                &arrayfire::col(&target, i),
                &arrayfire::col(&predicted_nlog, i),
                MatProp::NONE,
                MatProp::NONE
            );

            let mut out = [0.0_f32];
            d.host(&mut out);

            out[0]
        }).sum();

        -s / amount as f32
    }

    #[inline(always)]
    pub fn feedforward_with_hidden(
        &self,
        first_hidden: Array<f32>,
        input: &Array<f32>
    ) -> GPUGRUOutput
    {
        let mut outputs: Option<GPUGRUOutput> = None;

        for t in 0..(input.dims()[1] as i64)
        {
            let previous_hidden = if t == 0
            {
                first_hidden.clone()
            } else
            {
                arrayfire::col(&outputs.as_ref().unwrap().hidden, t - 1)
            };

            let inputs = arrayfire::col(input, t);
            let output = self.feedforward_single(&previous_hidden, &inputs);

            if outputs.is_none()
            {
                outputs = Some(output);
            } else
            {
                outputs.as_mut().map(|outputs| outputs.join(output));
            }
        }

        outputs.unwrap()
    }

    #[inline(always)]
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

        let output = SoftmaxedArray::softmax(&output);

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
    use crate::neural_network::{WeightsIterValue, InputOutputIter, input_output_associated};

    fn close_enough(a: f32, b: f32, epsilon: f32) -> bool
    {
        if (a == b) || ((a.min(b) == -0.0) && (a.max(b) == 0.0))
        {
            return true;
        }

        ((a - b).abs() / (a.abs() + b.abs())) < epsilon
    }

    fn close_enough_abs(a: f32, b: f32, epsilon: f32) -> bool
    {
        if (a == b) || ((a.min(b) == -0.0) && (a.max(b) == 0.0))
        {
            return true;
        }

        (a - b).abs() < epsilon
    }

    #[test]
    fn outer_product()
    {
        let prev = 5500;
        let this = 5750;

        let mut w0 = WeightsContainer::new_with(prev, this, || 0.0_f32);

        let mut a_i = 0;
        let a = LayerContainer::new_with(this, ||
        {
            let out = a_i;

            a_i += 1;

            out as f32
        });

        let mut b_i = 0;
        let b = LayerContainer::new_with(prev, ||
        {
            let out = b_i;

            b_i += 1;

            out as f32
        });

        eprintln!("cpu outer began");
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
        eprintln!("cpu outer ended");

        let a = a.as_arrayfire();
        let b = b.as_arrayfire();

        eprintln!("gpu outer began");
        let w1 = GPUGRU::outer_product(&a, &b);
        eprintln!("gpu outer ended");

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

    #[ignore]
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

        let gradients_info = GradientsInfo::new(inputs_amount);
        let gpu_adapter = gru.gpu_adapter(&gradients_info);
        
        let inputs = input_output_associated::join_array(inputs.map(|(a, b)|
            {
                (a.as_arrayfire(), b.as_arrayfire())
            })
        );

        fastrand::seed(12345);
        let gpu_accuracy = gpu_adapter.accuracy(inputs);

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

        let gradients_info = GradientsInfo::new(inputs_amount);
        let gpu_adapter = gru.gpu_adapter(&gradients_info);

        let inputs = input_output_associated::join_array(inputs.map(|(a, b)|
            {
                (a.as_arrayfire(), b.as_arrayfire())
            })
        );

        let gpu_loss = gpu_adapter.loss(inputs);

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

        eprintln!("cpu gradient began");
        let cpu_output = gru.gradients_cpu::<false>(inputs.iter().zip(expected.iter()));
        eprintln!("cpu gradient ended");

        let gradients_info = GradientsInfo::new(input_output_size);
        let gpu_adapter = gru.gpu_adapter(&gradients_info);

        let inputs = inputs.into_iter().map(|l| l.as_arrayfire()).collect::<Vec<_>>();
        let expected = expected.into_iter().map(|l| l.as_arrayfire()).collect::<Vec<_>>();

        let inputs = input_output_associated::join_array(inputs.iter().zip(expected.iter()));

        eprintln!("gpu gradient began");
        let gpu_output = gpu_adapter.gradients::<false>(inputs);
        eprintln!("gpu gradient ended");

        let comparer = |(cpu_result, gpu_result): (f32, f32)|
        {
            assert!(
                close_enough_abs(cpu_result, gpu_result, 0.0001),
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

        let gradients_info = GradientsInfo::new(inputs_amount);
        let gpu_adapter = gru.gpu_adapter(&gradients_info);

        let inputs = input_output_associated::join_array(
            inputs.into_iter().map(|l| l.as_arrayfire()).map(|l| (l.clone(), l))
        );

        let gpu_output = gpu_adapter.feedforward(&inputs.0);

        let comparer = |(cpu_result, gpu_result): (f32, f32)|
        {
            assert!(
                close_enough(cpu_result, gpu_result, 0.005),
                "cpu_result: {cpu_result}, gpu_result: {gpu_result}"
            );
        };

        let layer_comparer = |cpu_layer: &LayerContainer, gpu_layer: Array<f32>|
        {
            let mut gpu_layer_host = vec![0.0_f32; gpu_layer.elements()];
            gpu_layer.host(&mut gpu_layer_host);

            cpu_layer.into_iter().copied()
                .zip(gpu_layer_host.into_iter())
                .for_each(comparer);
        };

        cpu_output.into_iter().enumerate().for_each(|(i, cpu_output)|
        {
            let g = |a| arrayfire::col(a, i as i64);

            layer_comparer(&cpu_output.update, g(&gpu_output.update));
            layer_comparer(&cpu_output.reset, g(&gpu_output.reset));
            layer_comparer(&cpu_output.activation, g(&gpu_output.activation));
            layer_comparer(&cpu_output.hidden, g(&gpu_output.hidden));
            layer_comparer(&cpu_output.output, g(&gpu_output.output));
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

        let epsilon = 0.00001;

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
        let (correct, calculated) = (o[0], output[0]);
        assert!(
            close_enough(correct, calculated, epsilon),
            "correct: {correct}, calculated: {calculated}"
        );

        let (correct, calculated) = (o[1], output[1]);
        assert!(
            close_enough(correct, calculated, epsilon),
            "correct: {correct}, calculated: {calculated}"
        );
    }

    #[ignore]
    #[allow(dead_code)]
    // #[test]
    fn backprop_smol()
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

        let input = vec![
            LayerContainer::from(vec![-5.90, 1.78]),
            LayerContainer::from(vec![-1.23, 4.56]),
            LayerContainer::from(vec![9.99, -1.02]),
            LayerContainer::from(vec![0.01, 10.0])
        ];

        let input = InputOutputIter::new(input.iter());

        let output = gru.gradients_cpu::<false>(input);

        let single_match = |correct, calculated|
        {
            assert!(
                close_enough_abs(calculated, correct, 0.00001),
                "correct: {correct}, calculated: {calculated}"
            );
        };

        let layer_match = |correct: [f32; 4], calculated: WeightsContainer|
        {
            correct.iter().zip(calculated.iter()).for_each(|(correct, calculated)|
            {
                single_match(*correct, *calculated);
            });
        };

        let bias_match = |correct: [f32; 2], calculated: LayerContainer|
        {
            correct.iter().zip(calculated.iter()).for_each(|(correct, calculated)|
            {
                single_match(*correct, *calculated);
            });
        };

        layer_match([
            5.416323632651452e-8,
            -2.008012961152585e-7,
            0.0004701980745839663,
            -0.0017432416518287585,
        ], output.input_update_gradients);

        layer_match([
            -3.2916534783598936e-62,
            3.3608473953224137e-63,
            -4.376070777317816e-36,
            1.6512401517820905e-35,
        ], output.input_reset_gradients);

        layer_match([
            1.2026719991784244e-23,
            -4.4586864359785484e-23,
            -2.815569864494384e-9,
            2.8823767498944457e-10,
        ], output.input_activation_gradients);

        layer_match([
            -7.425982242345576e-23,
            -7.50496131841157e-18,
            -6.446831266464838e-19,
            -6.515396576798509e-14,
        ], output.hidden_update_gradients);

        layer_match([
            -8.867181401246852e-72,
            -3.2947707711316803e-63,
            2.1594354702353152e-47,
            8.02152580715418e-39,
        ], output.hidden_reset_gradients);

        layer_match([
            -1.6488992976193476e-38,
            -1.6658391023269182e-33,
            -5.1807365201048425e-64,
            -1.496633901915441e-39,
        ], output.hidden_activation_gradients);

        bias_match([
            -4.4035341451853064e-8,
            -0.000382287763564198,
        ], output.update_bias_gradients);

        bias_match([
            -3.29494842678668e-63,
            3.630957026725113e-36,
        ], output.reset_bias_gradients);

        bias_match([
            -9.777821131531905e-24,
            -2.8080948683310396e-10,
        ], output.activation_bias_gradients);

        layer_match([
            0.008777106286423676,
            -1.0378464082942112,
            -0.008777106286423085,
            1.037846408294211,
        ], output.output_gradients);
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

        let amount = target.len() as f32;

        let loss = GRU::cross_entropy(
            predicted.into_iter().map(|v| v.into_iter()),
            target.into_iter().map(|v| v.into_iter())
        ) / amount;

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

        let calculated_gradient = network.gradients_cpu::<false>(input);

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

        let calculated_gradient = network.gradients_cpu::<false>(input);

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

        let calculated_gradient = network.gradients_cpu::<false>(input);

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

        let calculated_gradient = network.gradients_cpu::<false>(input);

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

        let calculated_gradient = network.gradients_cpu::<false>(input);

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

        let calculated_gradient = network.gradients_cpu::<false>(input);

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

        let calculated_gradient = network.gradients_cpu::<false>(input);

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

        let calculated_gradient = network.gradients_cpu::<false>(input);

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

        let calculated_gradient = network.gradients_cpu::<false>(input);

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

        let calculated_gradient = network.gradients_cpu::<false>(input);
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
            let under_loss = network.loss_unscaled(input.clone());
            
            set_this_weight(network, bias + epsilon);
            let over_loss = network.loss_unscaled(input.clone());

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
            let under_loss = network.loss_unscaled(input.clone());
            
            set_this_weight(network, weight + epsilon);
            let over_loss = network.loss_unscaled(input.clone());

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
