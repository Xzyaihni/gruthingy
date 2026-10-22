use serde::{Serialize, Deserialize};

use super::{
    DECAY_FUNCTION,
    TensorPtr,
    DiffTensorPtr,
    LayerType,
    OperationsRecorder
};


#[allow(dead_code)]
pub enum DecayFunction
{
    Power,
    Division
}

impl DecayFunction
{
    fn decay(&self, value: f32, t: i32) -> f32
    {
        match self
        {
            DecayFunction::Power => value.powi(t),
            DecayFunction::Division => value / t as f32
        }
    }
}

pub trait NewableLayer
{
    fn new(previous: usize, current: usize) -> Self;
}

impl NewableLayer for ()
{
    fn new(_previous_size: usize, _this_size: usize) -> Self {}
}

pub struct OptimizerInfo<O>
{
    pub weights: Vec<DiffTensorPtr>,
    pub gradient_clip: Option<f32>,
    pub optimizer: O
}

impl<O: Optimizer> OptimizerInfo<O>
{
    pub fn add_operations(&self, recorder: &mut OperationsRecorder)
    {
        let batch_size = self.weights.first().map(|x| recorder.tensor_shape(x.as_value()).batch_size).unwrap_or(1);

        let batch_size_recip = recorder.set_new_value((batch_size as f32).recip()).as_value();

        self.weights.iter().for_each(|weight|
        {
            let gradient = weight.as_gradient().expect("all weights must have a gradient");
            let weight = weight.as_value();

            debug_assert_eq!(recorder.tensor_shape(weight).batch_size, batch_size);

            let average_gradient = recorder.mul_scalar_no_diff(gradient, batch_size_recip);
            recorder.name_tensor(average_gradient, "average_gradient");

            let clipped_gradient = self.gradient_clip.map(|gradient_clip|
            {
                let gradient_clip = recorder.set_new_value(gradient_clip).as_value();

                recorder.cap_magnitude(average_gradient, gradient_clip)
            }).unwrap_or(average_gradient);

            recorder.name_tensor(clipped_gradient, "clipped_gradient");

            let new_weight = self.optimizer.clone().add_operations(recorder, clipped_gradient);
            recorder.name_tensor(new_weight, "new_weight_override");

            recorder.store_tensor_until_end(new_weight);
            recorder.memory.allow_after_coalesce_use(new_weight);

            recorder.force_same_memory(new_weight, weight);
        });
    }
}

pub trait Optimizer: Default + Clone
{
    type WeightParam<T>;

    fn add_operations(self, recorder: &mut OperationsRecorder, gradient: TensorPtr) -> TensorPtr;

    fn set_learning_rate(&mut self, learning_rate: f32);
}

impl Optimizer for ()
{
    type WeightParam<T> = ();

    fn add_operations(self, _recorder: &mut OperationsRecorder, _gradient: TensorPtr) -> TensorPtr
    {
        unreachable!()
    }

    fn set_learning_rate(&mut self, _learning_rate: f32)
    {
        unreachable!()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sgd
{
    pub learning_rate: f32
}

impl Default for Sgd
{
    fn default() -> Self
    {
        Self{learning_rate: 0.001}
    }
}

impl Optimizer for Sgd
{
    type WeightParam<T> = ();

    fn add_operations(self, recorder: &mut OperationsRecorder, gradient: TensorPtr) -> TensorPtr
    {
        recorder.sgd_optimizer(gradient, self)
    }

    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.learning_rate = learning_rate;
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSignGradientInfo<T>
{
    m: T
}

impl NewableLayer for PowerSignGradientInfo<LayerType>
{
    fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            m: LayerType::new(this_size, previous_size)
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSign
{
    pub b1: f32,
    pub learning_rate: f32
}

impl Default for PowerSign
{
    fn default() -> Self
    {
        Self{
            b1: 0.9,
            learning_rate: 0.1
        }
    }
}

impl Optimizer for PowerSign
{
    type WeightParam<T> = PowerSignGradientInfo<T>;

    fn add_operations(self, recorder: &mut OperationsRecorder, gradient: TensorPtr) -> TensorPtr
    {
        todo!()
        /*let t = 1; let increase_me = ();

        gradient_info.m = gradient_info.m.mul_scalar(self.b1).add(gradient.mul_scalar(1.0 - self.b1).as_ref());

        let decay = DECAY_FUNCTION.decay(self.learning_rate, self.t);

        let mut this = gradient.signum().mul_componentwise(gradient_info.m.signum().as_ref()).mul_scalar(decay);
        this.as_mut().exp_inplace();

        this.mul_componentwise(gradient.as_ref())*/
    }

    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.learning_rate = learning_rate;
    }
}

const uncomment_me: () = ();
/*#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamXGradientInfo
{
    m: LayerType,
    v: LayerType,
    v_hat: Option<LayerType>
}

impl NewableLayer for AdamXGradientInfo
{
    fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            m: LayerType::new(this_size, previous_size),
            v: LayerType::new(this_size, previous_size),
            v_hat: None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamX
{
    pub a: f32,
    pub b1: f32,
    pub b2: f32,
    pub epsilon: f32,
    pub t: i32
}

impl Optimizer for AdamX
{
    type WeightParam = AdamXGradientInfo;

    fn new() -> Self
    {
        Self{
            a: 0.001,
            b1: 0.9,
            b2: 0.999,
            epsilon: 1e-8,
            t: 1
        }
    }

    fn gradient_to_change(
        &self,
        gradient_info: &mut Self::WeightParam,
        gradient: LayerType
    ) -> LayerType
    {
        let b1_t = DECAY_FUNCTION.decay(self.b1, self.t);
        let one_minus_b1_t = 1.0 - b1_t;

        gradient_info.m = gradient_info.m.mul_scalar(b1_t).add(gradient.mul_scalar(one_minus_b1_t).as_ref());
        gradient_info.v = gradient_info.v.mul_scalar(self.b2).add(gradient.mul_componentwise(gradient.as_ref()).mul_scalar(1.0 - self.b2).as_ref());

        if let Some(v_hat) = gradient_info.v_hat.as_mut()
        {
            let one_minus_b1_tlast: f32 = 1.0 - DECAY_FUNCTION.decay(self.b1, self.t - 1);

            let lhs = (one_minus_b1_t).powi(2) / (one_minus_b1_tlast).powi(2);

            let new_v_hat = v_hat.mul_scalar(lhs);

            *v_hat = new_v_hat.max(gradient_info.v.as_ref());
        } else
        {
            gradient_info.v_hat = Some(gradient_info.v.clone());
        }

        // it can be a / t.sqrt() but this is fine
        let a_t = self.a;

        let rhs = gradient_info.v_hat.as_ref().unwrap().sqrt_plus(self.epsilon);

        gradient_info.m.mul_scalar(a_t).div_componentwise(rhs.as_ref())
    }

    fn advance_time(&mut self)
    {
        self.t += 1;
    }

    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.a = learning_rate;
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamGradientInfo
{
    m: LayerType,
    v: LayerType
}

impl NewableLayer for AdamGradientInfo
{
    fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            m: LayerType::new(this_size, previous_size),
            v: LayerType::new(this_size, previous_size)
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam
{
    pub a: f32,
    pub b1: f32,
    pub b2: f32,
    pub epsilon: f32,
    pub t: i32
}

impl Optimizer for Adam
{
    type WeightParam = AdamGradientInfo;

    fn new() -> Self
    {
        Self{
            a: 0.001,
            b1: 0.9,
            b2: 0.999,
            epsilon: 1e-8,
            t: 1
        }
    }

    fn gradient_to_change(
        &self,
        gradient_info: &mut Self::WeightParam,
        gradient: LayerType
    ) -> LayerType
    {
        let one_minus_b1_t = 1.0 - DECAY_FUNCTION.decay(self.b1, self.t);
        let one_minus_b2_t: f32 = 1.0 - DECAY_FUNCTION.decay(self.b2, self.t);

        gradient_info.m = gradient_info.m.mul_scalar(self.b1).add(gradient.mul_scalar(1.0 - self.b1).as_ref());
        gradient_info.v = gradient_info.v.mul_scalar(self.b2).add(gradient.mul_componentwise(gradient.as_ref()).mul_scalar(1.0 - self.b2).as_ref());

        let a_t = self.a * one_minus_b2_t.sqrt() / one_minus_b1_t;

        (&gradient_info.m.mul_scalar(a_t)).div_componentwise(gradient_info.v.sqrt_plus(self.epsilon).as_ref())
    }

    fn advance_time(&mut self)
    {
        self.t += 1;
    }

    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.a = learning_rate;
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn adam_correct()
    {
        let mut old_weight = vec![3.21, 7.65];

        let mut m = vec![0.0, 0.0];
        let mut v = vec![0.0, 0.0];

        let mut g = vec![3.1_f32, -0.8_f32];

        let mut t = 1;

        for _ in 0..2
        {
            let a = 0.001;
            let b1 = 0.9;
            let b2 = 0.999;

            let epsilon = 10e-8;

            let adam_g = {
                let mut gradient_info = AdamGradientInfo{
                    m: LayerType::from_boxed(m.clone().into_boxed_slice(), 2, 1),
                    v: LayerType::from_boxed(v.clone().into_boxed_slice(), 2, 1)
                };

                let gradient = LayerType::from_boxed(g.clone().into_boxed_slice(), 2, 1);

                let adam = Adam{
                    a,
                    b1,
                    b2,
                    epsilon,
                    t
                };

                let change = adam.gradient_to_change(
                    &mut gradient_info,
                    gradient.clone()
                );

                LayerType::from_boxed(old_weight.clone().into_boxed_slice(), 2, 1).add(change.as_ref())
            };

            m = vec![
                b1 * m[0] + (1.0 - b1) * g[0],
                b1 * m[1] + (1.0 - b1) * g[1]
            ];

            v = vec![
                b2 * v[0] + (1.0 - b2) * g[0].powi(2),
                b2 * v[1] + (1.0 - b2) * g[1].powi(2)
            ];

            let m_hat = vec![
                m[0] / (1.0 - b1.powi(t)),
                m[1] / (1.0 - b1.powi(t))
            ];

            let v_hat = vec![
                v[0] / (1.0 - b2.powi(t)),
                v[1] / (1.0 - b2.powi(t))
            ];

            let new_weight = vec![
                old_weight[0] + a * m_hat[0] / (v_hat[0].sqrt() + epsilon),
                old_weight[1] + a * m_hat[1] / (v_hat[1].sqrt() + epsilon)
            ];

            if t == 1
            {
                let mut adam_g = adam_g.as_vec().into_iter();
                assert_eq!(new_weight[0], adam_g.next().unwrap());
                assert_eq!(new_weight[1], adam_g.next().unwrap());
            } else
            {
                let mut adam_g = adam_g.as_vec().into_iter();
                assert_eq!(new_weight[0], adam_g.next().unwrap());
                assert_eq!(new_weight[1], adam_g.next().unwrap());
            }

            t += 1;

            old_weight = new_weight;
            g = vec![
                g[0] - 1.1,
                g[1] - 2.34
            ];
        }
    }
}*/
