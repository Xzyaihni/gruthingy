use serde::{Serialize, Deserialize};

use super::{
    LayerType,
    DECAY_FUNCTION
};


pub trait NewableLayer
{
    fn new(previous: usize, current: usize) -> Self;
}

impl NewableLayer for ()
{
    fn new(_previous_size: usize, _this_size: usize) -> Self {}
}

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
            m: LayerType::new(previous_size, this_size),
            v: LayerType::new(previous_size, this_size)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
            m: LayerType::new(previous_size, this_size),
            v: LayerType::new(previous_size, this_size),
            v_hat: None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSignGradientInfo
{
    m: LayerType
}

impl NewableLayer for PowerSignGradientInfo
{
    fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self{
            m: LayerType::new(previous_size, this_size)
        }
    }
}

pub trait Optimizer
{
    type WeightParam;

    fn new() -> Self;

    fn gradient_to_change(
        &self,
        gradient_info: &mut Self::WeightParam,
        gradient: LayerType
    ) -> LayerType;

    fn advance_time(&mut self);
    fn set_learning_rate(&mut self, learning_rate: f32);
}

impl Optimizer for ()
{
    type WeightParam = ();

    fn new() -> Self {}

    fn gradient_to_change(
        &self,
        _gradient_info: &mut Self::WeightParam,
        _gradient: LayerType
    ) -> LayerType
    {
        unreachable!()
    }

    fn advance_time(&mut self)
    {
        unreachable!()
    }

    fn set_learning_rate(&mut self, _learning_rate: f32)
    {
        unreachable!()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Sgd
{
    learning_rate: f32
}

impl Optimizer for Sgd
{
    type WeightParam = ();

    fn new() -> Self
    {
        Self{learning_rate: 0.001}
    }

    fn gradient_to_change(
        &self,
        _gradient_info: &mut Self::WeightParam,
        gradient: LayerType
    ) -> LayerType
    {
        gradient * self.learning_rate
    }

    fn advance_time(&mut self) {}
    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.learning_rate = learning_rate;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PowerSign
{
    pub b1: f32,
    pub learning_rate: f32,
    pub t: i32
}

impl Optimizer for PowerSign
{
    type WeightParam = PowerSignGradientInfo;

    fn new() -> Self
    {
        Self{
            b1: 0.9,
            learning_rate: 0.1,
            t: 1
        }
    }

    fn gradient_to_change(
        &self,
        gradient_info: &mut Self::WeightParam,
        gradient: LayerType
    ) -> LayerType
    {
        gradient_info.m = &gradient_info.m * self.b1 + &gradient * (1.0 - self.b1);

        let decay = DECAY_FUNCTION.decay(self.learning_rate, self.t);

        let mut this = gradient.signum() * gradient_info.m.signum() * decay;
        this.exp();

        this * gradient
    }

    fn advance_time(&mut self)
    {
        self.t += 1;
    }

    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.learning_rate = learning_rate;
    }
}

#[derive(Debug, Serialize, Deserialize)]
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

        gradient_info.m = &gradient_info.m * b1_t + &gradient * one_minus_b1_t;
        gradient_info.v = &gradient_info.v * self.b2 + (&gradient * &gradient) * (1.0 - self.b2);

        if let Some(v_hat) = gradient_info.v_hat.as_mut()
        {
            let one_minus_b1_tlast = 1.0 - DECAY_FUNCTION.decay(self.b1, self.t - 1);

            let lhs = (one_minus_b1_t).powi(2) / (one_minus_b1_tlast).powi(2);

            let mut new_v_hat = &*v_hat * lhs;
            new_v_hat.max(&gradient_info.v);

            *v_hat = new_v_hat;
        } else
        {
            gradient_info.v_hat = Some(gradient_info.v.clone());
        }

        // it can be a / t.sqrt() but this is fine
        let a_t = self.a;

        let rhs = gradient_info.v_hat.as_ref().unwrap().clone_sqrt() + self.epsilon;

        (&gradient_info.m * a_t) / rhs
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

#[derive(Debug, Serialize, Deserialize)]
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
        let one_minus_b2_t = 1.0 - DECAY_FUNCTION.decay(self.b2, self.t);

        gradient_info.m = &gradient_info.m * self.b1 + &gradient * (1.0 - self.b1);
        gradient_info.v = &gradient_info.v * self.b2 + (&gradient * &gradient) * (1.0 - self.b2);

        let a_t = self.a * one_minus_b2_t.sqrt() / one_minus_b1_t;

        (&gradient_info.m * a_t) / (gradient_info.v.clone_sqrt() + self.epsilon)
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
                    m: LayerType::from_raw(m.clone().into_boxed_slice(), 2, 1),
                    v: LayerType::from_raw(v.clone().into_boxed_slice(), 2, 1)
                };

                let gradient = LayerType::from_raw(g.clone().into_boxed_slice(), 2, 1);

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

                LayerType::from_raw(old_weight.clone().into_boxed_slice(), 2, 1) + change
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
}
