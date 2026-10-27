use std::fmt;

use serde::{Serialize, Deserialize};

use super::{
    TensorPtr,
    DiffTensorPtr,
    ShapedTensorIndex,
    LayerType,
    OperationsRecorder
};


#[derive(Debug, Clone)]
pub enum OptimizerInfoPtr
{
    Ptr(TensorPtr),
    Index(ShapedTensorIndex)
}

fn force_ptr(x: &OptimizerInfoPtr) -> TensorPtr
{
    if let OptimizerInfoPtr::Ptr(x) = x
    {
        *x
    } else
    {
        panic!("info must be a ptr")
    }
}

pub trait NewableLayer: Sized
{
    fn new_values<F: FnMut(LayerType) -> OptimizerInfoPtr>(previous: usize, current: usize, f: F) -> Self;

    fn new(recorder: &mut OperationsRecorder, previous: usize, current: usize) -> Self
    {
        Self::new_values(previous, current, |x|
        {
            OptimizerInfoPtr::Ptr(recorder.set_new_tensor(x).as_value())
        })
    }
}

impl NewableLayer for ()
{
    fn new_values<F: FnMut(LayerType) -> OptimizerInfoPtr>(_previous: usize, _current: usize, _f: F) -> Self {}
}

pub struct OptimizerInfo<O, P>
{
    pub weights: Vec<DiffTensorPtr>,
    pub params: Vec<P>,
    pub gradient_clip: Option<f32>,
    pub optimizer: O
}

impl<O: Optimizer> OptimizerInfo<O, O::WeightParam<OptimizerInfoPtr>>
{
    pub fn add_operations(&self, recorder: &mut OperationsRecorder)
    {
        if O::is_none_optimizer()
        {
            return;
        }

        let batch_size = self.weights.first().map(|x| recorder.tensor_shape(x.as_value()).batch_size).unwrap_or(1);

        let batch_size_recip = recorder.set_new_value((batch_size as f32).recip()).as_value();

        self.weights.iter().zip(self.params.iter()).for_each(|(weight, param)|
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

            let new_weight = self.optimizer.clone().add_operations(recorder, param, clipped_gradient);
            recorder.name_tensor(new_weight, "new_weight_override");

            recorder.store_tensor_until_end(new_weight);
            recorder.memory.allow_after_coalesce_use(new_weight);
            recorder.memory.allow_after_coalesce_use(weight);

            O::map_data_ref(param, |x|
            {
                recorder.store_tensor_until_end(force_ptr(x));
            });

            recorder.force_same_memory(new_weight, weight);
        });
    }
}

pub trait Optimizer: fmt::Debug + Default + Clone
{
    type WeightParam<T>;

    fn is_none_optimizer() -> bool { false }

    fn map_data_ref<T, U, F: FnMut(&T) -> U>(x: &Self::WeightParam<T>, f: F) -> Self::WeightParam<U>;

    fn add_operations(self, recorder: &mut OperationsRecorder, param: &Self::WeightParam<OptimizerInfoPtr>, gradient: TensorPtr) -> TensorPtr;

    fn set_learning_rate(&mut self, learning_rate: f32);
}

impl Optimizer for ()
{
    type WeightParam<T> = ();

    fn is_none_optimizer() -> bool { true }

    fn map_data_ref<T, U, F: FnMut(&T) -> U>(_x: &Self::WeightParam<T>, _f: F) -> Self::WeightParam<U> {}

    fn add_operations(self, _recorder: &mut OperationsRecorder, _param: &Self::WeightParam<OptimizerInfoPtr>, _gradient: TensorPtr) -> TensorPtr
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
        Self{learning_rate: 0.0001}
    }
}

impl Optimizer for Sgd
{
    type WeightParam<T> = ();

    fn map_data_ref<T, U, F: FnMut(&T) -> U>(_x: &Self::WeightParam<T>, _f: F) -> Self::WeightParam<U> {}

    fn add_operations(self, recorder: &mut OperationsRecorder, _param: &Self::WeightParam<OptimizerInfoPtr>, gradient: TensorPtr) -> TensorPtr
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
    pub m: T
}

impl NewableLayer for PowerSignGradientInfo<OptimizerInfoPtr>
{
    fn new_values<F: FnMut(LayerType) -> OptimizerInfoPtr>(previous: usize, current: usize, mut f: F) -> Self
    {
        Self{
            m: f(LayerType::new(current, previous))
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
            learning_rate: 0.0001
        }
    }
}

impl Optimizer for PowerSign
{
    type WeightParam<T> = PowerSignGradientInfo<T>;

    fn map_data_ref<T, U, F: FnMut(&T) -> U>(x: &Self::WeightParam<T>, mut f: F) -> Self::WeightParam<U>
    {
        PowerSignGradientInfo{
            m: f(&x.m)
        }
    }

    fn add_operations(self, recorder: &mut OperationsRecorder, param: &Self::WeightParam<OptimizerInfoPtr>, gradient: TensorPtr) -> TensorPtr
    {
        recorder.powersign_optimizer(gradient, self, Self::map_data_ref(param, force_ptr))
    }

    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.learning_rate = learning_rate;
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamGradientInfo<T>
{
    pub m: T,
    pub v: T
}

impl NewableLayer for AdamGradientInfo<OptimizerInfoPtr>
{
    fn new_values<F: FnMut(LayerType) -> OptimizerInfoPtr>(previous: usize, current: usize, mut f: F) -> Self
    {
        Self{
            m: f(LayerType::new(current, previous)),
            v: f(LayerType::new(current, previous))
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam
{
    pub learning_rate: f32,
    pub b1: f32,
    pub b2: f32,
    pub epsilon: f32
}

impl Default for Adam
{
    fn default() -> Self
    {
        Self{
            learning_rate: 0.0001,
            b1: 0.9,
            b2: 0.999,
            epsilon: 1e-8
        }
    }
}

impl Optimizer for Adam
{
    type WeightParam<T> = AdamGradientInfo<T>;

    fn map_data_ref<T, U, F: FnMut(&T) -> U>(x: &Self::WeightParam<T>, mut f: F) -> Self::WeightParam<U>
    {
        AdamGradientInfo{
            m: f(&x.m),
            v: f(&x.v)
        }
    }

    fn add_operations(self, recorder: &mut OperationsRecorder, param: &Self::WeightParam<OptimizerInfoPtr>, gradient: TensorPtr) -> TensorPtr
    {
        recorder.adam_optimizer(gradient, self, Self::map_data_ref(param, force_ptr))
    }

    fn set_learning_rate(&mut self, learning_rate: f32)
    {
        self.learning_rate = learning_rate;
    }
}
