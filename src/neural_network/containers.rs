use std::{
    f32,
    iter,
    rc::Rc,
    fmt::Debug,
    cell::{self, RefCell},
    borrow::Borrow,
    ops::{Mul, Add, Sub, Div, AddAssign, DivAssign, Neg}
};

use serde::{Serialize, Deserialize};

#[allow(unused_imports)]
use matrix_wrapper::MatrixWrapper;

mod matrix_wrapper;


pub type LayerInnerType = MatrixWrapper;

pub const LEAKY_SLOPE: f32 = 0.01;

// i have no clue where else to put this
pub fn leaky_relu_d(value: f32) -> f32
{
    if value > 0.0
    {
        1.0
    } else
    {
        LEAKY_SLOPE
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum LayerChild
{
    Tensor(LayerType),
    Scalar(ScalarType)
}

impl LayerChild
{
    fn derivatives(&mut self, gradient: GradientType)
    {
        match self
        {
            Self::Tensor(x) =>
            {
                x.derivatives(LayerInnerType::from_gradient(gradient, || unreachable!()));
                return;
            },
            Self::Scalar(_) => ()
        }

        let value = f32::from_gradient(gradient, ||
        {
            match self
            {
                Self::Tensor(x) => x.value_clone(),
                _ => unreachable!()
            }
        });

        match self
        {
            Self::Scalar(x) => x.derivatives(value),
            _ => unreachable!()
        }
    }

    fn is_gradient(&self) -> bool
    {
        match self
        {
            Self::Tensor(x) => x.is_gradient(),
            Self::Scalar(x) => x.is_gradient()
        }
    }

    fn value_clone(&self) -> GradientType
    {
        match self
        {
            Self::Tensor(x) => GradientType::Tensor(x.value_clone()),
            Self::Scalar(x) => GradientType::Scalar(x.value_clone())
        }
    }
}

trait IntoChild<T=LayerChild>
{
    fn into_child(self, is_gradient: bool) -> T;
}

impl IntoChild for LayerType
{
    fn into_child(self, _is_gradient: bool) -> LayerChild
    {
        LayerChild::Tensor(self)
    }
}

impl<T> IntoChild<T> for T
{
    fn into_child(self, _is_gradient: bool) -> T
    {
        self
    }
}

impl IntoChild<LayerType> for &LayerType
{
    fn into_child(self, is_gradient: bool) -> LayerType
    {
        self.clone_maybe_gradientable(is_gradient)
    }
}

impl IntoChild for ScalarType
{
    fn into_child(self, _is_gradient: bool) -> LayerChild
    {
        LayerChild::Scalar(self)
    }
}

impl IntoChild for &LayerType
{
    fn into_child(self, is_gradient: bool) -> LayerChild
    {
        let this = self.clone_maybe_gradientable(is_gradient);

        LayerChild::Tensor(this)
    }
}

impl IntoChild for &mut LayerType
{
    fn into_child(self, is_gradient: bool) -> LayerChild
    {
        <&LayerType as IntoChild>::into_child(&self, is_gradient)
    }
}

impl IntoChild for &ScalarType
{
    fn into_child(self, is_gradient: bool) -> LayerChild
    {
        let this = self.clone_maybe_gradientable(is_gradient);

        LayerChild::Scalar(this)
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum LayerOps
{
    None,
    Diff,
    SumTensor(LayerType),
    Neg(LayerChild),
    Exp(LayerChild),
    Ln(LayerChild),
    LeakyRelu(LayerChild),
    Sigmoid(LayerChild),
    Tanh(LayerChild),
    Dot{lhs: LayerType, rhs: LayerType},
    Add{lhs: LayerChild, rhs: LayerChild},
    Sub{lhs: LayerChild, rhs: LayerChild},
    Mul{lhs: LayerChild, rhs: LayerChild},
    Div{lhs: LayerChild, rhs: LayerChild},
    Matmul{lhs: LayerType, rhs: LayerType},
    SoftmaxCrossEntropy{values: LayerType, targets: LayerInnerType}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientType
{
    Tensor(LayerInnerType),
    Scalar(f32)
}

impl GradientType
{
    pub fn reciprocal(&self) -> Self
    {
        match self
        {
            Self::Tensor(x) =>
            {
                let mut x = x.clone();
                x.reciprocal();

                Self::Tensor(x)
            },
            Self::Scalar(x) => Self::Scalar(x.recip())
        }
    }

    pub fn leaky_relu_d(&self) -> Self
    {
        match self
        {
            Self::Tensor(x) =>
            {
                let mut x = x.clone();
                x.leaky_relu_d();

                Self::Tensor(x)
            },
            Self::Scalar(x) => Self::Scalar(leaky_relu_d(*x))
        }
    }
}

impl From<LayerInnerType> for GradientType
{
    fn from(value: LayerInnerType) -> Self
    {
        Self::Tensor(value)
    }
}

impl From<f32> for GradientType
{
    fn from(value: f32) -> Self
    {
        Self::Scalar(value)
    }
}

impl AddAssign for GradientType
{
    fn add_assign(&mut self, rhs: Self)
    {
        match (self, rhs)
        {
            (Self::Tensor(lhs), Self::Tensor(rhs)) =>
            {
                *lhs += rhs;
            },
            (Self::Scalar(lhs), Self::Scalar(rhs)) =>
            {
                *lhs += rhs;
            },
            x => unimplemented!("{:?}", x)
        }
    }
}

impl Neg for GradientType
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        match self
        {
            Self::Tensor(x) => Self::Tensor(-x),
            Self::Scalar(x) => Self::Scalar(-x)
        }
    }
}

impl Div<f32> for GradientType
{
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output
    {
        match self
        {
            Self::Tensor(x) => Self::Tensor(x / rhs),
            Self::Scalar(x) => Self::Scalar(x / rhs)
        }
    }
}

impl Mul<f32> for GradientType
{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output
    {
        match self
        {
            Self::Tensor(x) => Self::Tensor(x * rhs),
            Self::Scalar(x) => Self::Scalar(x * rhs)
        }
    }
}

// is there an easier way to do this ;-; (maybe macros...)
impl<T> Mul<T> for &GradientType
where
    T: Borrow<GradientType>
{
    type Output = GradientType;

    fn mul(self, rhs: T) -> Self::Output
    {
        match (self, rhs.borrow())
        {
            (GradientType::Tensor(lhs), GradientType::Tensor(rhs)) =>
            {
                GradientType::Tensor(lhs * rhs)
            },
            (GradientType::Scalar(lhs), GradientType::Scalar(rhs)) =>
            {
                GradientType::Scalar(lhs * rhs)
            },
            (GradientType::Tensor(lhs), GradientType::Scalar(rhs)) =>
            {
                GradientType::Tensor(lhs * *rhs)
            },
            (GradientType::Scalar(lhs), GradientType::Tensor(rhs)) =>
            {
                GradientType::Tensor(rhs * *lhs)
            }
        }
    }
}

impl<T> Mul<T> for GradientType
where
    T: Borrow<Self>
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output
    {
        match (self, rhs.borrow())
        {
            (Self::Tensor(lhs), Self::Tensor(rhs)) => Self::Tensor(lhs * rhs),
            (Self::Scalar(lhs), Self::Scalar(rhs)) => Self::Scalar(lhs * rhs),
            (Self::Tensor(lhs), Self::Scalar(rhs)) => Self::Tensor(lhs * *rhs),
            (Self::Scalar(lhs), Self::Tensor(rhs)) => Self::Tensor(rhs * lhs)
        }
    }
}

impl Mul<&LayerInnerType> for GradientType
{
    type Output = Self;

    fn mul(self, rhs: &LayerInnerType) -> Self::Output
    {
        match self
        {
            Self::Tensor(x) => Self::Tensor(x * rhs),
            Self::Scalar(x) => Self::Tensor(rhs * x)
        }
    }
}

impl Mul<&LayerInnerType> for &GradientType
{
    type Output = GradientType;

    fn mul(self, rhs: &LayerInnerType) -> Self::Output
    {
        match self
        {
            GradientType::Tensor(x) => GradientType::Tensor(x * rhs),
            GradientType::Scalar(x) => GradientType::Tensor(rhs * *x)
        }
    }
}

impl Mul<&f32> for GradientType
{
    type Output = Self;

    fn mul(self, rhs: &f32) -> Self::Output
    {
        match self
        {
            Self::Tensor(x) => Self::Tensor(x * *rhs),
            Self::Scalar(x) => Self::Scalar(rhs * x)
        }
    }
}

impl Mul<&f32> for &GradientType
{
    type Output = GradientType;

    fn mul(self, rhs: &f32) -> Self::Output
    {
        match self
        {
            GradientType::Tensor(x) => GradientType::Tensor(x * *rhs),
            GradientType::Scalar(x) => GradientType::Scalar(rhs * x)
        }
    }
}

impl Mul<GradientType> for LayerInnerType
{
    type Output = Self;

    fn mul(self, rhs: GradientType) -> Self::Output
    {
        match rhs
        {
            GradientType::Tensor(rhs) => self * rhs,
            GradientType::Scalar(rhs) => self * rhs
        }
    }
}

pub trait Fillable
{
    fn fill(&mut self, value: f32);
}

impl Fillable for f32
{
    fn fill(&mut self, value: f32)
    {
        *self = value;
    }
}

impl Fillable for LayerInnerType
{
    fn fill(&mut self, value: f32)
    {
        self.fill(value);
    }
}

pub trait FromGradient
{
    fn from_gradient<F: FnOnce() -> LayerInnerType>(
        gradient: GradientType,
        value_getter: F
    ) -> Self;
}

impl FromGradient for f32
{
    fn from_gradient<F: FnOnce() -> LayerInnerType>(
        gradient: GradientType,
        _value_getter: F
    ) -> Self
    {
        match gradient
        {
            GradientType::Scalar(x) => x,
            GradientType::Tensor(tensor) =>
            {
                tensor.sum()
            }
        }
    }
}

impl FromGradient for LayerInnerType
{
    fn from_gradient<F: FnOnce() -> LayerInnerType>(
        gradient: GradientType,
        value_getter: F
    ) -> Self
    {
        match gradient
        {
            GradientType::Tensor(x) => x,
            GradientType::Scalar(scalar) =>
            {
                let mut value = value_getter();
                value.fill(scalar);

                value
            }
        }
    }
}

pub trait DiffBounds
where
    Self: Fillable + FromGradient + Clone + Into<GradientType>,
    Self: TryInto<LayerInnerType> + TryInto<f32>,
    <Self as TryInto<LayerInnerType>>::Error: Debug,
    <Self as TryInto<f32>>::Error: Debug,
    Self: AddAssign<Self> + Add<f32, Output=Self> + Neg<Output=Self>,
    for<'a> Self: Mul<&'a Self, Output=Self>,
    for<'a> &'a Self: Mul<&'a Self, Output=Self> + Neg<Output=Self>,
    for<'a> GradientType: Mul<&'a Self, Output=GradientType>,
    for<'a> &'a GradientType: Mul<&'a Self, Output=GradientType>
{
}

impl TryFrom<f32> for LayerInnerType
{
    type Error = ();

    fn try_from(_value: f32) -> Result<Self, Self::Error>
    {
        Err(())
    }
}

impl TryFrom<LayerInnerType> for f32
{
    type Error = ();

    fn try_from(_value: LayerInnerType) -> Result<Self, Self::Error>
    {
        Err(())
    }
}

impl DiffBounds for f32 {}
impl DiffBounds for LayerInnerType {}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiffType<T>
{
    inner: LayerOps,
    value: Option<T>,
    gradient: Option<T>,
    // oh this is so bad its so bad omg its so bad
    parents_amount: u32,
    calculate_gradient: bool
}

impl<T> DiffType<T>
where
    T: DiffBounds,
    <T as TryInto<LayerInnerType>>::Error: Debug,
    <T as TryInto<f32>>::Error: Debug,
    for<'a> T: Mul<&'a T, Output=T>,
    for<'a> &'a T: Mul<&'a T, Output=T> + Neg<Output=T>,
    for<'a> GradientType: Mul<&'a T, Output=GradientType>,
    for<'a> &'a GradientType: Mul<&'a T, Output=GradientType>
{
    pub fn calculate_gradients(&mut self)
    {
        let mut ones = self.value.clone().unwrap();
        ones.fill(1.0);

        self.derivatives(ones);
    }

    // long ass name but i wanna be descriptive about wut this does
    pub fn take_gradient(&mut self) -> T
    {
        match self.gradient.take()
        {
            Some(x) => x,
            None =>
            {
                let mut value = self.value.clone().unwrap();
                value.fill(0.0);

                return value;
            }
        }
    }

    fn derivatives(&mut self, starting_gradient: T)
    {
        if self.gradient.is_none()
        {
            self.gradient = Some(starting_gradient);
        } else
        {
            *self.gradient.as_mut().unwrap() += starting_gradient;
        }

        if self.parents_amount > 1
        {
            return;
        }

        let gradient = self.gradient.clone().unwrap();

        match &mut self.inner
        {
            LayerOps::Add{lhs, rhs} =>
            {
                if lhs.is_gradient()
                {
                    lhs.derivatives(gradient.clone().into());
                }

                if rhs.is_gradient()
                {
                    rhs.derivatives(gradient.into());
                }
            },
            LayerOps::Sub{lhs, rhs} =>
            {
                if lhs.is_gradient()
                {
                    lhs.derivatives(gradient.clone().into());
                }
                
                if rhs.is_gradient()
                {
                    rhs.derivatives((-gradient).into());
                }
            },
            LayerOps::Mul{lhs, rhs} =>
            {
                let rhs_cg = rhs.is_gradient();
                let lhs_value = rhs_cg.then(|| lhs.value_clone());

                if lhs.is_gradient()
                {
                    let d = rhs.value_clone() * &gradient;
                    lhs.derivatives(d.into());
                }

                if rhs_cg
                {
                    debug_assert!(lhs_value.is_some());
                    let lhs_value = unsafe{ lhs_value.unwrap_unchecked() };

                    let d = lhs_value * &gradient;
                    rhs.derivatives(d.into());
                }
            },
            LayerOps::Div{lhs, rhs} =>
            {
                let r_recip = rhs.value_clone().reciprocal();
                
                let rhs_cg = rhs.is_gradient();
                let lhs_value = rhs_cg.then(|| lhs.value_clone());

                if lhs.is_gradient()
                {
                    let d = &r_recip * &gradient;

                    lhs.derivatives(d);
                }

                if rhs_cg
                {
                    // my favorite syntax
                    let recip_squared =
                        <&GradientType as Mul<&GradientType>>::mul(&r_recip, &r_recip);
                    
                    debug_assert!(lhs_value.is_some());
                    let lhs_value = unsafe{ lhs_value.unwrap_unchecked() };

                    let d = -lhs_value * &gradient;
                    let d = <GradientType as Mul<GradientType>>::mul(d, recip_squared);

                    rhs.derivatives(d);
                }
            },
            LayerOps::Exp(x) =>
            {
                if x.is_gradient()
                {
                    let d = self.value.as_ref().unwrap();

                    x.derivatives((gradient * d).into());
                }
            },
            LayerOps::Sigmoid(x) =>
            {
                if x.is_gradient()
                {
                    let value = self.value.as_ref().unwrap();

                    // sigmoid(x) * (1.0 - sigmoid(x))
                    let d = (-value + 1.0) * value;

                    x.derivatives((gradient * &d).into());
                }
            },
            LayerOps::Tanh(x) =>
            {
                if x.is_gradient()
                {
                    let value = self.value.as_ref().unwrap();

                    // 1 - tanh^2(x)
                    let d = -(value * value) + 1.0;

                    x.derivatives((gradient * &d).into());
                }
            },
            LayerOps::LeakyRelu(x) =>
            {
                if x.is_gradient()
                {
                    let d = x.value_clone().leaky_relu_d();

                    x.derivatives(d * &gradient);
                }
            },
            LayerOps::Ln(x) =>
            {
                if x.is_gradient()
                {
                    let d = x.value_clone().reciprocal();

                    x.derivatives(d * &gradient);
                }
            },
            LayerOps::Neg(x) =>
            {
                if x.is_gradient()
                {
                    x.derivatives((-gradient).into());
                }
            },
            LayerOps::Matmul{lhs, rhs} =>
            {
                let gradient: LayerInnerType = gradient.try_into()
                    .expect("matmul must be a tensor");
                
                let rhs_cg = rhs.is_gradient();
                let rhs_d = rhs_cg.then(|| lhs.value().matmul_transposed(&gradient));

                if lhs.is_gradient()
                {
                    let d = gradient.matmul_by_transposed(&*rhs.value());
                    lhs.derivatives(d.into());
                }

                if rhs_cg
                {
                    debug_assert!(rhs_d.is_some());
                    let rhs_d = unsafe{ rhs_d.unwrap_unchecked() };

                    rhs.derivatives(rhs_d.into());
                }
            },
            LayerOps::SumTensor(x) =>
            {
                if x.is_gradient()
                {
                    let d = LayerInnerType::from_gradient(gradient.into(), ||
                    {
                        x.value_clone()
                    });

                    x.derivatives(d);
                }
            },
            LayerOps::Dot{lhs, rhs} =>
            {
                let gradient = LayerInnerType::from_gradient(gradient.into(), ||
                {
                    lhs.value_clone()
                });
                
                let rhs_cg = rhs.is_gradient();
                let lhs_value = rhs_cg.then(|| lhs.value_clone());

                if lhs.is_gradient()
                {
                    let d = &gradient * rhs.value_clone();
 
                    lhs.derivatives(d.into());
                }

                if rhs_cg
                {
                    debug_assert!(lhs_value.is_some());
                    let lhs_value = unsafe{ lhs_value.unwrap_unchecked() };

                    let d = &gradient * lhs_value;

                    rhs.derivatives(d.into());
                }
            },
            LayerOps::SoftmaxCrossEntropy{values, targets} =>
            {
                if values.is_gradient()
                {
                    let gradient = LayerInnerType::from_gradient(gradient.into(), ||
                    {
                        values.value_clone()
                    });

                    let softmaxed_values = {
                        // softmax
                        let mut values = values.value_clone();
                        values.exp();
                        let s = values.sum();

                        values /= s;

                        values
                    };

                    let d = gradient * (softmaxed_values - targets);
                    values.derivatives(d.into());
                }
            },
            LayerOps::Diff => (),
            LayerOps::None => ()
        }
    }
}

pub struct CloneableWrapper<T>(pub DiffWrapper<T>);

impl<T: Clone> Clone for CloneableWrapper<T>
{
    fn clone(&self) -> Self
    {
        let this = self.0.this_ref();

        let wrapper = if this.calculate_gradient
        {
            DiffWrapper::new_diff(this.value.clone().unwrap())
        } else
        {
            DiffWrapper::new_inner(this.value.clone().unwrap(), LayerOps::None, false)
        };

        Self(wrapper)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiffWrapper<T>(Option<Rc<RefCell<DiffType<T>>>>);

impl<T> Drop for DiffWrapper<T>
{
    fn drop(&mut self)
    {
        self.drop_child();
    }
}

impl<T> DiffWrapper<T>
// i dont even need these wut am i doing?
where
    T: DiffBounds,
    <T as TryInto<LayerInnerType>>::Error: Debug,
    <T as TryInto<f32>>::Error: Debug,
    for<'a> T: Mul<&'a T, Output=T> + Add<f32, Output=T> + Neg<Output=T>,
    for<'a> &'a T: Mul<&'a T, Output=T> + Neg<Output=T>,
    for<'a> GradientType: Mul<&'a T, Output=GradientType>,
    for<'a> &'a GradientType: Mul<&'a T, Output=GradientType>
{
    pub fn clear(&mut self)
    {
        *self = Self::new_diff(self.value_take());
    }

    pub fn take_gradient(&mut self) -> T
    {
        self.this_mut().take_gradient()
    }

    pub fn calculate_gradients(self)
    {
        let mut this = self.this_mut();

        this.calculate_gradients()
    }

    fn derivatives(&mut self, gradient: T)
    {
        {
            let mut this = self.this_mut();
            this.derivatives(gradient);
        }

        self.drop_child();
    }

    pub fn value(&self) -> cell::Ref<T>
    {
        cell::Ref::map(
            RefCell::borrow(self.0.as_ref().unwrap()),
            |v| v.value.as_ref().unwrap()
        )
    }

    pub fn value_mut(&mut self) -> cell::RefMut<T>
    {
        cell::RefMut::map(
            RefCell::borrow_mut(self.0.as_mut().unwrap()),
            |v| v.value.as_mut().unwrap()
        )
    }

    pub fn value_clone(&self) -> T
    {
        (*RefCell::borrow(self.0.as_ref().unwrap())).value.clone().unwrap()
    }

    pub fn value_take(&mut self) -> T
    {
        (*RefCell::borrow_mut(self.0.as_mut().unwrap())).value.take().unwrap()
    }
}

impl<T> DiffWrapper<T>
{
    pub fn new_diff(value: T) -> Self
    {
        Self::new_inner(value, LayerOps::Diff, true)
    }

    fn new_inner(value: T, ops: LayerOps, calculate_gradient: bool) -> Self
    {
        let diff = DiffType{
            value: Some(value),
            inner: ops,
            gradient: None,
            calculate_gradient,
            parents_amount: 1
        };

        Self(Some(Rc::new(RefCell::new(diff))))
    }

    // i love dropping children
    fn drop_child(&mut self)
    {
        if let Some(mut this) = self.this_mut_maybe()
        {
            if this.calculate_gradient
            {
                this.parents_amount -= 1;
            }
        }

        self.0 = None;
    }

    fn is_gradient(&self) -> bool
    {
        self.this_ref().calculate_gradient
    }

    pub fn clone_maybe_gradientable(&self, is_gradient: bool) -> Self
    {
        if is_gradient
        {
            self.clone_gradientable()
        } else
        {
            self.clone_non_gradientable()
        }
    }

    pub fn clone_gradientable(&self) -> Self
    {
        self.this_mut().parents_amount += 1;

        Self(self.0.clone())
    }

    pub fn clone_non_gradientable(&self) -> Self
    {
        Self(self.0.clone())
    }

    #[allow(dead_code)]
    fn this_ref(&self) -> cell::Ref<DiffType<T>>
    {
        RefCell::borrow(self.0.as_ref().unwrap())
    }

    fn this_mut_maybe(&self) -> Option<cell::RefMut<DiffType<T>>>
    {
        self.0.as_ref().map(|this|
        {
            RefCell::borrow_mut(this)
        })
    }

    fn this_mut(&self) -> cell::RefMut<DiffType<T>>
    {
        // this should kinda be &mut self but im lazy
        RefCell::borrow_mut(self.0.as_ref().unwrap())
    }
}

#[derive(Debug)]
pub struct Softmaxer;

impl Softmaxer
{
    pub fn softmax_temperature(layer: &mut LayerInnerType, temperature: f32) 
    {
        *layer /= temperature;

        Self::softmax(layer)
    }

    pub fn softmax(layer: &mut LayerInnerType) 
    {
        layer.exp();
        let s = layer.sum();

        *layer /= s;
    }

    pub fn pick_weighed_associated(values: &LayerInnerType) -> usize
    {
        Self::pick_weighed_inner(values.iter())
    }

    pub fn pick_weighed_inner<'b, I>(mut iter: I) -> usize
    where
        I: Iterator<Item=&'b f32> + ExactSizeIterator
    {
        let mut c = fastrand::f32();

        let max_index = iter.len() - 1;

        iter.position(|v|
        {
            c -= v;

            c <= 0.0
        }).unwrap_or(max_index)
    }

    pub fn highest_index<'b, I>(iter: I) -> usize
    where
        I: Iterator<Item=&'b f32>
    {
        iter.enumerate().max_by(|a, b|
        {
            a.1.partial_cmp(b.1).unwrap()
        }).unwrap().0
    }
}

pub type ScalarType = DiffWrapper<f32>;

impl ScalarType
{
    pub fn new(value: f32) -> Self
    {
        ScalarType::new_inner(value, LayerOps::None, false)
    }
}

// macros be like
macro_rules! op_impl
{
    (
        $this:ident, $other:ident,
        $out:ident,
        $op:ident, $fun:ident
    ) =>
    {
        impl $op<$other> for $this
        {
            type Output = $out;

            fn $fun(self, rhs: $other) -> Self::Output
            {
                op_impl_inner!(self, rhs, $out, $op, $fun)
            }
        }

        impl $op<&$other> for $this
        {
            type Output = $out;

            fn $fun(self, rhs: &$other) -> Self::Output
            {
                op_impl_inner!(self, rhs, $out, $op, $fun)
            }
        }

        impl $op<$other> for &$this
        {
            type Output = $out;

            fn $fun(self, rhs: $other) -> Self::Output
            {
                op_impl_inner!(self, rhs, $out, $op, $fun)
            }
        }

        impl $op<&$other> for &$this
        {
            type Output = $out;

            fn $fun(self, rhs: &$other) -> Self::Output
            {
                op_impl_inner!(self, rhs, $out, $op, $fun)
            }
        }
    }
}

macro_rules! op_impl_special
{
    (
        $this:ident, $other:ident,
        $out:ident,
        $op:ident, $fun:ident,
        $value_fun:ident
    ) =>
    {
        impl $op<$other> for $this
        {
            type Output = $out;

            fn $fun(self, rhs: $other) -> Self::Output
            {
                let value = {
                    let lhs = self.value();
                    let rhs = rhs.value();

                    $value_fun(&*lhs, &*rhs)
                };

                inner_from_value!(value, self, rhs, $out, $op)
            }
        }

        impl $op<&$other> for $this
        {
            type Output = $out;

            fn $fun(self, rhs: &$other) -> Self::Output
            {
                let value = {
                    let lhs = self.value();
                    let rhs = rhs.value();

                    $value_fun(&*lhs, &*rhs)
                };

                inner_from_value!(value, self, rhs, $out, $op)
            }
        }
    }
}

macro_rules! op_impl_inner
{
    (
        $lhs:expr, $rhs:expr,
        $out:ident,
        $op:ident, $fun:ident
    ) =>
    {
        {
            let value = {
                let lhs = $lhs.value();
                let lhs: &_ = &*lhs;

                let rhs = $rhs.value();
                let rhs: &_ = &*rhs;

                lhs.$fun(rhs)
            };

            inner_from_value!(value, $lhs, $rhs, $out, $op)
        }
    }
}

// MACRO MOMENT
macro_rules! inner_from_value
{
    (
        $value:expr,
        $lhs:expr, $rhs:expr,
        $out:ident,
        $op:ident
    ) =>
    {
        {
            let is_lhs_gradient = $lhs.is_gradient();
            let is_rhs_gradient = $rhs.is_gradient();

            $out::new_inner(
                $value,
                LayerOps::$op{
                    lhs: $lhs.into_child(is_lhs_gradient),
                    rhs: $rhs.into_child(is_rhs_gradient)
                },
                is_lhs_gradient || is_rhs_gradient
            )
        }
    }
}

macro_rules! inner_single_from_value
{
    ($value:expr, $this:expr, $out:ident, $op:ident) =>
    {
        {
            let is_gradient = $this.is_gradient();

            $out::new_inner(
                $value,
                LayerOps::$op($this.into_child(is_gradient)),
                is_gradient
            )
        }
    }
}

op_impl!{ScalarType, ScalarType, ScalarType, Add, add}
op_impl!{ScalarType, ScalarType, ScalarType, Div, div}

fn special_sub(lhs: &f32, rhs: &LayerInnerType) -> LayerInnerType
{
    (-rhs) + *lhs
}

op_impl_special!{ScalarType, LayerType, LayerType, Sub, sub, special_sub}

impl Neg for ScalarType
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        let value = -self.value_clone();
        inner_single_from_value!(value, self, Self, Neg)
    }
}

impl iter::Sum for ScalarType
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item=Self>
    {
        iter.reduce(|acc, value|
        {
            acc + value
        }).unwrap_or_else(|| ScalarType::new(f32::default()))
    }
}

impl AddAssign for ScalarType
{
    fn add_assign(&mut self, rhs: Self)
    {
        *self = &*self + rhs;
    }
}

pub type LayerType = DiffWrapper<LayerInnerType>;

op_impl!{LayerType, LayerType, LayerType, Add, add}
op_impl!{LayerType, ScalarType, LayerType, Add, add}
op_impl!{LayerType, LayerType, LayerType, Sub, sub}
op_impl!{LayerType, ScalarType, LayerType, Sub, sub}
op_impl!{LayerType, LayerType, LayerType, Mul, mul}
op_impl!{LayerType, ScalarType, LayerType, Mul, mul}
op_impl!{LayerType, LayerType, LayerType, Div, div}
op_impl!{LayerType, ScalarType, LayerType, Div, div}

impl AddAssign for LayerType
{
    fn add_assign(&mut self, rhs: Self)
    {
        *self = &*self + rhs;
    }
}

impl DivAssign<ScalarType> for LayerType
{
    fn div_assign(&mut self, rhs: ScalarType)
    {
        *self = &*self / rhs;
    }
}

impl LayerType
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        let value = MatrixWrapper::new(previous_size, this_size);
        Self::new_inner(value, LayerOps::None, false)
    }

    pub fn new_with<F: FnMut() -> f32>(
        previous_size: usize,
        this_size: usize,
        f: F
    )-> Self
    {
        let value = MatrixWrapper::new_with(previous_size, this_size, f);
        Self::new_inner(value, LayerOps::None, false)
    }

    pub fn from_raw<V: Into<Vec<f32>>>(
        values: V,
        previous_size: usize,
        this_size: usize
    ) -> Self
    {
        let value = MatrixWrapper::from_raw(values, previous_size, this_size);
        Self::new_inner(value, LayerOps::None, false)
    }

    pub fn softmax_cross_entropy(self, targets: LayerInnerType) -> ScalarType
    {
        let is_lhs_gradient = self.is_gradient();

        let value = self.value_clone().softmax_cross_entropy(&targets);

        ScalarType::new_inner(
            value,
            LayerOps::SoftmaxCrossEntropy{
                values: self,
                targets
            },
            is_lhs_gradient
        )
    }

    pub fn matmul(&self, rhs: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();

        op_impl_inner!(self, rhs, Self, Matmul, matmul)
    }

    pub fn dot(self, rhs: Self) -> ScalarType
    {
        let value = {
            let rhs = rhs.value();

            self.value_clone().dot(&*rhs)
        };

        inner_from_value!(value, self, rhs, ScalarType, Dot)
    }

    pub fn exp(&mut self)
    {
        let mut value = self.value_clone();
        value.exp();

        *self = inner_single_from_value!(value, self, Self, Exp);
    }

    pub fn ln(&mut self)
    {
        let mut value = self.value_clone();
        value.ln();

        *self = inner_single_from_value!(value, self, Self, Ln);
    }

    pub fn sigmoid(&mut self)
    {
        let mut value = self.value_clone();
        value.sigmoid();

        *self = inner_single_from_value!(value, self, Self, Sigmoid);
    }

    pub fn tanh(&mut self)
    {
        let mut value = self.value_clone();
        value.tanh();

        *self = inner_single_from_value!(value, self, Self, Tanh);
    }

    pub fn leaky_relu(&mut self)
    {
        let mut value = self.value_clone();
        value.leaky_relu();

        *self = inner_single_from_value!(value, self, Self, LeakyRelu);
    }

    pub fn sum(&self) -> ScalarType
    {
        let value = self.value().sum();
        inner_single_from_value!(value, self, ScalarType, SumTensor)
    }

    pub fn total_len(&self) -> usize
    {
        self.value().total_len()
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.value().as_vec()
    }

    pub fn pick_weighed(&self) -> usize
    {
        self.value().pick_weighed()
    }

    pub fn highest_index(&self) -> usize
    {
        self.value().highest_index()
    }
}

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::neural_network::gru::tests::close_enough_loose;

    const LAYER_PREV: usize = 10;
    const LAYER_CURR: usize = 10;

    fn compare_single(correct: f32, calculated: f32)
    {
        let epsilon = 0.2;
        assert!(
            close_enough_loose(correct, calculated, epsilon),
            "correct: {}, calculated: {}",
            correct, calculated
        );
    }

    fn compare_tensor(correct: LayerInnerType, calculated: LayerInnerType)
    {
        correct.as_vec().into_iter().zip(calculated.as_vec().into_iter())
            .for_each(|(correct, calculated)| compare_single(correct, calculated));
    }

    #[allow(dead_code)]
    fn check_tensor_with_dims(
        a_dims: (usize, usize),
        b_dims: (usize, usize),
        f: impl FnMut(&LayerType, &LayerType) -> LayerType
    )
    {
        let a = random_tensor(a_dims.0, a_dims.1);
        let b = random_tensor(b_dims.0, b_dims.1);

        check_tensor_inner(a, b, f);
    }

    fn check_tensor(f: impl FnMut(&LayerType, &LayerType) -> LayerType)
    {
        let a = random_tensor(LAYER_PREV, LAYER_CURR);
        let b = random_tensor(LAYER_PREV, LAYER_CURR);

        check_tensor_inner(a, b, f);
    }

    fn check_tensor_inner(
        mut a: LayerType,
        mut b: LayerType,
        mut f: impl FnMut(&LayerType, &LayerType) -> LayerType
    )
    {
        let out = f(&a, &b);

        out.calculate_gradients();

        let a_g = a.take_gradient();
        let b_g = b.take_gradient();

        let mut vals = |a: &mut LayerType, b: &mut LayerType|
        {
            a.clear();
            b.clear();

            f(&a, &b).value_take()
        };

        let orig = vals(&mut a, &mut b);

        let epsilon: f32 = 0.009;

        let fg = |value: LayerInnerType|
        {
            let value = value.sum();
            let orig = orig.clone().sum();

            (value - orig) / epsilon
        };

        let mut a_fg = vec![0.0; a.total_len()];
        for index in 0..a_fg.len()
        {
            let v = &a;
            let epsilon = one_hot(v.value_clone(), index, epsilon, 0.0);

            let this_fg = {
                let mut a = LayerType::new_diff(v.value_clone() + epsilon);
                fg(vals(&mut a, &mut b))
            };

            a_fg[index] = this_fg;
        }

        let mut b_fg = vec![0.0; b.total_len()];
        for index in 0..b_fg.len()
        {
            let v = &b;
            let epsilon = one_hot(v.value_clone(), index, epsilon, 0.0);

            let this_fg = {
                let mut b = LayerType::new_diff(v.value_clone() + epsilon);
                fg(vals(&mut a, &mut b))
            };

            b_fg[index] = this_fg;
        }

        let vec_to_layer = |v, layer_match: &LayerType|
        {
            let mut layer = layer_match.value_clone();

            layer.swap_raw_values(v);

            layer
        };

        let a_fg = vec_to_layer(a_fg, &a);
        let b_fg = vec_to_layer(b_fg, &b);

        eprintln!("derivative of a");
        compare_tensor(a_fg, a_g.clone());

        eprintln!("derivative of b");
        compare_tensor(b_fg, b_g.clone());
    }

    fn one_hot(
        dimensions_match: LayerInnerType,
        position: usize,
        value: f32,
        d_value: f32
    ) -> LayerInnerType
    {
        let values = dimensions_match.as_vec().into_iter().enumerate().map(|(i, _)|
        {
            if i == position
            {
                value
            } else
            {
                d_value
            }
        }).collect::<Vec<_>>();

        let mut layer = dimensions_match.clone();
        layer.swap_raw_values(values);

        layer
    }

    fn random_value() -> f32
    {
        /*let m = if fastrand::bool() {1.0} else {-1.0};
        (fastrand::f32() + 0.05) * m*/

        fastrand::u32(1..3) as f32
    }

    fn random_tensor(prev: usize, curr: usize) -> LayerType
    {
        LayerType::new_diff(
            LayerInnerType::new_with(
                prev,
                curr,
                random_value
            )
        )
    }

    #[test]
    fn subtraction()
    {
        check_tensor(|a, b| a - b)
    }

    #[test]
    fn addition()
    {
        check_tensor(|a, b| a + b)
    }

    #[test]
    fn multiplication()
    {
        check_tensor(|a, b| a * b)
    }

    #[test]
    fn division()
    {
        check_tensor(|a, b| a / b)
    }

    #[test]
    fn division_sum_inplace()
    {
        check_tensor(|a, b|
        {
            let mut a = a.clone_gradientable();
            a /= b.sum();

            a
        })
    }

    #[test]
    fn non_diff_subdiff()
    {
        check_tensor(|a, b| ScalarType::new(1.0) - (a + b))
    }

    #[test]
    fn basic_combined()
    {
        check_tensor(|a, b| a * b + a)
    }

    #[test]
    fn complex_combined()
    {
        check_tensor(|a, b| a * b + a + b - b * b + a)
    }

    #[test]
    fn sum_tensor_product()
    {
        check_tensor(|a, b| a * b.sum())
    }

    #[test]
    fn sum_tensor_addition()
    {
        check_tensor(|a, b| a + b.sum())
    }

    #[test]
    fn sum_tensor_product_negative()
    {
        check_tensor(|a, b| a * -b.sum())
    }

    #[test]
    fn dot_product()
    {
        check_tensor(|a, b| a + a.clone_gradientable().dot(b.clone_gradientable()))
    }

    #[test]
    fn scalar_minus_tensor()
    {
        check_tensor(|a, b| a.sum() - b)
    }

    #[test]
    fn scalar_minus_tensor_stuff()
    {
        check_tensor(|a, b| ScalarType::new(2.0) - (a.sum() - b))
    }

    #[test]
    fn exponential()
    {
        check_tensor(|a, b|
        {
            let mut a = a.clone_gradientable();
            a.exp();

            a + b
        })
    }

    #[test]
    fn natural_logarithm()
    {
        check_tensor(|a, b|
        {
            let a = a * a;

            let mut a = a.clone_gradientable();
            a.ln();

            a + b
        })
    }

    #[test]
    fn leaky_relu()
    {
        check_tensor(|a, b|
        {
            let mut a = a.clone_gradientable();
            a.leaky_relu();

            a + b
        })
    }

    // flexing my math functions name knowledge
    #[test]
    fn logistic_function()
    {
        check_tensor(|a, b|
        {
            let mut a = a.clone_gradientable();
            a.sigmoid();

            a + b
        })
    }

    #[test]
    fn hyperbolic_tangent()
    {
        check_tensor(|a, b|
        {
            let mut a = a.clone_gradientable();
            a.tanh();

            a + b
        })
    }

    #[test]
    fn sum_thingy()
    {
        check_tensor(|a, b|
        {
            let s: ScalarType = iter::repeat(a).take(5).zip(iter::repeat(b).take(5))
                .map(|(predicted, target)|
                {
                    let predicted = predicted.clone_gradientable();
                    predicted.dot(target.clone_gradientable())
                }).sum();

            a * s
        })
    }

    #[test]
    fn matrix_multiplication()
    {
        check_tensor_with_dims((2, 10), (10, 1), |a, b| a.matmul(b) + b.sum())
    }

    fn create_targets() -> LayerInnerType
    {
        let m = random_tensor(LAYER_PREV, LAYER_CURR).value_take();

        let pos = fastrand::usize(0..m.total_len());
        one_hot(m, pos, 1.0, 0.0)
    }

    #[test]
    fn softmax_cross_entropy()
    {
        let targets = create_targets();
        check_tensor(|a, b|
        {
            b + a.clone_gradientable().softmax_cross_entropy(targets.clone())
        })
    }

    #[test]
    fn softmax_cross_entropy_complicated()
    {
        let targets = create_targets();
        check_tensor(|a, b|
        {
            a + (a + b + ScalarType::new(2.0)).softmax_cross_entropy(targets.clone())
        })
    }
}
