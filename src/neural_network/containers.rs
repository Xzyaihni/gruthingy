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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    fn value_clone(&self) -> GradientType
    {
        match self
        {
            Self::Tensor(x) => GradientType::Tensor(x.value_clone()),
            Self::Scalar(x) => GradientType::Scalar(x.value_clone())
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    Matmul{lhs: LayerType, rhs: LayerType}
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
    Self: Fillable + FromGradient + Clone + Into<GradientType> + TryInto<LayerInnerType>,
    <Self as TryInto<LayerInnerType>>::Error: Debug,
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

impl DiffBounds for f32 {}
impl DiffBounds for LayerInnerType {}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiffType<T>
{
    inner: LayerOps,
    value: Option<T>,
    gradient: Option<T>,
    calculate_gradient: bool
}

impl<T> DiffType<T>
{
    fn maybe_gradient<A>(&self, rhs: &DiffType<A>) -> bool
    {
        self.calculate_gradient || rhs.calculate_gradient
    }
}

impl<T> DiffType<T>
where
    T: DiffBounds,
    <T as TryInto<LayerInnerType>>::Error: Debug,
    for<'a> T: Mul<&'a T, Output=T>,
    for<'a> &'a T: Mul<&'a T, Output=T> + Neg<Output=T>,
    for<'a> GradientType: Mul<&'a T, Output=GradientType>,
    for<'a> &'a GradientType: Mul<&'a T, Output=GradientType>
{
    pub fn calculate_gradients(mut self)
    {
        let mut ones = self.value.clone().unwrap();
        ones.fill(1.0);

        self.derivatives(ones, false);
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

    fn derivatives(&mut self, starting_gradient: T, multiple_parents: bool)
    {
        if !self.calculate_gradient
        {
            return;
        }

        let mut add_gradients = |gradient|
        {
            if self.gradient.is_none()
            {
                self.gradient = Some(gradient);
            } else
            {
                *self.gradient.as_mut().unwrap() += gradient;
            }
        };

        add_gradients(starting_gradient);

        if multiple_parents
        {
            return;
        }

        let gradient = self.gradient.clone().unwrap();

        match &mut self.inner
        {
            LayerOps::Add{lhs, rhs} =>
            {
                lhs.derivatives(gradient.clone().into());
                rhs.derivatives(gradient.into());
            },
            LayerOps::Sub{lhs, rhs} =>
            {
                lhs.derivatives(gradient.clone().into());
                rhs.derivatives((-gradient).into());
            },
            LayerOps::Mul{lhs, rhs} =>
            {
                let lhs_value = lhs.value_clone();
                {
                    let d = rhs.value_clone() * &gradient;
                    lhs.derivatives(d.into());
                }

                {
                    let d = lhs_value * &gradient;
                    rhs.derivatives(d.into());
                }
            },
            LayerOps::Div{lhs, rhs} =>
            {
                let r_recip = rhs.value_clone().reciprocal();
                let lhs_value = lhs.value_clone();

                {
                    let d = &r_recip * &gradient;

                    lhs.derivatives(d);
                }

                {
                    // my favorite syntax
                    let recip_squared =
                        <&GradientType as Mul<&GradientType>>::mul(&r_recip, &r_recip);

                    let d = -lhs_value * &gradient;
                    let d = <GradientType as Mul<GradientType>>::mul(d, recip_squared);

                    rhs.derivatives(d);
                }
            },
            LayerOps::Exp(x) =>
            {
                let d = self.value.as_ref().unwrap();

                x.derivatives((gradient * d).into());
            },
            LayerOps::Sigmoid(x) =>
            {
                let value = self.value.as_ref().unwrap();

                // sigmoid(x) * (1.0 - sigmoid(x))
                let d = (-value + 1.0) * value;

                x.derivatives((gradient * &d).into());
            },
            LayerOps::Tanh(x) =>
            {
                let value = self.value.as_ref().unwrap();

                // 1 - tanh^2(x)
                let d = -(value * value) + 1.0;

                x.derivatives((gradient * &d).into());
            },
            LayerOps::LeakyRelu(x) =>
            {
                let d = x.value_clone().leaky_relu_d();

                x.derivatives(d * &gradient);
            },
            LayerOps::Ln(x) =>
            {
                let d = x.value_clone().reciprocal();

                x.derivatives(d * &gradient);
            },
            LayerOps::Neg(x) =>
            {
                x.derivatives((-gradient).into());
            },
            LayerOps::Matmul{lhs, rhs} =>
            {
                let gradient: LayerInnerType = gradient.try_into()
                    .expect("matmul must be a tensor");

                let rhs_d = lhs.value().matmul_transposed(&gradient);

                {
                    let d = gradient.matmul_by_transposed(&*rhs.value());
                    lhs.derivatives(d.into());
                }

                rhs.derivatives(rhs_d.into());
            },
            LayerOps::SumTensor(x) =>
            {
                let d = LayerInnerType::from_gradient(gradient.into(), ||
                {
                    x.value_clone()
                });

                x.derivatives(d);
            },
            // wait is the derivative for this the EXACT same as normal elementwise multiplication?
            LayerOps::Dot{lhs, rhs} =>
            {
                let gradient = LayerInnerType::from_gradient(gradient.into(), ||
                {
                    lhs.value_clone()
                });

                let lhs_value = lhs.value_clone();
                // nice and short multplication calls
                {
                    let d = &gradient * rhs.value_clone();
 
                    lhs.derivatives(d.into());
                }

                {
                    let d = &gradient * lhs_value;

                    rhs.derivatives(d.into());
                }
            },
            LayerOps::Diff => (),
            LayerOps::None => ()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffWrapper<T>(Option<Rc<RefCell<DiffType<T>>>>);

impl<T> DiffWrapper<T>
where
    T: DiffBounds,
    <T as TryInto<LayerInnerType>>::Error: Debug,
    for<'a> T: Mul<&'a T, Output=T> + Add<f32, Output=T> + Neg<Output=T>,
    for<'a> &'a T: Mul<&'a T, Output=T> + Neg<Output=T>,
    for<'a> GradientType: Mul<&'a T, Output=GradientType>,
    for<'a> &'a GradientType: Mul<&'a T, Output=GradientType>
{
    pub fn new_diff(value: T) -> Self
    {
        Self::new_inner(value, LayerOps::Diff, true)
    }

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
        self.this_move().calculate_gradients()
    }

    fn this_move(self) -> DiffType<T>
    {
        Rc::into_inner(
            self.0.unwrap()
        ).expect("this wrapper must have no parents").into_inner()
    }

    fn new_inner(value: T, ops: LayerOps, calculate_gradient: bool) -> Self
    {
        let diff = DiffType{
            value: Some(value),
            inner: ops,
            gradient: None,
            calculate_gradient
        };

        Self(Some(Rc::new(RefCell::new(diff))))
    }

    fn derivatives(&mut self, gradient: T)
    {
        let multiple_parents = Rc::strong_count(self.0.as_ref().unwrap()) > 1;

        {
            let mut this = RefCell::borrow_mut(self.0.as_ref().unwrap());

            this.derivatives(gradient, multiple_parents);
        }

        self.drop_child();
    }

    // nice name lmao
    fn drop_child(&mut self)
    {
        // drop this pointer to decrease the parents amount
        let _ = self.0.take();
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
            RefCell::borrow_mut(self.0.as_ref().unwrap()),
            |v| v.value.as_mut().unwrap()
        )
    }

    pub fn value_clone(&self) -> T
    {
        (*RefCell::borrow(self.0.as_ref().unwrap())).value.clone().unwrap()
    }

    fn value_take(&mut self) -> T
    {
        (*RefCell::borrow_mut(self.0.as_ref().unwrap())).value.take().unwrap()
    }
}

impl<T> DiffWrapper<T>
{
    fn maybe_gradient<A>(&self, rhs: &DiffWrapper<A>) -> bool
    {
        self.this_ref().maybe_gradient(&rhs.this_ref())
    }

    fn is_gradient(&self) -> bool
    {
        self.this_ref().calculate_gradient
    }

    #[allow(dead_code)]
    fn this_ref(&self) -> cell::Ref<DiffType<T>>
    {
        RefCell::borrow(self.0.as_ref().unwrap())
    }

    fn this_mut(&mut self) -> cell::RefMut<DiffType<T>>
    {
        RefCell::borrow_mut(self.0.as_ref().unwrap())
    }
}

#[derive(Debug)]
pub struct SoftmaxedLayer(pub LayerType);

impl SoftmaxedLayer
{
    #[allow(dead_code)]
    pub fn new(mut layer: LayerType) -> Self
    {
        Self::softmax(&mut layer);
        Self(layer)
    }

    pub fn softmax(layer: &mut LayerType) 
    {
        layer.exp();
        let s = layer.sum();

        *layer /= s;
    }

    #[allow(dead_code)]
    pub fn from_raw(layer: LayerType) -> Self
    {
        Self(layer)
    }

    #[allow(dead_code)]
    pub fn new_empty(size: usize) -> Self
    {
        Self(LayerType::new(size, 1))
    }

    #[allow(dead_code)]
    pub fn pick_weighed(&self, temperature: f32) -> usize
    {
        Self::pick_weighed_associated(&self.0.value(), temperature)
    }

    pub fn pick_weighed_associated(values: &LayerInnerType, temperature: f32) -> usize
    {
        let values = values / temperature;

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

impl<T> Add<T> for ScalarType
where
    T: Borrow<Self>
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        let is_gradient = self.maybe_gradient(rhs);
        Self::new_inner(
            self.value_clone() + rhs.value_clone(),
            LayerOps::Add{
                lhs: LayerChild::Scalar(self),
                rhs: LayerChild::Scalar(rhs.clone())
            },
            is_gradient
        )
    }
}

impl<T> Sub<T> for ScalarType
where
    T: Borrow<LayerType>
{
    type Output = LayerType;

    fn sub(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        let is_gradient = self.maybe_gradient(rhs);
        LayerType::new_inner(
            -(&*rhs.value()) + self.value_clone(),
            LayerOps::Sub{
                lhs: LayerChild::Scalar(self),
                rhs: LayerChild::Tensor(rhs.clone())
            },
            is_gradient
        )
    }
}

impl Neg for ScalarType
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        let is_gradient = self.is_gradient();
        Self::new_inner(
            // im unwrapping cuz i dont want it to proceed if value is None
            -self.value_clone(),
            LayerOps::Neg(LayerChild::Scalar(self)),
            is_gradient
        )
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

pub type LayerType = DiffWrapper<LayerInnerType>;

impl<T> Add<T> for LayerType
where
    T: Borrow<Self>
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        let value = {
            let l_value = self.value();
            let r_value = rhs.value();

            &*l_value + &*r_value
        };

        let is_gradient = self.maybe_gradient(rhs);
        Self::new_inner(
            value,
            LayerOps::Add{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Tensor(rhs.clone())
            }.into(),
            is_gradient
        )
    }
}

impl<T> Add<T> for &LayerType
where
    T: Borrow<LayerType>
{
    type Output = LayerType;

    fn add(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        let value = {
            let l_value = self.value();
            let r_value = rhs.value();

            &*l_value + &*r_value
        };

        LayerType::new_inner(
            value,
            LayerOps::Add{
                lhs: LayerChild::Tensor(self.clone()),
                rhs: LayerChild::Tensor(rhs.clone())
            }.into(),
            self.maybe_gradient(rhs)
        )
    }
}

impl Add<ScalarType> for LayerType
{
    type Output = Self;

    fn add(self, rhs: ScalarType) -> Self::Output
    {
        let value = {
            let l_value = self.value();

            &*l_value + rhs.value_clone()
        };

        let is_gradient = self.maybe_gradient(&rhs);
        Self::new_inner(
            value,
            LayerOps::Add{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Scalar(rhs)
            },
            is_gradient
        )
    }
}

impl<T> Sub<T> for LayerType
where
    T: Borrow<Self>
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        let value = {
            let l_value = self.value();
            let r_value = rhs.value();

            &*l_value - &*r_value
        };

        let is_gradient = self.maybe_gradient(&rhs);
        Self::new_inner(
            value,
            LayerOps::Sub{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Tensor(rhs.clone())
            },
            is_gradient
        )
    }
}

impl<T> Sub<T> for &LayerType
where
    T: Borrow<LayerType>
{
    type Output = LayerType;

    fn sub(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        let value = {
            let l_value = self.value();
            let r_value = rhs.value();

            &*l_value - &*r_value
        };

        LayerType::new_inner(
            value,
            LayerOps::Sub{
                lhs: LayerChild::Tensor(self.clone()),
                rhs: LayerChild::Tensor(rhs.clone())
            },
            self.maybe_gradient(rhs)
        )
    }
}

impl Sub<ScalarType> for LayerType
{
    type Output = Self;

    fn sub(self, rhs: ScalarType) -> Self::Output
    {
        let value = {
            let l_value = self.value();

            &*l_value - rhs.value_clone()
        };

        let is_gradient = self.maybe_gradient(&rhs);
        Self::new_inner(
            value,
            LayerOps::Sub{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Scalar(rhs)
            },
            is_gradient
        )
    }
}

impl<T> Mul<T> for LayerType
where
    T: Borrow<Self>
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        let value = {
            let l_value = self.value();
            let r_value = rhs.value();

            &*l_value * &*r_value
        };

        let is_gradient = self.maybe_gradient(&rhs);
        Self::new_inner(
            value,
            LayerOps::Mul{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Tensor(rhs.borrow().clone())
            },
            is_gradient
        )
    }
}

impl<T> Mul<T> for &LayerType
where
    T: Borrow<LayerType>
{
    type Output = LayerType;

    fn mul(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        let value = {
            let l_value = self.value();
            let r_value = rhs.value();

            &*l_value * &*r_value
        };

        LayerType::new_inner(
            value,
            LayerOps::Mul{
                lhs: LayerChild::Tensor(self.clone()),
                rhs: LayerChild::Tensor(rhs.clone())
            },
            self.maybe_gradient(rhs)
        )
    }
}

impl Mul<ScalarType> for LayerType
{
    type Output = Self;

    fn mul(self, rhs: ScalarType) -> Self::Output
    {
        let value = {
            let l_value = self.value();

            &*l_value * rhs.value_clone()
        };

        let is_gradient = self.maybe_gradient(&rhs);
        Self::new_inner(
            value,
            LayerOps::Mul{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Scalar(rhs)
            },
            is_gradient
        )
    }
}

impl Mul<ScalarType> for &LayerType
{
    type Output = LayerType;

    fn mul(self, rhs: ScalarType) -> Self::Output
    {
        let value = {
            let l_value = self.value();

            &*l_value * rhs.value_clone()
        };

        let is_gradient = self.maybe_gradient(&rhs);
        LayerType::new_inner(
            value,
            LayerOps::Mul{
                lhs: LayerChild::Tensor(self.clone()),
                rhs: LayerChild::Scalar(rhs)
            },
            is_gradient
        )
    }
}

impl<T> Div<T> for LayerType
where
    T: Borrow<Self>
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        let value = {
            let l_value = self.value();
            let r_value = rhs.value();

            &*l_value / &*r_value
        };

        let is_gradient = self.maybe_gradient(&rhs);
        Self::new_inner(
            value,
            LayerOps::Div{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Tensor(rhs.clone())
            },
            is_gradient
        )
    }
}

impl<T> Div<T> for &LayerType
where
    T: Borrow<LayerType>
{
    type Output = LayerType;

    fn div(self, rhs: T) -> Self::Output
    {
        let rhs = rhs.borrow();

        let value = {
            let l_value = self.value();
            let r_value = rhs.value();

            &*l_value / &*r_value
        };

        LayerType::new_inner(
            value,
            LayerOps::Div{
                lhs: LayerChild::Tensor(self.clone()),
                rhs: LayerChild::Tensor(rhs.clone())
            },
            self.maybe_gradient(rhs)
        )
    }
}

impl Div<ScalarType> for LayerType
{
    type Output = Self;

    fn div(self, rhs: ScalarType) -> Self::Output
    {
        let value = {
            let l_value = self.value();

            &*l_value / rhs.value_clone()
        };

        let is_gradient = self.maybe_gradient(&rhs);
        Self::new_inner(
            value,
            LayerOps::Div{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Scalar(rhs)
            },
            is_gradient
        )
    }
}

impl Div<ScalarType> for &LayerType
{
    type Output = LayerType;

    fn div(self, rhs: ScalarType) -> Self::Output
    {
        let value = {
            let l_value = self.value();

            &*l_value / rhs.value_clone()
        };

        let is_gradient = self.maybe_gradient(&rhs);
        LayerType::new_inner(
            value,
            LayerOps::Div{
                lhs: LayerChild::Tensor(self.clone()),
                rhs: LayerChild::Scalar(rhs)
            },
            is_gradient
        )
    }
}

impl AddAssign for LayerType
{
    fn add_assign(&mut self, rhs: Self)
    {
        let is_gradient = self.maybe_gradient(&rhs);
        *self = Self::new_inner(
            self.value_clone() + rhs.value_clone(),
            LayerOps::Add{
                lhs: LayerChild::Tensor(self.clone()),
                rhs: LayerChild::Tensor(rhs)
            },
            is_gradient
        );
    }
}

impl DivAssign<ScalarType> for LayerType
{
    fn div_assign(&mut self, rhs: ScalarType)
    {
        let is_gradient = self.maybe_gradient(&rhs);
        *self = Self::new_inner(
            self.value_clone() / rhs.value_clone(),
            LayerOps::Div{
                lhs: LayerChild::Tensor(self.clone()),
                rhs: LayerChild::Scalar(rhs)
            },
            is_gradient
        );
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

    pub fn matmul(&self, rhs: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();

        let r_value = rhs.value();
        Self::new_inner(
            self.value().matmul(&*r_value),
            LayerOps::Matmul{
                lhs: self.clone(),
                rhs: rhs.clone()
            },
            self.maybe_gradient(rhs)
        )
    }

    pub fn dot(self, rhs: Self) -> ScalarType
    {
        let value = {
            let r_value = rhs.value();

            self.value_clone().dot(&r_value)
        };

        let is_gradient = self.maybe_gradient(&rhs);
        ScalarType::new_inner(
            value,
            LayerOps::Dot{
                lhs: self,
                rhs
            },
            is_gradient
        )
    }

    pub fn exp(&mut self)
    {
        let mut value = self.value_clone();
        value.exp();

        *self = Self::new_inner(
            value,
            LayerOps::Exp(LayerChild::Tensor(self.clone())),
            self.is_gradient()
        );
    }

    pub fn ln(&mut self)
    {
        let mut value = self.value_clone();
        value.ln();

        *self = Self::new_inner(
            value,
            LayerOps::Ln(LayerChild::Tensor(self.clone())),
            self.is_gradient()
        );
    }

    pub fn sigmoid(&mut self)
    {
        let mut value = self.value_clone();
        value.sigmoid();

        *self = Self::new_inner(
            value,
            LayerOps::Sigmoid(LayerChild::Tensor(self.clone())),
            self.is_gradient()
        );
    }

    pub fn tanh(&mut self)
    {
        let mut value = self.value_clone();
        value.tanh();

        *self = Self::new_inner(
            value,
            LayerOps::Tanh(LayerChild::Tensor(self.clone())),
            self.is_gradient()
        );
    }

    pub fn leaky_relu(&mut self)
    {
        let mut value = self.value_clone();
        value.leaky_relu();

        *self = Self::new_inner(
            value,
            LayerOps::LeakyRelu(LayerChild::Tensor(self.clone())),
            self.is_gradient()
        );
    }

    pub fn sum(&self) -> ScalarType
    {
        ScalarType::new_inner(
            self.value().sum(),
            LayerOps::SumTensor(self.clone()),
            self.is_gradient()
        )
    }

    pub fn total_len(&self) -> usize
    {
        self.value().total_len()
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.value().as_vec()
    }

    pub fn pick_weighed(&self, temperature: f32) -> usize
    {
        self.value().pick_weighed(temperature)
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
    use crate::neural_network::gru::tests::close_enough;

    const LAYER_PREV: usize = 10;
    const LAYER_CURR: usize = 10;

    fn compare_single(correct: f32, calculated: f32)
    {
        let epsilon = 0.1;
        assert!(
            close_enough(correct, calculated, epsilon),
            "correct: {}, calculated: {}",
            correct, calculated
        );
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

        let compare_tensor = |correct: LayerInnerType, calculated: LayerInnerType, index|
        {
            compare_single(correct.as_vec()[index], calculated.as_vec()[index]);
        };

        for epsilon_index in 0..orig.total_len()
        {
            let epsilon: f32 = 0.001;

            let mut fg = |value|
            {
                let epsilon = one_hot(orig.clone(), epsilon_index, epsilon, 1.0);
                (value - orig.clone()) / epsilon
            };

            let fg = &mut fg;

            let a_epsilon = one_hot(a.value_clone(), epsilon_index, epsilon, 0.0);
            let b_epsilon = one_hot(b.value_clone(), epsilon_index, epsilon, 0.0);

            let a_fg = {
                let mut a = LayerType::new_diff(a.value_clone() + a_epsilon);
                fg(vals(&mut a, &mut b))
            };

            let b_fg = {
                let mut b = LayerType::new_diff(b.value_clone() + b_epsilon);
                fg(vals(&mut a, &mut b))
            };

            eprintln!("derivative of a");
            compare_tensor(a_fg, a_g.clone(), epsilon_index);

            eprintln!("derivative of b");
            compare_tensor(b_fg, b_g.clone(), epsilon_index);
        }
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
        let m = if fastrand::bool() {1.0} else {-1.0};
        (fastrand::f32() + 0.05) * m
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
            let mut a = a.clone();
            a /= b.sum();

            a
        })
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
        check_tensor(|a, b| a.clone() + b.sum())
    }

    #[test]
    fn sum_tensor_product_negative()
    {
        check_tensor(|a, b| a * -b.sum())
    }

    #[test]
    fn dot_product()
    {
        check_tensor(|a, b| a.clone() + a.clone().dot(b.clone()))
    }

    #[test]
    fn dot_product_complex()
    {
        check_tensor(|a, b| a.clone() + a * a.clone().dot(b.clone()) - b)
    }

    #[test]
    fn scalar_minus_tensor()
    {
        check_tensor(|a, b| a.sum() - b)
    }

    #[test]
    fn exponential()
    {
        check_tensor(|a, b|
        {
            let mut a = a.clone();
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

            let mut a = a.clone();
            a.ln();

            a + b
        })
    }

    #[test]
    fn leaky_relu()
    {
        check_tensor(|a, b|
        {
            let mut a = a.clone();
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
            let mut a = a.clone();
            a.sigmoid();

            a + b
        })
    }

    #[test]
    fn hyperbolic_tangent()
    {
        check_tensor(|a, b|
        {
            let mut a = a.clone();
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
                    let predicted = predicted.clone();
                    predicted.dot(target.clone())
                }).sum();

            a * s
        })
    }

    #[test]
    fn matrix_multiplication()
    {
        check_tensor_with_dims((2, 10), (10, 1), |a, b| a.matmul(b) + b.sum())
    }
}
