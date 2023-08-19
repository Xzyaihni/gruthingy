use std::{
    f32,
    iter,
    rc::Rc,
    cell::{self, RefCell},
    borrow::Borrow,
    ops::{Mul, Add, Sub, Div, AddAssign, DivAssign, Neg}
};

use nalgebra::DMatrix;

use serde::{Serialize, Deserialize};


pub type LayerInnerType = MatrixWrapper;

pub const LEAKY_SLOPE: f32 = 0.01;

// i have no clue where else to put this
fn leaky_relu_d(value: f32) -> f32
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
            Self::Tensor(x) => x.derivatives(gradient),
            Self::Scalar(x) => x.derivatives(gradient)
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

#[derive(Debug, Serialize, Deserialize)]
pub struct DiffType<T>
{
    inner: LayerOps,
    value: Option<T>,
    gradient: Option<GradientType>
}

pub trait DiffBounds
where
    Self: Onesable + Clone,
    for<'a> Self: Mul<&'a Self, Output=Self> + Add<f32, Output=Self> + Neg<Output=Self>,
    for<'a> &'a Self: Mul<&'a Self, Output=Self> + Neg<Output=Self>,
    for<'a> GradientType: Mul<&'a Self, Output=GradientType>
{
}

impl DiffBounds for f32 {}
impl DiffBounds for LayerInnerType {}

impl<T> DiffType<T>
where
    T: DiffBounds,
    for<'a> T: Mul<&'a T, Output=T> + Add<f32, Output=T> + Neg<Output=T>,
    for<'a> &'a T: Mul<&'a T, Output=T> + Neg<Output=T>,
    for<'a> GradientType: Mul<&'a T, Output=GradientType>
{
    pub fn calculate_gradients(mut self)
    {
        let ones = self.value.as_ref().unwrap().ones();

        self.derivatives(ones);
    }

    // long ass name but i wanna be descriptive about wut this does
    pub fn take_gradient_tensor(&mut self) -> LayerInnerType
    {
        match self.gradient.take().expect("must have a gradient calculated")
        {
            GradientType::Tensor(x) => x,
            _ => panic!("invalid gradient type requested")
        }
    }

    fn derivatives(&mut self, gradient: GradientType)
    {
        if self.gradient.is_none()
        {
            self.gradient = Some(gradient.clone());
        } else
        {
            *self.gradient.as_mut().unwrap() += gradient.clone();
        }

        match &mut self.inner
        {
            LayerOps::Add{lhs, rhs} =>
            {
                lhs.derivatives(gradient.clone());
                rhs.derivatives(gradient);
            },
            LayerOps::Sub{lhs, rhs} =>
            {
                lhs.derivatives(gradient.clone());
                rhs.derivatives(-gradient);
            },
            LayerOps::Mul{lhs, rhs} =>
            {
                {
                    let d = &gradient * rhs.value_clone();
                    lhs.derivatives(d);
                }

                {
                    let d = &gradient * lhs.value_clone();
                    rhs.derivatives(d);
                }
            },
            LayerOps::Div{lhs, rhs} =>
            {
                let r_recip = rhs.value_clone().reciprocal();

                {
                    let d = &r_recip * &gradient;

                    lhs.derivatives(d);
                }

                {
                    // my favorite syntax
                    let recip_squared =
                        <&GradientType as Mul<&GradientType>>::mul(&r_recip, &r_recip);

                    let d = &gradient * -lhs.value_clone();
                    let d = <GradientType as Mul<GradientType>>::mul(d, recip_squared);

                    rhs.derivatives(d);
                }
            },
            LayerOps::Exp(x) =>
            {
                let d = self.value.as_ref().unwrap();

                x.derivatives(gradient * d);
            },
            LayerOps::Sigmoid(x) =>
            {
                let value = self.value.as_ref().unwrap();

                // sigmoid(x) * (1.0 - sigmoid(x))
                let d = (-value + 1.0) * value;

                x.derivatives(gradient * &d);
            },
            LayerOps::Tanh(x) =>
            {
                let value = self.value.as_ref().unwrap();

                // 1 - tanh^2(x)
                let d = -(value * value) + 1.0;

                x.derivatives(gradient * &d);
            },
            LayerOps::LeakyRelu(x) =>
            {
                let d = x.value_clone().leaky_relu_d();

                x.derivatives(<&GradientType as Mul<GradientType>>::mul(&gradient, d));
            },
            LayerOps::Ln(x) =>
            {
                let d = x.value_clone().reciprocal();

                // why does it want fully qualified syntax im so DONE
                x.derivatives(<&GradientType as Mul<GradientType>>::mul(&gradient, d));
            },
            LayerOps::Neg(x) =>
            {
                x.derivatives(-gradient);
            },
            LayerOps::Matmul{lhs, rhs} =>
            {
                let gradient = match gradient
                {
                    GradientType::Tensor(x) => x,
                    x => panic!("matmul must have a tensor gradient, instead it has {:?}", x)
                };

                {
                    let d = rhs.value().matmul_derivative(&gradient);
                    lhs.derivatives(d.into());
                }

                {
                    let d = lhs.value().matmul_transposed(gradient);
                    rhs.derivatives(d.into());
                }
            },
            LayerOps::SumTensor(x) =>
            {
                let d = match gradient
                {
                    GradientType::Tensor(gradient) => gradient,
                    GradientType::Scalar(gradient) =>
                    {
                        let mut x = x.value_clone();
                        x.fill(gradient);

                        x
                    }
                };

                x.derivatives(d.into());
            },
            // wait is the derivative for this the EXACT same as normal elementwise multiplication?
            LayerOps::Dot{lhs, rhs} =>
            {
                // nice and short multplication calls
                {
                    let d = <&GradientType as Mul<&LayerInnerType>>::mul(
                        &gradient,
                        &rhs.value_clone()
                    );

                    lhs.derivatives(d);
                }

                {
                    let d = <&GradientType as Mul<&LayerInnerType>>::mul(
                        &gradient,
                        &lhs.value_clone()
                    );

                    rhs.derivatives(d);
                }
            },
            LayerOps::Diff =>
            {
                dbg!("le diff matrix has arrived");
            },
            LayerOps::None =>
            {
                dbg!("it shouldnt print this after i optimize stuff");
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffWrapper<T>(Rc<RefCell<DiffType<T>>>);

impl<T> DiffWrapper<T>
where
    T: DiffBounds,
    for<'a> T: Mul<&'a T, Output=T> + Add<f32, Output=T> + Neg<Output=T>,
    for<'a> &'a T: Mul<&'a T, Output=T> + Neg<Output=T>,
    for<'a> GradientType: Mul<&'a T, Output=GradientType>
{
    pub fn new_diff(value: T) -> Self
    {
        Self::new_inner(value, LayerOps::Diff)
    }

    pub fn clear(&mut self)
    {
        *self = Self::new_diff(self.value_take());
    }

    pub fn take_gradient_tensor(&mut self) -> LayerInnerType
    {
        self.this_mut().take_gradient_tensor()
    }

    pub fn calculate_gradients(self)
    {
        self.this_move().calculate_gradients()
    }

    fn this_move(self) -> DiffType<T>
    {
        Rc::into_inner(self.0).expect("this wrapper must have no parents").into_inner()
    }

    fn new_inner(value: T, ops: LayerOps) -> Self
    {
        let diff = DiffType{
            value: Some(value),
            inner: ops,
            gradient: None
        };

        Self(Rc::new(RefCell::new(diff)))
    }

    fn derivatives(&mut self, gradient: GradientType)
    {
        RefCell::borrow_mut(&self.0).derivatives(gradient);
    }

    #[allow(dead_code)]
    fn this_ref(&self) -> cell::Ref<DiffType<T>>
    {
        RefCell::borrow(&self.0)
    }

    fn this_mut(&mut self) -> cell::RefMut<DiffType<T>>
    {
        RefCell::borrow_mut(&self.0)
    }

    pub fn value(&self) -> cell::Ref<T>
    {
        cell::Ref::map(RefCell::borrow(&self.0), |v| v.value.as_ref().unwrap())
    }

    pub fn value_mut(&mut self) -> cell::RefMut<T>
    {
        cell::RefMut::map(RefCell::borrow_mut(&self.0), |v| v.value.as_mut().unwrap())
    }

    pub fn value_clone(&self) -> T
    {
        (*RefCell::borrow(&self.0)).value.clone().unwrap()
    }

    fn value_take(&mut self) -> T
    {
        (*RefCell::borrow_mut(&self.0)).value.take().unwrap()
    }
}

pub trait Onesable
{
    fn ones(&self) -> GradientType;
}

impl Onesable for LayerInnerType
{
    fn ones(&self) -> GradientType
    {
        let mut x = self.clone();
        x.fill(1.0);

        GradientType::Tensor(x)
    }
}

impl Onesable for f32
{
    fn ones(&self) -> GradientType
    {
        GradientType::Scalar(1.0)
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixWrapper(DMatrix<f32>);

impl Add<f32> for MatrixWrapper
{
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output
    {
        Self(self.0.add_scalar(rhs))
    }
}

impl Add<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn add(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper(self.0.add_scalar(rhs))
    }
}

impl<T> Add<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output
    {
        Self(self.0 + &rhs.borrow().0)
    }
}

impl<T> Add<T> for &MatrixWrapper
where
    T: Borrow<MatrixWrapper>
{
    type Output = MatrixWrapper;

    fn add(self, rhs: T) -> Self::Output
    {
        MatrixWrapper(&self.0 + &rhs.borrow().0)
    }
}

impl<T> Sub<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output
    {
        Self(self.0 - &rhs.borrow().0)
    }
}

impl<T> Sub<T> for &MatrixWrapper
where
    T: Borrow<MatrixWrapper>
{
    type Output = MatrixWrapper;

    fn sub(self, rhs: T) -> Self::Output
    {
        MatrixWrapper(&self.0 - &rhs.borrow().0)
    }
}

impl Sub<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn sub(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper(self.0.add_scalar(-rhs))
    }
}

impl<T> Mul<T> for &MatrixWrapper
where
    T: Borrow<MatrixWrapper>
{
    type Output = MatrixWrapper;

    fn mul(self, rhs: T) -> Self::Output
    {
        MatrixWrapper(self.0.component_mul(&rhs.borrow().0))
    }
}

impl Mul<f32> for MatrixWrapper
{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output
    {
        Self(self.0 * rhs)
    }
}

impl Mul<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn mul(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper(&self.0 * rhs)
    }
}

impl<T> Mul<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output
    {
        Self(self.0.component_mul(&rhs.borrow().0))
    }
}

impl Div<f32> for MatrixWrapper
{
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output
    {
        Self(self.0 / rhs)
    }
}

impl Div<f32> for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn div(self, rhs: f32) -> Self::Output
    {
        MatrixWrapper(&self.0 / rhs)
    }
}

impl<T> Div<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output
    {
        Self(self.0.component_div(&rhs.borrow().0))
    }
}

impl<T> Div<T> for &MatrixWrapper
where
    T: Borrow<MatrixWrapper>
{
    type Output = MatrixWrapper;

    fn div(self, rhs: T) -> Self::Output
    {
        MatrixWrapper(self.0.component_div(&rhs.borrow().0))
    }
}

impl<T> AddAssign<T> for MatrixWrapper
where
    T: Borrow<Self>
{
    fn add_assign(&mut self, rhs: T)
    {
        self.0 += &rhs.borrow().0;
    }
}

impl DivAssign<f32> for MatrixWrapper
{
    fn div_assign(&mut self, rhs: f32)
    {
        self.0 /= rhs;
    }
}

impl Neg for MatrixWrapper
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        Self(-self.0)
    }
}

impl Neg for &MatrixWrapper
{
    type Output = MatrixWrapper;

    fn neg(self) -> Self::Output
    {
        MatrixWrapper(-&self.0)
    }
}

impl MatrixWrapper
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        Self(DMatrix::zeros(previous_size, this_size))
    }

    pub fn new_with<F: FnMut() -> f32>(
        previous_size: usize,
        this_size: usize,
        mut f: F
    )-> Self
    {
        Self(DMatrix::from_fn(previous_size, this_size, |_, _| f()))
    }

    pub fn from_raw<V: Into<Vec<f32>>>(values: V, previous_size: usize, this_size: usize) -> Self
    {
        Self(DMatrix::from_vec(previous_size, this_size, values.into()))
    }

    pub fn swap_raw_values<V: Into<Vec<f32>>>(&mut self, values: V)
    {
        self.0.copy_from_slice(&values.into());
    }

    pub fn fill(&mut self, value: f32)
    {
        self.0.fill(value);
    }

    pub fn matmul(&self, rhs: impl Borrow<Self>) -> Self
    {
        Self(&self.0 * &rhs.borrow().0)
    }

    pub fn matmul_transposed(&self, rhs: impl Borrow<Self>) -> Self
    {
        Self(self.0.tr_mul(&rhs.borrow().0))
    }

    pub fn matmul_derivative(&self, rhs: impl Borrow<Self>) -> Self
    {
        Self(&rhs.borrow().0 * self.0.transpose())
    }

    pub fn dot(self, rhs: &Self) -> f32
    {
        self.0.dot(&rhs.0)
    }

    pub fn sqrt(&mut self)
    {
        self.0.apply(|v| *v = v.sqrt());
    }

    pub fn clone_sqrt(&self) -> Self
    {
        let mut out = self.clone();
        out.sqrt();

        out
    }

    pub fn exp(&mut self)
    {
        self.0.apply(|v| *v = v.exp());
    }

    pub fn ln(&mut self)
    {
        self.0.apply(|v| *v = v.ln());
    }

    pub fn reciprocal(&mut self)
    {
        self.0.apply(|v| *v = 1.0 / *v);
    }

    pub fn sigmoid(&mut self)
    {
        self.0.apply(|v| *v = 1.0 / (1.0 + (-*v).exp()));
    }

    pub fn tanh(&mut self)
    {
        self.0.apply(|v| *v = v.tanh());
    }

    pub fn leaky_relu(&mut self)
    {
        self.0.apply(|v| *v = v.max(LEAKY_SLOPE * *v));
    }

    pub fn leaky_relu_d(&mut self)
    {
        self.0.apply(|v| *v = leaky_relu_d(*v));
    }

    pub fn sum(&self) -> f32
    {
        self.0.sum()
    }

    pub fn total_len(&self) -> usize
    {
        self.0.as_slice().len()
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.0.as_slice().to_vec()
    }

    pub fn iter(&self) -> impl Iterator<Item=&f32> + ExactSizeIterator
    {
        self.0.as_slice().iter()
    }

    pub fn pick_weighed(&self, temperature: f32) -> usize
    {
        SoftmaxedLayer::pick_weighed_associated(self, temperature)
    }

    pub fn highest_index(&self) -> usize
    {
        SoftmaxedLayer::highest_index(self.iter())
    }
}

pub type ScalarType = DiffWrapper<f32>;

impl ScalarType
{
    pub fn new(value: f32) -> Self
    {
        ScalarType::new_inner(value, LayerOps::None)
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
        Self::new_inner(
            self.value_clone() + rhs.value_clone(),
            LayerOps::Add{
                lhs: LayerChild::Scalar(self),
                rhs: LayerChild::Scalar(rhs.clone())
            }
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
        LayerType::new_inner(
            -(&*rhs.value()) + self.value_clone(),
            LayerOps::Sub{
                lhs: LayerChild::Scalar(self),
                rhs: LayerChild::Tensor(rhs.clone())
            }
        )
    }
}

impl Neg for ScalarType
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        Self::new_inner(
            // im unwrapping cuz i dont want it to proceed if value is None
            -self.value_clone(),
            LayerOps::Neg(LayerChild::Scalar(self))
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

        Self::new_inner(
            value,
            LayerOps::Add{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Tensor(rhs.clone())
            }.into()
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
            }.into()
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

        Self::new_inner(
            value,
            LayerOps::Add{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Scalar(rhs)
            }
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

        Self::new_inner(
            value,
            LayerOps::Sub{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Tensor(rhs.clone())
            }
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
            }
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

        Self::new_inner(
            value,
            LayerOps::Sub{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Scalar(rhs)
            }
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

        Self::new_inner(
            value,
            LayerOps::Mul{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Tensor(rhs.borrow().clone())
            }
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
            }
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

        Self::new_inner(
            value,
            LayerOps::Mul{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Scalar(rhs)
            }
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

        LayerType::new_inner(
            value,
            LayerOps::Mul{
                lhs: LayerChild::Tensor(self.clone()),
                rhs: LayerChild::Scalar(rhs)
            }
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

        Self::new_inner(
            value,
            LayerOps::Div{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Tensor(rhs.clone())
            }
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
            }
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

        Self::new_inner(
            value,
            LayerOps::Div{
                lhs: LayerChild::Tensor(self),
                rhs: LayerChild::Scalar(rhs)
            }
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

        LayerType::new_inner(
            value,
            LayerOps::Div{
                lhs: LayerChild::Tensor(self.clone()),
                rhs: LayerChild::Scalar(rhs)
            }
        )
    }
}

impl AddAssign for LayerType
{
    fn add_assign(&mut self, rhs: Self)
    {
        let lhs = self.clone();

        let mut this = self.this_mut();

        *this.value.as_mut().unwrap() += rhs.value_clone();
        this.inner = LayerOps::Add{
            lhs: LayerChild::Tensor(lhs),
            rhs: LayerChild::Tensor(rhs)
        }.into();
    }
}

impl DivAssign<ScalarType> for LayerType
{
    fn div_assign(&mut self, rhs: ScalarType)
    {
        let lhs = self.clone();

        let mut this = self.this_mut();

        *this.value.as_mut().unwrap() /= rhs.value_clone();
        this.inner = LayerOps::Div{
            lhs: LayerChild::Tensor(lhs),
            rhs: LayerChild::Scalar(rhs)
        }.into();
    }
}

impl LayerType
{
    pub fn new(previous_size: usize, this_size: usize) -> Self
    {
        let value = MatrixWrapper::new(previous_size, this_size);
        Self::new_inner(value, LayerOps::None)
    }

    pub fn new_with<F: FnMut() -> f32>(
        previous_size: usize,
        this_size: usize,
        f: F
    )-> Self
    {
        let value = MatrixWrapper::new_with(previous_size, this_size, f);
        Self::new_inner(value, LayerOps::None)
    }

    pub fn from_raw<V: Into<Vec<f32>>>(
        values: V,
        previous_size: usize,
        this_size: usize
    ) -> Self
    {
        let value = MatrixWrapper::from_raw(values, previous_size, this_size);
        Self::new_inner(value, LayerOps::None)
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
            }
        )
    }

    pub fn dot(self, rhs: Self) -> ScalarType
    {
        let value = {
            let r_value = rhs.value();

            self.value_clone().dot(&r_value)
        };

        ScalarType::new_inner(
            value,
            LayerOps::Dot{
                lhs: self,
                rhs
            }
        )
    }

    pub fn exp(&mut self)
    {
        let mut value = self.value_clone();
        value.exp();

        *self = Self::new_inner(value, LayerOps::Exp(LayerChild::Tensor(self.clone())));
    }

    pub fn ln(&mut self)
    {
        let mut value = self.value_clone();
        value.ln();

        *self = Self::new_inner(value, LayerOps::Ln(LayerChild::Tensor(self.clone())));
    }

    pub fn sigmoid(&mut self)
    {
        let mut value = self.value_clone();
        value.sigmoid();

        *self = Self::new_inner(value, LayerOps::Sigmoid(LayerChild::Tensor(self.clone())));
    }

    pub fn tanh(&mut self)
    {
        let mut value = self.value_clone();
        value.tanh();

        *self = Self::new_inner(value, LayerOps::Tanh(LayerChild::Tensor(self.clone())));
    }

    pub fn leaky_relu(&mut self)
    {
        let mut value = self.value_clone();
        value.leaky_relu();

        *self = Self::new_inner(value, LayerOps::LeakyRelu(LayerChild::Tensor(self.clone())));
    }

    pub fn sum(&self) -> ScalarType
    {
        ScalarType::new_inner(self.value().sum(), LayerOps::SumTensor(self.clone()))
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
        let epsilon = 0.05;
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

        let a_g = a.take_gradient_tensor();
        let b_g = b.take_gradient_tensor();

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
            let epsilon: f32 = 0.0005;

            let mut fg = |value|
            {
                let epsilon = one_hot(orig.clone(), epsilon_index, epsilon, 1.0);
                (value - orig.clone()) / epsilon
            };

            let fg = &mut fg;

            let epsilon = one_hot(orig.clone(), epsilon_index, epsilon, 0.0);

            let a_fg = {
                let mut a = LayerType::new_diff(a.value_clone() + &epsilon);
                fg(vals(&mut a, &mut b))
            };

            let b_fg = {
                let mut b = LayerType::new_diff(b.value_clone() + epsilon);
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

    // i dont know wut the correct derivative is im so DONEEEEEEEEEEEEEEE
    /*#[test]
    fn matrix_multiplication()
    {
        check_tensor_with_dims((3, 10), (10, 1), |a, b| a.matmul(b))
    }*/
}
