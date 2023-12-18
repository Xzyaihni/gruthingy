use std::{
    f32,
    mem,
    iter,
    rc::Rc,
    fmt::Debug,
    cell::{self, RefCell},
    borrow::Borrow,
    ops::{Mul, Add, Sub, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg}
};

use serde::{Serialize, Deserialize};

use matrix_wrapper::MatrixWrapper;

mod matrix_wrapper;


pub type LayerType = MatrixWrapper;

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

pub trait Softmaxable
where
    Self: DivAssign<f32>
{
    fn exp(&mut self);
    fn sum(&self) -> f32;
}

#[derive(Debug)]
pub struct Softmaxer;

impl Softmaxer
{
    #[allow(dead_code)]
    pub fn softmax_temperature(layer: &mut LayerType, temperature: f32) 
    {
        *layer /= temperature;

        Self::softmax(layer)
    }

    pub fn softmax(layer: &mut impl Softmaxable) 
    {
        layer.exp();
        let s = layer.sum();

        *layer /= s;
    }

    pub fn pick_weighed_inner<I, T>(mut iter: I) -> usize
    where
        T: Borrow<f32>,
        I: Iterator<Item=T> + ExactSizeIterator
    {
        let mut c = fastrand::f32();

        let max_index = iter.len() - 1;

        iter.position(|v|
        {
            c -= v.borrow();

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

impl LayerType
{
    pub fn softmax_cross_entropy(mut self, targets: &OneHotLayer) -> (Self, f32)
    {
        Softmaxer::softmax(&mut self);
        let softmaxed = self.clone();

        // assumes that targets r either 0 or 1
        self.ln_onehot(targets);

        let s = self.dot_onehot(targets);

        (softmaxed, -s)
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum AnyDiffType
{
    Tensor(DiffType<LayerType>),
    Scalar(DiffType<f32>)
}

impl From<DiffType<LayerType>> for AnyDiffType
{
    fn from(value: DiffType<LayerType>) -> Self
    {
        Self::Tensor(value)
    }
}

impl From<DiffType<f32>> for AnyDiffType
{
    fn from(value: DiffType<f32>) -> Self
    {
        Self::Scalar(value)
    }
}

impl AnyDiffType
{
    pub fn take_gradient(&mut self) -> GradientType
    {
        match self
        {
            Self::Tensor(x) => GradientType::Tensor(x.take_gradient()),
            Self::Scalar(x) => GradientType::Scalar(x.take_gradient())
        }
    }

    pub fn take_gradient_tensor(&mut self) -> LayerType
    {
        match self
        {
            Self::Tensor(x) => x.take_gradient(),
            Self::Scalar(_) => panic!("expected tensor, got scalar")
        }
    }

    pub fn take_gradient_scalar(&mut self) -> f32
    {
        match self
        {
            Self::Scalar(x) => x.take_gradient(),
            Self::Tensor(_) => panic!("expected scalar, got tensor")
        }
    }

    pub fn calculate_gradients(self)
    {
        match self
        {
            Self::Tensor(x) => x.calculate_gradients(),
            Self::Scalar(x) => x.calculate_gradients()
        }
    }

    pub fn parent(&self) -> &Ops
    {
        match self
        {
            Self::Tensor(x) => &x.parent,
            Self::Scalar(x) => &x.parent
        }
    }

    fn derivatives(&mut self, gradient: GradientType, children_amount: usize)
    {
        match self
        {
            Self::Tensor(x) =>
            {
                x.derivatives(
                    LayerType::from_gradient(gradient, || unreachable!()),
                    children_amount
                );

                return;
            },
            Self::Scalar(_) => ()
        }

        let value = f32::from_gradient(gradient, ||
        {
            match self
            {
                Self::Tensor(x) => x.value.clone(),
                _ => unreachable!()
            }
        });

        match self
        {
            Self::Scalar(x) => x.derivatives(value, children_amount),
            _ => unreachable!()
        }
    }

    fn set_calculate_gradient(&mut self, value: bool)
    {
        match self
        {
            Self::Tensor(x) => x.calculate_gradient = value,
            Self::Scalar(x) => x.calculate_gradient = value
        }
    }

    fn is_gradient(&self) -> bool
    {
        match self
        {
            Self::Tensor(x) => x.calculate_gradient,
            Self::Scalar(x) => x.calculate_gradient
        }
    }

    fn value_clone(&self) -> GradientType
    {
        match self
        {
            Self::Tensor(x) => GradientType::Tensor(x.value.clone()),
            Self::Scalar(x) => GradientType::Scalar(x.value)
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Ops
{
    None,
    SumTensor{value: DiffWrapper},
    Neg{value: DiffWrapper},
    Exp{value: DiffWrapper},
    Ln{value: DiffWrapper},
    Sqrt{value: DiffWrapper},
    LeakyRelu{value: DiffWrapper},
    Sigmoid{value: DiffWrapper},
    Tanh{value: DiffWrapper},
    Pow{lhs: DiffWrapper, power: u32},
    Dot{lhs: DiffWrapper, rhs: DiffWrapper},
    Add{lhs: DiffWrapper, rhs: DiffWrapper},
    Sub{lhs: DiffWrapper, rhs: DiffWrapper},
    Mul{lhs: DiffWrapper, rhs: DiffWrapper},
    Div{lhs: DiffWrapper, rhs: DiffWrapper},
    Matmulv{lhs: DiffWrapper, rhs: DiffWrapper},
    MatmulvAdd{lhs: DiffWrapper, rhs: DiffWrapper, added: DiffWrapper},
    MatmulOneHotvAdd{lhs: DiffWrapper, rhs: OneHotLayer, added: DiffWrapper},
    SoftmaxCrossEntropy{
        values: DiffWrapper,
        softmaxed_values: LayerType,
        targets: OneHotLayer
    }
}

impl Ops
{
    pub fn is_none(&self) -> bool
    {
        // i am NOT using the matches macro its TRASH that gives u true if u flip args on accident
        #[allow(clippy::match_like_matches_macro)]
        match self
        {
            Self::None => true,
            _ => false
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientType
{
    Tensor(LayerType),
    Scalar(f32)
}

impl GradientType
{
    pub fn reciprocal(&self) -> Self
    {
        match self
        {
            Self::Tensor(x) => Self::Tensor(x.clone().reciprocal()),
            Self::Scalar(x) => Self::Scalar(x.recip())
        }
    }

    pub fn pow(&self, power: u32) -> Self
    {
        match self
        {
            Self::Tensor(x) =>
            {
                let mut x = x.clone();
                x.pow(power);

                Self::Tensor(x)
            },
            Self::Scalar(x) => Self::Scalar(x.powi(power as i32))
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

    pub fn tensor(&self) -> &LayerType
    {
        match self
        {
            Self::Tensor(ref x) => x,
            Self::Scalar(_) => panic!("expected tensor, got scalar")
        }
    }

    pub fn scalar(&self) -> &f32
    {
        match self
        {
            Self::Scalar(ref x) => x,
            Self::Tensor(_) => panic!("expected scalar, got tensor")
        }
    }
}

impl From<LayerType> for GradientType
{
    fn from(value: LayerType) -> Self
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

macro_rules! inplace_binary_operator_same
{
    ($this_t:ident, $trait_name:ident, $func:ident) =>
    {
        impl $trait_name for $this_t
        {
            fn $func(&mut self, rhs: Self)
            {
                match (self, rhs)
                {
                    (Self::Tensor(lhs), Self::Tensor(rhs)) =>
                    {
                        lhs.$func(rhs);
                    },
                    (Self::Scalar(lhs), Self::Scalar(rhs)) =>
                    {
                        lhs.$func(rhs);
                    },
                    x => unimplemented!("{x:?}")
                }
            }
        }
    }
}

macro_rules! binary_operator_fixed_rhs
{
    ($this_t:ident, $op_output_t:ident, $trait_name:ident, $func:ident, $output_t:ident, $rhs_t:ident) =>
    {
        impl $trait_name<$rhs_t> for $this_t
        {
            type Output = $op_output_t;

            fn $func(self, rhs: $rhs_t) -> Self::Output
            {
                match self
                {
                    $this_t::Tensor(x) => $op_output_t::Tensor(x.$func(rhs)),
                    $this_t::Scalar(x) => $op_output_t::$output_t(rhs.$func(x))
                }
            }
        }

        impl $trait_name<&$rhs_t> for $this_t
        {
            type Output = $op_output_t;

            fn $func(self, rhs: &$rhs_t) -> Self::Output
            {
                match self
                {
                    $this_t::Tensor(x) => $op_output_t::Tensor(x.$func(rhs)),
                    $this_t::Scalar(x) => $op_output_t::$output_t(rhs.$func(x))
                }
            }
        }

        impl $trait_name<$rhs_t> for &$this_t
        {
            type Output = $op_output_t;

            fn $func(self, rhs: $rhs_t) -> Self::Output
            {
                match self
                {
                    $this_t::Tensor(x) => $op_output_t::Tensor(x.$func(rhs)),
                    $this_t::Scalar(x) => $op_output_t::$output_t(rhs.$func(x))
                }
            }
        }

        impl $trait_name<&$rhs_t> for &$this_t
        {
            type Output = $op_output_t;

            fn $func(self, rhs: &$rhs_t) -> Self::Output
            {
                match self
                {
                    $this_t::Tensor(x) => $op_output_t::Tensor(x.$func(rhs)),
                    $this_t::Scalar(x) => $op_output_t::$output_t(rhs.$func(x))
                }
            }
        }
    }
}

macro_rules! binary_operator_diff
{
    ($this_t:ident, $op_output_t:ident, $trait_name:ident, $func:ident) =>
    {
        impl $trait_name<$this_t> for $this_t
        {
            type Output = $op_output_t;

            fn $func(self, rhs: $this_t) -> Self::Output
            {
                match (self, rhs)
                {
                    ($this_t::Tensor(lhs), $this_t::Tensor(rhs)) =>
                    {
                        $op_output_t::Tensor(lhs.$func(rhs))
                    },
                    ($this_t::Scalar(lhs), $this_t::Scalar(rhs)) =>
                    {
                        $op_output_t::Scalar(lhs.$func(rhs))
                    },
                    ($this_t::Tensor(lhs), $this_t::Scalar(rhs)) =>
                    {
                        $op_output_t::Tensor(lhs.$func(rhs))
                    },
                    ($this_t::Scalar(lhs), $this_t::Tensor(rhs)) =>
                    {
                        $op_output_t::Tensor(rhs.$func(lhs))
                    }
                }
            }
        }

        impl $trait_name<&$this_t> for $this_t
        {
            type Output = $op_output_t;

            fn $func(self, rhs: &$this_t) -> Self::Output
            {
                match (self, rhs)
                {
                    ($this_t::Tensor(lhs), $this_t::Tensor(rhs)) =>
                    {
                        $op_output_t::Tensor(lhs.$func(rhs))
                    },
                    ($this_t::Scalar(lhs), $this_t::Scalar(rhs)) =>
                    {
                        $op_output_t::Scalar(lhs.$func(rhs))
                    },
                    ($this_t::Tensor(lhs), $this_t::Scalar(rhs)) =>
                    {
                        $op_output_t::Tensor(lhs.$func(rhs))
                    },
                    ($this_t::Scalar(lhs), $this_t::Tensor(rhs)) =>
                    {
                        $op_output_t::Tensor(rhs.$func(lhs))
                    }
                }
            }
        }

        impl $trait_name<$this_t> for &$this_t
        {
            type Output = $op_output_t;

            fn $func(self, rhs: $this_t) -> Self::Output
            {
                match (self, rhs)
                {
                    ($this_t::Tensor(lhs), $this_t::Tensor(rhs)) =>
                    {
                        $op_output_t::Tensor(lhs.$func(rhs))
                    },
                    ($this_t::Scalar(lhs), $this_t::Scalar(rhs)) =>
                    {
                        $op_output_t::Scalar(lhs.$func(rhs))
                    },
                    ($this_t::Tensor(lhs), $this_t::Scalar(rhs)) =>
                    {
                        $op_output_t::Tensor(lhs.$func(rhs))
                    },
                    ($this_t::Scalar(lhs), $this_t::Tensor(rhs)) =>
                    {
                        $op_output_t::Tensor(rhs.$func(lhs))
                    }
                }
            }
        }

        impl $trait_name<&$this_t> for &$this_t
        {
            type Output = $op_output_t;

            fn $func(self, rhs: &$this_t) -> Self::Output
            {
                match (self, rhs)
                {
                    ($this_t::Tensor(lhs), $this_t::Tensor(rhs)) =>
                    {
                        $op_output_t::Tensor(lhs.$func(rhs))
                    },
                    ($this_t::Scalar(lhs), $this_t::Scalar(rhs)) =>
                    {
                        $op_output_t::Scalar(lhs.$func(rhs))
                    },
                    ($this_t::Tensor(lhs), $this_t::Scalar(rhs)) =>
                    {
                        $op_output_t::Tensor(lhs.$func(rhs))
                    },
                    ($this_t::Scalar(lhs), $this_t::Tensor(rhs)) =>
                    {
                        $op_output_t::Tensor(rhs.$func(lhs))
                    }
                }
            }
        }
    }
}

macro_rules! unary_operator
{
    ($this_t:ident, $op_output_t:ident, $trait_name:ident, $func:ident) =>
    {
        impl $trait_name for $this_t
        {
            type Output = $op_output_t;

            fn $func(self) -> Self::Output
            {
                match self
                {
                    $this_t::Tensor(x) => $op_output_t::Tensor(x.$func()),
                    $this_t::Scalar(x) => $op_output_t::Scalar(x.$func())
                }
            }
        }

        impl $trait_name for &$this_t
        {
            type Output = $op_output_t;

            fn $func(self) -> Self::Output
            {
                match self
                {
                    $this_t::Tensor(x) => $op_output_t::Tensor(x.$func()),
                    $this_t::Scalar(x) => $op_output_t::Scalar(x.$func())
                }
            }
        }
    }
}

// code bloat? nah we rust
inplace_binary_operator_same!(GradientType, AddAssign, add_assign);
inplace_binary_operator_same!(GradientType, SubAssign, sub_assign);

binary_operator_fixed_rhs!(GradientType, GradientType, Div, div, Scalar, f32);
binary_operator_fixed_rhs!(GradientType, GradientType, Mul, mul, Scalar, f32);

binary_operator_fixed_rhs!(GradientType, GradientType, Mul, mul, Tensor, LayerType);

binary_operator_diff!(GradientType, GradientType, Add, add);
binary_operator_diff!(GradientType, GradientType, Sub, sub);
binary_operator_diff!(GradientType, GradientType, Mul, mul);
binary_operator_diff!(GradientType, GradientType, Div, div);

unary_operator!(GradientType, GradientType, Neg, neg);


binary_operator_diff!(AnyDiffType, GradientType, Add, add);
binary_operator_diff!(AnyDiffType, GradientType, Sub, sub);
binary_operator_diff!(AnyDiffType, GradientType, Mul, mul);
binary_operator_diff!(AnyDiffType, GradientType, Div, div);

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

impl Fillable for LayerType
{
    fn fill(&mut self, value: f32)
    {
        self.fill(value);
    }
}

pub trait FromGradient
{
    fn from_gradient<F: FnOnce() -> LayerType>(
        gradient: GradientType,
        value_getter: F
    ) -> Self;
}

impl FromGradient for f32
{
    fn from_gradient<F: FnOnce() -> LayerType>(
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

impl FromGradient for LayerType
{
    fn from_gradient<F: FnOnce() -> LayerType>(
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
    Self: TryInto<LayerType> + TryInto<f32>,
    Self: AddAssign<Self> + Add<f32, Output=Self> + Neg<Output=Self>
{
    fn reciprocal(self) -> Self;
}

impl TryFrom<f32> for LayerType
{
    type Error = ();

    fn try_from(_value: f32) -> Result<Self, Self::Error>
    {
        Err(())
    }
}

impl TryFrom<LayerType> for f32
{
    type Error = ();

    fn try_from(_value: LayerType) -> Result<Self, Self::Error>
    {
        Err(())
    }
}

impl DiffBounds for f32
{
    fn reciprocal(self) -> Self
    {
        self.recip()
    }
}

impl DiffBounds for LayerType
{
    fn reciprocal(mut self) -> Self
    {
        LayerType::reciprocal(&mut self);

        self
    }
}

// damn that sure is one hot layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneHotLayer
{
    pub positions: Box<[usize]>,
    pub size: usize
}

impl OneHotLayer
{
    pub fn new(positions: impl Into<Box<[usize]>>, size: usize) -> Self
    {
        Self{positions: positions.into(), size}
    }

    pub fn into_layer(self) -> LayerType
    {
        let size = self.size;
        let mut layer = vec![0.0; size];

        for position in self.positions.into_iter()
        {
            layer[*position] = 1.0;
        }

        LayerType::from_raw(layer, 1, size)
    }
}

#[derive(Debug, Clone)]
pub enum InputType
{
    Normal(DiffWrapper),
    OneHot(OneHotLayer)
}

impl InputType
{
    pub fn into_one_hot(self) -> OneHotLayer
    {
        match self
        {
            Self::OneHot(value) => value,
            _ => panic!("expected onehot")
        }
    }

    pub fn into_normal(self) -> DiffWrapper
    {
        match self
        {
            Self::Normal(value) => value,
            _ => panic!("expected normal")
        }
    }

    pub fn as_one_hot(&self) -> &OneHotLayer
    {
        match self
        {
            Self::OneHot(value) => value,
            _ => panic!("expected onehot")
        }
    }
}

impl From<DiffWrapper> for InputType
{
    fn from(value: DiffWrapper) -> Self
    {
        Self::Normal(value)
    }
}

impl From<OneHotLayer> for InputType
{
    fn from(value: OneHotLayer) -> Self
    {
        Self::OneHot(value)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiffType<T>
{
    parent: Ops,
    value: T,
    gradient: Option<T>,
    calculate_gradient: bool
}

impl<T> DiffType<T>
{
    fn new(value: T, ops: Ops, calculate_gradient: bool) -> Self
    {
        DiffType{
            value,
            parent: ops,
            gradient: None,
            calculate_gradient
        }
    }
}

impl<T> DiffType<T>
where
    T: DiffBounds,
    for<'a> T: Mul<&'a T, Output=T>,
    for<'a> &'a T: Mul<&'a T, Output=T> + Mul<f32, Output=T> + Neg<Output=T>,
    GradientType: Mul<GradientType, Output=GradientType>,
    for<'a> GradientType: Mul<&'a T, Output=GradientType> + Mul<&'a LayerType, Output=GradientType>,
    for<'a> &'a GradientType: Mul<&'a T, Output=GradientType> + Mul<&'a GradientType, Output=GradientType>
{
    pub fn calculate_gradients(mut self)
    {
        let mut ones = self.value.clone();
        ones.fill(1.0);

        self.derivatives(ones, 0);
    }

    pub fn take_gradient(&mut self) -> T
    {
        match self.gradient.take()
        {
            Some(x) => x,
            None =>
            {
                let mut value = self.value.clone();
                value.fill(0.0);

                value
            }
        }
    }

    fn derivatives(&mut self, starting_gradient: T, children_amount: usize)
    {
        if self.gradient.is_none()
        {
            self.gradient = Some(starting_gradient);
        } else
        {
            *self.gradient.as_mut().unwrap() += starting_gradient;
        }

        if children_amount > 1
        {
            return;
        }

        let gradient = self.gradient.clone().unwrap();

        match mem::replace(&mut self.parent, Ops::None)
        {
            Ops::Add{lhs, rhs} =>
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
            Ops::Sub{lhs, rhs} =>
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
            Ops::Mul{lhs, rhs} =>
            {
                let lhs_value = rhs.is_gradient().then(|| lhs.value_clone());

                if lhs.is_gradient()
                {
                    let d = rhs.value_clone() * &gradient;
                    lhs.derivatives(d);
                }

                if let Some(lhs_value) = lhs_value
                {
                    let d = lhs_value * &gradient;
                    rhs.derivatives(d);
                }
            },
            Ops::Div{lhs, rhs} =>
            {
                let r_recip = rhs.value_clone().reciprocal();
                
                let lhs_value = rhs.is_gradient().then(|| lhs.value_clone());

                if lhs.is_gradient()
                {
                    let d = &r_recip * &gradient;

                    lhs.derivatives(d);
                }

                if let Some(lhs_value) = lhs_value
                {
                    // my favorite syntax
                    let recip_squared = &r_recip * &r_recip;

                    let d = -lhs_value * &gradient;
                    let d = d * recip_squared;

                    rhs.derivatives(d);
                }
            },
            Ops::Exp{value: x} =>
            {
                if x.is_gradient()
                {
                    x.derivatives((gradient * &self.value).into());
                }
            },
            Ops::Sigmoid{value: x} =>
            {
                if x.is_gradient()
                {
                    // sigmoid(x) * (1.0 - sigmoid(x))
                    let d = (-&self.value + 1.0) * &self.value;

                    x.derivatives((gradient * &d).into());
                }
            },
            Ops::Tanh{value: x} =>
            {
                if x.is_gradient()
                {
                    // 1 - tanh^2(x)
                    let d = -(&self.value * &self.value) + 1.0;

                    x.derivatives((gradient * &d).into());
                }
            },
            Ops::LeakyRelu{value: x} =>
            {
                if x.is_gradient()
                {
                    let d = x.value_clone().leaky_relu_d();

                    x.derivatives(d * &gradient);
                }
            },
            Ops::Ln{value: x} =>
            {
                if x.is_gradient()
                {
                    let d = x.value_clone().reciprocal();

                    x.derivatives(d * &gradient);
                }
            },
            Ops::Sqrt{value: x} =>
            {
                if x.is_gradient()
                {
                    let m = &self.value * 2.0;

                    let d = m.reciprocal();

                    x.derivatives((d * &gradient).into());
                }
            },
            Ops::Neg{value: x} =>
            {
                if x.is_gradient()
                {
                    x.derivatives((-gradient).into());
                }
            },
            Ops::Pow{lhs, power} =>
            {
                if lhs.is_gradient()
                {
                    let p = lhs.value_clone().pow(power - 1);
                    let d = p * power as f32;

                    lhs.derivatives(d * &gradient);
                }
            },
            Ops::Matmulv{lhs, rhs} =>
            {
                let gradient: LayerType = gradient.try_into()
                    .ok().expect("matmul must be a tensor");
                
                let rhs_d = rhs.is_gradient().then(|| lhs.tensor().matmulv_transposed(&gradient));

                if lhs.is_gradient()
                {
                    let d = gradient.outer_product(&*rhs.tensor());
                    lhs.derivatives(d.into());
                }

                if let Some(rhs_d) = rhs_d
                {
                    rhs.derivatives(rhs_d.into());
                }
            },
            Ops::MatmulvAdd{lhs, rhs, added} =>
            {
                let gradient: LayerType = gradient.try_into()
                    .ok().expect("matmul must be a tensor");
                
                let rhs_d = rhs.is_gradient().then(|| lhs.tensor().matmulv_transposed(&gradient));

                if lhs.is_gradient()
                {
                    let d = gradient.outer_product(&*rhs.tensor());
                    lhs.derivatives(d.into());
                }

                if let Some(rhs_d) = rhs_d
                {
                    rhs.derivatives(rhs_d.into());
                }

                if added.is_gradient()
                {
                    added.derivatives(gradient.into());
                }
            },
            Ops::MatmulOneHotvAdd{lhs, rhs, added} =>
            {
                let gradient: LayerType = gradient.try_into()
                    .ok().expect("matmul must be a tensor");
                
                if lhs.is_gradient()
                {
                    let d = gradient.outer_product_one_hot(&rhs);
                    lhs.derivatives(d.into());
                }

                if added.is_gradient()
                {
                    added.derivatives(gradient.into());
                }
            },
            Ops::SumTensor{value: x} =>
            {
                if x.is_gradient()
                {
                    let d = LayerType::from_gradient(gradient.into(), ||
                    {
                        x.tensor().clone()
                    });

                    x.derivatives(d.into());
                }
            },
            Ops::Dot{lhs, rhs} =>
            {
                let gradient = LayerType::from_gradient(gradient.into(), ||
                {
                    lhs.tensor().clone()
                });
                
                let lhs_value = rhs.is_gradient().then(|| lhs.value_clone());

                if lhs.is_gradient()
                {
                    let d = rhs.value_clone() * &gradient;
 
                    lhs.derivatives(d);
                }

                if let Some(lhs_value) = lhs_value
                {
                    let d = lhs_value * &gradient;

                    rhs.derivatives(d);
                }
            },
            Ops::SoftmaxCrossEntropy{values, softmaxed_values, targets} =>
            {
                if values.is_gradient()
                {
                    let gradient = LayerType::from_gradient(gradient.into(), ||
                    {
                        values.tensor().clone()
                    });

                    let d = gradient * (softmaxed_values - targets.into_layer());
                    values.derivatives(d.into());
                }
            },
            Ops::None => ()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffWrapper(Rc<RefCell<AnyDiffType>>);

impl From<&DiffWrapper> for DiffWrapper
{
    fn from(value: &DiffWrapper) -> Self
    {
        Self(value.0.clone())
    }
}

impl From<&mut DiffWrapper> for DiffWrapper
{
    fn from(value: &mut DiffWrapper) -> Self
    {
        Self(value.0.clone())
    }
}

impl DiffWrapper
{
    pub fn take_gradient(&mut self) -> GradientType
    {
        self.this_mut().take_gradient()
    }

    pub fn calculate_gradients(self)
    {
        self.take_inner().calculate_gradients()
    }

    pub fn parent(&self) -> cell::Ref<Ops>
    {
        cell::Ref::map(self.this_ref(), |x| x.parent())
    }

    fn derivatives(mut self, gradient: GradientType)
    {
        let children_amount = Rc::strong_count(&self.0);

        self.this_mut().derivatives(gradient, children_amount);
    }

    pub fn take_gradient_tensor(&mut self) -> LayerType
    {
        self.this_mut().take_gradient_tensor()
    }

    pub fn take_gradient_scalar(&mut self) -> f32
    {
        self.this_mut().take_gradient_scalar()
    }

    fn take_inner(self) -> AnyDiffType
    {
        Rc::into_inner(self.0)
            .expect("the value must have no children")
            .into_inner()
    }

    pub fn value_clone(&self) -> GradientType
    {
        self.this_ref().value_clone()
    }

    pub fn recreate(&self) -> Self
    {
        let this = self.this_ref();

        if this.is_gradient()
        {
            DiffWrapper::new_diff(this.value_clone())
        } else
        {
            DiffWrapper::new_inner(this.value_clone(), Ops::None, false)
        }
    }

    pub fn new_diff(value: GradientType) -> Self
    {
        Self::new_inner(value, Ops::None, true)
    }

    pub fn new_undiff(value: GradientType) -> Self
    {
        Self::new_inner(value, Ops::None, false)
    }

    fn new_inner(value: GradientType, ops: Ops, calculate_gradient: bool) -> Self
    {
        let diff = match value
        {
            GradientType::Tensor(x) => DiffType::new(x, ops, calculate_gradient).into(),
            GradientType::Scalar(x) => DiffType::new(x, ops, calculate_gradient).into()
        };

        Self(Rc::new(RefCell::new(diff)))
    }

    pub fn enable_gradients(&mut self)
    {
        self.this_mut().set_calculate_gradient(true);
    }

    pub fn disable_gradients(&mut self)
    {
        self.this_mut().set_calculate_gradient(false);
    }

    fn is_gradient(&self) -> bool
    {
        self.this_ref().is_gradient()
    }

    #[allow(dead_code)]
    fn this_ref(&self) -> cell::Ref<AnyDiffType>
    {
        RefCell::borrow(&self.0)
    }

    fn this_mut(&mut self) -> cell::RefMut<AnyDiffType>
    {
        RefCell::borrow_mut(&self.0)
    }

    pub fn tensor(&self) -> cell::Ref<LayerType>
    {
        cell::Ref::map(self.this_ref(), |diff|
        {
            match diff
            {
                AnyDiffType::Tensor(x) => &x.value,
                AnyDiffType::Scalar(_) => panic!("expected tensor, got scalar")
            }
        })
    }

    pub fn scalar(&self) -> cell::Ref<f32>
    {
        cell::Ref::map(self.this_ref(), |diff|
        {
            match diff
            {
                AnyDiffType::Scalar(x) => &x.value,
                AnyDiffType::Tensor(_) => panic!("expected scalar, got tensor")
            }
        })
    }
}

macro_rules! op_impl
{
    ($this_t:ident, $this_g:ident, $other_g:ident, $output_g:ident, $op:ident, $fun:ident) =>
    {
        impl $op<$this_t<$other_g>> for $this_t<$this_g>
        {
            type Output = $output_g;

            fn $fun(self, rhs: $this_t<$other_g>) -> Self::Output
            {
                self.value.$fun(rhs.value)
            }
        }

        impl $op<&$this_t<$other_g>> for $this_t<$this_g>
        {
            type Output = $output_g;

            fn $fun(self, rhs: &$this_t<$other_g>) -> Self::Output
            {
                self.value.$fun(&rhs.value)
            }
        }

        impl $op<$this_t<$other_g>> for &$this_t<$this_g>
        {
            type Output = $output_g;

            fn $fun(self, rhs: $this_t<$other_g>) -> Self::Output
            {
                (&self.value).$fun(rhs.value)
            }
        }

        impl $op<&$this_t<$other_g>> for &$this_t<$this_g>
        {
            type Output = $output_g;

            fn $fun(self, rhs: &$this_t<$other_g>) -> Self::Output
            {
                (&self.value).$fun(&rhs.value)
            }
        }
    }
}

// inb4 50 megabyte executable
op_impl!{DiffType, f32, f32, f32, Add, add}
op_impl!{DiffType, f32, f32, f32, Sub, sub}
op_impl!{DiffType, f32, f32, f32, Mul, mul}
op_impl!{DiffType, f32, f32, f32, Div, div}

op_impl!{DiffType, LayerType, LayerType, LayerType, Add, add}
op_impl!{DiffType, LayerType, LayerType, LayerType, Sub, sub}
op_impl!{DiffType, LayerType, LayerType, LayerType, Mul, mul}
op_impl!{DiffType, LayerType, LayerType, LayerType, Div, div}

op_impl!{DiffType, LayerType, f32, LayerType, Add, add}
op_impl!{DiffType, LayerType, f32, LayerType, Sub, sub}
op_impl!{DiffType, LayerType, f32, LayerType, Mul, mul}
op_impl!{DiffType, LayerType, f32, LayerType, Div, div}

macro_rules! wrapper_op_inplace_impl
{
    ($op:ident, $fun:ident, $fun_inner:ident) =>
    {
        // i dont know if i can do this inplace cuz i kinda need a copy
        impl $op for DiffWrapper
        {
            fn $fun(&mut self, rhs: DiffWrapper)
            {
                *self = (&*self).$fun_inner(rhs);
            }
        }

        impl $op<&DiffWrapper> for DiffWrapper
        {
            fn $fun(&mut self, rhs: &DiffWrapper)
            {
                *self = (&*self).$fun_inner(rhs);
            }
        }
    }
}

macro_rules! wrapper_op_impl
{
    ($op:ident, $fun:ident) =>
    {
        impl $op for DiffWrapper
        {
            type Output = DiffWrapper;

            fn $fun(self, rhs: DiffWrapper) -> Self::Output
            {
                op_impl_inner!(self, rhs, $op, $fun)
            }
        }

        impl $op<&DiffWrapper> for DiffWrapper
        {
            type Output = DiffWrapper;

            fn $fun(self, rhs: &DiffWrapper) -> Self::Output
            {
                op_impl_inner!(self, rhs, $op, $fun)
            }
        }

        impl $op<DiffWrapper> for &DiffWrapper
        {
            type Output = DiffWrapper;

            fn $fun(self, rhs: DiffWrapper) -> Self::Output
            {
                op_impl_inner!(self, rhs, $op, $fun)
            }
        }

        impl $op<&DiffWrapper> for &DiffWrapper
        {
            type Output = DiffWrapper;

            fn $fun(self, rhs: &DiffWrapper) -> Self::Output
            {
                op_impl_inner!(self, rhs, $op, $fun)
            }
        }
    }
}

macro_rules! op_impl_inner
{
    (
        $lhs:expr, $rhs:expr,
        $op:ident, $fun:ident
    ) =>
    {
        {
            let value = (&*$lhs.this_ref()).$fun(&*$rhs.this_ref());

            inner_from_value!(
                value,
                $op,
                lhs, $lhs,
                rhs, $rhs
            )
        }
    }
}

macro_rules! inner_single_from_value
{
    ($value:expr, $op:ident, $field_value:expr) =>
    {
        inner_from_value!{$value, $op, value, $field_value}
    }
}

macro_rules! inner_from_value
{
    (
        $value:expr,
        $op:ident,
        $($field:ident, $field_value:expr),+
    ) =>
    {
        inner_from_value_special_op!{
            $value,
            Ops::$op{
                $($field: $field_value.into(),)+
            },
            $($field_value),+
        }
    }
}

macro_rules! inner_from_value_special_op
{
    (
        $value:expr,
        $op:expr,
        $($field_value:expr),+
    ) =>
    {
        {
            let is_gradient = false $(|| $field_value.is_gradient())+;

            let ops = if is_gradient
            {
                $op
            } else
            {
                Ops::None
            };

            DiffWrapper::new_inner(
                $value.into(),
                ops,
                is_gradient
            )
        }
    }
}

impl Neg for DiffWrapper
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        let value = -self.value_clone();

        inner_from_value!(
            value, Neg,
            value, self
        )
    }
}

impl iter::Sum for DiffWrapper
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item=Self>
    {
        iter.reduce(|acc, value|
        {
            acc + value
        }).unwrap_or_else(|| unimplemented!())
    }
}

wrapper_op_inplace_impl!{AddAssign, add_assign, add}
wrapper_op_inplace_impl!{SubAssign, sub_assign, sub}
wrapper_op_inplace_impl!{MulAssign, mul_assign, mul}
wrapper_op_inplace_impl!{DivAssign, div_assign, div}

wrapper_op_impl!{Add, add}
wrapper_op_impl!{Sub, sub}
wrapper_op_impl!{Mul, mul}
wrapper_op_impl!{Div, div}

impl From<&DiffWrapper> for Vec<f32>
{
    fn from(other: &DiffWrapper) -> Self
    {
        other.as_vec()
    }
}

impl DiffWrapper
{
    pub fn softmax_cross_entropy(self, targets: OneHotLayer) -> DiffWrapper
    {
        let (softmaxed, value) = self.tensor().clone().softmax_cross_entropy(&targets);

        inner_from_value_special_op!(value, Ops::SoftmaxCrossEntropy{
            values: self,
            softmaxed_values: softmaxed,
            targets
        }, self)
    }

    pub fn matmulv(&self, rhs: impl Borrow<Self>) -> Self
    {
        let rhs = rhs.borrow();

        let value = {
            let rhs = rhs.tensor();

            self.tensor().matmulv(&*rhs)
        };

        inner_from_value!(value, Matmulv, lhs, self, rhs, rhs)
    }

    pub fn matmulv_add(&self, rhs: &InputType, added: impl Borrow<Self>) -> Self
    {
        match rhs
        {
            InputType::Normal(value) => self.matmul_normalv_add(value, added),
            InputType::OneHot(value) => self.matmul_onehotv_add(value, added)
        }
    }

    pub fn matmul_normalv_add(&self, rhs: &Self, added: impl Borrow<Self>) -> Self
    {
        let added = added.borrow();

        let value = {
            let rhs = rhs.tensor();
            let added = added.tensor();

            self.tensor().matmulv_add(&*rhs, &*added)
        };

        inner_from_value!(value, MatmulvAdd, lhs, self, rhs, rhs, added, added)
    }

    pub fn matmul_onehotv_add(&self, rhs: &OneHotLayer, added: impl Borrow<Self>) -> Self
    {
        let added = added.borrow();

        let value = {
            let added = added.tensor();

            self.tensor().matmul_onehotv_add(&rhs, &*added)
        };

        inner_from_value_special_op!(value, Ops::MatmulOneHotvAdd{
            lhs: self.into(),
            rhs: rhs.clone(),
            added: added.into()
        }, self)
    }

    pub fn dot(self, rhs: Self) -> DiffWrapper
    {
        let value = {
            let rhs = rhs.tensor();

            self.tensor().clone().dot(&rhs)
        };

        inner_from_value!(value, Dot, lhs, self, rhs, rhs)
    }

    pub fn pow(&mut self, power: u32)
    {
        let mut value = self.tensor().clone();
        value.pow(power);

        *self = inner_from_value_special_op!(value, Ops::Pow{
            lhs: self.into(),
            power
        }, self);
    }

    pub fn exp(&mut self)
    {
        let mut value = self.tensor().clone();
        value.exp();

        *self = inner_single_from_value!(value, Exp, self);
    }

    pub fn ln(&mut self)
    {
        let mut value = self.tensor().clone();
        value.ln();

        *self = inner_single_from_value!(value, Ln, self);
    }

    pub fn sqrt(&mut self)
    {
        let mut value = self.tensor().clone();
        value.sqrt();

        *self = inner_single_from_value!(value, Sqrt, self);
    }

    pub fn sigmoid(&mut self)
    {
        let mut value = self.tensor().clone();
        value.sigmoid();

        *self = inner_single_from_value!(value, Sigmoid, self);
    }

    pub fn tanh(&mut self)
    {
        let mut value = self.tensor().clone();
        value.tanh();

        *self = inner_single_from_value!(value, Tanh, self);
    }

    pub fn leaky_relu(&mut self)
    {
        let mut value = self.tensor().clone();
        value.leaky_relu();

        *self = inner_single_from_value!(value, LeakyRelu, self);
    }

    pub fn sum(&self) -> DiffWrapper
    {
        let value = self.tensor().sum();
        inner_single_from_value!(value, SumTensor, self)
    }

    pub fn cosine_similarity(&self, other: Self) -> f32
    {
        let lhs = self.tensor();
        let rhs = other.tensor();

        let top = lhs.clone().dot(&rhs);

        let bottom = lhs.magnitude() * rhs.magnitude();

        top / bottom
    }

    pub fn total_len(&self) -> usize
    {
        self.tensor().total_len()
    }

    pub fn as_vec(&self) -> Vec<f32>
    {
        self.tensor().as_vec()
    }

    pub fn pick_weighed(&self) -> usize
    {
        self.tensor().pick_weighed()
    }

    pub fn highest_index(&self) -> usize
    {
        self.tensor().highest_index()
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    const LAYER_PREV: usize = 3;
    const LAYER_CURR: usize = 2;

    pub fn close_enough_loose(a: f32, b: f32, epsilon: f32) -> bool
    {
        if a == 0.0 || a == -0.0
        {
            return b.abs() < epsilon;
        }

        if b == 0.0 || b == -0.0
        {
            return a.abs() < epsilon;
        }

        ((a - b).abs() / (a.abs() + b.abs())) < epsilon
    }

    fn compare_single(correct: f32, calculated: f32)
    {
        let epsilon = 0.2;
        assert!(
            close_enough_loose(correct, calculated, epsilon),
            "correct: {}, calculated: {}",
            correct, calculated
        );
    }

    fn compare_tensor(correct: LayerType, calculated: LayerType)
    {
        correct.as_vec().into_iter().zip(calculated.as_vec().into_iter())
            .for_each(|(correct, calculated)| compare_single(correct, calculated));
    }

    #[allow(dead_code)]
    fn check_tensor_with_dims(
        a_dims: (usize, usize),
        b_dims: (usize, usize),
        f: impl FnMut(&DiffWrapper, &DiffWrapper) -> DiffWrapper
    )
    {
        let a = random_tensor(a_dims.0, a_dims.1);
        let b = random_tensor(b_dims.0, b_dims.1);

        check_tensor_inner(a, b, f);
    }

    fn check_vector(f: impl FnMut(&DiffWrapper, &DiffWrapper) -> DiffWrapper)
    {
        let a = random_tensor(1, LAYER_CURR);
        let b = random_tensor(1, LAYER_CURR);

        check_tensor_inner(a, b, f);
    }

    fn check_tensor(f: impl FnMut(&DiffWrapper, &DiffWrapper) -> DiffWrapper)
    {
        let a = random_tensor(LAYER_PREV, LAYER_CURR);
        let b = random_tensor(LAYER_PREV, LAYER_CURR);

        check_tensor_inner(a, b, f);
    }

    fn check_tensor_inner(
        mut a: DiffWrapper,
        mut b: DiffWrapper,
        mut f: impl FnMut(&DiffWrapper, &DiffWrapper) -> DiffWrapper
    )
    {
        let out = f(&a, &b);

        out.calculate_gradients();

        let a_g = a.take_gradient();
        let b_g = b.take_gradient();

        a.disable_gradients();
        b.disable_gradients();

        let mut vals = |a: &mut DiffWrapper, b: &mut DiffWrapper|
        {
            assert!(a.parent().is_none());
            assert!(b.parent().is_none());

            f(&a, &b).tensor().clone()
        };

        let orig = vals(&mut a, &mut b).sum();

        let epsilon: f32 = 0.009;

        let fg = |value: LayerType|
        {
            let value = value.sum();

            (value - orig) / epsilon
        };

        let mut a_fg = vec![0.0; a.total_len()];
        for index in 0..a_fg.len()
        {
            let v = &a;
            let epsilon = one_hot(v.tensor().clone(), index, epsilon, 0.0);

            let this_fg = {
                let mut a = DiffWrapper::new_undiff((v.tensor().clone() + epsilon).into());
                fg(vals(&mut a, &mut b))
            };

            a_fg[index] = this_fg;
        }

        let mut b_fg = vec![0.0; b.total_len()];
        for index in 0..b_fg.len()
        {
            let v = &b;
            let epsilon = one_hot(v.tensor().clone(), index, epsilon, 0.0);

            let this_fg = {
                let mut b = DiffWrapper::new_undiff((v.tensor().clone() + epsilon).into());
                fg(vals(&mut a, &mut b))
            };

            b_fg[index] = this_fg;
        }

        let vec_to_layer = |v, layer_match: &DiffWrapper|
        {
            let mut layer = layer_match.tensor().clone();

            layer.swap_raw_values(v);

            layer
        };

        let a_fg = vec_to_layer(a_fg, &a);
        let b_fg = vec_to_layer(b_fg, &b);

        eprintln!("derivative of a");
        compare_tensor(a_fg, a_g.tensor().clone());

        eprintln!("derivative of b");
        compare_tensor(b_fg, b_g.tensor().clone());
    }

    fn one_hot(
        dimensions_match: LayerType,
        position: usize,
        value: f32,
        d_value: f32
    ) -> LayerType
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
        fastrand::u32(1..5) as f32
    }

    fn random_tensor(prev: usize, curr: usize) -> DiffWrapper
    {
        DiffWrapper::new_diff(
            LayerType::new_with(
                prev,
                curr,
                random_value
            ).into()
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
    fn non_diff_subdiff()
    {
        check_tensor(|a, b| DiffWrapper::new_undiff(1.0.into()) - (a + b))
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
        check_vector(|a, b| a + a.clone().dot(b.clone()))
    }

    #[test]
    fn scalar_minus_tensor()
    {
        check_tensor(|a, b| a.sum() - b)
    }

    #[test]
    fn scalar_minus_tensor_stuff()
    {
        check_tensor(|a, b| DiffWrapper::new_undiff(2.0.into()) - (a.sum() - b))
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
    fn sqrt()
    {
        check_tensor(|a, b|
        {
            let mut a = a.clone();
            a.sqrt();

            a + b
        })
    }

    #[test]
    fn pow()
    {
        check_tensor(|a, b|
        {
            let mut a = a.clone();
            a.pow(3);

            a + b
        })
    }

    #[test]
    fn matrix_multiplication()
    {
        check_tensor_with_dims((4, 2), (1, 4), |a, b| a.matmulv(b) + b.sum())
    }

    fn create_targets() -> OneHotLayer
    {
        let pos = fastrand::usize(0..LAYER_CURR);

        OneHotLayer::new([pos], LAYER_CURR)
    }

    #[test]
    fn softmax_cross_entropy()
    {
        let targets = create_targets();
        check_vector(|a, b|
        {
            b + a.clone().softmax_cross_entropy(targets.clone())
        })
    }

    #[test]
    fn softmax_cross_entropy_complicated()
    {
        let targets = create_targets();
        check_vector(|a, b|
        {
            a + (b + DiffWrapper::new_undiff(2.0.into())).softmax_cross_entropy(targets.clone())
        })
    }
}
