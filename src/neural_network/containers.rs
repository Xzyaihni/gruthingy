use std::{
    f32,
    mem,
    iter,
    rc::Rc,
    fmt::Debug,
    cell::{self, RefCell},
    borrow::Borrow,
    collections::HashSet,
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

#[derive(Debug, PartialEq, Eq)]
enum RecorderState
{
    Recording,
    AwaitingGradient,
    Ready
}

#[derive(Debug)]
pub struct OperationsRecorder
{
    state: RecorderState,
    values: Vec<f32>,
    tensors: Vec<LayerType>,
    recording_operations: Vec<Op>,
    gradient_operations: Vec<GradientOp>
}

// avoids borrow checker
macro_rules! new_tensor_index
{
    ($this:expr, $rows:expr, $columns:expr) =>
    {
        {
            let id = $this.tensors.len();

            $this.tensors.push(LayerType::repeat($rows, $columns, 0.0));

            TensorIndex(id)
        }
    }
}

macro_rules! new_value_index
{
    ($this:expr) =>
    {
        {
            let id = $this.values.len();

            $this.values.push(0.0);

            ValueIndex(id)
        }
    }
}

macro_rules! new_tensor
{
    ($this:expr, $source:expr, $gradient:expr, $rows:expr, $columns:expr) =>
    {
        {
            DiffTensor{
                index: new_tensor_index!($this, $rows, $columns),
                gradient: $gradient.then(|| new_tensor_index!($this, $rows, $columns)),
                source: $source
            }
        }
    }
}

macro_rules! new_value
{
    ($this:expr, $source:expr, $gradient:expr) =>
    {
        {
            DiffScalar{
                index: new_value_index!($this),
                gradient: $gradient.then(|| new_value_index!($this)),
                source: $source
            }
        }
    }
}

macro_rules! impl_pair_tensor_op
{
    ($this:expr, $a:expr, $b:expr, $name:ident) =>
    {
        {
            let source = Some($this.current_source());

            let a_shape@(a_rows, a_columns) = $this.tensor_shape($a.as_value());
            let b_shape = $this.tensor_shape($b.as_value());

            debug_assert!(a_shape == b_shape);

            let output = $this.new_tensor_op(source, a_rows, a_columns);

            $this.recording_operations.push(Op::$name{lhs: $a, rhs: $b, output});

            output
        }
    }
}

macro_rules! impl_map_tensor_op
{
    ($this:expr, $a:expr, $name:ident) =>
    {
        {
            let source = Some($this.current_source());

            let (rows, columns) = $this.tensor_shape($a.as_value());

            let output = $this.new_tensor_op(source, rows, columns);

            $this.recording_operations.push(Op::$name{value: $a, output});

            output
        }
    }
}

impl OperationsRecorder
{
    pub fn new() -> Self
    {
        Self{
            state: RecorderState::Recording,
            values: Vec::new(),
            tensors: Vec::new(),
            recording_operations: Vec::new(),
            gradient_operations: Vec::new()
        }
    }

    pub fn new_tensor(&mut self, rows: usize, columns: usize) -> DiffTensor
    {
        self.new_tensor_with_source(None, true, rows, columns)
    }

    pub fn new_value(&mut self) -> DiffScalar
    {
        self.new_value_with_source(None, true)
    }

    fn new_tensor_op(
        &mut self,
        source: Option<OperationIndex>,
        rows: usize,
        columns: usize
    ) -> DiffTensor
    {
        self.new_tensor_with_source(source, true, rows, columns)
    }

    fn new_value_op(
        &mut self,
        source: Option<OperationIndex>
    ) -> DiffScalar
    {
        self.new_value_with_source(source, true)
    }

    fn new_tensor_with_source(
        &mut self,
        source: Option<OperationIndex>,
        is_gradient: bool,
        rows: usize,
        columns: usize
    ) -> DiffTensor
    {
        new_tensor!(self, source, is_gradient, rows, columns)
    }

    fn new_value_with_source(
        &mut self,
        source: Option<OperationIndex>,
        is_gradient: bool
    ) -> DiffScalar
    {
        new_value!(self, source, is_gradient)
    }

    pub fn set_tensor(&mut self, index: TensorIndex, value: LayerType)
    {
        self.tensors[index.0] = value;
    }

    pub fn set_value(&mut self, index: ValueIndex, value: f32)
    {
        self.values[index.0] = value;
    }

    pub fn set_new_tensor(&mut self, value: LayerType) -> DiffTensor
    {
        let tensor = self.new_tensor_with_source(None, false, value.rows(), value.columns());
        self.set_tensor(tensor.as_value(), value);

        tensor
    }

    pub fn set_new_value(&mut self, value: f32) -> DiffScalar
    {
        let scalar = self.new_value_with_source(None, false);
        self.set_value(scalar.as_value(), value);

        scalar
    }

    fn set_ones(&mut self, wrapper: DiffWrapper)
    {
        match wrapper
        {
            DiffWrapper::Tensor(DiffTensor{index, gradient, ..}) =>
            {
                let (rows, columns) = self.tensor_shape(index);
                self.set_tensor(gradient.expect("gradient must exist"), LayerType::repeat(rows, columns, 1.0))
            },
            DiffWrapper::Value(DiffScalar{gradient, ..}) => self.set_value(gradient.expect("gradient must exist"), 1.0)
        }
    }

    pub fn get_tensor(&self, index: TensorIndex) -> &LayerType
    {
        &self.tensors[index.0]
    }

    pub fn get_value(&self, index: ValueIndex) -> f32
    {
        self.values[index.0]
    }

    fn current_source(&self) -> OperationIndex
    {
        debug_assert!(self.state == RecorderState::Recording);

        OperationIndex(self.recording_operations.len())
    }

    pub fn add_scalar(&mut self, a: DiffTensor, b: DiffScalar) -> DiffTensor
    {
        let source = Some(self.current_source());

        let (a_rows, a_columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(source, a_rows, a_columns);

        self.recording_operations.push(Op::AddScalar{lhs: a, rhs: b, output});

        output
    }

    pub fn add(&mut self, a: DiffTensor, b: DiffTensor) -> DiffTensor
    {
        impl_pair_tensor_op!(self, a, b, Add)
    }

    pub fn sub(&mut self, a: DiffTensor, b: DiffTensor) -> DiffTensor
    {
        impl_pair_tensor_op!(self, a, b, Sub)
    }

    pub fn sub_from_scalar(&mut self, a: DiffScalar, b: DiffTensor) -> DiffTensor
    {
        let source = Some(self.current_source());

        let (a_rows, a_columns) = self.tensor_shape(b.as_value());

        let output = self.new_tensor_op(source, a_rows, a_columns);

        self.recording_operations.push(Op::SubFromScalar{lhs: a, rhs: b, output});

        output
    }

    pub fn mul_scalars(&mut self, a: DiffScalar, b: DiffScalar) -> DiffScalar
    {
        let source = Some(self.current_source());

        let output = self.new_value_op(source);

        self.recording_operations.push(Op::MulScalars{lhs: a, rhs: b, output});

        output
    }

    pub fn mul_scalar(&mut self, a: DiffTensor, b: DiffScalar) -> DiffTensor
    {
        let source = Some(self.current_source());

        let (a_rows, a_columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(source, a_rows, a_columns);

        self.recording_operations.push(Op::MulScalar{lhs: a, rhs: b, output});

        output
    }

    pub fn mul_componentwise(&mut self, a: DiffTensor, b: DiffTensor) -> DiffTensor
    {
        impl_pair_tensor_op!(self, a, b, MulComponentwise)
    }

    pub fn sum_tensor(&mut self, a: DiffTensor) -> DiffScalar
    {
        let source = Some(self.current_source());

        let output = self.new_value_op(source);

        self.recording_operations.push(Op::SumTensor{value: a, output});

        output
    }

    pub fn pow(&mut self, a: DiffTensor, power: i32) -> DiffTensor
    {
        let source = Some(self.current_source());

        let (rows, columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(source, rows, columns);

        self.recording_operations.push(Op::Pow{lhs: a, power, output});

        output
    }

    pub fn sigmoid(&mut self, a: DiffTensor) -> DiffTensor
    {
        impl_map_tensor_op!(self, a, Sigmoid)
    }

    pub fn tanh(&mut self, a: DiffTensor) -> DiffTensor
    {
        impl_map_tensor_op!(self, a, Tanh)
    }

    pub fn leaky_relu(&mut self, a: DiffTensor) -> DiffTensor
    {
        impl_map_tensor_op!(self, a, LeakyRelu)
    }

    pub fn tensor_shape(&self, tensor: TensorIndex) -> (usize, usize)
    {
        let tensor = &self.tensors[tensor.0];

        (tensor.rows(), tensor.columns())
    }

    pub fn calculate(&mut self)
    {
        self.gradient_operations.iter().for_each(|gradient_op|
        {
            match gradient_op
            {
                GradientOp::Copy{src, dst} =>
                {
                    let this_should_not_even_exist = ();
                    self.tensors[dst.0] = self.tensors[src.0].clone();
                },
                GradientOp::AddScalars{lhs, rhs, output} =>
                {
                    self.values[output.0] = self.values[lhs.0] + self.values[rhs.0];
                },
                GradientOp::AddScalar{lhs, rhs, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = &self.tensors[lhs.0] + self.values[rhs.0];
                },
                GradientOp::Add{lhs, rhs, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = &self.tensors[lhs.0] + &self.tensors[rhs.0];
                },
                GradientOp::Sub{lhs, rhs, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = &self.tensors[lhs.0] - &self.tensors[rhs.0];
                },
                GradientOp::SubFromScalar{lhs, rhs, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = -&self.tensors[rhs.0] + self.values[lhs.0];
                },
                GradientOp::MulScalar{lhs, rhs, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = &self.tensors[lhs.0] * &self.values[rhs.0];
                },
                GradientOp::MulScalars{lhs, rhs, output} =>
                {
                    self.values[output.0] = self.values[lhs.0] * self.values[rhs.0];
                },
                GradientOp::MulComponentwise{lhs, rhs, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = &self.tensors[lhs.0] * &self.tensors[rhs.0];
                },
                GradientOp::SumTensor{value, output} =>
                {
                    self.values[output.0] = self.tensors[value.0].sum();
                },
                GradientOp::Fill{value, output} =>
                {
                    self.tensors[output.0].fill(self.values[value.0]);
                },
                GradientOp::Pow{lhs, power, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = self.tensors[lhs.0].pow(*power);
                },
                GradientOp::Sigmoid{value, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = self.tensors[value.0].sigmoid();
                },
                GradientOp::SigmoidDiff{value, gradient, output} =>
                {
                    let [output, value, gradient] = self.tensors.get_disjoint_mut([output.0, value.0, gradient.0]).unwrap();

                    output.sigmoid_gradient_inplace(value, gradient);
                },
                GradientOp::Tanh{value, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = self.tensors[value.0].tanh();
                },
                GradientOp::TanhDiff{value, gradient, output} =>
                {
                    let [output, value, gradient] = self.tensors.get_disjoint_mut([output.0, value.0, gradient.0]).unwrap();

                    output.tanh_gradient_inplace(value, gradient);
                },
                GradientOp::LeakyRelu{value, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = self.tensors[value.0].leaky_relu();
                },
                GradientOp::LeakyReluDiff{value, gradient, output} =>
                {
                    let [output, value, gradient] = self.tensors.get_disjoint_mut([output.0, value.0, gradient.0]).unwrap();

                    output.leaky_relu_gradient_inplace(value, gradient);
                }
            }
        });
    }

    pub fn finish(&mut self)
    {
        debug_assert!(self.state == RecorderState::Recording);

        self.gradient_operations.extend(self.recording_operations.iter().map(|op|
        {
            match op
            {
                Op::AddScalar{lhs, rhs, output} =>
                {
                    GradientOp::AddScalar{lhs: lhs.as_value(), rhs: rhs.as_value(), output: output.as_value()}
                },
                Op::Add{lhs, rhs, output} =>
                {
                    GradientOp::Add{lhs: lhs.as_value(), rhs: rhs.as_value(), output: output.as_value()}
                },
                Op::Sub{lhs, rhs, output} =>
                {
                    GradientOp::Sub{lhs: lhs.as_value(), rhs: rhs.as_value(), output: output.as_value()}
                },
                Op::SubFromScalar{lhs, rhs, output} =>
                {
                    GradientOp::SubFromScalar{lhs: lhs.as_value(), rhs: rhs.as_value(), output: output.as_value()}
                },
                Op::MulScalar{lhs, rhs, output} =>
                {
                    GradientOp::MulScalar{lhs: lhs.as_value(), rhs: rhs.as_value(), output: output.as_value()}
                },
                Op::MulScalars{lhs, rhs, output} =>
                {
                    GradientOp::MulScalars{lhs: lhs.as_value(), rhs: rhs.as_value(), output: output.as_value()}
                },
                Op::MulComponentwise{lhs, rhs, output} =>
                {
                    GradientOp::MulComponentwise{lhs: lhs.as_value(), rhs: rhs.as_value(), output: output.as_value()}
                },
                Op::SumTensor{value, output} =>
                {
                    GradientOp::SumTensor{value: value.as_value(), output: output.as_value()}
                },
                Op::Pow{lhs, power, output} =>
                {
                    GradientOp::Pow{lhs: lhs.as_value(), power: *power as u32, output: output.as_value()}
                },
                Op::Sigmoid{value, output} =>
                {
                    GradientOp::Sigmoid{value: value.as_value(), output: output.as_value()}
                },
                Op::Tanh{value, output} =>
                {
                    GradientOp::Tanh{value: value.as_value(), output: output.as_value()}
                },
                Op::LeakyRelu{value, output} =>
                {
                    GradientOp::LeakyRelu{value: value.as_value(), output: output.as_value()}
                }
            }
        }));

        self.state = RecorderState::AwaitingGradient;
    }

    fn combine_same_outputs(&mut self)
    {
        let mut handled = Vec::new();

        let mut i = 1;
        while i < self.gradient_operations.len()
        {
            let output_of = |op: &GradientOp| -> DiffValue
            {
                match *op
                {
                    GradientOp::Copy{dst: output, ..}
                    | GradientOp::AddScalar{output, ..}
                    | GradientOp::Add{output, ..}
                    | GradientOp::Sub{output, ..}
                    | GradientOp::SubFromScalar{output, ..}
                    | GradientOp::MulScalar{output, ..}
                    | GradientOp::MulComponentwise{output, ..}
                    | GradientOp::Fill{output, ..}
                    | GradientOp::Pow{output, ..}
                    | GradientOp::Sigmoid{output, ..}
                    | GradientOp::SigmoidDiff{output, ..}
                    | GradientOp::Tanh{output, ..}
                    | GradientOp::TanhDiff{output, ..}
                    | GradientOp::LeakyRelu{output, ..}
                    | GradientOp::LeakyReluDiff{output, ..} => output.into(),
                    GradientOp::AddScalars{output, ..}
                    | GradientOp::MulScalars{output, ..}
                    | GradientOp::SumTensor{output, ..} => output.into()
                }
            };

            let this = &self.gradient_operations[i];

            if let Some(previous) = (0..i).find(|previous|
            {
                let is_shared_output = output_of(&self.gradient_operations[*previous]) == output_of(this);

                is_shared_output && !handled.contains(previous)
            })
            {
                let maybe_shape = if let DiffValue::Tensor(x) = output_of(this)
                {
                    Some(self.tensor_shape(x))
                } else
                {
                    None
                };

                match &mut self.gradient_operations[i]
                {
                    GradientOp::Copy{dst: output, ..}
                    | GradientOp::AddScalar{output, ..}
                    | GradientOp::Add{output, ..}
                    | GradientOp::Sub{output, ..}
                    | GradientOp::SubFromScalar{output, ..}
                    | GradientOp::MulScalar{output, ..}
                    | GradientOp::MulComponentwise{output, ..}
                    | GradientOp::Fill{output, ..}
                    | GradientOp::Pow{output, ..}
                    | GradientOp::Sigmoid{output, ..}
                    | GradientOp::SigmoidDiff{output, ..}
                    | GradientOp::Tanh{output, ..}
                    | GradientOp::TanhDiff{output, ..}
                    | GradientOp::LeakyRelu{output, ..}
                    | GradientOp::LeakyReluDiff{output, ..} =>
                    {
                        let (rows, columns) = maybe_shape.unwrap();

                        let final_output = *output;

                        let temporary_add_index = new_tensor_index!(self, rows, columns);

                        *output = temporary_add_index;

                        let these_are_illegal = (); let replace_with_a_separate_op = ();
                        self.gradient_operations.insert(
                            i + 1,
                            GradientOp::Add{lhs: temporary_add_index, rhs: final_output, output: final_output}
                        );
                    },
                    GradientOp::AddScalars{output, ..}
                    | GradientOp::MulScalars{output, ..}
                    | GradientOp::SumTensor{output, ..} =>
                    {
                        let final_output = *output;

                        let temporary_add_index = new_value_index!(self);

                        *output = temporary_add_index;

                        self.gradient_operations.insert(
                            i + 1,
                            GradientOp::AddScalars{lhs: temporary_add_index, rhs: final_output, output: final_output}
                        );
                    }
                }

                handled.push(i);
                handled.push(i + 1);

                i += 1;
            }

            i += 1;
        }
    }

    pub fn gradient_with_respect(&mut self, respect: DiffWrapper)
    {
        debug_assert!(self.state == RecorderState::AwaitingGradient);

        self.set_ones(respect);

        self.calculate_gradient(respect);

        self.recording_operations.clear();

        self.combine_same_outputs();

        self.state = RecorderState::Ready;
    }

    fn calculate_gradient(&mut self, respect: DiffWrapper)
    {
        let (this_value, gradient, source): (DiffValue, DiffValue, Option<OperationIndex>) = match respect
        {
            DiffWrapper::Tensor(DiffTensor{index, gradient, source}) =>
            {
                let gradient = if let Some(x) = gradient { x } else { return; };

                (index.into(), gradient.into(), source)
            },
            DiffWrapper::Value(DiffScalar{index, gradient, source}) =>
            {
                let gradient = if let Some(x) = gradient { x } else { return; };

                (index.into(), gradient.into(), source)
            }
        };

        if let Some(source) = source
        {
            let this_operation = &self.recording_operations[source.0];
            match this_operation
            {
                Op::Add{lhs, ..}
                | Op::AddScalar{lhs, ..} =>
                {
                    let gradient = gradient.as_tensor();

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        self.gradient_operations.push(GradientOp::Copy{src: gradient, dst: lhs_gradient});
                    }

                    if let Op::Add{rhs, ..} = this_operation
                    {
                        if let Some(rhs_gradient) = rhs.as_gradient()
                        {
                            self.gradient_operations.push(GradientOp::Copy{src: gradient, dst: rhs_gradient});
                        }
                    } else if let Op::AddScalar{rhs, ..} = this_operation
                    {
                        if let Some(rhs_gradient) = rhs.as_gradient()
                        {
                            self.gradient_operations.push(GradientOp::SumTensor{value: gradient, output: rhs_gradient});
                        }
                    } else
                    {
                        unreachable!()
                    }

                    let lhs = *lhs;

                    self.calculate_gradient(if let Op::Add{rhs, ..} = this_operation
                    {
                        (*rhs).into()
                    } else if let Op::AddScalar{rhs, ..} = this_operation
                    {
                        (*rhs).into()
                    } else
                    {
                        unreachable!()
                    });

                    self.calculate_gradient(lhs.into());
                },
                Op::Sub{rhs, ..}
                | Op::SubFromScalar{rhs, ..} =>
                {
                    let gradient = gradient.as_tensor();

                    if let Op::Sub{lhs, ..} = this_operation
                    {
                        if let Some(lhs_gradient) = lhs.as_gradient()
                        {
                            self.gradient_operations.push(GradientOp::Copy{src: gradient, dst: lhs_gradient});
                        }
                    } else if let Op::SubFromScalar{lhs, ..} = this_operation
                    {
                        if let Some(lhs_gradient) = lhs.as_gradient()
                        {
                            self.gradient_operations.push(GradientOp::SumTensor{value: gradient, output: lhs_gradient});
                        }
                    } else
                    {
                        unreachable!()
                    }

                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        let m1_index = new_value_index!(self);
                        self.values[m1_index.0] = -1.0;

                        self.gradient_operations.push(GradientOp::MulScalar{lhs: gradient, rhs: m1_index, output: rhs_gradient});
                    }

                    let rhs = *rhs;

                    self.calculate_gradient(if let Op::Sub{lhs, ..} = this_operation
                    {
                        (*lhs).into()
                    } else if let Op::SubFromScalar{lhs, ..} = this_operation
                    {
                        (*lhs).into()
                    } else
                    {
                        unreachable!()
                    });

                    self.calculate_gradient(rhs.into());
                },
                Op::MulScalars{lhs, rhs, output: _} =>
                {
                    let gradient = gradient.as_value();

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        self.gradient_operations.push(GradientOp::MulScalars{lhs: rhs.as_value(), rhs: gradient, output: lhs_gradient});
                    }

                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        self.gradient_operations.push(GradientOp::MulScalars{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                    }

                    let lhs = *lhs;
                    self.calculate_gradient((*rhs).into());
                    self.calculate_gradient(lhs.into());
                },
                Op::MulComponentwise{lhs, ..}
                | Op::MulScalar{lhs, ..} =>
                {
                    let gradient = gradient.as_tensor();

                    let (rows, columns) = self.tensor_shape(gradient);

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        if let Op::MulComponentwise{rhs, ..} = this_operation
                        {
                            self.gradient_operations.push(GradientOp::MulComponentwise{lhs: rhs.as_value(), rhs: gradient, output: lhs_gradient});
                        } else if let Op::MulScalar{rhs, ..} = this_operation
                        {
                            self.gradient_operations.push(GradientOp::MulScalar{lhs: gradient, rhs: rhs.as_value(), output: lhs_gradient});
                        } else
                        {
                            unreachable!()
                        }
                    }

                    if let Op::MulComponentwise{rhs, ..} = this_operation
                    {
                        if let Some(rhs_gradient) = rhs.as_gradient()
                        {
                            self.gradient_operations.push(GradientOp::MulComponentwise{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                        }
                    } else if let Op::MulScalar{rhs, ..} = this_operation
                    {
                        if let Some(rhs_gradient) = rhs.as_gradient()
                        {
                            let pre_fold = new_tensor_index!(self, rows, columns);
                            self.gradient_operations.push(GradientOp::MulComponentwise{lhs: lhs.as_value(), rhs: gradient, output: pre_fold});

                            self.gradient_operations.push(GradientOp::SumTensor{value: pre_fold, output: rhs_gradient});
                        }
                    } else
                    {
                        unreachable!()
                    }

                    let lhs = *lhs;

                    self.calculate_gradient(if let Op::MulComponentwise{rhs, ..} = this_operation
                    {
                        (*rhs).into()
                    } else if let Op::MulScalar{rhs, ..} = this_operation
                    {
                        (*rhs).into()
                    } else
                    {
                        unreachable!()
                    });

                    self.calculate_gradient(lhs.into());
                },
                Op::SumTensor{value, output: _} =>
                {
                    let gradient = gradient.as_value();

                    if let Some(value_gradient) = value.as_gradient()
                    {
                        self.gradient_operations.push(GradientOp::Fill{value: gradient, output: value_gradient});

                        self.calculate_gradient((*value).into());
                    }
                },
                Op::Pow{lhs, power, output: _} =>
                {
                    let gradient = gradient.as_tensor();

                    let (rows, columns) = self.tensor_shape(lhs.as_value());

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        let power_index = new_value_index!(self);
                        self.values[power_index.0] = *power as f32;

                        let pow_d_lhs = new_tensor_index!(self, rows, columns);
                        self.gradient_operations.push(GradientOp::Pow{lhs: lhs.as_value(), power: (power - 1) as u32, output: pow_d_lhs});

                        let pow_d = new_tensor_index!(self, rows, columns);
                        self.gradient_operations.push(GradientOp::MulScalar{lhs: pow_d_lhs, rhs: power_index.into(), output: pow_d});

                        self.gradient_operations.push(GradientOp::MulComponentwise{lhs: pow_d, rhs: gradient, output: lhs_gradient});

                        self.calculate_gradient((*lhs).into());
                    }
                },
                Op::Sigmoid{value, output} =>
                {
                    // sigmoid(x) * (1.0 - sigmoid(x))
                    let gradient = gradient.as_tensor();

                    if let Some(value_gradient) = value.as_gradient()
                    {
                        self.gradient_operations.push(GradientOp::SigmoidDiff{value: output.as_value(), gradient, output: value_gradient});
                    }
                },
                Op::Tanh{value, output} =>
                {
                    // 1 - tanh^2(x)
                    let gradient = gradient.as_tensor();

                    if let Some(value_gradient) = value.as_gradient()
                    {
                        self.gradient_operations.push(GradientOp::TanhDiff{value: output.as_value(), gradient, output: value_gradient});
                    }
                },
                Op::LeakyRelu{value, output: _} =>
                {
                    let gradient = gradient.as_tensor();

                    if let Some(value_gradient) = value.as_gradient()
                    {
                        self.gradient_operations.push(GradientOp::LeakyReluDiff{value: value.as_value(), gradient, output: value_gradient});
                    }
                }
            }
        } else
        {
            let this_gradient_is_one = ();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffValue
{
    Tensor(TensorIndex),
    Value(ValueIndex)
}

impl From<TensorIndex> for DiffValue
{
    fn from(index: TensorIndex) -> Self
    {
        Self::Tensor(index)
    }
}

impl From<ValueIndex> for DiffValue
{
    fn from(index: ValueIndex) -> Self
    {
        Self::Value(index)
    }
}

impl From<DiffWrapper> for DiffValue
{
    fn from(value: DiffWrapper) -> Self
    {
        match value
        {
            DiffWrapper::Tensor(DiffTensor{index, ..}) => Self::Tensor(index),
            DiffWrapper::Value(DiffScalar{index, ..}) => Self::Value(index)
        }
    }
}

impl DiffValue
{
    fn as_tensor(self) -> TensorIndex
    {
        if let Self::Tensor(x) = self { x } else { panic!("as_tensor must be called on a tensor") }
    }

    fn as_value(self) -> ValueIndex
    {
        if let Self::Value(x) = self { x } else { panic!("as_value must be called on a value") }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct OperationIndex(usize);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DiffTensor
{
    index: TensorIndex,
    gradient: Option<TensorIndex>,
    source: Option<OperationIndex>
}

impl DiffTensor
{
    pub fn as_value(&self) -> TensorIndex
    {
        self.index
    }

    pub fn as_gradient(&self) -> Option<TensorIndex>
    {
        self.gradient
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DiffScalar
{
    index: ValueIndex,
    gradient: Option<ValueIndex>,
    source: Option<OperationIndex>
}

impl DiffScalar
{
    pub fn as_value(&self) -> ValueIndex
    {
        self.index
    }

    pub fn as_gradient(&self) -> Option<ValueIndex>
    {
        self.gradient
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DiffWrapper
{
    Tensor(DiffTensor),
    Value(DiffScalar)
}

impl From<DiffTensor> for DiffWrapper
{
    fn from(value: DiffTensor) -> Self
    {
        Self::Tensor(value)
    }
}

impl From<DiffScalar> for DiffWrapper
{
    fn from(value: DiffScalar) -> Self
    {
        Self::Value(value)
    }
}

impl DiffWrapper
{
    pub fn as_tensor(&self) -> TensorIndex
    {
        if let Self::Tensor(DiffTensor{index, ..}) = self
        {
            *index
        } else
        {
            panic!("as_tensor must only be called on a tensor")
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum GradientOp
{
    Copy{src: TensorIndex, dst: TensorIndex},
    AddScalar{lhs: TensorIndex, rhs: ValueIndex, output: TensorIndex},
    AddScalars{lhs: ValueIndex, rhs: ValueIndex, output: ValueIndex},
    Add{lhs: TensorIndex, rhs: TensorIndex, output: TensorIndex},
    Sub{lhs: TensorIndex, rhs: TensorIndex, output: TensorIndex},
    SubFromScalar{lhs: ValueIndex, rhs: TensorIndex, output: TensorIndex},
    MulScalar{lhs: TensorIndex, rhs: ValueIndex, output: TensorIndex},
    MulScalars{lhs: ValueIndex, rhs: ValueIndex, output: ValueIndex},
    MulComponentwise{lhs: TensorIndex, rhs: TensorIndex, output: TensorIndex},
    SumTensor{value: TensorIndex, output: ValueIndex},
    Fill{value: ValueIndex, output: TensorIndex},
    Pow{lhs: TensorIndex, power: u32, output: TensorIndex},
    LeakyRelu{value: TensorIndex, output: TensorIndex},
    LeakyReluDiff{value: TensorIndex, gradient: TensorIndex, output: TensorIndex},
    Sigmoid{value: TensorIndex, output: TensorIndex},
    SigmoidDiff{value: TensorIndex, gradient: TensorIndex, output: TensorIndex},
    Tanh{value: TensorIndex, output: TensorIndex},
    TanhDiff{value: TensorIndex, gradient: TensorIndex, output: TensorIndex}
}

const UNCOMMENT_US: () = {let a = (); ()};
#[derive(Debug, Serialize, Deserialize)]
pub enum Op
{
    AddScalar{lhs: DiffTensor, rhs: DiffScalar, output: DiffTensor},
    Add{lhs: DiffTensor, rhs: DiffTensor, output: DiffTensor},
    Sub{lhs: DiffTensor, rhs: DiffTensor, output: DiffTensor},
    SubFromScalar{lhs: DiffScalar, rhs: DiffTensor, output: DiffTensor},
    MulScalar{lhs: DiffTensor, rhs: DiffScalar, output: DiffTensor},
    MulScalars{lhs: DiffScalar, rhs: DiffScalar, output: DiffScalar},
    MulComponentwise{lhs: DiffTensor, rhs: DiffTensor, output: DiffTensor},
    SumTensor{value: DiffTensor, output: DiffScalar},
    Pow{lhs: DiffTensor, power: i32, output: DiffTensor},
    LeakyRelu{value: DiffTensor, output: DiffTensor},
    Sigmoid{value: DiffTensor, output: DiffTensor},
    Tanh{value: DiffTensor, output: DiffTensor}
    /*
    Dot{lhs: DiffWrapper, rhs: DiffWrapper},
    Matmulv{lhs: DiffWrapper, rhs: DiffWrapper},
    MatmulvAdd{lhs: DiffWrapper, rhs: DiffWrapper, added: DiffWrapper},
    MatmulOneHotvAdd{lhs: DiffWrapper, rhs: OneHotLayer, added: DiffWrapper},
    SoftmaxCrossEntropy{
        values: DiffWrapper,
        softmaxed_values: LayerType,
        targets: OneHotLayer
    }*/
}

const REMOVE_ME_TOO: () = {let a = (); ()};
/*impl DiffBounds for f32
{
    fn into_layer_type<'a>(&self, value_getter: impl FnOnce() -> cell::Ref<'a, LayerType>) -> LayerType
    {
        let value = value_getter();

        LayerType::repeat(value.columns(), value.rows(), *self)
    }

    fn component_mul(&self, rhs: LayerType) -> LayerType
    {
        rhs * *self
    }

    fn reciprocal(self) -> Self
    {
        self.recip()
    }

    fn negate_operation(&self) -> impl DiffOperation
    {
        NegateF32Operation(*self)
    }

    fn mul_ba_operation<'a>(&'a self, a: cell::Ref<'a, AnyDiffType>) -> impl DiffOperation
    {
        MulGF32Operation{a, b: *self}
    }
}

impl DiffBounds for LayerType
{
    fn into_layer_type<'a>(&self, _value_getter: impl FnOnce() -> cell::Ref<'a, LayerType>) -> LayerType
    {
        self.clone()
    }

    fn component_mul(&self, rhs: LayerType) -> LayerType
    {
        self * rhs
    }

    fn matmul_v(&self, lhs: DiffWrapper, mut rhs: DiffWrapper)
    {
        let is_rhs_gradient = rhs.is_gradient();

        if is_rhs_gradient
        {
            rhs.derivatives_set_gradient(MatMulVTransposed{a: &lhs.tensor(), b: self});
        }

        if lhs.is_gradient()
        {
            lhs.derivatives(OuterProduct{a: self, b: &rhs.tensor()});
        }

        if is_rhs_gradient
        {
            rhs.derivatives_skip_gradient();
        }
    }

    fn matmul_v_add(&self, lhs: DiffWrapper, mut rhs: DiffWrapper, added: DiffWrapper)
    {
        let is_rhs_gradient = rhs.is_gradient();

        if is_rhs_gradient
        {
            rhs.derivatives_set_gradient(MatMulVTransposed{a: &lhs.tensor(), b: self});
        }

        if lhs.is_gradient()
        {
            lhs.derivatives(OuterProduct{a: self, b: &rhs.tensor()});
        }

        if is_rhs_gradient
        {
            rhs.derivatives_skip_gradient();
        }

        if added.is_gradient()
        {
            added.derivatives(self);
        }
    }

    fn matmul_one_hot_v_add(&self, lhs: DiffWrapper, rhs: OneHotLayer, added: DiffWrapper)
    {
        if lhs.is_gradient()
        {
            lhs.derivatives(OuterProductOneHot{a: self, b: rhs});
        }

        if added.is_gradient()
        {
            added.derivatives(self);
        }
    }

    fn reciprocal(mut self) -> Self
    {
        LayerType::reciprocal(&mut self);

        self
    }

    fn sigmoid_operation<'a>(&'a self, a: &'a Self) -> impl DiffOperation
    {
        SigmoidOperation{a, gradient: self}
    }

    fn tanh_operation<'a>(&'a self, a: &'a Self) -> impl DiffOperation
    {
        TanhOperation{a, gradient: self}
    }

    fn leaky_relu_operation(&self) -> impl DiffOperation
    {
        LeakyReluOperation(self)
    }

    fn negate_operation(&self) -> impl DiffOperation
    {
        NegateTensorOperation(self)
    }

    fn mul_ba_operation<'a>(&'a self, a: cell::Ref<'a, AnyDiffType>) -> impl DiffOperation
    {
        MulGTensorOperation{a, b: self}
    }
}*/

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
        let this = Self{positions: positions.into(), size};

        debug_assert!(
        {
            let s: HashSet<_> = this.positions.iter().collect();

            s.len() == this.positions.len()
        }, "positions must be unique: {:?}", this.positions.iter().collect::<Vec<_>>());

        this
    }

    pub fn into_layer(self) -> LayerType
    {
        let size = self.size;
        let mut layer = vec![0.0; size];

        for position in self.positions.iter()
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

const REMOVE_ME: () = {let a = (); ()};
/*
    fn derivatives_skip_gradient_no_check(&mut self)
    {
        let gradient = self.gradient.as_ref().unwrap();

        match mem::replace(&mut self.parent, Ops::None)
        {
            Ops::Matmulv{lhs, rhs} =>
            {
                gradient.matmul_v(lhs, rhs);
            },
            Ops::MatmulvAdd{lhs, rhs, added} =>
            {
                gradient.matmul_v_add(lhs, rhs, added);
            },
            Ops::MatmulOneHotvAdd{lhs, rhs, added} =>
            {
                gradient.matmul_one_hot_v_add(lhs, rhs, added);
            },
            Ops::Dot{lhs, rhs} =>
            {
                let gradient = gradient.into_layer_type(|| lhs.tensor());

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
                    let d = gradient.component_mul(softmaxed_values - targets.into_layer());
                    values.derivatives(d);
                }
            }
        }
    }
*/

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
        f: impl FnMut(&mut OperationsRecorder, DiffTensor, DiffTensor) -> DiffTensor
    )
    {
        let mut recorder = OperationsRecorder::new();

        let a = random_tensor(&mut recorder, a_dims.0, a_dims.1);
        let b = random_tensor(&mut recorder, b_dims.0, b_dims.1);

        check_tensor_inner(&mut recorder, a, b, f);
    }

    fn check_vector(f: impl FnMut(&mut OperationsRecorder, DiffTensor, DiffTensor) -> DiffTensor)
    {
        let mut recorder = OperationsRecorder::new();

        let a = random_tensor(&mut recorder, 1, LAYER_CURR);
        let b = random_tensor(&mut recorder, 1, LAYER_CURR);

        check_tensor_inner(&mut recorder, a, b, f);
    }

    fn check_tensor(f: impl FnMut(&mut OperationsRecorder, DiffTensor, DiffTensor) -> DiffTensor)
    {
        let mut recorder = OperationsRecorder::new();

        let a = random_tensor(&mut recorder, LAYER_PREV, LAYER_CURR);
        let b = random_tensor(&mut recorder, LAYER_PREV, LAYER_CURR);

        check_tensor_inner(&mut recorder, a, b, f);
    }

    fn copy_tensor(
        old_recorder: &mut OperationsRecorder,
        new_recorder: &mut OperationsRecorder,
        old_tensor: DiffTensor
    ) -> DiffTensor
    {
        let (rows, columns) = old_recorder.tensor_shape(old_tensor.as_value());
        let old_value = old_recorder.get_tensor(old_tensor.as_value()).clone();

        new_recorder.set_new_tensor(old_value)
    }

    fn check_tensor_inner(
        recorder: &mut OperationsRecorder,
        a: DiffTensor,
        b: DiffTensor,
        mut f: impl FnMut(&mut OperationsRecorder, DiffTensor, DiffTensor) -> DiffTensor
    )
    {
        let out = f(recorder, a, b);

        recorder.finish();
        recorder.gradient_with_respect(out.into());

        recorder.calculate();

        let a_g = recorder.get_tensor(a.as_gradient().unwrap()).clone();
        let b_g = recorder.get_tensor(b.as_gradient().unwrap()).clone();

        let mut vals = |old_recorder: &mut OperationsRecorder, a: DiffTensor, b: DiffTensor|
        {
            assert!(a.source.is_none());
            assert!(b.source.is_none());

            let mut new_recorder = OperationsRecorder::new();

            let new_a = copy_tensor(old_recorder, &mut new_recorder, a);
            let new_b = copy_tensor(old_recorder, &mut new_recorder, b);

            let output = f(&mut new_recorder, new_a, new_b);

            new_recorder.finish();
            new_recorder.gradient_with_respect(output.into());

            new_recorder.calculate();

            new_recorder.get_tensor(output.as_value()).clone()
        };

        let orig = vals(recorder, a, b).sum();

        let epsilon: f32 = 0.009;

        let fg = |value: LayerType|
        {
            let value = value.sum();

            (value - orig) / epsilon
        };

        let mut temp_recorder = OperationsRecorder::new();

        let mut a_fg = vec![0.0; recorder.get_tensor(a.as_value()).total_len()];
        for index in 0..a_fg.len()
        {
            let v = recorder.get_tensor(a.as_value());
            let epsilon = one_hot(v.clone(), index, epsilon, 0.0);

            let this_fg = {
                let a = temp_recorder.set_new_tensor(v.clone() + epsilon);
                let b = copy_tensor(recorder, &mut temp_recorder, b);
                fg(vals(&mut temp_recorder, a, b))
            };

            a_fg[index] = this_fg;
        }

        let mut b_fg = vec![0.0; recorder.get_tensor(b.as_value()).total_len()];
        for index in 0..b_fg.len()
        {
            let v = recorder.get_tensor(b.as_value());
            let epsilon = one_hot(v.clone(), index, epsilon, 0.0);

            let this_fg = {
                let b = temp_recorder.set_new_tensor(v.clone() + epsilon);
                let a = copy_tensor(recorder, &mut temp_recorder, a);
                fg(vals(&mut temp_recorder, a, b))
            };

            b_fg[index] = this_fg;
        }

        let vec_to_layer = |v, layer_match: DiffTensor|
        {
            let mut layer = recorder.get_tensor(layer_match.as_value()).clone();

            layer.swap_raw_values(v);

            layer
        };

        let a_fg = vec_to_layer(a_fg, a);
        let b_fg = vec_to_layer(b_fg, b);

        dbg!(&a_fg, &a_g);
        eprintln!("derivative of a");
        compare_tensor(a_fg, a_g);

        eprintln!("derivative of b");
        compare_tensor(b_fg, b_g);
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

    fn random_tensor(recorder: &mut OperationsRecorder, columns: usize, rows: usize) -> DiffTensor
    {
        let tensor = recorder.new_tensor(rows, columns);
        recorder.set_tensor(tensor.as_value(), LayerType::new_with(rows, columns, random_value));

        tensor
    }

    #[test]
    fn subtraction()
    {
        check_tensor(|recorder, a, b| recorder.sub(a, b))
    }

    #[test]
    fn addition()
    {
        check_tensor(|recorder, a, b| recorder.add(a, b))
    }

    #[test]
    fn multiplication()
    {
        check_tensor(|recorder, a, b| recorder.mul_componentwise(a, b))
    }

    #[test]
    fn non_diff_subdiff()
    {
        check_tensor(|recorder, a, b|
        {
            let one = recorder.set_new_value(1.0);
            let inner_sum = recorder.add(a, b);

            recorder.sub_from_scalar(one, inner_sum)
        })
    }

    #[test]
    fn basic_combined()
    {
        check_tensor(|recorder, a, b|
        {
            let mul_result = recorder.mul_componentwise(a, b);
            recorder.add(mul_result, a)
        })
    }

    #[test]
    fn complex_combined()
    {
        check_tensor(|recorder, a, b|
        {
            let left = recorder.mul_componentwise(a, b);
            let bb = recorder.mul_componentwise(b, b);

            let lefta = recorder.add(left, a);
            let leftab = recorder.add(lefta, b);

            let right = recorder.add(bb, a);

            recorder.sub(leftab, right)
        })
    }

    #[test]
    fn sum_tensor_product()
    {
        check_tensor(|recorder, a, b|
        {
            let s = recorder.sum_tensor(b);

            recorder.mul_scalar(a, s)
        })
    }

    #[test]
    fn sum_tensor_addition()
    {
        check_tensor(|recorder, a, b|
        {
            let s = recorder.sum_tensor(b);

            recorder.add_scalar(a, s)
        })
    }

    #[test]
    fn sum_tensor_product_negative()
    {
        check_tensor(|recorder, a, b|
        {
            let s = recorder.sum_tensor(b);

            let m1 = recorder.set_new_value(-1.0);
            let sn = recorder.mul_scalars(s, m1);

            recorder.mul_scalar(a, sn)
        })
    }

    const UNCOMMENT_ME_AAA: () = { let a = (); () };
    /*#[test]
    fn dot_product()
    {
        check_vector(|a, b| a + a.clone().dot(b.clone()))
    }*/

    #[test]
    fn scalar_minus_tensor()
    {
        check_tensor(|recorder, a, b|
        {
            let s = recorder.sum_tensor(a);
            recorder.sub_from_scalar(s, b)
        })
    }

    #[test]
    fn scalar_minus_tensor_stuff()
    {
        check_tensor(|recorder, a, b|
        {
            let s = recorder.sum_tensor(a);
            let right = recorder.sub_from_scalar(s, b);

            let two = recorder.set_new_value(2.0);

            recorder.sub_from_scalar(two, right)
        })
    }

    #[test]
    fn leaky_relu()
    {
        check_tensor(|recorder, a, b|
        {
            let x = recorder.leaky_relu(a);
            recorder.add(x, b)
        })
    }

    // flexing my math functions name knowledge
    #[test]
    fn logistic_function()
    {
        check_tensor(|recorder, a, b|
        {
            let x = recorder.sigmoid(a);
            recorder.add(x, b)
        })
    }

    #[test]
    fn hyperbolic_tangent()
    {
        check_tensor(|recorder, a, b|
        {
            let x = recorder.tanh(a);
            recorder.add(x, b)
        })
    }

    #[test]
    fn pow()
    {
        check_tensor(|recorder, a, b|
        {
            let output = recorder.pow(a, 3);
            recorder.add(output, b)
        })
    }

    const UNCOMMENT_ME_THREE: () = { let a = (); () };
    /*#[test]
    fn matrix_multiplication()
    {
        check_tensor_with_dims((4, 2), (1, 4), |a, b| a.matmulv(b) + b.sum())
    }*/

    fn create_targets() -> OneHotLayer
    {
        let pos = fastrand::usize(0..LAYER_CURR);

        OneHotLayer::new([pos], LAYER_CURR)
    }

    const UNCOMMENT_ME_FOUR: () = { let a = (); () };
    /*#[test]
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
    }*/
}
