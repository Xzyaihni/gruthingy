use std::{
    f32,
    mem,
    convert,
    fmt::{self, Debug},
    borrow::Borrow,
    collections::HashSet,
    ops::DivAssign
};

#[allow(unused_imports)]
use std::iter;

use serde::{Serialize, Deserialize};

use matrix_wrapper::MatrixWrapper;

mod matrix_wrapper;


pub type LayerType = MatrixWrapper;

pub const LEAKY_SLOPE: f32 = 0.01;

const OPT_INFO: bool = false;

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
        let this_is_horrible = ();
        Softmaxer::softmax(&mut self);
        let softmaxed = self.clone();

        // assumes that targets r either 0 or 1
        self.ln_onehot(targets);

        let s = self.dot_onehot(targets);

        (softmaxed, -s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecorderState
{
    Recording,
    AwaitingGradient,
    AwaitingResolve,
    Ready
}

impl RecorderState
{
    fn before_or_at(self, other: Self) -> bool
    {
        let as_id = |s|
        {
            match s
            {
                Self::Recording => 0,
                Self::AwaitingGradient => 1,
                Self::AwaitingResolve => 2,
                Self::Ready => 3
            }
        };

        as_id(self) <= as_id(other)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockIndex(usize);

#[allow(dead_code)]
impl BlockIndex
{
    pub fn undefined() -> Self { Self(usize::MAX) }

    pub fn into_index(self) -> usize { self.0 }
}

struct ForceNoPretty<T>(T);
impl<T: Debug> Debug for ForceNoPretty<T>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        write!(f, "{:?}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct LiveRange
{
    start: Option<i32>,
    end: Option<i32>
}

impl LiveRange
{
    fn valid_range(&self) -> bool
    {
        let start = if let Some(x) = self.start { x } else { return false };

        let end = if let Some(x) = self.end { x } else { return false };

        start < end
    }

    fn overlaps(&self, other: &Self) -> bool
    {
        debug_assert!(self.valid_range() && other.valid_range());

        let this_start = self.start.unwrap();
        let this_end = self.end.unwrap();

        let other_start = other.start.unwrap();
        let other_end = other.end.unwrap();

        this_start <= other_end && this_end >= other_start
    }
}

#[derive(Clone)]
pub struct OperationsBlock
{
    block_inputs: Vec<TensorPtr>,
    block_outputs: Vec<TensorPtr>,
    value_live_ranges: Vec<LiveRange>,
    tensor_live_ranges: Vec<LiveRange>,
    recording_operations: Vec<Op>,
    gradient_operations: Vec<GradientOp<TensorPtr>>,
    raw_operations: Vec<GradientOp<TensorIndex>>,
    feedforward_operations_count: usize
}

impl Debug for OperationsBlock
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        f.debug_struct("OperationsBlock")
            .field("block_inputs", &self.block_inputs.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
            .field("block_outputs", &self.block_outputs.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
            .field("value_live_ranges", &self.value_live_ranges.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
            .field("tensor_live_ranges", &self.tensor_live_ranges.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
            .field("recording_operations", &self.recording_operations.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
            .field("gradient_operations", &self.gradient_operations.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
            .field("raw_operations", &self.raw_operations.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
            .field("feedforward_operations_count", &self.feedforward_operations_count)
            .finish()
    }
}

impl Default for OperationsBlock
{
    fn default() -> Self
    {
        OperationsBlock{
            block_inputs: Vec::new(),
            block_outputs: Vec::new(),
            value_live_ranges: Vec::new(),
            tensor_live_ranges: Vec::new(),
            recording_operations: Vec::new(),
            gradient_operations: Vec::new(),
            raw_operations: Vec::new(),
            feedforward_operations_count: 0
        }
    }
}

struct LayerNoLong<'a>(usize, &'a LayerType);

impl Debug for LayerNoLong<'_>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        if self.1.total_len() > self.0
        {
            write!(f, "{{rows: {}, columns: {}, values: (has {} values)}}", self.1.rows(), self.1.columns(), self.1.total_len())
        } else
        {
            LayerType::fmt(self.1, f)
        }
    }
}

#[derive(Debug, Clone)]
enum TensorMemoryValue
{
    Value(LayerType),
    Size{rows: usize, columns: usize}
}

impl TensorMemoryValue
{
    fn tensor_shape(&self) -> (usize, usize)
    {
        match self
        {
            Self::Value(tensor) => tensor.shape(),
            Self::Size{rows, columns} => (*rows, *columns)
        }
    }
}

#[derive(Debug, Clone)]
struct TensorMemorySlot
{
    memory: Option<TensorIndex>,
    value: TensorMemoryValue
}

struct SlotNoLong<'a>(usize, &'a TensorMemorySlot);

impl Debug for SlotNoLong<'_>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let mut head = f.debug_struct("TensorMemorySlot");

        let head = head.field("memory", &self.1.memory);

        match &self.1.value
        {
            TensorMemoryValue::Value(x) => head.field("value", &LayerNoLong(self.0, x)).finish(),
            x => head.field("value", x).finish()
        }
    }
}

#[derive(Clone)]
pub struct OperationsRecorder
{
    state: RecorderState,
    current_block: BlockIndex,
    global_value_live_ranges: Vec<LiveRange>,
    global_tensor_live_ranges: Vec<LiveRange>,
    tensors_memory: Vec<TensorMemorySlot>,
    values: Vec<f32>,
    tensors: Vec<LayerType>,
    one_hot_layers: Vec<OneHotLayer>,
    operations_blocks: Vec<OperationsBlock>,
    #[cfg(debug_assertions)]
    used_values: Vec<ValueIndex>
}

impl Debug for OperationsRecorder
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let max_length = 50;

        f.debug_struct("OperationsRecorder")
            .field("state", &self.state)
            .field("global_value_live_ranges", &self.global_value_live_ranges.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
            .field("global_tensor_live_ranges", &self.global_tensor_live_ranges.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
            .field("tensors_memory", &self.tensors_memory.iter().map(|x| ForceNoPretty(SlotNoLong(max_length, x))).collect::<Vec<_>>())
            .field("values", &ForceNoPretty(&self.values))
            .field("tensors", &self.tensors.iter().map(|x| LayerNoLong(max_length, x)).collect::<Vec<_>>())
            .field("one_hot_layers", &self.one_hot_layers)
            .field("operations_blocks", &self.operations_blocks)
            .finish()
    }
}

// avoids borrow checker
macro_rules! new_tensor_index
{
    ($this:expr, $rows:expr, $columns:expr) =>
    {
        new_tensor_index!($this, $rows, $columns, TensorMemoryValue::Size{rows: $rows, columns: $columns})
    };
    ($this:expr, $rows:expr, $columns:expr, $value:expr) =>
    {
        {
            let id = $this.tensors_memory.len();

            $this.global_tensor_live_ranges.push(LiveRange{start: None, end: None});
            $this.tensors_memory.push(TensorMemorySlot{value: $value, memory: None});

            TensorPtr(id)
        }
    }
}

macro_rules! new_value_index
{
    ($this:expr) =>
    {
        {
            let id = $this.values.len();

            $this.global_value_live_ranges.push(LiveRange{start: None, end: None});
            $this.values.push(0.0);

            ValueIndex(id)
        }
    }
}

macro_rules! new_tensor
{
    ($this:expr, $source:expr, $gradient:expr, $rows:expr, $columns:expr) =>
    {
        new_tensor!($this, $source, $gradient, $rows, $columns, TensorMemoryValue::Size{rows: $rows, columns: $columns})
    };
    ($this:expr, $source:expr, $gradient:expr, $rows:expr, $columns:expr, $value:expr) =>
    {
        {
            DiffTensorPtr{
                index: new_tensor_index!($this, $rows, $columns, $value),
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

macro_rules! tensor_shape
{
    ($this:expr, $tensor:expr) =>
    {
        {
            debug_assert!(matches!($tensor, TensorPtr(_)));
            $this.tensors_memory[$tensor.0].value.tensor_shape()
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

            debug_assert_eq!(a_shape, b_shape);

            let output = $this.new_tensor_op(source, a_rows, a_columns);

            $this.operations_blocks[$this.current_block.0].recording_operations.push(Op::$name{lhs: $a, rhs: $b, output});

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

            $this.operations_blocks[$this.current_block.0].recording_operations.push(Op::$name{value: $a, output});

            output
        }
    }
}

#[allow(dead_code)]
impl OperationsRecorder
{
    pub fn new() -> Self
    {
        Self{
            state: RecorderState::Recording,
            current_block: BlockIndex(0),
            global_value_live_ranges: Vec::new(),
            global_tensor_live_ranges: Vec::new(),
            tensors_memory: Vec::new(),
            values: Vec::new(),
            tensors: Vec::new(),
            one_hot_layers: Vec::new(),
            operations_blocks: vec![OperationsBlock::default()],
            #[cfg(debug_assertions)]
            used_values: Vec::new()
        }
    }

    pub fn new_tensor(&mut self, rows: usize, columns: usize) -> DiffTensorPtr
    {
        let input = self.new_tensor_with_source(None, true, rows, columns);
        self.global_tensor_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn new_tensor_no_gradient(&mut self, rows: usize, columns: usize) -> DiffTensorPtr
    {
        let input = self.new_tensor_with_source(None, false, rows, columns);
        self.global_tensor_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn new_value(&mut self) -> DiffScalar
    {
        let input = self.new_value_with_source(None, true);
        self.global_value_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn new_one_hot(&mut self) -> OneHotIndex
    {
        let id = self.one_hot_layers.len();

        self.one_hot_layers.push(OneHotLayer::empty());

        OneHotIndex(id)
    }

    fn new_tensor_op(
        &mut self,
        source: Option<OperationIndex>,
        rows: usize,
        columns: usize
    ) -> DiffTensorPtr
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
    ) -> DiffTensorPtr
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

    pub fn set_one_hot(&mut self, index: OneHotIndex, value: OneHotLayer)
    {
        self.one_hot_layers[index.0] = value;
    }

    pub fn set_new_tensor_gradientable(&mut self, value: LayerType) -> DiffTensorPtr
    {
        let rows = value.rows();
        let columns = value.columns();

        let input = new_tensor!(self, None, true, rows, columns, TensorMemoryValue::Value(value));
        self.global_tensor_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn set_new_tensor(&mut self, value: LayerType) -> DiffTensorPtr
    {
        let rows = value.rows();
        let columns = value.columns();

        let input = new_tensor!(self, None, false, rows, columns, TensorMemoryValue::Value(value));
        self.global_tensor_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn set_new_value(&mut self, value: f32) -> DiffScalar
    {
        let scalar = self.new_value_with_source(None, false);
        self.set_value(scalar.as_value(), value);

        self.global_value_live_ranges[scalar.as_value().0].start = Some(-1);

        scalar
    }

    fn set_ones(&mut self, wrapper: DiffWrapper)
    {
        match wrapper
        {
            DiffWrapper::Tensor(DiffTensorPtr{index, gradient, ..}) =>
            {
                let (rows, columns) = self.tensor_shape(index);

                let new_value = TensorMemoryValue::Value(LayerType::repeat(rows, columns, 1.0));

                let gradient_ptr: TensorPtr = gradient.expect("gradient must exist");

                self.global_tensor_live_ranges[gradient_ptr.0].start = Some(-1);
                self.tensors_memory[gradient_ptr.0].value = new_value;
            },
            DiffWrapper::Value(DiffScalar{gradient, ..}) =>
            {
                let gradient = gradient.expect("gradient must exist");

                self.global_value_live_ranges[gradient.0].start = Some(-1);

                self.set_value(gradient, 1.0)
            }
        }
    }

    pub fn get_tensor(&self, index: TensorIndex) -> &LayerType
    {
        &self.tensors[index.0]
    }

    pub fn get_tensor_mut(&mut self, index: TensorIndex) -> &mut LayerType
    {
        &mut self.tensors[index.0]
    }

    pub fn get_value(&self, index: ValueIndex) -> f32
    {
        #[cfg(debug_assertions)]
        debug_assert!(self.used_values.contains(&index), "store_value_until_end must be called on {index:?}");

        self.values[index.0]
    }

    pub fn get_one_hot(&self, index: OneHotIndex) -> &OneHotLayer
    {
        &self.one_hot_layers[index.0]
    }

    fn current_source(&self) -> OperationIndex
    {
        debug_assert_eq!(self.state, RecorderState::Recording);

        OperationIndex(self.current_block, self.operations_blocks[self.current_block.0].recording_operations.len())
    }

    pub fn add_scalars(&mut self, a: DiffScalar, b: DiffScalar) -> DiffScalar
    {
        let source = Some(self.current_source());

        let output = self.new_value_op(source);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::AddScalars{lhs: a, rhs: b, output});

        output
    }

    pub fn add_scalar(&mut self, a: DiffTensorPtr, b: DiffScalar) -> DiffTensorPtr
    {
        let source = Some(self.current_source());

        let (a_rows, a_columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(source, a_rows, a_columns);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::AddScalar{lhs: a, rhs: b, output});

        output
    }

    pub fn add(&mut self, a: DiffTensorPtr, b: DiffTensorPtr) -> DiffTensorPtr
    {
        impl_pair_tensor_op!(self, a, b, Add)
    }

    pub fn sub(&mut self, a: DiffTensorPtr, b: DiffTensorPtr) -> DiffTensorPtr
    {
        impl_pair_tensor_op!(self, a, b, Sub)
    }

    pub fn sub_from_scalar(&mut self, a: DiffScalar, b: DiffTensorPtr) -> DiffTensorPtr
    {
        let source = Some(self.current_source());

        let (a_rows, a_columns) = self.tensor_shape(b.as_value());

        let output = self.new_tensor_op(source, a_rows, a_columns);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::SubFromScalar{lhs: a, rhs: b, output});

        output
    }

    pub fn mul_scalars(&mut self, a: DiffScalar, b: DiffScalar) -> DiffScalar
    {
        let source = Some(self.current_source());

        let output = self.new_value_op(source);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::MulScalars{lhs: a, rhs: b, output});

        output
    }

    pub fn mul_scalar(&mut self, a: DiffTensorPtr, b: DiffScalar) -> DiffTensorPtr
    {
        let source = Some(self.current_source());

        let (a_rows, a_columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(source, a_rows, a_columns);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::MulScalar{lhs: a, rhs: b, output});

        output
    }

    pub fn mul_componentwise(&mut self, a: DiffTensorPtr, b: DiffTensorPtr) -> DiffTensorPtr
    {
        impl_pair_tensor_op!(self, a, b, MulComponentwise)
    }

    pub fn matmulv(&mut self, a: DiffTensorPtr, b: DiffTensorPtr) -> DiffTensorPtr
    {
        let source = Some(self.current_source());

        let (a_rows, a_columns) = self.tensor_shape(a.as_value());
        let (b_rows, b_columns) = self.tensor_shape(b.as_value());

        debug_assert_eq!(a_columns, b_rows);

        let output = self.new_tensor_op(source, a_rows, b_columns);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::Matmulv{lhs: a, rhs: b, output});

        output
    }

    pub fn matmulv_add(&mut self, a: DiffTensorPtr, b: DiffTensorPtr, added: DiffTensorPtr) -> DiffTensorPtr
    {
        let source = Some(self.current_source());

        let (a_rows, a_columns) = self.tensor_shape(a.as_value());
        let (b_rows, b_columns) = self.tensor_shape(b.as_value());

        let (rows, columns) = self.tensor_shape(added.as_value());

        debug_assert_eq!(a_columns, b_rows);

        debug_assert_eq!(a_rows, rows);
        debug_assert_eq!(b_columns, columns);

        let output = self.new_tensor_op(source, rows, columns);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::MatmulvAdd{lhs: a, rhs: b, added, output});

        output
    }

    pub fn matmul_onehotv_add(&mut self, a: DiffTensorPtr, b: OneHotIndex, added: DiffTensorPtr) -> DiffTensorPtr
    {
        let source = Some(self.current_source());

        let (a_rows, _a_columns) = self.tensor_shape(a.as_value());
        let b_columns = 1;

        let (rows, columns) = self.tensor_shape(added.as_value());

        debug_assert_eq!(a_rows, rows);
        debug_assert_eq!(b_columns, columns);

        let output = self.new_tensor_op(source, rows, columns);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::MatmulOneHotvAdd{lhs: a, rhs: b, added, output});

        output
    }

    pub fn sum_tensor(&mut self, a: DiffTensorPtr) -> DiffScalar
    {
        let source = Some(self.current_source());

        let output = self.new_value_op(source);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::SumTensor{value: a, output});

        output
    }

    pub fn dot(&mut self, a: DiffTensorPtr, b: DiffTensorPtr) -> DiffScalar
    {
        let source = Some(self.current_source());

        debug_assert_eq!(self.tensor_shape(a.as_value()), self.tensor_shape(b.as_value()));

        let output = self.new_value_op(source);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::Dot{lhs: a, rhs: b, output});

        output
    }

    pub fn pow(&mut self, a: DiffTensorPtr, power: i32) -> DiffTensorPtr
    {
        let source = Some(self.current_source());

        let (rows, columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(source, rows, columns);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::Pow{lhs: a, power, output});

        output
    }

    pub fn sigmoid(&mut self, a: DiffTensorPtr) -> DiffTensorPtr
    {
        impl_map_tensor_op!(self, a, Sigmoid)
    }

    pub fn tanh(&mut self, a: DiffTensorPtr) -> DiffTensorPtr
    {
        impl_map_tensor_op!(self, a, Tanh)
    }

    pub fn leaky_relu(&mut self, a: DiffTensorPtr) -> DiffTensorPtr
    {
        impl_map_tensor_op!(self, a, LeakyRelu)
    }

    pub fn softmax_cross_entropy(&mut self, values: DiffTensorPtr, targets: OneHotIndex) -> (DiffTensorPtr, DiffScalar)
    {
        let source = Some(self.current_source());

        let (rows, columns) = self.tensor_shape(values.as_value());

        let softmaxed_output = self.new_tensor_with_source(source, false, rows, columns);
        let output = self.new_value_op(source);

        self.operations_blocks[self.current_block.0].recording_operations.push(Op::SoftmaxCrossEntropy{values, targets, softmaxed_output, output});

        (softmaxed_output, output)
    }

    pub fn tensor_shape(&self, tensor: TensorPtr) -> (usize, usize)
    {
        tensor_shape!(self, tensor)
    }

    pub fn resolve_tensor_ptr(&self, index_ptr: TensorPtr) -> TensorIndex
    {
        debug_assert_eq!(self.state, RecorderState::Ready);

        self.tensors_memory[index_ptr.0].memory.expect("must be resolved")
    }

    pub fn resolve_diff_tensor_ptr(&self, diff: DiffTensorPtr) -> DiffTensor
    {
        DiffTensor{
            index: self.resolve_tensor_ptr(diff.index),
            gradient: diff.gradient.map(|x| self.resolve_tensor_ptr(x))
        }
    }

    pub fn set_block_input(&mut self, block: BlockIndex, index_ptr: TensorPtr)
    {
        debug_assert!(self.state.before_or_at(RecorderState::AwaitingGradient));

        let inputs = &mut self.operations_blocks[block.0].block_inputs;

        if !inputs.contains(&index_ptr)
        {
            inputs.push(index_ptr);
        }
    }

    pub fn store_tensor_until_end(&mut self, index_ptr: TensorPtr)
    {
        debug_assert!(self.state.before_or_at(RecorderState::AwaitingGradient));

        self.global_tensor_live_ranges[index_ptr.0].end = Some(i32::MAX);
    }

    pub fn store_tensor_until_end_in_block(&mut self, block: BlockIndex, index_ptr: TensorPtr)
    {
        debug_assert!(self.state.before_or_at(RecorderState::AwaitingGradient));

        let outputs = &mut self.operations_blocks[block.0].block_outputs;

        if !outputs.contains(&index_ptr)
        {
            outputs.push(index_ptr);
        }
    }

    pub fn store_value_until_end(&mut self, index: ValueIndex)
    {
        debug_assert!(self.state.before_or_at(RecorderState::AwaitingGradient));

        self.global_value_live_ranges[index.0].end = Some(i32::MAX);

        #[cfg(debug_assertions)]
        self.used_values.push(index);
    }

    pub fn new_block(&mut self) -> BlockIndex
    {
        let id = BlockIndex(self.operations_blocks.len());

        self.operations_blocks.push(OperationsBlock::default());

        id
    }

    pub fn set_current_block(&mut self, index: BlockIndex)
    {
        self.current_block = index;
    }

    pub fn current_block(&self) -> BlockIndex
    {
        self.current_block
    }

    pub fn blocks_iter(&self) -> impl Iterator<Item=BlockIndex> + use<>
    {
        (0..self.blocks_count()).map(BlockIndex)
    }

    pub fn blocks_count(&self) -> usize
    {
        self.operations_blocks.len()
    }

    pub fn calculate_feedforward(&mut self, block: BlockIndex)
    {
        let count = self.operations_blocks[block.0].feedforward_operations_count;

        self.calculate_steps(block, count);
    }

    pub fn calculate(&mut self, block: BlockIndex)
    {
        let count = self.operations_blocks[block.0].raw_operations.len();

        self.calculate_steps(block, count);
    }

    fn calculate_steps(&mut self, block: BlockIndex, steps: usize)
    {
        debug_assert_eq!(self.state, RecorderState::Ready);

        self.operations_blocks[block.0].raw_operations.iter().take(steps).for_each(|gradient_op|
        {
            macro_rules! get_disjoint_mut
            {
                ($($name:ident),+) =>
                {
                    {
                        #[cfg(debug_assertions)]
                        {
                            self.tensors.get_disjoint_mut([$($name.0,)+]).unwrap()
                        }

                        #[cfg(not(debug_assertions))]
                        {
                            unsafe{ self.tensors.get_disjoint_unchecked_mut([$($name.0,)+]) }
                        }
                    }
                }
            }

            //let before_instant = std::time::Instant::now();
            match gradient_op
            {
                GradientOp::None => unreachable!(),
                GradientOp::Copy{src, dst} => self.tensors[dst.0] = self.tensors[src.0].clone(),
                GradientOp::CopyScalar{src, dst} =>
                {
                    self.values[dst.0] = self.values[src.0].clone();
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
                    let [output, lhs, rhs] = get_disjoint_mut!(output, lhs, rhs);

                    output.add_to(lhs, rhs);
                },
                GradientOp::AddInplace{value, output} =>
                {
                    let [output, value] = get_disjoint_mut!(output, value);

                    *output += value;
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
                    let [output, lhs, rhs] = get_disjoint_mut!(output, lhs, rhs);

                    output.component_mul_into(lhs, rhs);
                },
                GradientOp::SumTensor{value, output} =>
                {
                    self.values[output.0] = self.tensors[value.0].sum();
                },
                GradientOp::Dot{lhs, rhs, output} =>
                {
                    self.values[output.0] = self.tensors[lhs.0].dot(&self.tensors[rhs.0]);
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
                    let [output, value, gradient] = get_disjoint_mut!(output, value, gradient);

                    output.sigmoid_gradient_inplace(value, gradient);
                },
                GradientOp::Tanh{value, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = self.tensors[value.0].tanh();
                },
                GradientOp::TanhDiff{value, gradient, output} =>
                {
                    let [output, value, gradient] = get_disjoint_mut!(output, value, gradient);

                    output.tanh_gradient_inplace(value, gradient);
                },
                GradientOp::LeakyRelu{value, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = self.tensors[value.0].leaky_relu();
                },
                GradientOp::LeakyReluDiff{value, gradient, output} =>
                {
                    let [output, value, gradient] = get_disjoint_mut!(output, value, gradient);

                    output.leaky_relu_gradient_inplace(value, gradient);
                },
                GradientOp::SoftmaxCrossEntropy{values, targets, softmaxed_output, output} =>
                {
                    let optimize_this = ();
                    (self.tensors[softmaxed_output.0], self.values[output.0]) = self.tensors[values.0].clone()
                        .softmax_cross_entropy(&self.one_hot_layers[targets.0]);
                },
                GradientOp::SoftmaxCrossEntropyNoSoftmaxed{values, targets, output} =>
                {
                    self.values[output.0] = self.tensors[values.0].clone().softmax_cross_entropy(&self.one_hot_layers[targets.0]).1;
                },
                GradientOp::SoftmaxCrossEntropyDiff{softmaxed_values, gradient, targets, output} =>
                {
                    debug_assert_eq!(
                        self.tensors[softmaxed_values.0].shape(), (self.one_hot_layers[targets.0].size, 1),
                        "softmaxed: {softmaxed_values:?}, targets: {targets:?}"
                    );

                    let optimize_this = ();
                    self.tensors[output.0] = (&self.tensors[softmaxed_values.0] - self.one_hot_layers[targets.0].clone().into_layer()) * self.values[gradient.0];
                },
                GradientOp::Matmulv{lhs, rhs, output} =>
                {
                    let [output, lhs, rhs] = get_disjoint_mut!(output, lhs, rhs);

                    output.matmulv_into(lhs, rhs);
                },
                GradientOp::MatmulvAdd{lhs, rhs, added, output} =>
                {
                    let [output, lhs, rhs, added] = get_disjoint_mut!(output, lhs, rhs, added);

                    output.matmulv_add_into(lhs, rhs, added);
                },
                GradientOp::MatmulOneHotvAdd{lhs, rhs, added, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = self.tensors[lhs.0].matmul_onehotv_add(&self.one_hot_layers[rhs.0], &self.tensors[added.0]);
                },
                GradientOp::MatmulvTransposed{lhs, rhs, output} =>
                {
                    let [output, lhs, rhs] = get_disjoint_mut!(output, lhs, rhs);

                    output.matmulv_transposed_into(lhs, rhs);
                },
                GradientOp::OuterProduct{lhs, rhs, output} =>
                {
                    let [output, lhs, rhs] = get_disjoint_mut!(output, lhs, rhs);

                    output.outer_product_into(lhs, rhs);
                },
                GradientOp::OuterProductOneHot{lhs, rhs, output} =>
                {
                    let [output, lhs] = get_disjoint_mut!(output, lhs);

                    output.outer_product_one_hot_into(lhs, &self.one_hot_layers[rhs.0]);
                }
            }

            //eprintln!("{}, elapsed {:.3} us",format!("{gradient_op:?}").split(' ').next().unwrap(),before_instant.elapsed().as_nanos()as f64/1000.0);
        });
    }

    pub fn finish(&mut self)
    {
        debug_assert_eq!(self.state, RecorderState::Recording);

        self.operations_blocks.iter_mut().for_each(|block|
        {
            block.gradient_operations = block.recording_operations.iter().map(|op|
            {
                match op
                {
                    Op::AddScalar{lhs, rhs, output} =>
                    {
                        GradientOp::AddScalar{lhs: lhs.as_value(), rhs: rhs.as_value(), output: output.as_value()}
                    },
                    Op::AddScalars{lhs, rhs, output} =>
                    {
                        GradientOp::AddScalars{lhs: lhs.as_value(), rhs: rhs.as_value(), output: output.as_value()}
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
                    Op::Dot{lhs, rhs, output} =>
                    {
                        GradientOp::Dot{lhs: lhs.as_value(), rhs: rhs.as_value(), output: output.as_value()}
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
                    },
                    Op::SoftmaxCrossEntropy{values, targets, softmaxed_output, output} =>
                    {
                        GradientOp::SoftmaxCrossEntropy{
                            values: values.as_value(),
                            targets: targets.clone(),
                            softmaxed_output: softmaxed_output.as_value(),
                            output: output.as_value()
                        }
                    },
                    Op::Matmulv{lhs, rhs, output} =>
                    {
                        GradientOp::Matmulv{lhs: lhs.as_value(), rhs: rhs.as_value(), output: output.as_value()}
                    },
                    Op::MatmulvAdd{lhs, rhs, added, output} =>
                    {
                        GradientOp::MatmulvAdd{lhs: lhs.as_value(), rhs: rhs.as_value(), added: added.as_value(), output: output.as_value()}
                    },
                    Op::MatmulOneHotvAdd{lhs, rhs, added, output} =>
                    {
                        GradientOp::MatmulOneHotvAdd{lhs: lhs.as_value(), rhs: *rhs, added: added.as_value(), output: output.as_value()}
                    }
                }
            }).collect();
        });

        self.state = RecorderState::AwaitingGradient;
    }

    fn combine_same_outputs(&mut self, block: BlockIndex)
    {
        let this_block = &mut self.operations_blocks[block.0];

        let mut i = 1;
        while i < this_block.gradient_operations.len()
        {
            let this = &this_block.gradient_operations[i];

            if let Some(previous) = (0..i).find(|previous|
            {
                this_block.gradient_operations[*previous].diff_output_of() == this.diff_output_of()
            })
            {
                fn match_operation<T>(
                    op: &mut GradientOp<TensorPtr>,
                    on_tensor: impl FnOnce(&mut TensorPtr) -> T,
                    on_scalar: impl FnOnce(&mut ValueIndex) -> T
                ) -> T
                {
                    match op
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
                        | GradientOp::LeakyReluDiff{output, ..}
                        | GradientOp::SoftmaxCrossEntropyDiff{output, ..}
                        | GradientOp::Matmulv{output, ..}
                        | GradientOp::MatmulvAdd{output, ..}
                        | GradientOp::MatmulOneHotvAdd{output, ..}
                        | GradientOp::MatmulvTransposed{output, ..}
                        | GradientOp::OuterProduct{output, ..}
                        | GradientOp::OuterProductOneHot{output, ..} =>
                        {
                            on_tensor(output)
                        },
                        GradientOp::CopyScalar{dst: output, ..}
                        | GradientOp::AddScalars{output, ..}
                        | GradientOp::MulScalars{output, ..}
                        | GradientOp::SumTensor{output, ..}
                        | GradientOp::Dot{output, ..}
                        | GradientOp::SoftmaxCrossEntropy{output, ..} =>
                        {
                            on_scalar(output)
                        },
                        GradientOp::None
                        | GradientOp::AddInplace{..}
                        | GradientOp::SoftmaxCrossEntropyNoSoftmaxed{..} => unreachable!()
                    }
                }

                let late_call = match_operation(&mut this_block.gradient_operations[i], |this_output| -> Box<dyn FnOnce(&mut OperationsBlock)>
                {
                    let (rows, columns) = tensor_shape!(self, this_output);

                    let final_output = *this_output;

                    let temporary_lhs_index = new_tensor_index!(self, rows, columns);
                    let temporary_rhs_index = new_tensor_index!(self, rows, columns);

                    *this_output = temporary_rhs_index;

                    Box::new(move |this_block|
                    {
                        let tail = match_operation(&mut this_block.gradient_operations[previous], |previous_output|
                        {
                            *previous_output = temporary_lhs_index;

                            move |this_block: &mut OperationsBlock|
                            {
                                this_block.gradient_operations[(previous + 1)..i].iter_mut().for_each(|between_op|
                                {
                                    *between_op = between_op.clone().map_args(
                                        |arg| if arg == final_output { temporary_lhs_index } else { arg },
                                        convert::identity
                                    );
                                });

                                this_block.gradient_operations.insert(
                                    i + 1,
                                    GradientOp::Add{lhs: temporary_lhs_index, rhs: temporary_rhs_index, output: final_output}
                                );
                            }
                        }, |_output|
                        {
                            unreachable!()
                        });

                        (tail)(this_block);
                    })
                }, |output|
                {
                    let final_output = *output;

                    let temporary_lhs_index = new_value_index!(self);
                    let temporary_rhs_index = new_value_index!(self);

                    *output = temporary_rhs_index;

                    Box::new(move |this_block|
                    {
                        let tail = match_operation(&mut this_block.gradient_operations[previous], |_output|
                        {
                            unreachable!()
                        }, |previous_output|
                        {
                            *previous_output = temporary_lhs_index;

                            move |this_block: &mut OperationsBlock|
                            {
                                this_block.gradient_operations[(previous + 1)..i].iter_mut().for_each(|between_op|
                                {
                                    *between_op = between_op.clone().map_args(
                                        convert::identity,
                                        |arg| if arg == final_output { temporary_lhs_index } else { arg }
                                    );
                                });

                                this_block.gradient_operations.insert(
                                    i + 1,
                                    GradientOp::AddScalars{lhs: temporary_lhs_index, rhs: temporary_rhs_index, output: final_output}
                                );
                            }
                        });

                        (tail)(this_block);
                    })
                });

                (late_call)(this_block);

                i += 1;
            }

            i += 1;
        }
    }

    fn copy_coalesce(&mut self, block: BlockIndex)
    {
        let this_block = &mut self.operations_blocks[block.0];

        let mut i = 1;
        while i < this_block.gradient_operations.len()
        {
            let this = &this_block.gradient_operations[i];

            if let GradientOp::Copy{src, dst} = *this
            {
                let dst_is_output = this_block.block_outputs.contains(&dst)
                    || self.global_tensor_live_ranges[dst.0].end == Some(i32::MAX);

                if !dst_is_output
                {
                    let mut overlaps_args = false;

                    // dst can only appear starting from this clone so no need to check before it
                    for check_op in &this_block.gradient_operations[(i + 1)..]
                    {
                        let mut any_is_src = false;
                        let mut any_is_dst = false;

                        let mut f = |v|
                        {
                            if v == src { any_is_src = true }
                            if v == dst { any_is_dst = true }
                        };

                        check_op.for_args(&mut f, |_| {});
                        check_op.for_outputs(f, |_| {});

                        overlaps_args = any_is_src && any_is_dst;

                        if overlaps_args
                        {
                            break;
                        }
                    }

                    if !overlaps_args
                    {
                        for check_op in this_block.gradient_operations[(i + 1)..].iter_mut()
                        {
                            *check_op = check_op.clone().map_args(|arg| if arg == dst { src } else { arg }, convert::identity);
                        }

                        this_block.gradient_operations.remove(i);
                        continue;
                    }
                }
            }

            i += 1;
        }
    }

    pub fn is_ready(&self) -> bool
    {
        self.state == RecorderState::Ready
    }

    fn calculate_live_ranges_block(&mut self, block_index: BlockIndex) -> bool
    {
        let block = &mut self.operations_blocks[block_index.0];

        block.value_live_ranges = self.global_value_live_ranges.clone();
        block.tensor_live_ranges = self.global_tensor_live_ranges.clone();

        block.block_inputs.iter().for_each(|block_input|
        {
            block.tensor_live_ranges[block_input.0].start = Some(-1);
        });

        block.block_outputs.iter().for_each(|block_output|
        {
            block.tensor_live_ranges[block_output.0].end = Some(i32::MAX);
        });

        block.gradient_operations.iter().enumerate().rev().for_each(|(op_index, op)|
        {
            let handle_output = |live_range: &mut LiveRange, err_name: String|
            {
                let start = &mut live_range.start;

                debug_assert!(start.is_none(), "in {block_index:?}: {err_name} was reused at operation {} and {op_index}", start.unwrap());

                *start = Some(op_index as i32);
            };

            let mut tensor_ptrs = Vec::new();
            op.for_outputs(|tensor_ptr|
            {
                tensor_ptrs.push(tensor_ptr);
            }, |value_index|
            {
                handle_output(&mut block.value_live_ranges[value_index.0], format!("{value_index:?}"));
            });

            tensor_ptrs.into_iter().for_each(|tensor_ptr|
            {
                handle_output(&mut block.tensor_live_ranges[tensor_ptr.0], format!("{tensor_ptr:?}"));
            });

            let handle_arg = |live_range: &mut LiveRange, err_name: String|
            {
                if let Some(start) = live_range.start
                {
                    if start >= op_index as i32
                    {
                        panic!("in {block_index:?}: {err_name} was defined at {start} after being used at {op_index}");
                    }
                }

                if live_range.end.is_none()
                {
                    live_range.end = Some(op_index as i32);
                }
            };

            op.for_args(|tensor_ptr|
            {
                handle_arg(&mut block.tensor_live_ranges[tensor_ptr.0], format!("{tensor_ptr:?}"));
            }, |value_index|
            {
                handle_arg(&mut block.value_live_ranges[value_index.0], format!("{value_index:?}"));
            });
        });

        let mut any_unused = false;
        block.gradient_operations.iter_mut().for_each(|op|
        {
            let mut all_unused: Option<bool> = None;

            let mut handle_output = |live_range: &mut LiveRange, err_name: String|
            {
                let is_unused = live_range.end.is_none();

                debug_assert!(live_range.start != Some(-1), "{err_name} is an unused input");

                if let Some(all_unused) = all_unused.as_mut()
                {
                    *all_unused &= is_unused;
                } else
                {
                    all_unused = Some(is_unused);
                }
            };

            let mut tensor_ptrs = Vec::new();
            op.for_outputs(|tensor_ptr|
            {
                tensor_ptrs.push(tensor_ptr);
            }, |value_index|
            {
                handle_output(&mut block.value_live_ranges[value_index.0], format!("{value_index:?}"));
            });

            tensor_ptrs.into_iter().for_each(|tensor_ptr|
            {
                handle_output(&mut block.tensor_live_ranges[tensor_ptr.0], format!("{tensor_ptr:?}"));
            });

            let is_unused = all_unused.unwrap_or(false);

            if is_unused
            {
                *op = GradientOp::None;
                any_unused = true;
            }
        });

        any_unused
    }

    fn calculate_live_ranges(&mut self)
    {
        (0..self.operations_blocks.len()).map(BlockIndex).for_each(|index|
        {
            loop
            {
                let any_unused = self.calculate_live_ranges_block(index);

                if !any_unused
                {
                    break;
                }
            }
        });
    }

    fn greedy_graph_color_block(&mut self, block: BlockIndex)
    {
        let nodes_count = self.global_tensor_live_ranges.len();

        let mut graph_connections: Vec<Vec<usize>> = iter::from_fn(|| Some(Vec::new()))
            .take(nodes_count)
            .collect();

        let verify_range = |range: &LiveRange, index|
        {
            if let Some(end) = range.end
            {
                if range.start.is_none()
                {
                    panic!("in {block:?}: TensorPtr({index}) was used at OperationIndex({end}) but never set");
                }
            }
        };

        (0..self.operations_blocks.len()).for_each(|check_block_index|
        {
            let this_block = &self.operations_blocks[check_block_index];

            (0..nodes_count).for_each(|node_index|
            {
                let this_range = &this_block.tensor_live_ranges[node_index];

                {
                    verify_range(&this_range, node_index);

                    if this_range.end.is_none()
                    {
                        return;
                    }
                }

                ((node_index + 1)..nodes_count).for_each(|check_index|
                {
                    let is_overlap = {
                        let other_range = &this_block.tensor_live_ranges[check_index];

                        verify_range(&other_range, check_index);

                        if other_range.end.is_none()
                        {
                            return;
                        }

                        this_range.overlaps(other_range)
                    };

                    if is_overlap
                    {
                        graph_connections[node_index].push(check_index);
                        graph_connections[check_index].push(node_index);
                    }
                });
            });
        });

        let mut connections_count_sorted: Vec<usize> = (0..nodes_count).collect();
        connections_count_sorted.sort_unstable_by_key(|node_index| graph_connections[*node_index].len());
        connections_count_sorted.reverse();

        connections_count_sorted.into_iter().for_each(|node_index|
        {
            if self.tensors_memory[node_index].memory.is_some()
            {
                return;
            }

            if self.operations_blocks[block.0].tensor_live_ranges[node_index].end.is_none()
            {
                return;
            }

            let this_color = (0..).find(|color|
            {
                let all_connected_unconflicted = graph_connections[node_index].iter().all(|connected_node_index|
                {
                    let connected_node_color: Option<usize> = self.tensors_memory[*connected_node_index].memory.map(|x| x.0);

                    connected_node_color != Some(*color)
                });

                let spot_size_matches = self.tensors.get(*color).map(|spot_tensor|
                {
                    let this_shape = self.tensors_memory[node_index].value.tensor_shape();

                    spot_tensor.shape() == this_shape
                }).unwrap_or(true);

                all_connected_unconflicted && spot_size_matches
            }).unwrap();

            if this_color == self.tensors.len()
            {
                let tensor = match &self.tensors_memory[node_index].value
                {
                    TensorMemoryValue::Value(x) => x.clone(),
                    TensorMemoryValue::Size{rows, columns} => LayerType::new(*rows, *columns)
                };

                self.tensors.push(tensor);
            } else if let TensorMemoryValue::Value(x) = &self.tensors_memory[node_index].value
            {
                self.tensors[this_color] = x.clone();
            }

            debug_assert!(self.tensors_memory[node_index].memory.is_none(), "in {block:?}: tried to replace slot of TensorPtr({node_index})");
            self.tensors_memory[node_index].memory = Some(TensorIndex(this_color));
        });
    }

    fn greedy_graph_color(&mut self)
    {
        let mut blocks_by_length: Vec<usize> = (0..self.operations_blocks.len()).collect();
        blocks_by_length.sort_by_key(|block_index| self.operations_blocks[*block_index].gradient_operations.len());
        blocks_by_length.reverse();

        for block_index in blocks_by_length
        {
            self.greedy_graph_color_block(BlockIndex(block_index));
        }
    }

    pub fn resolve_memory(&mut self)
    {
        debug_assert_eq!(self.state, RecorderState::AwaitingResolve);

        self.calculate_live_ranges();

        self.greedy_graph_color();

        if OPT_INFO
        {
            eprintln!("using {} memory spots", self.tensors.len());
        }

        self.operations_blocks.iter_mut().enumerate().for_each(|(_block_index, block)|
        {
            block.value_live_ranges.clear();
            block.tensor_live_ranges.clear();

            block.raw_operations = mem::take(&mut block.gradient_operations).into_iter().filter_map(|op| -> Option<GradientOp<TensorIndex>>
            {
                match op
                {
                    GradientOp::None => None,
                    GradientOp::SoftmaxCrossEntropy{
                        values,
                        targets,
                        softmaxed_output,
                        output
                    } if self.tensors_memory[softmaxed_output.0].memory.is_none() =>
                    {
                        Some(GradientOp::SoftmaxCrossEntropyNoSoftmaxed{
                            values: self.tensors_memory[values.0].memory.expect("must be resolved"),
                            targets,
                            output
                        })
                    },
                    x => Some(x.map_tensors(|tensor_ptr| self.tensors_memory[tensor_ptr.0].memory.expect("must be resolved")))
                }
            }).collect();

            #[cfg(debug_assertions)]
            {
                block.raw_operations.iter().for_each(|op|
                {
                    let mut tensor_args = Vec::new();
                    let mut value_args = Vec::new();

                    op.for_args(|t_arg| tensor_args.push(t_arg), |v_arg| value_args.push(v_arg));

                    op.for_outputs(|t_out|
                    {
                        debug_assert!(!tensor_args.contains(&t_out), "in BlockIndex({_block_index}): {op:?} has overlap between args and outputs")
                    }, |v_out|
                    {
                        debug_assert!(!value_args.contains(&v_out), "in BlockIndex({_block_index}): {op:?} has overlap between args and outputs")
                    });
                });
            }

            block.feedforward_operations_count = block.raw_operations.len();
        });

        self.global_value_live_ranges.clear();
        self.global_tensor_live_ranges.clear();

        self.state = RecorderState::Ready;
    }

    pub fn no_gradient(&mut self)
    {
        debug_assert_eq!(self.state, RecorderState::AwaitingGradient);

        let blocks_count = self.operations_blocks.len();
        (0..blocks_count).for_each(|block| self.operations_blocks[block].recording_operations.clear());

        self.state = RecorderState::AwaitingResolve;
    }

    pub fn gradient_with_respect(&mut self, respect: Vec<DiffWrapper>)
    {
        debug_assert_eq!(self.state, RecorderState::AwaitingGradient);

        let blocks_count = self.operations_blocks.len();
        (0..blocks_count).map(BlockIndex).for_each(|block|
        {
            self.set_ones(respect[block.0]);

            self.calculate_gradient(block, respect[block.0]);

            self.combine_same_outputs(block);

            self.copy_coalesce(block);
        });

        (0..blocks_count).for_each(|block| self.operations_blocks[block].recording_operations.clear());

        self.state = RecorderState::AwaitingResolve;
    }

    fn calculate_gradient(&mut self, block: BlockIndex, respect: DiffWrapper)
    {
        let (_this_value, gradient, source): (DiffValue, DiffValue, Option<OperationIndex>) = match respect
        {
            DiffWrapper::Tensor(DiffTensorPtr{index, gradient, source}) =>
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
            let this_operation = self.operations_blocks[source.0.0].recording_operations[source.1].clone();
            let this_gradient_operations = &mut self.operations_blocks[block.0].gradient_operations;

            match this_operation
            {
                Op::Add{lhs, ..}
                | Op::AddScalar{lhs, ..} =>
                {
                    let gradient = gradient.as_tensor();

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::Copy{src: gradient, dst: lhs_gradient});
                    }

                    if let Op::Add{rhs, ..} = this_operation
                    {
                        if let Some(rhs_gradient) = rhs.as_gradient()
                        {
                            this_gradient_operations.push(GradientOp::Copy{src: gradient, dst: rhs_gradient});
                        }
                    } else if let Op::AddScalar{rhs, ..} = this_operation
                    {
                        if let Some(rhs_gradient) = rhs.as_gradient()
                        {
                            this_gradient_operations.push(GradientOp::SumTensor{value: gradient, output: rhs_gradient});
                        }
                    } else
                    {
                        unreachable!()
                    }

                    let rhs = if let Op::Add{rhs, ..} = this_operation
                    {
                        rhs.into()
                    } else if let Op::AddScalar{rhs, ..} = this_operation
                    {
                        rhs.into()
                    } else
                    {
                        unreachable!()
                    };

                    self.calculate_gradient(block, rhs);
                    self.calculate_gradient(block, lhs.into());
                },
                Op::AddScalars{lhs, rhs, output: _} =>
                {
                    let gradient = gradient.as_value();

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::CopyScalar{src: gradient, dst: lhs_gradient});
                    }

                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::CopyScalar{src: gradient, dst: rhs_gradient});
                    }

                    self.calculate_gradient(block, lhs.into());
                    self.calculate_gradient(block, rhs.into());
                },
                Op::Sub{rhs, ..}
                | Op::SubFromScalar{rhs, ..} =>
                {
                    let gradient = gradient.as_tensor();

                    if let Op::Sub{lhs, ..} = this_operation
                    {
                        if let Some(lhs_gradient) = lhs.as_gradient()
                        {
                            this_gradient_operations.push(GradientOp::Copy{src: gradient, dst: lhs_gradient});
                        }
                    } else if let Op::SubFromScalar{lhs, ..} = this_operation
                    {
                        if let Some(lhs_gradient) = lhs.as_gradient()
                        {
                            this_gradient_operations.push(GradientOp::SumTensor{value: gradient, output: lhs_gradient});
                        }
                    } else
                    {
                        unreachable!()
                    }

                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        let m1_index = new_value_index!(self);
                        self.values[m1_index.0] = -1.0;

                        this_gradient_operations.push(GradientOp::MulScalar{lhs: gradient, rhs: m1_index, output: rhs_gradient});
                    }

                    let lhs = if let Op::Sub{lhs, ..} = this_operation
                    {
                        lhs.into()
                    } else if let Op::SubFromScalar{lhs, ..} = this_operation
                    {
                        lhs.into()
                    } else
                    {
                        unreachable!()
                    };

                    self.calculate_gradient(block, lhs);
                    self.calculate_gradient(block, rhs.into());
                },
                Op::MulScalars{lhs, rhs, output: _} =>
                {
                    let gradient = gradient.as_value();

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::MulScalars{lhs: rhs.as_value(), rhs: gradient, output: lhs_gradient});
                    }

                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::MulScalars{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                    }

                    self.calculate_gradient(block, rhs.into());
                    self.calculate_gradient(block, lhs.into());
                },
                Op::MulComponentwise{lhs, ..}
                | Op::MulScalar{lhs, ..} =>
                {
                    let gradient = gradient.as_tensor();

                    let (rows, columns) = tensor_shape!(self, gradient);

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        if let Op::MulComponentwise{rhs, ..} = this_operation
                        {
                            this_gradient_operations.push(GradientOp::MulComponentwise{lhs: rhs.as_value(), rhs: gradient, output: lhs_gradient});
                        } else if let Op::MulScalar{rhs, ..} = this_operation
                        {
                            this_gradient_operations.push(GradientOp::MulScalar{lhs: gradient, rhs: rhs.as_value(), output: lhs_gradient});
                        } else
                        {
                            unreachable!()
                        }
                    }

                    if let Op::MulComponentwise{rhs, ..} = this_operation
                    {
                        if let Some(rhs_gradient) = rhs.as_gradient()
                        {
                            this_gradient_operations.push(GradientOp::MulComponentwise{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                        }
                    } else if let Op::MulScalar{rhs, ..} = this_operation
                    {
                        if let Some(rhs_gradient) = rhs.as_gradient()
                        {
                            let pre_fold = new_tensor_index!(self, rows, columns);
                            this_gradient_operations.push(GradientOp::MulComponentwise{lhs: lhs.as_value(), rhs: gradient, output: pre_fold});

                            this_gradient_operations.push(GradientOp::SumTensor{value: pre_fold, output: rhs_gradient});
                        }
                    } else
                    {
                        unreachable!()
                    }

                    let rhs = if let Op::MulComponentwise{rhs, ..} = this_operation
                    {
                        rhs.into()
                    } else if let Op::MulScalar{rhs, ..} = this_operation
                    {
                        rhs.into()
                    } else
                    {
                        unreachable!()
                    };

                    self.calculate_gradient(block, rhs);
                    self.calculate_gradient(block, lhs.into());
                },
                Op::SumTensor{value, output: _} =>
                {
                    let gradient = gradient.as_value();

                    if let Some(value_gradient) = value.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::Fill{value: gradient, output: value_gradient});

                        self.calculate_gradient(block, value.into());
                    }
                },
                Op::Dot{lhs, rhs, output: _} =>
                {
                    let gradient = gradient.as_value();

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::MulScalar{lhs: rhs.as_value(), rhs: gradient, output: lhs_gradient});
                    }

                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::MulScalar{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                    }

                    self.calculate_gradient(block, rhs.into());
                    self.calculate_gradient(block, lhs.into());
                },
                Op::Pow{lhs, power, output: _} =>
                {
                    let gradient = gradient.as_tensor();

                    let (rows, columns) = tensor_shape!(self, lhs.as_value());

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        let power_index = new_value_index!(self);
                        self.values[power_index.0] = power as f32;

                        let pow_d_lhs = new_tensor_index!(self, rows, columns);
                        this_gradient_operations.push(GradientOp::Pow{lhs: lhs.as_value(), power: (power - 1) as u32, output: pow_d_lhs});

                        let pow_d = new_tensor_index!(self, rows, columns);
                        this_gradient_operations.push(GradientOp::MulScalar{lhs: pow_d_lhs, rhs: power_index.into(), output: pow_d});

                        this_gradient_operations.push(GradientOp::MulComponentwise{lhs: pow_d, rhs: gradient, output: lhs_gradient});

                        self.calculate_gradient(block, lhs.into());
                    }
                },
                Op::Sigmoid{value, output} =>
                {
                    // sigmoid(x) * (1.0 - sigmoid(x))
                    let gradient = gradient.as_tensor();

                    if let Some(value_gradient) = value.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::SigmoidDiff{value: output.as_value(), gradient, output: value_gradient});

                        self.calculate_gradient(block, value.into());
                    }
                },
                Op::Tanh{value, output} =>
                {
                    // 1 - tanh^2(x)
                    let gradient = gradient.as_tensor();

                    if let Some(value_gradient) = value.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::TanhDiff{value: output.as_value(), gradient, output: value_gradient});

                        self.calculate_gradient(block, value.into());
                    }
                },
                Op::LeakyRelu{value, output: _} =>
                {
                    let gradient = gradient.as_tensor();

                    if let Some(value_gradient) = value.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::LeakyReluDiff{value: value.as_value(), gradient, output: value_gradient});

                        self.calculate_gradient(block, value.into());
                    }
                },
                Op::SoftmaxCrossEntropy{values, targets, softmaxed_output, output: _} =>
                {
                    let gradient = gradient.as_value();

                    if let Some(values_gradient) = values.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::SoftmaxCrossEntropyDiff{
                            softmaxed_values: softmaxed_output.as_value(),
                            gradient,
                            targets: targets.clone(),
                            output: values_gradient
                        });

                        self.calculate_gradient(block, values.into());
                    }
                },
                Op::Matmulv{lhs, rhs, output: _} =>
                {
                    let gradient = gradient.as_tensor();

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::OuterProduct{lhs: gradient, rhs: rhs.as_value(), output: lhs_gradient});
                    }

                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::MatmulvTransposed{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                    }

                    self.calculate_gradient(block, lhs.into());
                    self.calculate_gradient(block, rhs.into());
                },
                Op::MatmulvAdd{lhs, rhs, added, output: _} =>
                {
                    let gradient = gradient.as_tensor();

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::OuterProduct{lhs: gradient, rhs: rhs.as_value(), output: lhs_gradient});
                    }

                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::MatmulvTransposed{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                    }

                    if let Some(added_gradient) = added.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::Copy{src: gradient, dst: added_gradient});
                    }

                    self.calculate_gradient(block, lhs.into());
                    self.calculate_gradient(block, rhs.into());
                    self.calculate_gradient(block, added.into());
                },
                Op::MatmulOneHotvAdd{lhs, rhs, added, output: _} =>
                {
                    let gradient = gradient.as_tensor();

                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::OuterProductOneHot{lhs: gradient, rhs: rhs, output: lhs_gradient});
                    }

                    if let Some(added_gradient) = added.as_gradient()
                    {
                        this_gradient_operations.push(GradientOp::Copy{src: gradient, dst: added_gradient});
                    }

                    self.calculate_gradient(block, lhs.into());
                    self.calculate_gradient(block, added.into());
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OneHotIndex(usize);

#[allow(dead_code)]
impl OneHotIndex
{
    pub fn undefined() -> Self { Self(usize::MAX) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorPtr(usize);

impl TensorPtr
{
    pub fn undefined() -> Self { Self(usize::MAX) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorIndex(usize);

impl TensorIndex
{
    pub fn undefined() -> Self { Self(usize::MAX) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueIndex(usize);

#[allow(dead_code)]
impl ValueIndex
{
    pub fn undefined() -> Self { Self(usize::MAX) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffValueRaw
{
    Tensor(TensorIndex),
    Value(ValueIndex)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffValue
{
    Tensor(TensorPtr),
    Value(ValueIndex)
}

impl From<TensorPtr> for DiffValue
{
    fn from(index: TensorPtr) -> Self
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
            DiffWrapper::Tensor(DiffTensorPtr{index, ..}) => Self::Tensor(index),
            DiffWrapper::Value(DiffScalar{index, ..}) => Self::Value(index)
        }
    }
}

impl DiffValue
{
    fn as_tensor(self) -> TensorPtr
    {
        if let Self::Tensor(x) = self { x } else { panic!("as_tensor must be called on a tensor") }
    }

    fn as_value(self) -> ValueIndex
    {
        if let Self::Value(x) = self { x } else { panic!("as_value must be called on a value") }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct OperationIndex(BlockIndex, usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffTensorPtr
{
    index: TensorPtr,
    gradient: Option<TensorPtr>,
    source: Option<OperationIndex>
}

#[allow(dead_code)]
impl DiffTensorPtr
{
    pub fn no_gradient(index: TensorPtr) -> Self
    {
        Self{
            index,
            gradient: None,
            source: None
        }
    }

    pub fn undefined() -> Self
    {
        Self{
            index: TensorPtr::undefined(),
            gradient: None,
            source: None
        }
    }

    pub fn as_value(&self) -> TensorPtr
    {
        debug_assert_ne!(self.index, TensorPtr::undefined());

        self.index
    }

    pub fn as_gradient(&self) -> Option<TensorPtr>
    {
        self.gradient
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffTensor
{
    index: TensorIndex,
    gradient: Option<TensorIndex>
}

#[allow(dead_code)]
impl DiffTensor
{
    pub fn no_gradient(index: TensorIndex) -> Self
    {
        Self{
            index,
            gradient: None
        }
    }

    pub fn undefined() -> Self
    {
        Self{
            index: TensorIndex::undefined(),
            gradient: None
        }
    }

    pub fn as_value(&self) -> TensorIndex
    {
        debug_assert_ne!(self.index, TensorIndex::undefined());

        self.index
    }

    pub fn as_gradient(&self) -> Option<TensorIndex>
    {
        self.gradient
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffScalar
{
    index: ValueIndex,
    gradient: Option<ValueIndex>,
    source: Option<OperationIndex>
}

#[allow(dead_code)]
impl DiffScalar
{
    pub fn undefined() -> Self
    {
        Self{
            index: ValueIndex(usize::MAX),
            gradient: None,
            source: None
        }
    }

    pub fn as_value(&self) -> ValueIndex
    {
        debug_assert_ne!(self.index, ValueIndex::undefined());

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
    Tensor(DiffTensorPtr),
    Value(DiffScalar)
}

impl From<DiffTensorPtr> for DiffWrapper
{
    fn from(value: DiffTensorPtr) -> Self
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OwnedDiffValue
{
    Tensor(LayerType),
    Value(f32)
}

impl From<LayerType> for OwnedDiffValue
{
    fn from(x: LayerType) -> Self
    {
        Self::Tensor(x)
    }
}

impl From<f32> for OwnedDiffValue
{
    fn from(x: f32) -> Self
    {
        Self::Value(x)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientOp<T>
{
    None,
    Copy{src: T, dst: T},
    CopyScalar{src: ValueIndex, dst: ValueIndex},
    AddScalar{lhs: T, rhs: ValueIndex, output: T},
    AddScalars{lhs: ValueIndex, rhs: ValueIndex, output: ValueIndex},
    Add{lhs: T, rhs: T, output: T},
    AddInplace{value: T, output: T},
    Sub{lhs: T, rhs: T, output: T},
    SubFromScalar{lhs: ValueIndex, rhs: T, output: T},
    MulScalar{lhs: T, rhs: ValueIndex, output: T},
    MulScalars{lhs: ValueIndex, rhs: ValueIndex, output: ValueIndex},
    MulComponentwise{lhs: T, rhs: T, output: T},
    SumTensor{value: T, output: ValueIndex},
    Fill{value: ValueIndex, output: T},
    Pow{lhs: T, power: u32, output: T},
    LeakyRelu{value: T, output: T},
    LeakyReluDiff{value: T, gradient: T, output: T},
    Sigmoid{value: T, output: T},
    SigmoidDiff{value: T, gradient: T, output: T},
    Tanh{value: T, output: T},
    TanhDiff{value: T, gradient: T, output: T},
    Dot{lhs: T, rhs: T, output: ValueIndex},
    SoftmaxCrossEntropy{values: T, targets: OneHotIndex, softmaxed_output: T, output: ValueIndex},
    SoftmaxCrossEntropyNoSoftmaxed{values: T, targets: OneHotIndex, output: ValueIndex},
    SoftmaxCrossEntropyDiff{softmaxed_values: T, gradient: ValueIndex, targets: OneHotIndex, output: T},
    Matmulv{lhs: T, rhs: T, output: T},
    MatmulvAdd{lhs: T, rhs: T, added: T, output: T},
    MatmulOneHotvAdd{lhs: T, rhs: OneHotIndex, added: T, output: T},
    MatmulvTransposed{lhs: T, rhs: T, output: T},
    OuterProduct{lhs: T, rhs: T, output: T},
    OuterProductOneHot{lhs: T, rhs: OneHotIndex, output: T}
}

impl<T> GradientOp<T>
{
    fn map_tensors<U>(self, mut f: impl FnMut(T) -> U) -> GradientOp<U>
    {
        match self
        {
            Self::None => GradientOp::None,
            Self::Copy{src, dst} => GradientOp::Copy{src: f(src), dst: f(dst)},
            Self::CopyScalar{src, dst} => GradientOp::CopyScalar{src, dst},
            Self::AddScalar{lhs, rhs, output} => GradientOp::AddScalar{lhs: f(lhs), rhs, output: f(output)},
            Self::AddScalars{lhs, rhs, output} => GradientOp::AddScalars{lhs, rhs, output},
            Self::Add{lhs, rhs, output} => GradientOp::Add{lhs: f(lhs), rhs: f(rhs), output: f(output)},
            Self::Sub{lhs, rhs, output} => GradientOp::Sub{lhs: f(lhs), rhs: f(rhs), output: f(output)},
            Self::SubFromScalar{lhs, rhs, output} => GradientOp::SubFromScalar{lhs, rhs: f(rhs), output: f(output)},
            Self::MulScalar{lhs, rhs, output} => GradientOp::MulScalar{lhs: f(lhs), rhs, output: f(output)},
            Self::MulScalars{lhs, rhs, output} => GradientOp::MulScalars{lhs, rhs, output},
            Self::MulComponentwise{lhs, rhs, output} => GradientOp::MulComponentwise{lhs: f(lhs), rhs: f(rhs), output: f(output)},
            Self::SumTensor{value, output} => GradientOp::SumTensor{value: f(value), output},
            Self::Fill{value, output} => GradientOp::Fill{value, output: f(output)},
            Self::Pow{lhs, power, output} => GradientOp::Pow{lhs: f(lhs), power, output: f(output)},
            Self::LeakyRelu{value, output} => GradientOp::LeakyRelu{value: f(value), output: f(output)},
            Self::LeakyReluDiff{value, gradient, output} => GradientOp::LeakyReluDiff{value: f(value), gradient: f(gradient), output: f(output)},
            Self::Sigmoid{value, output} => GradientOp::Sigmoid{value: f(value), output: f(output)},
            Self::SigmoidDiff{value, gradient, output} => GradientOp::SigmoidDiff{value: f(value), gradient: f(gradient), output: f(output)},
            Self::Tanh{value, output} => GradientOp::Tanh{value: f(value), output: f(output)},
            Self::TanhDiff{value, gradient, output} => GradientOp::TanhDiff{value: f(value), gradient: f(gradient), output: f(output)},
            Self::Dot{lhs, rhs, output} => GradientOp::Dot{lhs: f(lhs), rhs: f(rhs), output},
            Self::SoftmaxCrossEntropy{values, targets, softmaxed_output, output} =>
            {
                GradientOp::SoftmaxCrossEntropy{values: f(values), targets, softmaxed_output: f(softmaxed_output), output}
            },
            Self::SoftmaxCrossEntropyDiff{softmaxed_values, gradient, targets, output} =>
            {
                GradientOp::SoftmaxCrossEntropyDiff{softmaxed_values: f(softmaxed_values), gradient, targets, output: f(output)}
            },
            Self::Matmulv{lhs, rhs, output} => GradientOp::Matmulv{lhs: f(lhs), rhs: f(rhs), output: f(output)},
            Self::MatmulvAdd{lhs, rhs, added, output} => GradientOp::MatmulvAdd{lhs: f(lhs), rhs: f(rhs), added: f(added), output: f(output)},
            Self::MatmulOneHotvAdd{lhs, rhs, added, output} =>
            {
                GradientOp::MatmulOneHotvAdd{lhs: f(lhs), rhs, added: f(added), output: f(output)}
            },
            Self::MatmulvTransposed{lhs, rhs, output} => GradientOp::MatmulvTransposed{lhs: f(lhs), rhs: f(rhs), output: f(output)},
            Self::OuterProduct{lhs, rhs, output} => GradientOp::OuterProduct{lhs: f(lhs), rhs: f(rhs), output: f(output)},
            Self::OuterProductOneHot{lhs, rhs, output} => GradientOp::OuterProductOneHot{lhs: f(lhs), rhs, output: f(output)},
            Self::AddInplace{..}
            | Self::SoftmaxCrossEntropyNoSoftmaxed{..} => unreachable!()
        }
    }
}

impl<T: Clone> GradientOp<T>
{
    fn for_outputs(&self, mut tf: impl FnMut(T), mut vf: impl FnMut(ValueIndex))
    {
        self.clone().map_outputs(|x| { tf(x.clone()); x }, |x| { vf(x.clone()); x });
    }

    fn map_outputs(self, mut tf: impl FnMut(T) -> T, mut vf: impl FnMut(ValueIndex) -> ValueIndex) -> Self
    {
        match self
        {
            Self::Copy{dst, src} => Self::Copy{dst: tf(dst), src},
            Self::AddScalar{output, lhs, rhs} => Self::AddScalar{output: tf(output), lhs, rhs},
            Self::Add{output, lhs, rhs} => Self::Add{output: tf(output), lhs, rhs},
            Self::Sub{output, lhs, rhs} => Self::Sub{output: tf(output), lhs, rhs},
            Self::SubFromScalar{output, lhs, rhs} => Self::SubFromScalar{output: tf(output), lhs, rhs},
            Self::MulScalar{output, lhs, rhs} => Self::MulScalar{output: tf(output), lhs, rhs},
            Self::MulComponentwise{output, lhs, rhs} => Self::MulComponentwise{output: tf(output), lhs, rhs},
            Self::Fill{output, value} => Self::Fill{output: tf(output), value},
            Self::Pow{output, power, lhs} => Self::Pow{output: tf(output), power, lhs},
            Self::Sigmoid{output, value} => Self::Sigmoid{output: tf(output), value},
            Self::SigmoidDiff{output, gradient, value} => Self::SigmoidDiff{output: tf(output), gradient, value},
            Self::Tanh{output, value} => Self::Tanh{output: tf(output), value},
            Self::TanhDiff{output, gradient, value} => Self::TanhDiff{output: tf(output), gradient, value},
            Self::LeakyRelu{output, value} => Self::LeakyRelu{output: tf(output), value},
            Self::LeakyReluDiff{output, gradient, value} => Self::LeakyReluDiff{output: tf(output), gradient, value},
            Self::SoftmaxCrossEntropyDiff{output, softmaxed_values, gradient, targets} =>
            {
                Self::SoftmaxCrossEntropyDiff{output: tf(output), softmaxed_values, gradient, targets}
            },
            Self::Matmulv{output, lhs, rhs} => Self::Matmulv{output: tf(output), lhs, rhs},
            Self::MatmulvAdd{output, lhs, rhs, added} => Self::MatmulvAdd{output: tf(output), lhs, rhs, added},
            Self::MatmulOneHotvAdd{output, lhs, rhs, added} => Self::MatmulOneHotvAdd{output: tf(output), lhs, rhs, added},
            Self::MatmulvTransposed{output, lhs, rhs} => Self::MatmulvTransposed{output: tf(output), lhs, rhs},
            Self::OuterProduct{output, lhs, rhs} => Self::OuterProduct{output: tf(output), lhs, rhs},
            Self::OuterProductOneHot{output, lhs, rhs} => Self::OuterProductOneHot{output: tf(output), lhs, rhs},
            Self::CopyScalar{dst, src} => Self::CopyScalar{dst: vf(dst), src},
            Self::AddScalars{output, lhs, rhs} => Self::AddScalars{output: vf(output), lhs, rhs},
            Self::MulScalars{output, lhs, rhs} => Self::MulScalars{output: vf(output), lhs, rhs},
            Self::SumTensor{output, value} => Self::SumTensor{output: vf(output), value},
            Self::Dot{output, lhs, rhs} => Self::Dot{output: vf(output), lhs, rhs},
            Self::SoftmaxCrossEntropy{softmaxed_output, output, targets, values} =>
            {
                Self::SoftmaxCrossEntropy{softmaxed_output: tf(softmaxed_output), output: vf(output), targets, values}
            },
            Self::SoftmaxCrossEntropyNoSoftmaxed{output, targets, values} =>
            {
                Self::SoftmaxCrossEntropyNoSoftmaxed{output: vf(output), targets, values}
            },
            Self::None => Self::None,
            Self::AddInplace{..} => unreachable!()
        }
    }

    fn for_args(&self, mut tf: impl FnMut(T), mut vf: impl FnMut(ValueIndex))
    {
        self.clone().map_args(|x| { tf(x.clone()); x }, |x| { vf(x.clone()); x });
    }

    fn map_args(self, mut tf: impl FnMut(T) -> T, mut vf: impl FnMut(ValueIndex) -> ValueIndex) -> Self
    {
        match self
        {
            Self::Copy{src, dst} => Self::Copy{src: tf(src), dst},
            Self::AddScalar{lhs, rhs, output} => Self::AddScalar{lhs: tf(lhs), rhs: vf(rhs), output},
            Self::Add{lhs, rhs, output} => Self::Add{lhs: tf(lhs), rhs: tf(rhs), output},
            Self::Sub{lhs, rhs, output} => Self::Sub{lhs: tf(lhs), rhs: tf(rhs), output},
            Self::SubFromScalar{lhs, rhs, output} => Self::SubFromScalar{rhs: tf(rhs), lhs: vf(lhs), output},
            Self::MulScalar{lhs, rhs, output} => Self::MulScalar{lhs: tf(lhs), rhs: vf(rhs), output},
            Self::MulComponentwise{lhs, rhs, output} => Self::MulComponentwise{lhs: tf(lhs), rhs: tf(rhs), output},
            Self::SumTensor{value, output} => Self::SumTensor{value: tf(value), output},
            Self::Pow{lhs, power, output} => Self::Pow{lhs: tf(lhs), power, output},
            Self::LeakyRelu{value, output} => Self::LeakyRelu{value: tf(value), output},
            Self::LeakyReluDiff{value, gradient, output} => Self::LeakyReluDiff{value: tf(value), gradient: tf(gradient), output},
            Self::Sigmoid{value, output} => Self::Sigmoid{value: tf(value), output},
            Self::SigmoidDiff{value, gradient, output} => Self::SigmoidDiff{value: tf(value), gradient: tf(gradient), output},
            Self::Tanh{value, output} => Self::Tanh{value: tf(value), output},
            Self::TanhDiff{value, gradient, output} => Self::TanhDiff{value: tf(value), gradient: tf(gradient), output},
            Self::Dot{lhs, rhs, output} => Self::Dot{lhs: tf(lhs), rhs: tf(rhs), output},
            Self::SoftmaxCrossEntropy{values, targets, softmaxed_output, output} =>
            {
                Self::SoftmaxCrossEntropy{values: tf(values), targets, softmaxed_output, output}
            },
            Self::SoftmaxCrossEntropyDiff{softmaxed_values, gradient, targets, output} =>
            {
                Self::SoftmaxCrossEntropyDiff{softmaxed_values: tf(softmaxed_values), gradient: vf(gradient), targets, output}
            },
            Self::Matmulv{lhs, rhs, output} => Self::Matmulv{lhs: tf(lhs), rhs: tf(rhs), output},
            Self::MatmulvAdd{lhs, rhs, added, output} => Self::MatmulvAdd{lhs: tf(lhs), rhs: tf(rhs), added: tf(added), output},
            Self::MatmulOneHotvAdd{lhs, rhs, added, output} => Self::MatmulOneHotvAdd{lhs: tf(lhs), rhs, added: tf(added), output},
            Self::MatmulvTransposed{lhs, rhs, output} => Self::MatmulvTransposed{lhs: tf(lhs), rhs: tf(rhs), output},
            Self::OuterProduct{lhs, rhs, output} => Self::OuterProduct{lhs: tf(lhs), rhs: tf(rhs), output},
            Self::OuterProductOneHot{lhs, rhs, output} => Self::OuterProductOneHot{lhs: tf(lhs), rhs, output},
            Self::CopyScalar{src, dst} => Self::CopyScalar{src: vf(src), dst},
            Self::AddScalars{lhs, rhs, output} => Self::AddScalars{lhs: vf(lhs), rhs: vf(rhs), output},
            Self::MulScalars{lhs, rhs, output} => Self::MulScalars{lhs: vf(lhs), rhs: vf(rhs), output},
            Self::Fill{value, output} => Self::Fill{value: vf(value), output},
            Self::SoftmaxCrossEntropyNoSoftmaxed{values, targets, output} =>
            {
                Self::SoftmaxCrossEntropyNoSoftmaxed{values: tf(values), targets, output}
            },
            Self::None => Self::None,
            Self::AddInplace{..} => unreachable!()
        }
    }
}

impl GradientOp<TensorPtr>
{
    fn diff_output_of(&self) -> DiffValue
    {
        match *self
        {
            Self::Copy{dst: output, ..}
            | Self::AddScalar{output, ..}
            | Self::Add{output, ..}
            | Self::Sub{output, ..}
            | Self::SubFromScalar{output, ..}
            | Self::MulScalar{output, ..}
            | Self::MulComponentwise{output, ..}
            | Self::Fill{output, ..}
            | Self::Pow{output, ..}
            | Self::Sigmoid{output, ..}
            | Self::SigmoidDiff{output, ..}
            | Self::Tanh{output, ..}
            | Self::TanhDiff{output, ..}
            | Self::LeakyRelu{output, ..}
            | Self::LeakyReluDiff{output, ..}
            | Self::SoftmaxCrossEntropyDiff{output, ..}
            | Self::Matmulv{output, ..}
            | Self::MatmulvAdd{output, ..}
            | Self::MatmulOneHotvAdd{output, ..}
            | Self::MatmulvTransposed{output, ..}
            | Self::OuterProduct{output, ..}
            | Self::OuterProductOneHot{output, ..} => output.into(),
            Self::CopyScalar{dst: output, ..}
            | Self::AddScalars{output, ..}
            | Self::MulScalars{output, ..}
            | Self::SumTensor{output, ..}
            | Self::Dot{output, ..}
            | Self::SoftmaxCrossEntropy{output, ..} => output.into(),
            Self::None
            | Self::AddInplace{..}
            | Self::SoftmaxCrossEntropyNoSoftmaxed{..} => unreachable!()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Op
{
    AddScalar{lhs: DiffTensorPtr, rhs: DiffScalar, output: DiffTensorPtr},
    AddScalars{lhs: DiffScalar, rhs: DiffScalar, output: DiffScalar},
    Add{lhs: DiffTensorPtr, rhs: DiffTensorPtr, output: DiffTensorPtr},
    Sub{lhs: DiffTensorPtr, rhs: DiffTensorPtr, output: DiffTensorPtr},
    SubFromScalar{lhs: DiffScalar, rhs: DiffTensorPtr, output: DiffTensorPtr},
    MulScalar{lhs: DiffTensorPtr, rhs: DiffScalar, output: DiffTensorPtr},
    MulScalars{lhs: DiffScalar, rhs: DiffScalar, output: DiffScalar},
    MulComponentwise{lhs: DiffTensorPtr, rhs: DiffTensorPtr, output: DiffTensorPtr},
    SumTensor{value: DiffTensorPtr, output: DiffScalar},
    Pow{lhs: DiffTensorPtr, power: i32, output: DiffTensorPtr},
    LeakyRelu{value: DiffTensorPtr, output: DiffTensorPtr},
    Sigmoid{value: DiffTensorPtr, output: DiffTensorPtr},
    Tanh{value: DiffTensorPtr, output: DiffTensorPtr},
    Dot{lhs: DiffTensorPtr, rhs: DiffTensorPtr, output: DiffScalar},
    SoftmaxCrossEntropy{values: DiffTensorPtr, targets: OneHotIndex, softmaxed_output: DiffTensorPtr, output: DiffScalar},
    Matmulv{lhs: DiffTensorPtr, rhs: DiffTensorPtr, output: DiffTensorPtr},
    MatmulvAdd{lhs: DiffTensorPtr, rhs: DiffTensorPtr, added: DiffTensorPtr, output: DiffTensorPtr},
    MatmulOneHotvAdd{lhs: DiffTensorPtr, rhs: OneHotIndex, added: DiffTensorPtr, output: DiffTensorPtr}
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
        let this = Self{positions: positions.into(), size};

        debug_assert!(
        {
            let s: HashSet<_> = this.positions.iter().collect();

            s.len() == this.positions.len()
        }, "positions must be unique: {:?}", this.positions.iter().collect::<Vec<_>>());

        this
    }

    pub fn empty() -> Self
    {
        Self::new([], 0)
    }

    pub fn into_layer(self) -> LayerType
    {
        let size = self.size;
        let mut layer = vec![0.0; size];

        for position in self.positions.iter()
        {
            layer[*position] = 1.0;
        }

        LayerType::from_raw(layer, size, 1)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum InputTypePtr
{
    Normal(TensorPtr),
    OneHot(OneHotIndex)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum InputType
{
    Normal(TensorIndex),
    OneHot(OneHotIndex)
}

#[allow(dead_code)]
impl InputType
{
    pub fn undefined() -> Self { Self::Normal(TensorIndex::undefined()) }
}

impl From<TensorIndex> for InputType
{
    fn from(value: TensorIndex) -> Self
    {
        Self::Normal(value)
    }
}

impl From<OneHotIndex> for InputType
{
    fn from(value: OneHotIndex) -> Self
    {
        Self::OneHot(value)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum DiffInputType
{
    Normal(DiffTensorPtr),
    OneHot(OneHotIndex)
}

#[allow(dead_code)]
impl DiffInputType
{
    pub fn into_one_hot(self) -> OneHotIndex
    {
        match self
        {
            Self::OneHot(value) => value,
            _ => panic!("expected onehot")
        }
    }
}

#[derive(Debug, Clone)]
pub enum OwnedInputType
{
    Normal(LayerType),
    OneHot(OneHotLayer)
}

#[allow(dead_code)]
impl OwnedInputType
{
    pub fn into_one_hot(self) -> OneHotLayer
    {
        match self
        {
            Self::OneHot(value) => value,
            _ => panic!("expected onehot")
        }
    }

    pub fn into_normal(self) -> LayerType
    {
        match self
        {
            Self::Normal(value) => value,
            _ => panic!("expected normal")
        }
    }
}

impl From<LayerType> for OwnedInputType
{
    fn from(value: LayerType) -> Self
    {
        Self::Normal(value)
    }
}

impl From<OneHotLayer> for OwnedInputType
{
    fn from(value: OneHotLayer) -> Self
    {
        Self::OneHot(value)
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
        f: impl FnMut(&mut OperationsRecorder, DiffTensorPtr, DiffTensorPtr) -> DiffTensorPtr
    )
    {
        let mut recorder = OperationsRecorder::new();

        let a = random_tensor(&mut recorder, a_dims.0, a_dims.1);
        let b = random_tensor(&mut recorder, b_dims.0, b_dims.1);

        check_tensor_inner(&mut recorder, a, b, f);
    }

    fn check_vector(f: impl FnMut(&mut OperationsRecorder, DiffTensorPtr, DiffTensorPtr) -> DiffTensorPtr)
    {
        let mut recorder = OperationsRecorder::new();

        let a = random_tensor(&mut recorder, 1, LAYER_CURR);
        let b = random_tensor(&mut recorder, 1, LAYER_CURR);

        check_tensor_inner(&mut recorder, a, b, f);
    }

    fn check_tensor(f: impl FnMut(&mut OperationsRecorder, DiffTensorPtr, DiffTensorPtr) -> DiffTensorPtr)
    {
        let mut recorder = OperationsRecorder::new();

        let a = random_tensor(&mut recorder, LAYER_PREV, LAYER_CURR);
        let b = random_tensor(&mut recorder, LAYER_PREV, LAYER_CURR);

        check_tensor_inner(&mut recorder, a, b, f);
    }

    fn check_tensor_inner(
        recorder: &mut OperationsRecorder,
        (a_value, a): (LayerType, DiffTensorPtr),
        (b_value, b): (LayerType, DiffTensorPtr),
        mut f: impl FnMut(&mut OperationsRecorder, DiffTensorPtr, DiffTensorPtr) -> DiffTensorPtr
    )
    {
        let out = f(recorder, a, b);

        let a_gradient = a.as_gradient().unwrap();
        let b_gradient = b.as_gradient().unwrap();

        recorder.finish();

        recorder.store_tensor_until_end(a_gradient);
        recorder.store_tensor_until_end(b_gradient);

        recorder.gradient_with_respect(vec![out.into()]);

        recorder.resolve_memory();

        let a_gradient = recorder.resolve_tensor_ptr(a_gradient);
        let b_gradient = recorder.resolve_tensor_ptr(b_gradient);

        let current_block = recorder.current_block();
        recorder.calculate(current_block);

        let a_g = recorder.get_tensor(a_gradient).clone();
        let b_g = recorder.get_tensor(b_gradient).clone();

        let mut vals = |a: LayerType, b: LayerType|
        {
            let mut new_recorder = OperationsRecorder::new();

            let new_a = new_recorder.set_new_tensor(a);
            let new_b = new_recorder.set_new_tensor(b);

            let output = f(&mut new_recorder, new_a, new_b);

            let output_value = output.as_value();

            new_recorder.store_tensor_until_end(output_value);

            new_recorder.finish();
            new_recorder.gradient_with_respect(vec![output.into()]);

            new_recorder.resolve_memory();

            let output_value = new_recorder.resolve_tensor_ptr(output_value);

            let current_block = new_recorder.current_block();
            new_recorder.calculate(current_block);

            new_recorder.get_tensor(output_value).clone()
        };

        let orig = vals(a_value.clone(), b_value.clone()).sum();

        let epsilon: f32 = 0.009;

        let fg = |value: LayerType|
        {
            let value = value.sum();

            (value - orig) / epsilon
        };

        let mut a_fg = vec![0.0; a_value.total_len()];
        for index in 0..a_fg.len()
        {
            let v = a_value.clone();
            let epsilon = one_hot(v.clone(), index, epsilon, 0.0);

            let this_fg = fg(vals(v.clone() + epsilon, b_value.clone()));

            a_fg[index] = this_fg;
        }

        let mut b_fg = vec![0.0; b_value.total_len()];
        for index in 0..b_fg.len()
        {
            let v = b_value.clone();
            let epsilon = one_hot(v.clone(), index, epsilon, 0.0);

            let this_fg = fg(vals(a_value.clone(), v.clone() + epsilon));

            b_fg[index] = this_fg;
        }

        let vec_to_layer = |v, mut layer: LayerType|
        {
            layer.swap_raw_values(v);

            layer
        };

        let a_fg = vec_to_layer(a_fg, a_value);
        let b_fg = vec_to_layer(b_fg, b_value);

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

    fn random_tensor(recorder: &mut OperationsRecorder, columns: usize, rows: usize) -> (LayerType, DiffTensorPtr)
    {
        let value = LayerType::new_with(rows, columns, random_value);

        (value.clone(), recorder.set_new_tensor_gradientable(value))
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
            let bb = recorder.pow(b, 2);

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

    #[test]
    fn dot_product()
    {
        check_vector(|recorder, a, b|
        {
            let a_dot_b = recorder.dot(a, b);
            recorder.add_scalar(a, a_dot_b)
        })
    }

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

    #[test]
    fn matrix_multiplication_easy()
    {
        check_tensor_with_dims((4, 2), (1, 4), |recorder, a, b|
        {
            let s = recorder.sum_tensor(b);
            let mm = recorder.matmulv(a, b);

            recorder.add_scalar(mm, s)
        })
    }

    #[test]
    fn matrix_multiplication_more()
    {
        check_tensor_with_dims((4, 2), (1, 4), |recorder, a, b|
        {
            let s = recorder.sum_tensor(b);
            let mm = recorder.matmulv(a, b);
            let left = recorder.add_scalar(mm, s);

            let k = recorder.matmulv(a, b);
            let l = recorder.matmulv(a, b);

            let right = recorder.add(k, l);

            recorder.mul_componentwise(left, right)
        })
    }

    #[test]
    fn matrix_multiplication_lots()
    {
        check_tensor_with_dims((4, 2), (1, 4), |recorder, a, b|
        {
            let s = recorder.sum_tensor(b);
            let mm = recorder.matmulv(a, b);
            let left = recorder.add_scalar(mm, s);

            let k = recorder.matmulv(a, b);
            let right = recorder.matmulv_add(a, b, k);

            recorder.mul_componentwise(left, right)
        })
    }

    fn create_targets() -> OneHotLayer
    {
        create_targets_with_size(LAYER_CURR)
    }

    fn create_targets_with_size(size: usize) -> OneHotLayer
    {
        let pos = fastrand::usize(0..size);

        OneHotLayer::new([pos], size)
    }

    #[test]
    fn matrix_multiplication_one_hot()
    {
        let a_columns = 4;
        let targets = create_targets_with_size(a_columns);
        check_tensor_with_dims((a_columns, 2), (1, a_columns), |recorder, a, b|
        {
            let targets_index = recorder.new_one_hot();
            recorder.set_one_hot(targets_index, targets.clone());

            let s = recorder.sum_tensor(b);
            let mm = recorder.matmulv(a, b);
            let left = recorder.add_scalar(mm, s);

            let k = recorder.matmulv(a, b);
            let right = recorder.matmul_onehotv_add(a, targets_index, k);

            recorder.mul_componentwise(left, right)
        })
    }

    #[test]
    fn softmax_cross_entropy_easy()
    {
        let targets = create_targets();
        check_vector(|recorder, a, b|
        {
            let targets_index = recorder.new_one_hot();
            recorder.set_one_hot(targets_index, targets.clone());

            let sm = recorder.softmax_cross_entropy(a, targets_index).1;
            recorder.add_scalar(b, sm)
        })
    }

    #[test]
    fn softmax_cross_entropy_complicated()
    {
        let targets = create_targets();
        check_vector(|recorder, a, b|
        {
            let targets_index = recorder.new_one_hot();
            recorder.set_one_hot(targets_index, targets.clone());

            let two = recorder.set_new_value(2.0);
            let btwo = recorder.add_scalar(b, two);

            let sm = recorder.softmax_cross_entropy(btwo, targets_index).1;

            recorder.add_scalar(a, sm)
        })
    }
}
