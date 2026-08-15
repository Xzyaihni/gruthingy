use std::{
    f32,
    mem,
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum RecorderState
{
    Recording,
    AwaitingGradient,
    AwaitingResolve,
    Ready
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockIndex(usize);

impl BlockIndex
{
    pub fn undefined() -> Self { Self(usize::MAX) }
}

struct ForceNoPretty<'a, T>(&'a T);
impl<'a, T: Debug> Debug for ForceNoPretty<'a, T>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        write!(f, "{:?}", self.0)
    }
}

#[derive(Debug, Clone)]
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

        start <= end
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
    live_ranges: Vec<LiveRange>,
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
            .field("live_ranges", &self.live_ranges.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
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
            live_ranges: Vec::new(),
            recording_operations: Vec::new(),
            gradient_operations: Vec::new(),
            raw_operations: Vec::new(),
            feedforward_operations_count: 0
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
            Self::Value(tensor) => (tensor.rows(), tensor.columns()),
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

#[derive(Clone)]
pub struct OperationsRecorder
{
    state: RecorderState,
    current_block: BlockIndex,
    global_live_ranges: Vec<LiveRange>,
    tensors_memory: Vec<TensorMemorySlot>,
    values: Vec<f32>,
    tensors: Vec<LayerType>,
    one_hot_layers: Vec<OneHotLayer>,
    operations_blocks: Vec<OperationsBlock>,
//    #[cfg(debug_assertions)]
//    checked_inputs: Vec<DiffValue>,
//    #[cfg(debug_assertions)]
//    pub allow_uninitialized: Vec<DiffValue>
}

impl Debug for OperationsRecorder
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        f.debug_struct("OperationsRecorder")
            .field("state", &self.state)
            .field("global_live_ranges", &self.global_live_ranges.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
            .field("tensors_memory", &self.tensors_memory.iter().map(|x| ForceNoPretty(x)).collect::<Vec<_>>())
            .field("values", &ForceNoPretty(&self.values))
            .field("tensors", &self.tensors)
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

            $this.global_live_ranges.push(LiveRange{start: None, end: None});
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
            global_live_ranges: Vec::new(),
            tensors_memory: Vec::new(),
            values: Vec::new(),
            tensors: Vec::new(),
            one_hot_layers: Vec::new(),
            operations_blocks: vec![OperationsBlock::default()],
//            #[cfg(debug_assertions)]
//            checked_inputs: Vec::new(),
//            #[cfg(debug_assertions)]
//            allow_uninitialized: Vec::new()
        }
    }

    pub fn new_tensor(&mut self, rows: usize, columns: usize) -> DiffTensorPtr
    {
        let input = self.new_tensor_with_source(None, true, rows, columns);
        self.global_live_ranges[input.as_value().0].start = Some(-1);

let put_me_back = ();
/*        #[cfg(debug_assertions)]
        {
            self.checked_inputs.push(input.as_value().into());
            self.checked_inputs.push(input.as_gradient().unwrap().into());
        }*/

        input
    }

    pub fn new_tensor_no_gradient(&mut self, rows: usize, columns: usize) -> DiffTensorPtr
    {
        let input = self.new_tensor_with_source(None, false, rows, columns);
        self.global_live_ranges[input.as_value().0].start = Some(-1);

let put_me_back = ();
//        #[cfg(debug_assertions)]
//        self.checked_inputs.push(input.as_value().into());

        input
    }

    pub fn new_value(&mut self) -> DiffScalar
    {
        let input = self.new_value_with_source(None, true);

let put_me_back = ();
/*        #[cfg(debug_assertions)]
        {
            self.checked_inputs.push(input.as_value().into());
            self.checked_inputs.push(input.as_gradient().unwrap().into());
        }*/

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
        self.global_live_ranges[input.as_value().0].start = Some(-1);

let put_me_back = ();
/*        #[cfg(debug_assertions)]
        {
            self.checked_inputs.push(input.as_value().into());
            self.checked_inputs.push(input.as_gradient().unwrap().into());
        }*/

        input
    }

    pub fn set_new_tensor(&mut self, value: LayerType) -> DiffTensorPtr
    {
        let rows = value.rows();
        let columns = value.columns();

        let input = new_tensor!(self, None, false, rows, columns, TensorMemoryValue::Value(value));
        self.global_live_ranges[input.as_value().0].start = Some(-1);

let put_me_back = ();
//        #[cfg(debug_assertions)]
//        self.checked_inputs.push(input.as_value().into());

        input
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
            DiffWrapper::Tensor(DiffTensorPtr{index, gradient, ..}) =>
            {
                let (rows, columns) = self.tensor_shape(index);

                let new_value = TensorMemoryValue::Value(LayerType::repeat(rows, columns, 1.0));

                let gradient_ptr: TensorPtr = gradient.expect("gradient must exist");

                self.global_live_ranges[gradient_ptr.0].start = Some(-1);
                self.tensors_memory[gradient_ptr.0].value = new_value;
            },
            DiffWrapper::Value(DiffScalar{gradient, ..}) => self.set_value(gradient.expect("gradient must exist"), 1.0)
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

    pub fn store_tensor_until_end(&mut self, index_ptr: TensorPtr)
    {
        self.global_live_ranges[index_ptr.0].end = Some(i32::MAX);
    }

    pub fn new_block(&mut self) -> BlockIndex
    {
        let id = BlockIndex(self.operations_blocks.len());

        self.operations_blocks.push(OperationsBlock::default());

        self.current_block = id;

        id
    }

    pub fn current_block(&self) -> BlockIndex
    {
        self.current_block
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
            //let before_instant = std::time::Instant::now();
            match gradient_op
            {
                GradientOp::None => (),
                GradientOp::Copy{src, dst} =>
                {
                    let this_should_not_even_exist = ();
                    self.tensors[dst.0] = self.tensors[src.0].clone();
                },
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
                    let optimize_this = ();
                    self.tensors[output.0] = &self.tensors[lhs.0] + &self.tensors[rhs.0];
                },
                GradientOp::AddInplace{value, output} =>
                {
                    let [output, value] = self.tensors.get_disjoint_mut([output.0, value.0]).unwrap();

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
                    let optimize_this = ();
                    self.tensors[output.0] = &self.tensors[lhs.0] * &self.tensors[rhs.0];
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
                },
                GradientOp::SoftmaxCrossEntropy{values, targets, softmaxed_output, output} =>
                {
                    let optimize_this = ();
                    (self.tensors[softmaxed_output.0], self.values[output.0]) = self.tensors[values.0].clone()
                        .softmax_cross_entropy(&self.one_hot_layers[targets.0]);
                },
                GradientOp::SoftmaxCrossEntropyDiff{softmaxed_values, gradient, targets, output} =>
                {
                    debug_assert_eq!(
                        tensor_shape!(self, *softmaxed_values), (self.one_hot_layers[targets.0].size, 1),
                        "softmaxed: {softmaxed_values:?}, targets: {targets:?}"
                    );

                    let optimize_this = ();
                    self.tensors[output.0] = (&self.tensors[softmaxed_values.0] - self.one_hot_layers[targets.0].clone().into_layer()) * self.values[gradient.0];
                },
                GradientOp::Matmulv{lhs, rhs, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = self.tensors[lhs.0].matmulv(&self.tensors[rhs.0]);
                },
                GradientOp::MatmulvAdd{lhs, rhs, added, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = self.tensors[lhs.0].matmulv(&self.tensors[rhs.0]) + &self.tensors[added.0];
                },
                GradientOp::MatmulOneHotvAdd{lhs, rhs, added, output} =>
                {
                    let optimize_this = ();
                    self.tensors[output.0] = self.tensors[lhs.0].matmul_onehotv_add(&self.one_hot_layers[rhs.0], &self.tensors[added.0]);
                },
                GradientOp::MatmulvTransposed{lhs, rhs, output} =>
                {
                    let [output, lhs, rhs] = self.tensors.get_disjoint_mut([output.0, lhs.0, rhs.0]).unwrap();

                    output.matmulv_transposed_into(lhs, rhs);
                },
                GradientOp::OuterProduct{lhs, rhs, output} =>
                {
                    let [output, lhs, rhs] = self.tensors.get_disjoint_mut([output.0, lhs.0, rhs.0]).unwrap();

                    output.outer_product_into(lhs, rhs);
                },
                GradientOp::OuterProductOneHot{lhs, rhs, output} =>
                {
                    let [output, lhs] = self.tensors.get_disjoint_mut([output.0, lhs.0]).unwrap();

                    output.outer_product_one_hot_into(lhs, &self.one_hot_layers[rhs.0]);
                }
            }
            //eprintln!("{}, elapsed {:.3} us", format!("{gradient_op:?}").chars().take_while(|c| *c != ' ').collect::<String>(), before_instant.elapsed().as_nanos() as f64 / 1000.0);
        });

        // this might give a false positive but its a very important check
/*        #[cfg(debug_assertions)]
        {
            let belongs_to_block = |value: &DiffValueRaw| -> bool
            {
                self.operations_blocks[block.0].raw_operations.iter().take(steps).any(|op| *value == op.output_of())
            };

            self.values.iter().enumerate().map(|(index, _)| DiffValueRaw::Value(ValueIndex(index)))
                .chain(self.tensors.iter().enumerate().map(|(index, _)| DiffValueRaw::Tensor(TensorIndex(index))))
                .filter(belongs_to_block)
                .zip(iter::repeat(false))
                .chain(self.checked_inputs.iter().cloned().zip(iter::repeat(true)))
                .filter(|(value, _)|
                {
                    match value
                    {
                        DiffValueRaw::Tensor(x) => self.tensors[x.0].iter().all(|inner| *inner == 0.0),
                        DiffValueRaw::Value(x) => self.values[x.0] == 0.0
                    }
                })
                .filter(|(value, _)| !self.allow_uninitialized.contains(value))
                .for_each(|(value, is_input)|
                {
                    if is_input
                    {
                        eprintln!("index {value:?} might be uninitialized");
                    } else
                    {
                        panic!("index {value:?} is uninitialized");
                    }
                });
        }*/
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

        let mut handled = Vec::new();

        let mut i = 1;
        while i < this_block.gradient_operations.len()
        {
            let this = &this_block.gradient_operations[i];

            if let Some(previous) = (0..i).find(|previous|
            {
                let is_shared_output = this_block.gradient_operations[*previous].diff_output_of() == this.diff_output_of();

                is_shared_output && !handled.contains(previous)
            })
            {
                match &mut this_block.gradient_operations[i]
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
                        let (rows, columns) = tensor_shape!(self, output);

                        let final_output = *output;

                        let temporary_add_index = new_tensor_index!(self, rows, columns);

                        *output = temporary_add_index;

                        this_block.gradient_operations.insert(
                            i + 1,
                            GradientOp::Add{lhs: temporary_add_index, rhs: final_output, output: final_output}
                        );
                    },
                    GradientOp::CopyScalar{dst: output, ..}
                    | GradientOp::AddScalars{output, ..}
                    | GradientOp::MulScalars{output, ..}
                    | GradientOp::SumTensor{output, ..}
                    | GradientOp::Dot{output, ..}
                    | GradientOp::SoftmaxCrossEntropy{output, ..} =>
                    {
                        let final_output = *output;

                        let temporary_add_index = new_value_index!(self);

                        *output = temporary_add_index;

                        this_block.gradient_operations.insert(
                            i + 1,
                            GradientOp::AddScalars{lhs: temporary_add_index, rhs: final_output, output: final_output}
                        );
                    },
                    GradientOp::None
                    | GradientOp::AddInplace{..} => unreachable!()
                }

                handled.push(i);
                handled.push(i + 1);

                i += 1;
            }

            i += 1;
        }
    }

    pub fn is_ready(&self) -> bool
    {
        self.state == RecorderState::Ready
    }

    fn calculate_live_ranges_block(&mut self, block: BlockIndex) -> bool
    {
        let block = &mut self.operations_blocks[block.0];

        block.live_ranges = self.global_live_ranges.clone();

        block.gradient_operations.iter().enumerate().rev().for_each(|(op_index, op)|
        {
            op.for_tensor_outputs(|tensor_ptr|
            {
                let start = &mut block.live_ranges[tensor_ptr.0].start;

                debug_assert!(start.is_none());

                *start = Some(op_index as i32);
            });

            op.for_tensor_args(|tensor_ptr|
            {
                let end = &mut block.live_ranges[tensor_ptr.0].end;

                if end.is_none()
                {
                    *end = Some(op_index as i32);
                }
            });
        });

        let mut any_unused = false;
        block.gradient_operations.iter_mut().for_each(|op|
        {
            let mut all_unused: Option<bool> = None;
            op.for_tensor_outputs(|tensor_ptr|
            {
                let is_unused = block.live_ranges[tensor_ptr.0].end == None;

                if let Some(all_unused) = all_unused.as_mut()
                {
                    *all_unused &= is_unused;
                } else
                {
                    all_unused = Some(is_unused);
                }
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
        let nodes_count = self.global_live_ranges.len();

        let mut graph_connections: Vec<Vec<usize>> = iter::from_fn(|| Some(Vec::new()))
            .take(nodes_count)
            .collect();

        let live_ranges = &mut self.operations_blocks[block.0].live_ranges;

        (0..nodes_count).for_each(|node_index|
        {
            {
                let this_range = &live_ranges[node_index];

                if this_range.start.is_none()
                {
                    debug_assert!(this_range.end.is_none());
                    return;
                }
            }

            (0..nodes_count).for_each(|check_index|
            {
                if node_index == check_index
                {
                    return;
                }

                let is_overlap = {
                    let other_range = &live_ranges[check_index];

                    if other_range.start.is_none()
                    {
                        debug_assert!(other_range.end.is_none());
                        return;
                    }

                    live_ranges[node_index].overlaps(other_range)
                };

                if is_overlap
                {
                    graph_connections[node_index].push(check_index);
                }
            });
        });

        let mut connections_count_sorted: Vec<usize> = (0..nodes_count).collect();
        connections_count_sorted.sort_unstable_by_key(|node_index| graph_connections[*node_index].len());
        connections_count_sorted.reverse();

        connections_count_sorted.into_iter().for_each(|node_index|
        {
            if live_ranges[node_index].start.is_none()
            {
                debug_assert!(live_ranges[node_index].end.is_none());

                return;
            }

            let this_color = (0..).find(|color|
            {
                let all_connected_unconflicted = graph_connections[node_index].iter().all(|connected_node_index|
                {
                    let connected_node_color: Option<usize> = self.tensors_memory[*connected_node_index].memory.map(|x| x.0);

                    connected_node_color != Some(*color)
                });

                let actually_set_this = ();
                let spot_size_matches = true;

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
                debug_assert!(self.tensors[this_color].iter().all(|x| *x == 0.0));

                self.tensors[this_color] = x.clone();
            }

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

        self.operations_blocks.iter_mut().for_each(|block|
        {
            block.live_ranges.clear();

            block.raw_operations = mem::take(&mut block.gradient_operations).into_iter().filter_map(|op| -> Option<GradientOp<TensorIndex>>
            {
                match op
                {
                    GradientOp::None => None,
                    x => Some(x.map_tensors(|tensor_ptr| self.tensors_memory[tensor_ptr.0].memory.expect("must be resolved")))
                }
            }).collect();

            block.feedforward_operations_count = block.raw_operations.len();
        });

        self.global_live_ranges.clear();

        self.state = RecorderState::Ready;
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

impl OneHotIndex
{
    pub fn undefined() -> Self { Self(usize::MAX) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorPtr(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorIndex(usize);

impl TensorIndex
{
    pub fn undefined() -> Self { Self(usize::MAX) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueIndex(usize);

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

    pub fn as_value(&self) -> TensorPtr
    {
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
            Self::AddInplace{..} => unreachable!()
        }
    }
}

impl GradientOp<TensorPtr>
{
    fn for_tensor_outputs(&self, mut f: impl FnMut(TensorPtr))
    {
        match self
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
            | Self::SoftmaxCrossEntropy{softmaxed_output: output, ..}
            | Self::SoftmaxCrossEntropyDiff{output, ..}
            | Self::Matmulv{output, ..}
            | Self::MatmulvAdd{output, ..}
            | Self::MatmulOneHotvAdd{output, ..}
            | Self::MatmulvTransposed{output, ..}
            | Self::OuterProduct{output, ..}
            | Self::OuterProductOneHot{output, ..} => f(*output),
            Self::None
            | Self::CopyScalar{..}
            | Self::AddScalars{..}
            | Self::MulScalars{..}
            | Self::SumTensor{..}
            | Self::Dot{..} => (),
            Self::AddInplace{..} => unreachable!()
        }
    }

    fn for_tensor_args(&self, mut f: impl FnMut(TensorPtr))
    {
        match *self
        {
            Self::Copy{src, ..} => f(src),
            Self::AddScalar{lhs, ..} => f(lhs),
            Self::Add{lhs, rhs, ..} => { f(lhs); f(rhs) },
            Self::Sub{lhs, rhs, ..} => { f(lhs); f(rhs) },
            Self::SubFromScalar{rhs, ..} => f(rhs),
            Self::MulScalar{lhs, ..} => f(lhs),
            Self::MulComponentwise{lhs, rhs, ..} => { f(lhs); f(rhs) },
            Self::SumTensor{value, ..} => f(value),
            Self::Pow{lhs, ..} => f(lhs),
            Self::LeakyRelu{value, ..} => f(value),
            Self::LeakyReluDiff{value, gradient, ..} => { f(value); f(gradient) },
            Self::Sigmoid{value, ..} => f(value),
            Self::SigmoidDiff{value, gradient, ..} => { f(value); f(gradient) },
            Self::Tanh{value, ..} => f(value),
            Self::TanhDiff{value, gradient, ..} => { f(value); f(gradient) },
            Self::Dot{lhs, rhs, ..} => { f(lhs); f(rhs) },
            Self::SoftmaxCrossEntropy{values, ..} => f(values),
            Self::SoftmaxCrossEntropyDiff{softmaxed_values, ..} => f(softmaxed_values),
            Self::Matmulv{lhs, rhs, ..} => { f(lhs); f(rhs) },
            Self::MatmulvAdd{lhs, rhs, added, ..} => { f(lhs); f(rhs); f(added) },
            Self::MatmulOneHotvAdd{lhs, added, ..} => { f(lhs); f(added) },
            Self::MatmulvTransposed{lhs, rhs, ..} => { f(lhs); f(rhs) },
            Self::OuterProduct{lhs, rhs, ..} => { f(lhs); f(rhs) },
            Self::OuterProductOneHot{lhs, ..} => f(lhs),
            Self::None
            | Self::CopyScalar{..}
            | Self::AddScalars{..}
            | Self::MulScalars{..}
            | Self::Fill{..} => (),
            Self::AddInplace{..} => unreachable!()
        }
    }

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
            | Self::AddInplace{..} => unreachable!()
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

#[derive(Debug, Clone, Copy)]
pub enum InputType
{
    Normal(TensorIndex),
    OneHot(OneHotIndex)
}

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

#[derive(Debug, Clone, Copy)]
pub enum DiffInputType
{
    Normal(DiffTensor),
    OneHot(OneHotIndex)
}

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

    fn copy_tensor(
        old_recorder: &mut OperationsRecorder,
        new_recorder: &mut OperationsRecorder,
        old_tensor: DiffTensor
    ) -> DiffTensor
    {
        let old_value = old_recorder.get_tensor(old_tensor.as_value()).clone();

        new_recorder.set_new_tensor(old_value)
    }

    fn check_tensor_inner(
        recorder: &mut OperationsRecorder,
        a: DiffTensorPtr,
        b: DiffTensorPtr,
        mut f: impl FnMut(&mut OperationsRecorder, DiffTensorPtr, DiffTensorPtr) -> DiffTensorPtr
    )
    {
        let out = f(recorder, a, b);

        recorder.finish();
        recorder.gradient_with_respect(vec![out.into()]);

        recorder.resolve_memory();

        let current_block = recorder.current_block();
        recorder.calculate(current_block);

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
            new_recorder.gradient_with_respect(vec![output.into()]);

            let current_block = new_recorder.current_block();
            new_recorder.calculate(current_block);

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
    fn matrix_multiplication()
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
    fn softmax_cross_entropy()
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
