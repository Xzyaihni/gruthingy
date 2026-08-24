use std::{
    f32,
    mem,
    convert,
    fmt::{self, Debug},
    borrow::Borrow,
    collections::{VecDeque, HashSet},
    ops::{DivAssign, Range}
};

#[allow(unused_imports)]
use std::iter;

use serde::{Serialize, Deserialize};

use matrix_wrapper::{MatrixWrapper, MatrixWrapperRef, MatrixWrapperMut, VectorWrapper, VectorWrapperMut};

mod matrix_wrapper;


pub type LayerType = MatrixWrapper;
pub type LayerTypeRef<'a> = MatrixWrapperRef<'a>;
pub type LayerTypeMut<'a> = MatrixWrapperMut<'a>;

pub type LayerTypeVector<'a> = VectorWrapper<'a>;
pub type LayerTypeVectorMut<'a> = VectorWrapperMut<'a>;

pub const LEAKY_SLOPE: f32 = 0.01;

const OPT_INFO: bool = true;
const NO_COLORING: bool = false;
const PRINT_CALCULATE_VALUES: bool = true;


macro_rules! get_disjoint_mut_with
{
    ($this:expr, $(($target_type:ident, $name:expr, $tmp_name:ident)),+$(,)?) =>
    {
        {
            let indices = [$($name.range(),)+];

            let [$($tmp_name,)+] = {
                #[cfg(debug_assertions)]
                {
                    $this.tensors_raw_data.get_disjoint_mut(indices).unwrap()
                }

                #[cfg(not(debug_assertions))]
                {
                    unsafe{ $this.tensors_raw_data.get_disjoint_unchecked_mut(indices) }
                }
            };

            ($($target_type::from_data($tmp_name, $name),)+)
        }
    }
}

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

struct DebugStringRaw(String);

impl Debug for DebugStringRaw
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        write!(f, "{}", &self.0)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum StoreCheckKey<P, R>
{
    PreResolve(P),
    Resolved(R)
}

#[cfg(debug_assertions)]
fn verify_store_check<T: Eq, K: Eq + Debug + Copy>(
    store_checks: &Vec<StoreCheckKey<T, K>>,
    index: K,
    name: &str
)
{
    debug_assert!(
        store_checks.contains(&StoreCheckKey::Resolved(index)),
        "store_{name}_until_end must be called on {index:?}"
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TensorRawDataPointer
{
    raw_index: TensorIndexRaw,
    rows: usize,
    columns: usize
}

impl Debug for TensorRawDataPointer
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        write!(f, "{{({},{}) {:?}}}", self.rows, self.columns, self.raw_index)
    }
}

impl TensorRawDataPointer
{
    fn undefined() -> Self
    {
        Self{
            raw_index: TensorIndexRaw(usize::MAX),
            rows: 0,
            columns: 0
        }
    }

    pub fn range(&self) -> Range<usize>
    {
        self.raw_index.0..(self.raw_index.0 + self.size())
    }

    pub fn size(&self) -> usize
    {
        self.rows * self.columns
    }
}

#[derive(Debug, Clone)]
struct LoopInfo
{
    times: usize,
    input_values: Vec<LayerType>,
    inputs: Vec<InputType>
}

#[derive(Debug, Clone)]
struct RawJumpInfo
{
    loop_index: LoopIndex,
    operation_index: GradientOperationIndex
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopIndex(usize);

#[derive(Debug, Clone)]
enum OperationsTarget
{
    Normal,
    Loop(LoopOperationIndex)
}

#[derive(Debug, Clone)]
enum InputCheckType
{
    Ptr(TensorPtr),
    Index(TensorIndex)
}

impl From<TensorPtr> for InputCheckType
{
    fn from(x: TensorPtr) -> Self
    {
        Self::Ptr(x)
    }
}

impl From<TensorIndex> for InputCheckType
{
    fn from(x: TensorIndex) -> Self
    {
        Self::Index(x)
    }
}

#[derive(Clone)]
pub struct OperationsRecorder
{
    state: RecorderState,
    operations_target: OperationsTarget,
    value_live_ranges: Vec<LiveRange>,
    tensor_live_ranges: Vec<LiveRange>,
    tensors_memory: Vec<TensorMemorySlot>,
    values: Vec<f32>,
    tensors: Vec<TensorRawDataPointer>,
    tensors_raw_data: Vec<f32>,
    one_hot_layers: Vec<OneHotLayer>,
    loops: Vec<LoopInfo>,
    recording_operations: Vec<Op>,
    gradient_operations: Vec<GradientOp<TensorPtr, JumpInfo>>,
    raw_operations: Vec<GradientOp<TensorRawDataPointer, RawJumpInfo>>,
    feedforward_operations_count: usize,
    #[cfg(debug_assertions)]
    tensor_inputs: Vec<TensorPtr>,
    #[cfg(debug_assertions)]
    set_tensors_check: Vec<InputCheckType>,
    #[cfg(debug_assertions)]
    store_tensors_check: Vec<StoreCheckKey<TensorPtr, TensorIndex>>,
    #[cfg(debug_assertions)]
    store_values_check: Vec<StoreCheckKey<ValueIndex, ValueIndex>>
}

impl Debug for OperationsRecorder
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let max_length = 50;

        let mut s = f.debug_struct("OperationsRecorder");

        s.field("state", &self.state)
            .field("operations_target", &self.operations_target)
            .field("value_live_ranges", &self.value_live_ranges.iter().map(ForceNoPretty).collect::<Vec<_>>())
            .field("tensor_live_ranges", &self.tensor_live_ranges.iter().map(ForceNoPretty).collect::<Vec<_>>())
            .field("tensors_memory", &self.tensors_memory.iter().map(|x| ForceNoPretty(SlotNoLong(max_length, x))).collect::<Vec<_>>())
            .field("values", &ForceNoPretty(&self.values))
            .field("tensors", &self.tensors.iter().map(ForceNoPretty).collect::<Vec<_>>())
            .field("tensors_raw_data", &DebugStringRaw(format!("{} values", self.tensors_raw_data.len())))
            .field("one_hot_layers", &self.one_hot_layers)
            .field("loops", &self.loops)
            .field("recording_operations", &self.recording_operations.iter().map(ForceNoPretty).collect::<Vec<_>>())
            .field("gradient_operations", &self.gradient_operations.iter().map(ForceNoPretty).collect::<Vec<_>>())
            .field("raw_operations", &self.raw_operations.iter().map(ForceNoPretty).collect::<Vec<_>>())
            .field("feedforward_operations_count", &self.feedforward_operations_count);

        #[cfg(debug_assertions)]
        {
            s.field("tensor_inputs", &self.tensor_inputs.iter().map(ForceNoPretty).collect::<Vec<_>>())
                .field("set_tensors_check", &self.set_tensors_check.iter().map(ForceNoPretty).collect::<Vec<_>>())
                .field("store_tensors_check", &self.store_tensors_check.iter().map(ForceNoPretty).collect::<Vec<_>>())
                .field("store_values_check", &self.store_values_check.iter().map(ForceNoPretty).collect::<Vec<_>>());
        }

        s.finish()
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

            $this.tensor_live_ranges.push(LiveRange{start: None, end: None});
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

            $this.value_live_ranges.push(LiveRange{start: None, end: None});
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

            $this.add_recording_operation(Op::$name{lhs: $a, rhs: $b, output});

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

            $this.add_recording_operation(Op::$name{value: $a, output});

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
            operations_target: OperationsTarget::Normal,
            value_live_ranges: Vec::new(),
            tensor_live_ranges: Vec::new(),
            tensors_memory: Vec::new(),
            values: Vec::new(),
            tensors: Vec::new(),
            tensors_raw_data: Vec::new(),
            one_hot_layers: Vec::new(),
            loops: Vec::new(),
            recording_operations: Vec::new(),
            gradient_operations: Vec::new(),
            raw_operations: Vec::new(),
            feedforward_operations_count: 0,
            #[cfg(debug_assertions)]
            tensor_inputs: Vec::new(),
            #[cfg(debug_assertions)]
            set_tensors_check: Vec::new(),
            #[cfg(debug_assertions)]
            store_tensors_check: Vec::new(),
            #[cfg(debug_assertions)]
            store_values_check: Vec::new()
        }
    }

    pub fn new_tensor(&mut self, rows: usize, columns: usize) -> DiffTensorPtr
    {
        let input = self.new_tensor_with_source(None, true, rows, columns);
        self.tensor_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn new_tensor_no_gradient(&mut self, rows: usize, columns: usize) -> DiffTensorPtr
    {
        let input = self.new_tensor_with_source(None, false, rows, columns);
        self.tensor_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn new_value(&mut self) -> DiffScalar
    {
        let input = self.new_value_with_source(None, true);
        self.value_live_ranges[input.as_value().0].start = Some(-1);

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
        debug_assert_eq!(self.state, RecorderState::Ready);

        #[cfg(debug_assertions)]
        {
            self.set_tensors_check.push(index.into());
        }

        let dst = LayerTypeMut::from_data_with_start(&mut self.tensors_raw_data, self.tensors[index.0]);
        let src = LayerTypeRef::from(&value);

        dst.copy_from(src);
    }

    pub fn set_tensor_from(&mut self, index: TensorIndex, src: TensorIndex)
    {
        debug_assert_eq!(self.state, RecorderState::Ready);

        #[cfg(debug_assertions)]
        {
            self.set_tensors_check.push(index.into());
        }

        let dst = self.tensors[index.0];
        let src = self.tensors[src.0];

        let (dst, src) = get_disjoint_mut_with!(self, (LayerTypeMut, dst, x0), (LayerTypeRef, src, x1));

        dst.copy_from(src);
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

        #[cfg(debug_assertions)]
        {
            self.set_tensors_check.push(input.as_value().into());
        }

        self.tensor_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn set_new_tensor(&mut self, value: LayerType) -> DiffTensorPtr
    {
        let rows = value.rows();
        let columns = value.columns();

        let input = new_tensor!(self, None, false, rows, columns, TensorMemoryValue::Value(value));

        #[cfg(debug_assertions)]
        {
            self.set_tensors_check.push(input.as_value().into());
        }

        self.tensor_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn set_new_value(&mut self, value: f32) -> DiffScalar
    {
        let scalar = self.new_value_with_source(None, false);
        self.set_value(scalar.as_value(), value);

        self.value_live_ranges[scalar.as_value().0].start = Some(-1);

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

                #[cfg(debug_assertions)]
                {
                    self.set_tensors_check.push(gradient_ptr.into());
                }

                self.tensor_live_ranges[gradient_ptr.0].start = Some(-1);
                self.tensors_memory[gradient_ptr.0].value = new_value;
            },
            DiffWrapper::Value(DiffScalar{gradient, ..}) =>
            {
                let gradient = gradient.expect("gradient must exist");

                self.value_live_ranges[gradient.0].start = Some(-1);

                self.set_value(gradient, 1.0)
            }
        }
    }

    pub fn get_tensor_memory_value(&self, index: TensorPtr) -> LayerTypeRef<'_>
    {
        debug_assert_eq!(self.state, RecorderState::Ready);

        if let TensorMemoryValue::Value(x) = &self.tensors_memory[index.0].value
        {
            LayerTypeRef::from(x)
        } else
        {
            panic!("{index:?} has no memory value");
        }
    }

    pub fn is_undefined_location(&self, index: TensorIndex) -> bool
    {
        self.tensors[index.0] == TensorRawDataPointer::undefined()
    }

    pub fn get_tensor(&self, index: TensorIndex) -> LayerTypeRef<'_>
    {
        debug_assert_eq!(self.state, RecorderState::Ready);
        debug_assert_ne!(index, TensorIndex::undefined());

        #[cfg(debug_assertions)]
        {
            verify_store_check(&self.store_tensors_check, index, "tensor");
        }

        let info = self.tensors[index.0];
        debug_assert_ne!(info, TensorRawDataPointer::undefined(), "{index:?} location is undefined");

        LayerTypeRef::from_data_with_start(&self.tensors_raw_data, info)
    }

    pub fn get_tensor_mut<const USES_VALUE: bool>(&mut self, index: TensorIndex) -> LayerTypeMut<'_>
    {
        debug_assert_eq!(self.state, RecorderState::Ready);
        debug_assert_ne!(index, TensorIndex::undefined());

        #[cfg(debug_assertions)]
        {
            if USES_VALUE
            {
                verify_store_check(&self.store_tensors_check, index, "tensor");
            }
        }

        #[cfg(debug_assertions)]
        {
            if !USES_VALUE
            {
                self.set_tensors_check.push(index.into());
            }
        }

        let info = self.tensors[index.0];
        debug_assert_ne!(info, TensorRawDataPointer::undefined(), "{index:?} location is undefined");

        LayerTypeMut::from_data_with_start(&mut self.tensors_raw_data, info)
    }

    pub fn get_value(&self, index: ValueIndex) -> f32
    {
        debug_assert_eq!(self.state, RecorderState::Ready);
        debug_assert_ne!(index, ValueIndex::undefined());

        #[cfg(debug_assertions)]
        {
            verify_store_check(&self.store_values_check, index, "value");
        }

        self.values[index.0]
    }

    pub fn get_one_hot(&self, index: OneHotIndex) -> &OneHotLayer
    {
        debug_assert_eq!(self.state, RecorderState::Ready);
        debug_assert_ne!(index, OneHotIndex::undefined());

        &self.one_hot_layers[index.0]
    }

    fn current_source(&self) -> OperationIndex
    {
        debug_assert_eq!(self.state, RecorderState::Recording);

        if let OperationsTarget::Loop(operation_index) = self.operations_target
        {
            let inside_index = if let Op::Loop{ops, ..} = &self.recording_operations[operation_index.0]
            {
                ops.len()
            } else
            {
                unreachable!()
            };

            OperationIndex(self.recording_operations.len() - 1, Some(inside_index))
        } else
        {
            OperationIndex(self.recording_operations.len(), None)
        }
    }

    fn add_recording_operation(&mut self, op: Op)
    {
        let ops_target = match self.operations_target
        {
            OperationsTarget::Normal => &mut self.recording_operations,
            OperationsTarget::Loop(operation_index) =>
            {
                if let Op::Loop{ops, ..} = &mut self.recording_operations[operation_index.0]
                {
                    ops
                } else
                {
                    unreachable!()
                }
            }
        };

        ops_target.push(op);
    }

    fn remove_gradient_operation(&mut self, i: usize)
    {
        self.gradient_operations.remove(i);
        if i < self.feedforward_operations_count
        {
            self.feedforward_operations_count -= 1;
        }
    }

    pub fn add_scalars(&mut self, a: DiffScalar, b: DiffScalar) -> DiffScalar
    {
        let source = Some(self.current_source());

        let output = self.new_value_op(source);

        self.add_recording_operation(Op::AddScalars{lhs: a, rhs: b, output});

        output
    }

    pub fn add_scalar(&mut self, a: DiffTensorPtr, b: DiffScalar) -> DiffTensorPtr
    {
        let source = Some(self.current_source());

        let (a_rows, a_columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(source, a_rows, a_columns);

        self.add_recording_operation(Op::AddScalar{lhs: a, rhs: b, output});

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

        self.add_recording_operation(Op::SubFromScalar{lhs: a, rhs: b, output});

        output
    }

    pub fn mul_scalars(&mut self, a: DiffScalar, b: DiffScalar) -> DiffScalar
    {
        let source = Some(self.current_source());

        let output = self.new_value_op(source);

        self.add_recording_operation(Op::MulScalars{lhs: a, rhs: b, output});

        output
    }

    pub fn mul_scalar(&mut self, a: DiffTensorPtr, b: DiffScalar) -> DiffTensorPtr
    {
        let source = Some(self.current_source());

        let (a_rows, a_columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(source, a_rows, a_columns);

        self.add_recording_operation(Op::MulScalar{lhs: a, rhs: b, output});

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

        self.add_recording_operation(Op::Matmulv{lhs: a, rhs: b, output});

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

        self.add_recording_operation(Op::MatmulvAdd{lhs: a, rhs: b, added, output});

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

        self.add_recording_operation(Op::MatmulOneHotvAdd{lhs: a, rhs: b, added, output});

        output
    }

    pub fn sum_tensor(&mut self, a: DiffTensorPtr) -> DiffScalar
    {
        let source = Some(self.current_source());

        let output = self.new_value_op(source);

        self.add_recording_operation(Op::SumTensor{value: a, output});

        output
    }

    pub fn dot(&mut self, a: DiffTensorPtr, b: DiffTensorPtr) -> DiffScalar
    {
        let source = Some(self.current_source());

        debug_assert_eq!(self.tensor_shape(a.as_value()), self.tensor_shape(b.as_value()));

        let output = self.new_value_op(source);

        self.add_recording_operation(Op::Dot{lhs: a, rhs: b, output});

        output
    }

    pub fn pow(&mut self, a: DiffTensorPtr, power: i32) -> DiffTensorPtr
    {
        let source = Some(self.current_source());

        let (rows, columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(source, rows, columns);

        self.add_recording_operation(Op::Pow{lhs: a, power, output});

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

        self.add_recording_operation(Op::SoftmaxCrossEntropy{values, targets, softmaxed_output, output});

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
        debug_assert!(self.state.before_or_at(RecorderState::AwaitingGradient));

        #[cfg(debug_assertions)]
        {
            let key = StoreCheckKey::PreResolve(index_ptr);

            if !self.store_tensors_check.contains(&key)
            {
                self.store_tensors_check.push(key);
            }
        }

        self.tensor_live_ranges[index_ptr.0].end = Some(i32::MAX);
    }

    pub fn store_value_until_end(&mut self, index: ValueIndex)
    {
        debug_assert!(self.state.before_or_at(RecorderState::AwaitingGradient));

        #[cfg(debug_assertions)]
        {
            let key = StoreCheckKey::PreResolve(index);

            if !self.store_values_check.contains(&key)
            {
                self.store_values_check.push(key);
            }
        }

        self.value_live_ranges[index.0].end = Some(i32::MAX);
    }

    pub fn begin_loop(&mut self, inputs: Vec<InputTypePtr>) -> LoopIndex
    {
        debug_assert_eq!(self.state, RecorderState::Recording);
        debug_assert!(matches!(self.operations_target, OperationsTarget::Normal));

        let id = LoopIndex(self.loops.len());
        let operation_index = LoopOperationIndex(self.recording_operations.len());

        self.loops.push(LoopInfo{
            times: 0,
            input_values: Vec::new(),
            inputs: Vec::new()
        });

        self.recording_operations.push(Op::Loop{index: id, inputs, ops: Vec::new()});

        self.operations_target = OperationsTarget::Loop(operation_index);

        id
    }

    pub fn end_loop(&mut self, _index: LoopIndex)
    {
        debug_assert_eq!(self.state, RecorderState::Recording);

        self.operations_target = OperationsTarget::Normal;
    }

    pub fn set_loop_times(&mut self, index: LoopIndex, times: usize)
    {
        self.loops[index.0].times = times;
    }

    pub fn set_loop_inputs(&mut self, index: LoopIndex, inputs: Vec<LayerType>)
    {
        self.loops[index.0].input_values = inputs;
    }

    pub fn calculate_feedforward(&mut self)
    {
        let count = self.feedforward_operations_count;

        self.calculate_steps(0, count);
    }

    pub fn calculate_backpropagate(&mut self)
    {
        let total = self.raw_operations.len();
        let count = self.feedforward_operations_count;

        self.calculate_steps(count, total);
    }

    pub fn calculate(&mut self)
    {
        let total = self.raw_operations.len();

        self.calculate_steps(0, total);
    }

    fn calculate_steps(&mut self, start: usize, end: usize)
    {
        debug_assert_eq!(self.state, RecorderState::Ready);

        for loop_index in 0..self.loops.len()
        {
            let total_count;
            let current_index;
            let inputs_count;

            {
                let loop_info = &self.loops[loop_index];

                debug_assert!(loop_info.times > 0);
                debug_assert_eq!(loop_info.times * loop_info.inputs.len(), loop_info.input_values.len());

                inputs_count = loop_info.inputs.len();

                total_count = loop_info.input_values.len() / loop_info.inputs.len();
                current_index = total_count - loop_info.times;
            }

            for input_index in 0..inputs_count
            {
                match self.loops[loop_index].inputs[input_index]
                {
                    InputType::Normal(input) =>
                    {
                        let value = self.loops[loop_index].input_values[current_index + input_index].clone();

                        self.set_tensor(input, value)
                    },
                    InputType::OneHot(input) => todo!()
                }
            }
        }

        dbg!(&self);
        #[cfg(debug_assertions)]
        {
            self.tensor_inputs.iter().for_each(|input_tensor_ptr|
            {
                let contains_ptr = self.set_tensors_check.iter()
                    .filter_map(|x| if let InputCheckType::Ptr(x) = x { Some(x) } else { None })
                    .any(|x| x == input_tensor_ptr);

                if contains_ptr
                {
                    return;
                }

                if let Some(input_memory_index) = self.tensors_memory[input_tensor_ptr.0].memory
                {
                    let contains_index = self.set_tensors_check.iter()
                        .filter_map(|x| if let InputCheckType::Index(x) = x { Some(x) } else { None })
                        .any(|x| *x == input_memory_index);

                    assert!(contains_index, "{input_tensor_ptr:?} ({input_memory_index:?}) wasnt set");
                } else
                {
                    panic!("input {input_tensor_ptr:?} wasnt allocated");
                }
            });
        }

        let mut current_index = start;
        while current_index < end
        {
            let gradient_op = &self.raw_operations[current_index];
            macro_rules! copy_tensor
            {
                ($src:expr, $dst:expr) =>
                {
                    self.tensors_raw_data.copy_within($src.range(), $dst.raw_index.0)
                }
            }

            macro_rules! get_disjoint_mut
            {
                ($(($target_type:ident, $name:ident, $tmp_name:ident)),+) =>
                {
                    get_disjoint_mut_with!(self, $(($target_type, *$name, $tmp_name),)+)
                }
            }

            macro_rules! debug_calculate_common
            {
                (($($t_name:ident),*$(,)?),($($v_name:ident),*$(,)?)) =>
                {
                    #[cfg(debug_assertions)]
                    {
                        #[allow(unused_assignments)]
                        if PRINT_CALCULATE_VALUES
                        {
                            let mut is_first = true;

                            $(
                                if !is_first { eprint!(", "); }
                                eprint!("{}: {:?} = {:?}", stringify!($t_name), $t_name, &self.tensors_raw_data[$t_name.range()]);
                                is_first = false;
                            )*

                            $(
                                if !is_first { eprint!(", "); }
                                eprint!("{}: {:?} = {:?}", stringify!($v_name), $v_name, &self.values[$v_name.0]);
                                is_first = false;
                            )*
                        }
                    }
                }
            }

            macro_rules! debug_calculate_values
            {
                ($name:ident, ($($t_name:ident),*),($($v_name:ident),*)) =>
                {
                    #[cfg(debug_assertions)]
                    {
                        if PRINT_CALCULATE_VALUES
                        {
                            eprint!("GradientOp::{} (BEFORE ", stringify!($name));
                        }
                    }

                    {
                        debug_calculate_common!(($($t_name,)*),($($v_name,)*));
                    }

                    #[cfg(debug_assertions)]
                    {
                        if PRINT_CALCULATE_VALUES
                        {
                            eprint!(") (AFTER ");
                        }
                    }
                }
            }

            macro_rules! debug_calculate_values_result
            {
                (($($t_name:ident),*),($($v_name:ident),*)) =>
                {
                    {
                        debug_calculate_common!(($($t_name,)*),($($v_name,)*));
                    }

                    #[cfg(debug_assertions)]
                    {
                        if PRINT_CALCULATE_VALUES
                        {
                            eprintln!(")");
                        }
                    }
                }
            }

            //let before_instant = std::time::Instant::now();
            match gradient_op
            {
                GradientOp::None => unreachable!(),
                GradientOp::Jump(RawJumpInfo{loop_index, operation_index}) =>
                {
                    if self.loops[loop_index.0].times > 1
                    {
                        current_index = operation_index.0;

                        self.loops[loop_index.0].times -= 1;

                        continue;
                    }
                },
                GradientOp::Copy{src, dst} =>
                {
                    debug_calculate_values!(Copy, (src, dst),());
                    copy_tensor!(src, dst);
                    debug_calculate_values_result!((dst),());
                },
                GradientOp::CopyScalar{src, dst} =>
                {
                    debug_calculate_values!(CopyScalar, (),(src, dst));
                    self.values[dst.0] = self.values[src.0];
                    debug_calculate_values_result!((),(dst));
                },
                GradientOp::AddScalars{lhs, rhs, output} =>
                {
                    debug_calculate_values!(AddScalars, (),(lhs, rhs, output));
                    self.values[output.0] = self.values[lhs.0] + self.values[rhs.0];
                    debug_calculate_values_result!((),(output));
                },
                GradientOp::AddScalar{lhs, rhs, output} =>
                {
                    debug_calculate_values!(AddScalar, (lhs, output),(rhs));
                    copy_tensor!(lhs, output);

                    LayerTypeMut::from_data_with_start(&mut self.tensors_raw_data, *output).add_scalar(self.values[rhs.0]);
                    debug_calculate_values_result!((output),());
                },
                GradientOp::Add{lhs, rhs, output} =>
                {
                    debug_calculate_values!(Add, (lhs, rhs, output),());

                    {
                        let (output, lhs, rhs) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeRef, rhs, x2)
                        );

                        output.add_to(lhs, rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::AddInplace{value: _, output: _} =>
                {
                    unimplemented!()
                },
                GradientOp::Sub{lhs, rhs, output} =>
                {
                    debug_calculate_values!(Sub, (lhs, rhs, output),());

                    {
                        let (mut output, lhs, rhs) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeRef, rhs, x2)
                        );

                        output.sub_to(lhs, rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::SubFromScalar{lhs, rhs, output} =>
                {
                    debug_calculate_values!(SubFromScalar, (rhs, output),(lhs));

                    {
                        let (output, rhs) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, rhs, x1)
                        );

                        output.sub_from_scalar(self.values[lhs.0], rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::MulScalar{lhs, rhs, output} =>
                {
                    debug_calculate_values!(MulScalar, (lhs, output),(rhs));
                    copy_tensor!(lhs, output);

                    LayerTypeMut::from_data_with_start(&mut self.tensors_raw_data, *output).scale(self.values[rhs.0]);
                    debug_calculate_values_result!((output),());
                },
                GradientOp::MulScalars{lhs, rhs, output} =>
                {
                    debug_calculate_values!(MulScalars, (),(lhs, rhs, output));
                    self.values[output.0] = self.values[lhs.0] * self.values[rhs.0];
                    debug_calculate_values_result!((),(output));
                },
                GradientOp::MulComponentwise{lhs, rhs, output} =>
                {
                    debug_calculate_values!(MulComponentwise, (lhs, rhs, output),());

                    {
                        let (output, lhs, rhs) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeRef, rhs, x2)
                        );

                        output.component_mul_into(lhs, rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::MulComponentwiseAdd{lhs, rhs, added, output} =>
                {
                    debug_calculate_values!(MulComponentwise, (lhs, rhs, added, output),());

                    {
                        let (output, lhs, rhs, added) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeRef, rhs, x2),
                            (LayerTypeRef, added, x3)
                        );

                        output.component_mul_add_into(lhs, rhs, added);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::SumTensor{value, output} =>
                {
                    debug_calculate_values!(SumTensor, (value),(output));
                    self.values[output.0] = self.tensors_raw_data[value.range()].iter().sum();
                    debug_calculate_values_result!((),(output));
                },
                GradientOp::Dot{lhs, rhs, output} =>
                {
                    debug_calculate_values!(Dot, (lhs, rhs),(output));
                    let lhs = LayerTypeRef::from_data_with_start(&self.tensors_raw_data, *lhs);
                    let rhs = LayerTypeRef::from_data_with_start(&self.tensors_raw_data, *rhs);

                    self.values[output.0] = lhs.dot(rhs);
                    debug_calculate_values_result!((),(output));
                },
                GradientOp::Fill{value, output} =>
                {
                    debug_calculate_values!(Fill, (output),(value));
                    self.tensors_raw_data[output.range()].fill(self.values[value.0]);
                    debug_calculate_values_result!((output),());
                },
                GradientOp::Pow{lhs, power, output} =>
                {
                    debug_calculate_values!(Pow, (lhs, output),());
                    copy_tensor!(lhs, output);

                    LayerTypeMut::from_data_with_start(&mut self.tensors_raw_data, *output).pow_inplace(*power);
                    debug_calculate_values_result!((output),());
                },
                GradientOp::Sigmoid{value, output} =>
                {
                    debug_calculate_values!(Sigmoid, (value, output),());
                    copy_tensor!(value, output);

                    LayerTypeMut::from_data_with_start(&mut self.tensors_raw_data, *output).sigmoid_inplace();
                    debug_calculate_values_result!((output),());
                },
                GradientOp::SigmoidDiff{value, gradient, output} =>
                {
                    debug_calculate_values!(SigmoidDiff, (value, gradient, output),());

                    {
                        let (output, value, gradient) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, value, x1),
                            (LayerTypeRef, gradient, x2)
                        );

                        output.sigmoid_gradient_inplace(value, gradient);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::Tanh{value, output} =>
                {
                    debug_calculate_values!(Tanh, (value, output),());
                    copy_tensor!(value, output);

                    LayerTypeMut::from_data_with_start(&mut self.tensors_raw_data, *output).tanh_inplace();
                    debug_calculate_values_result!((output),());
                },
                GradientOp::TanhDiff{value, gradient, output} =>
                {
                    debug_calculate_values!(TanhDiff, (value, gradient, output),());

                    {
                        let (output, value, gradient) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, value, x1),
                            (LayerTypeRef, gradient, x2)
                        );

                        output.tanh_gradient_inplace(value, gradient);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::LeakyRelu{value, output} =>
                {
                    debug_calculate_values!(LeakyRelu, (value, output),());
                    copy_tensor!(value, output);

                    LayerTypeMut::from_data_with_start(&mut self.tensors_raw_data, *output).leaky_relu_inplace();
                    debug_calculate_values_result!((output),());
                },
                GradientOp::LeakyReluDiff{value, gradient, output} =>
                {
                    debug_calculate_values!(LeakyReluDiff, (value, gradient, output),());

                    {
                        let (output, value, gradient) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, value, x1),
                            (LayerTypeRef, gradient, x2)
                        );

                        output.leaky_relu_gradient_inplace(value, gradient);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::SoftmaxCrossEntropy{values, targets, softmaxed_output, output} =>
                {
                    debug_calculate_values!(SoftmaxCrossEntropy, (values, softmaxed_output),(output));

                    {
                        copy_tensor!(values, softmaxed_output);

                        let softmaxed_output = LayerTypeMut::from_data_with_start(&mut self.tensors_raw_data, *softmaxed_output);

                        self.values[output.0] = softmaxed_output.softmax_cross_entropy_inplace(&self.one_hot_layers[targets.0]);
                    }

                    debug_calculate_values_result!((softmaxed_output),(output));
                },
                GradientOp::SoftmaxCrossEntropyNoSoftmaxed{values, targets, output} =>
                {
                    debug_calculate_values!(SoftmaxCrossEntropyNoSoftmaxed, (values),(output));
                    let values = LayerTypeRef::from_data_with_start(&self.tensors_raw_data, *values);

                    self.values[output.0] = values.softmax_cross_entropy(&self.one_hot_layers[targets.0]);
                    debug_calculate_values_result!((),(output));
                },
                GradientOp::SoftmaxCrossEntropyDiff{softmaxed_values, gradient, targets, output} =>
                {
                    debug_calculate_values!(SoftmaxCrossEntropyDiff, (softmaxed_values, output),(gradient));

                    {
                        debug_assert_eq!(
                            (softmaxed_values.rows, softmaxed_values.columns), (self.one_hot_layers[targets.0].size, 1),
                            "softmaxed: {softmaxed_values:?}, targets: {targets:?}"
                        );

                        let optimize_this = ();

                        let (mut output, softmaxed_values) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, softmaxed_values, x1)
                        );

                        output.sub_to(softmaxed_values, MatrixWrapperRef::from(&self.one_hot_layers[targets.0].clone().into_layer()));

                        output.scale(self.values[gradient.0]);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::Matmulv{lhs, rhs, output} =>
                {
                    debug_calculate_values!(Matmulv, (lhs, rhs, output),());

                    {
                        let (output, lhs, rhs) = get_disjoint_mut!(
                            (LayerTypeVectorMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeVector, rhs, x2)
                        );

                        output.matmulv_into(lhs, rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::MatmulvAdd{lhs, rhs, added, output} =>
                {
                    debug_calculate_values!(MatmulvAdd, (lhs, rhs, added, output),());

                    {
                        let (output, lhs, rhs, added) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeRef, rhs, x2),
                            (LayerTypeRef, added, x3)
                        );

                        output.matmulv_add_into(lhs, rhs, added);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::MatmulOneHotvAdd{lhs, rhs, added, output} =>
                {
                    debug_calculate_values!(MatmulOneHotvAdd, (lhs, added, output),());

                    {
                        let (output, lhs, added) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeRef, added, x2)
                        );

                        output.matmul_onehotv_add_into(lhs, &self.one_hot_layers[rhs.0], added);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::MatmulvTransposed{lhs, rhs, output} =>
                {
                    debug_calculate_values!(MatmulvTransposed, (lhs, rhs, output),());

                    {
                        let (output, lhs, rhs) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeRef, rhs, x2)
                        );

                        output.matmulv_transposed_into(lhs, rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::OuterProduct{lhs, rhs, output} =>
                {
                    debug_calculate_values!(OuterProduct, (lhs, rhs, output),());

                    {
                        let (output, lhs, rhs) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeRef, rhs, x2)
                        );

                        output.outer_product_into(lhs, rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::OuterProductOneHot{lhs, rhs, output} =>
                {
                    debug_calculate_values!(OuterProductOneHot, (lhs, output),());

                    {
                        let (output, lhs) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, lhs, x1)
                        );

                        output.outer_product_one_hot_into(lhs, &self.one_hot_layers[rhs.0]);
                    }

                    debug_calculate_values_result!((output),());
                }
            }

            //eprintln!("{}, elapsed {:.3} us",format!("{gradient_op:?}").split(' ').next().unwrap(),before_instant.elapsed().as_nanos()as f64/1000.0);

            current_index += 1;
        }
    }

    pub fn finish(&mut self)
    {
        debug_assert_eq!(self.state, RecorderState::Recording);

        self.gradient_operations = Vec::new();

        fn handle_op(target: &mut Vec<GradientOp<TensorPtr, JumpInfo>>, op: &Op)
        {
            let new_op = match op
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
                },
                Op::Loop{index, inputs, ops} =>
                {
                    target.push(GradientOp::Jump(JumpInfo::Target{inputs: inputs.clone(), index: *index}));

                    ops.iter().for_each(|inner_op|
                    {
                        handle_op(target, inner_op);
                    });

                    target.push(GradientOp::Jump(JumpInfo::Source(*index)));

                    return;
                }
            };

            target.push(new_op);
        }

        self.recording_operations.iter().for_each(|op| handle_op(&mut self.gradient_operations, op));

        self.feedforward_operations_count = self.gradient_operations.len();

        self.state = RecorderState::AwaitingGradient;
    }

    fn is_ptr_output(&self, ptr: TensorPtr) -> bool
    {
        self.tensor_live_ranges[ptr.0].end == Some(i32::MAX)
    }

    fn copy_coalesce(&mut self)
    {
        let mut i = 0;
        while i < self.gradient_operations.len()
        {
            if let GradientOp::Copy{src, dst} = self.gradient_operations[i]
            {
                let dst_is_output = self.is_ptr_output(dst);

                if !dst_is_output
                {
                    let mut overlaps_args = false;

                    // dst can only appear starting from this clone so no need to check before it
                    for check_op in &self.gradient_operations[(i + 1)..]
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
                        for check_op in self.gradient_operations[(i + 1)..].iter_mut()
                        {
                            *check_op = check_op.clone().map_args(|arg| if arg == dst { src } else { arg }, convert::identity);
                        }

                        self.remove_gradient_operation(i);

                        continue;
                    }
                }
            }

            i += 1;
        }
    }

    fn combine_ops(&mut self)
    {
        let mut i = 1;
        while i < self.gradient_operations.len()
        {
            let previous_i = i - 1;

            let previous = self.gradient_operations[previous_i].clone();
            let this = self.gradient_operations[i].clone();
            match (previous, this)
            {
                (
                    GradientOp::MulComponentwise{output, lhs: mul_lhs, rhs: mul_rhs},
                    GradientOp::Add{lhs, rhs, output: add_output}
                ) if ((output == lhs && mul_lhs != rhs && mul_rhs != rhs) || (output == rhs && mul_lhs != lhs && mul_rhs != lhs))
                    && !self.is_ptr_output(output) =>
                {
                    let mut used_after = false;

                    for check_op in &self.gradient_operations[(i + 1)..]
                    {
                        check_op.for_args(|v| if v == output { used_after = true }, |_| {});

                        if used_after
                        {
                            break;
                        }
                    }

                    if !used_after
                    {
                        let added = if output == lhs
                        {
                            rhs
                        } else if output == rhs
                        {
                            lhs
                        } else
                        {
                            unreachable!()
                        };

                        self.gradient_operations[previous_i] = GradientOp::MulComponentwiseAdd{
                            lhs: mul_lhs,
                            rhs: mul_rhs,
                            added,
                            output: add_output
                        };

                        self.remove_gradient_operation(i);

                        continue;
                    }
                },
                _ => ()
            }

            i += 1;
        }
    }

    pub fn is_ready(&self) -> bool
    {
        self.state == RecorderState::Ready
    }

    pub fn tensors_raw_data(&self) -> &[f32]
    {
        &self.tensors_raw_data
    }

    fn calculate_live_ranges_once(&mut self) -> bool
    {
        self.gradient_operations.iter().enumerate().rev().for_each(|(op_index, op)|
        {
            let handle_output = |live_range: &mut LiveRange, err_name: String|
            {
                let start = &mut live_range.start;

                debug_assert!(start.is_none(), "{err_name} was reused at operation {} and {op_index}", start.unwrap());

                *start = Some(op_index as i32);
            };

            let mut tensor_ptrs = Vec::new();
            op.for_outputs(|tensor_ptr|
            {
                tensor_ptrs.push(tensor_ptr);
            }, |value_index|
            {
                handle_output(&mut self.value_live_ranges[value_index.0], format!("{value_index:?}"));
            });

            tensor_ptrs.into_iter().for_each(|tensor_ptr|
            {
                handle_output(&mut self.tensor_live_ranges[tensor_ptr.0], format!("{tensor_ptr:?}"));
            });

            let handle_arg = |live_range: &mut LiveRange, err_name: String|
            {
                if let Some(start) = live_range.start
                {
                    if start >= op_index as i32
                    {
                        panic!("{err_name} was defined at {start} after being used at {op_index}");
                    }
                }

                if live_range.end.is_none()
                {
                    live_range.end = Some(op_index as i32);
                }
            };

            op.for_args(|tensor_ptr|
            {
                handle_arg(&mut self.tensor_live_ranges[tensor_ptr.0], format!("{tensor_ptr:?}"));
            }, |value_index|
            {
                handle_arg(&mut self.value_live_ranges[value_index.0], format!("{value_index:?}"));
            });
        });

        let mut any_unused = false;
        self.gradient_operations.iter_mut().for_each(|op|
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
                handle_output(&mut self.value_live_ranges[value_index.0], format!("{value_index:?}"));
            });

            tensor_ptrs.into_iter().for_each(|tensor_ptr|
            {
                handle_output(&mut self.tensor_live_ranges[tensor_ptr.0], format!("{tensor_ptr:?}"));
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
        let tensor_live_ranges = self.tensor_live_ranges.clone();
        let value_live_ranges = self.value_live_ranges.clone();

        loop
        {
            let any_unused = self.calculate_live_ranges_once();

            if !any_unused
            {
                break;
            }

            self.tensor_live_ranges = tensor_live_ranges.clone();
            self.value_live_ranges = value_live_ranges.clone();
        }
    }

    fn greedy_graph_color(&mut self, memory_assignments: &mut Vec<TensorMemoryValue>)
    {
        let nodes_count = self.tensor_live_ranges.len();

        let mut graph_connections: Vec<Vec<usize>> = iter::from_fn(|| Some(Vec::new()))
            .take(nodes_count)
            .collect();

        let verify_range = |range: &LiveRange, index|
        {
            if let Some(end) = range.end
            {
                if range.start.is_none()
                {
                    panic!("TensorPtr({index}) was used at OperationIndex({end}) but never set");
                }
            }
        };

        (0..nodes_count).for_each(|node_index|
        {
            let this_range = &self.tensor_live_ranges[node_index];

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
                    let other_range = &self.tensor_live_ranges[check_index];

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

        let mut connections_count_sorted: Vec<usize> = (0..nodes_count).collect();

        let kf = |node_index: &usize| graph_connections[*node_index].len();

        #[cfg(debug_assertions)]
        {
            connections_count_sorted.sort_by_key(kf);
        }

        #[cfg(not(debug_assertions))]
        {
            connections_count_sorted.sort_unstable_by_key(kf);
        }

        connections_count_sorted.reverse();

        connections_count_sorted.into_iter().for_each(|node_index|
        {
            if self.tensors_memory[node_index].memory.is_some()
            {
                return;
            }

            if self.tensor_live_ranges[node_index].end.is_none()
            {
                return;
            }

            let this_color = if NO_COLORING
            {
                memory_assignments.len()
            } else
            {
                (0..).find(|color|
                {
                    let all_connected_unconflicted = graph_connections[node_index].iter().all(|connected_node_index|
                    {
                        let connected_node_color: Option<usize> = self.tensors_memory[*connected_node_index].memory.map(|x| x.0);

                        connected_node_color != Some(*color)
                    });

                    let spot_size_matches = memory_assignments.get(*color).map(|spot_tensor|
                    {
                        spot_tensor.tensor_shape() == self.tensors_memory[node_index].value.tensor_shape()
                    }).unwrap_or(true);

                    all_connected_unconflicted && spot_size_matches
                }).unwrap()
            };

            if this_color == memory_assignments.len()
            {
                memory_assignments.push(self.tensors_memory[node_index].value.clone());
            } else if let TensorMemoryValue::Value(x) = &self.tensors_memory[node_index].value
            {
                memory_assignments[this_color] = TensorMemoryValue::Value(x.clone());
            }

            debug_assert!(self.tensors_memory[node_index].memory.is_none(), "tried to replace slot of TensorPtr({node_index})");
            self.tensors_memory[node_index].memory = Some(TensorIndex(this_color));
        });
    }

    fn operations_to_raw(&mut self, memory_assignments: &mut Vec<TensorMemoryValue>)
    {
        let mut usage_counts: Vec<(TensorIndex, usize, Vec<(TensorIndex, usize)>)> = (0..self.tensors.len())
            .map(|x| (TensorIndex(x), 0, Vec::new()))
            .collect();

        self.gradient_operations.iter().for_each(|op|
        {
            let mut local = Vec::new();

            let mut f_local = |ptr: TensorPtr|
            {
                if let Some(this_index) = self.tensors_memory[ptr.0].memory
                {
                    local.push(this_index);
                }
            };

            op.for_args(&mut f_local, |_| {});
            op.for_outputs(f_local, |_| {});

            let mut f = |ptr: TensorPtr|
            {
                if let Some(this_index) = self.tensors_memory[ptr.0].memory
                {
                    usage_counts[this_index.0].1 += 1;
                    let this_local = &mut usage_counts[this_index.0].2;

                    local.iter().for_each(|local|
                    {
                        if let Some(local_total) = this_local.iter_mut().find(|x| x.0 == *local)
                        {
                            local_total.1 += 1;
                        } else
                        {
                            this_local.push((*local, 1));
                        }
                    });
                }
            };

            op.for_args(&mut f, |_| {});
            op.for_outputs(f, |_| {});
        });

        usage_counts.sort_by_key(|x| x.1);

        let mut create_tensor = |this_index: TensorIndex|
        {
            if self.tensors[this_index.0] != TensorRawDataPointer::undefined()
            {
                return;
            }

            let this_tensor: &TensorMemoryValue = &memory_assignments[this_index.0];

            let (rows, columns) = this_tensor.tensor_shape();
            let size = rows * columns;

            let id = TensorIndexRaw(self.tensors_raw_data.len());

            match this_tensor
            {
                TensorMemoryValue::Value(x) => self.tensors_raw_data.extend(x.as_slice()),
                TensorMemoryValue::Size{..} => self.tensors_raw_data.resize(self.tensors_raw_data.len() + size, 0.0)
            }

            debug_assert_eq!(self.tensors[this_index.0], TensorRawDataPointer::undefined());
            self.tensors[this_index.0] = TensorRawDataPointer{
                raw_index: id,
                rows,
                columns
            };
        };

        usage_counts.into_iter().rev().for_each(|(this_index, _, local)|
        {
            create_tensor(this_index);

            let mut local_sorted: Vec<(TensorIndex, usize)> = local.iter().map(|(a, b)| (*a, *b)).filter(|(x, _)| *x != this_index).collect();
            local_sorted.sort_by_key(|x| x.1);

            local_sorted.into_iter().rev().map(|(x, _)| x).for_each(&mut create_tensor)
        });

        let access_tensor = |ptr: TensorPtr| -> TensorRawDataPointer
        {
            let this_index: TensorIndex = self.tensors_memory[ptr.0].memory.expect("must be resolved");

            let current_value = self.tensors[this_index.0];

            debug_assert_ne!(current_value, TensorRawDataPointer::undefined());

            current_value
        };

        let mut loops_labels: Vec<(LoopIndex, GradientOperationIndex)> = Vec::new();

        self.raw_operations.reserve_exact(self.gradient_operations.len());

        for (index, gradient_op) in mem::take(&mut self.gradient_operations).into_iter().enumerate()
        {
            let operation_index = GradientOperationIndex(self.raw_operations.len());

            let new_op = match gradient_op
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
                        values: access_tensor(values),
                        targets,
                        output
                    })
                },
                x =>
                {
                    let mut ignore_output = false;

                    let loops = &mut self.loops;
                    let output = x.map_tensors(&access_tensor, |jump_info|
                    {
                        match jump_info
                        {
                            JumpInfo::Target{inputs, index} =>
                            {
                                loops[index.0].inputs = inputs.into_iter().map(|x|
                                {
                                    match x
                                    {
                                        InputTypePtr::Normal(x) => InputType::Normal(self.tensors_memory[x.0].memory.expect("must be resolved")),
                                        InputTypePtr::OneHot(x) => InputType::OneHot(x)
                                    }
                                }).collect();

                                loops_labels.push((index, GradientOperationIndex(operation_index.0 + 1)));

                                ignore_output = true;

                                RawJumpInfo{loop_index: LoopIndex(usize::MAX), operation_index: GradientOperationIndex(usize::MAX)}
                            },
                            JumpInfo::Source(index) =>
                            {
                                let target_index: GradientOperationIndex = loops_labels.iter().find(|(loop_index, _)| index == *loop_index)
                                    .expect("loop must be defined before being used")
                                    .1;

                                RawJumpInfo{loop_index: index, operation_index: target_index}
                            }
                        }
                    });

                    (!ignore_output).then_some(output)
                }
            };

            if let Some(raw_op) = new_op
            {
                self.raw_operations.push(raw_op);
            }

            if (index + 1) == self.feedforward_operations_count
            {
                self.feedforward_operations_count = self.raw_operations.len();
            }
        }
    }

    pub fn resolve_memory(&mut self)
    {
        debug_assert_eq!(self.state, RecorderState::AwaitingResolve);

        self.calculate_live_ranges();

        let mut memory_assignments = Vec::new();
        self.greedy_graph_color(&mut memory_assignments);

        self.tensors.resize(memory_assignments.len(), TensorRawDataPointer::undefined());

        if OPT_INFO
        {
            eprintln!("using {} memory spots", self.tensors.len());
        }

        #[cfg(debug_assertions)]
        {
            let mut new_store_tensors_check = Vec::new();

            self.store_tensors_check.iter().for_each(|k|
            {
                let this_index: TensorIndex = match *k
                {
                    StoreCheckKey::PreResolve(ptr) => self.tensors_memory[ptr.0].memory.expect("must be resolved"),
                    StoreCheckKey::Resolved(_) => unreachable!()
                };

                let new_k = StoreCheckKey::Resolved(this_index);

                // new_k has overlap with other ptrs so this check will give some false negatives
                // disable graph coloring for an exact check

                if !new_store_tensors_check.contains(&new_k)
                {
                    new_store_tensors_check.push(new_k);
                }
            });

            self.store_tensors_check = new_store_tensors_check;

            self.store_values_check = self.store_values_check.iter().map(|k|
            {
                match *k
                {
                    StoreCheckKey::PreResolve(x) => StoreCheckKey::Resolved(x),
                    StoreCheckKey::Resolved(_) => unreachable!()
                }
            }).collect();
        }

        self.operations_to_raw(&mut memory_assignments);

        #[cfg(debug_assertions)]
        {
            self.raw_operations.iter().for_each(|op|
            {
                let mut tensor_args = Vec::new();
                let mut value_args = Vec::new();

                op.for_args(|t_arg| tensor_args.push(t_arg), |v_arg| value_args.push(v_arg));

                op.for_outputs(|t_out|
                {
                    debug_assert!(!tensor_args.contains(&t_out), "{op:?} has overlap between args and outputs")
                }, |v_out|
                {
                    debug_assert!(!value_args.contains(&v_out), "{op:?} has overlap between args and outputs")
                });
            });

            self.tensor_inputs = self.tensor_live_ranges.iter().enumerate()
                .filter(|(_, x)| x.start == Some(-1))
                .map(|(index, _)| TensorPtr(index))
                .collect();
        }

        self.value_live_ranges = Vec::new();
        self.tensor_live_ranges = Vec::new();

        self.state = RecorderState::Ready;
    }

    pub fn no_gradient(&mut self)
    {
        debug_assert_eq!(self.state, RecorderState::AwaitingGradient);

        self.copy_coalesce();

        self.combine_ops();

        self.recording_operations = Vec::new();

        self.state = RecorderState::AwaitingResolve;
    }

    pub fn gradient_with_respect(&mut self, respect: DiffWrapper)
    {
        debug_assert_eq!(self.state, RecorderState::AwaitingGradient);

        self.set_ones(respect);

        {
            let mut assigned_gradients = Vec::new();
            let mut next_nodes: VecDeque<DiffWrapper> = VecDeque::from(vec![respect]);

            while let Some(current_node) = next_nodes.pop_front()
            {
                self.calculate_gradient(&mut assigned_gradients, &mut next_nodes, current_node);
            }
        }

        self.copy_coalesce();

        self.combine_ops();

        self.recording_operations = Vec::new();

        self.state = RecorderState::AwaitingResolve;
    }

    fn calculate_gradient(
        &mut self,
        assigned_gradients: &mut Vec<(DiffValue, GradientOperationIndex)>,
        next_nodes: &mut VecDeque<DiffWrapper>,
        respect: DiffWrapper
    )
    {
        let (gradient, source) = if let Some(x) = respect.destructure() { x } else { return; };

        if let Some(source) = source
        {
            self.calculate_gradient_op(assigned_gradients, next_nodes, gradient, source, false)
        }
    }

    fn calculate_gradient_op(
        &mut self,
        assigned_gradients: &mut Vec<(DiffValue, GradientOperationIndex)>,
        next_nodes: &mut VecDeque<DiffWrapper>,
        gradient: DiffValue,
        source: OperationIndex,
        is_nested: bool
    )
    {
        let mut add_gradient_operation = |this: &mut Self, output: DiffValue, gradient_op: GradientOp<TensorPtr, JumpInfo>|
        {
            if let Some((_, previous_operation_index)) = assigned_gradients.iter_mut().find(|(value, _)| *value == output)
            {
                let previous_op = this.gradient_operations[previous_operation_index.0].clone();

                let new_id = match output
                {
                    DiffValue::Tensor(output) =>
                    {
                        let mut handle_tensor = |new_tensor_out: &mut Option<TensorPtr>, x: TensorPtr| -> TensorPtr
                        {
                            let (rows, columns) = tensor_shape!(this, x);

                            let new_tensor = new_tensor_index!(this, rows, columns);

                            *new_tensor_out = Some(new_tensor);

                            new_tensor
                        };

                        let mut lhs = None;
                        this.gradient_operations[previous_operation_index.0] = previous_op.map_outputs(|x|
                        {
                            handle_tensor(&mut lhs, x)
                        }, convert::identity);

                        let mut rhs = None;
                        this.gradient_operations.push(gradient_op.map_outputs(|x|
                        {
                            handle_tensor(&mut rhs, x)
                        }, convert::identity));

                        let new_id = this.gradient_operations.len();

                        this.gradient_operations.push(GradientOp::Add{lhs: lhs.unwrap(), rhs: rhs.unwrap(), output});

                        new_id
                    },
                    DiffValue::Value(output) =>
                    {
                        let mut handle_scalar = |new_value_out: &mut Option<ValueIndex>, x: ValueIndex| -> ValueIndex
                        {
                            let new_value = new_value_index!(this);

                            *new_value_out = Some(new_value);

                            new_value
                        };

                        let mut lhs = None;
                        this.gradient_operations[previous_operation_index.0] = previous_op.map_outputs(convert::identity, |x|
                        {
                            handle_scalar(&mut lhs, x)
                        });

                        let mut rhs = None;
                        this.gradient_operations.push(gradient_op.map_outputs(convert::identity, |x|
                        {
                            handle_scalar(&mut rhs, x)
                        }));

                        let new_id = this.gradient_operations.len();

                        this.gradient_operations.push(GradientOp::AddScalars{lhs: lhs.unwrap(), rhs: rhs.unwrap(), output});

                        new_id
                    }
                };

                *previous_operation_index = GradientOperationIndex(new_id);
            } else
            {
                let this_op_index = GradientOperationIndex(this.gradient_operations.len());
                this.gradient_operations.push(gradient_op);

                assigned_gradients.push((output, this_op_index));
            }
        };

        let this_operation = &self.recording_operations[source.0];

        let this_operation = if is_nested
        {
            if let Op::Loop{ops, ..} = this_operation
            {
                ops[source.1.unwrap()].clone()
            } else
            {
                unreachable!()
            }
        } else
        {
            this_operation.clone()
        };

        match this_operation
        {
            Op::Add{lhs, ..}
            | Op::AddScalar{lhs, ..} =>
            {
                let gradient = gradient.as_tensor();

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, lhs_gradient.into(), GradientOp::Copy{src: gradient, dst: lhs_gradient});
                }

                if let Op::Add{rhs, ..} = this_operation
                {
                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        add_gradient_operation(self, rhs_gradient.into(), GradientOp::Copy{src: gradient, dst: rhs_gradient});
                    }
                } else if let Op::AddScalar{rhs, ..} = this_operation
                {
                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        add_gradient_operation(self, rhs_gradient.into(), GradientOp::SumTensor{value: gradient, output: rhs_gradient});
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

                next_nodes.push_back(rhs);
                next_nodes.push_back(lhs.into());
            },
            Op::AddScalars{lhs, rhs, output: _} =>
            {
                let gradient = gradient.as_value();

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, lhs_gradient.into(), GradientOp::CopyScalar{src: gradient, dst: lhs_gradient});
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    add_gradient_operation(self, rhs_gradient.into(), GradientOp::CopyScalar{src: gradient, dst: rhs_gradient});
                }

                next_nodes.push_back(lhs.into());
                next_nodes.push_back(rhs.into());
            },
            Op::Sub{rhs, ..}
            | Op::SubFromScalar{rhs, ..} =>
            {
                let gradient = gradient.as_tensor();

                if let Op::Sub{lhs, ..} = this_operation
                {
                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        add_gradient_operation(self, lhs_gradient.into(), GradientOp::Copy{src: gradient, dst: lhs_gradient});
                    }
                } else if let Op::SubFromScalar{lhs, ..} = this_operation
                {
                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        add_gradient_operation(self, lhs_gradient.into(), GradientOp::SumTensor{value: gradient, output: lhs_gradient});
                    }
                } else
                {
                    unreachable!()
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    let m1_index = new_value_index!(self);
                    self.values[m1_index.0] = -1.0;

                    add_gradient_operation(self, rhs_gradient.into(), GradientOp::MulScalar{lhs: gradient, rhs: m1_index, output: rhs_gradient});
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

                next_nodes.push_back(lhs);
                next_nodes.push_back(rhs.into());
            },
            Op::MulScalars{lhs, rhs, output: _} =>
            {
                let gradient = gradient.as_value();

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, lhs_gradient.into(), GradientOp::MulScalars{lhs: rhs.as_value(), rhs: gradient, output: lhs_gradient});
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    add_gradient_operation(self, rhs_gradient.into(), GradientOp::MulScalars{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                }

                next_nodes.push_back(rhs.into());
                next_nodes.push_back(lhs.into());
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
                        add_gradient_operation(self, lhs_gradient.into(), GradientOp::MulComponentwise{lhs: rhs.as_value(), rhs: gradient, output: lhs_gradient});
                    } else if let Op::MulScalar{rhs, ..} = this_operation
                    {
                        add_gradient_operation(self, lhs_gradient.into(), GradientOp::MulScalar{lhs: gradient, rhs: rhs.as_value(), output: lhs_gradient});
                    } else
                    {
                        unreachable!()
                    }
                }

                if let Op::MulComponentwise{rhs, ..} = this_operation
                {
                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        add_gradient_operation(self, rhs_gradient.into(), GradientOp::MulComponentwise{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                    }
                } else if let Op::MulScalar{rhs, ..} = this_operation
                {
                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        let pre_fold = new_tensor_index!(self, rows, columns);
                        self.gradient_operations.push(GradientOp::MulComponentwise{lhs: lhs.as_value(), rhs: gradient, output: pre_fold});

                        add_gradient_operation(self, rhs_gradient.into(), GradientOp::SumTensor{value: pre_fold, output: rhs_gradient});
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

                next_nodes.push_back(rhs);
                next_nodes.push_back(lhs.into());
            },
            Op::SumTensor{value, output: _} =>
            {
                let gradient = gradient.as_value();

                if let Some(value_gradient) = value.as_gradient()
                {
                    add_gradient_operation(self, value_gradient.into(), GradientOp::Fill{value: gradient, output: value_gradient});

                    next_nodes.push_back(value.into());
                }
            },
            Op::Dot{lhs, rhs, output: _} =>
            {
                let gradient = gradient.as_value();

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, lhs_gradient.into(), GradientOp::MulScalar{lhs: rhs.as_value(), rhs: gradient, output: lhs_gradient});
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    add_gradient_operation(self, rhs_gradient.into(), GradientOp::MulScalar{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                }

                next_nodes.push_back(rhs.into());
                next_nodes.push_back(lhs.into());
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
                    self.gradient_operations.push(GradientOp::Pow{lhs: lhs.as_value(), power: (power - 1) as u32, output: pow_d_lhs});

                    let pow_d = new_tensor_index!(self, rows, columns);
                    self.gradient_operations.push(GradientOp::MulScalar{lhs: pow_d_lhs, rhs: power_index.into(), output: pow_d});

                    add_gradient_operation(self, lhs_gradient.into(), GradientOp::MulComponentwise{lhs: pow_d, rhs: gradient, output: lhs_gradient});

                    next_nodes.push_back(lhs.into());
                }
            },
            Op::Sigmoid{value, output} =>
            {
                // sigmoid(x) * (1.0 - sigmoid(x))
                let gradient = gradient.as_tensor();

                if let Some(value_gradient) = value.as_gradient()
                {
                    add_gradient_operation(self, value_gradient.into(), GradientOp::SigmoidDiff{value: output.as_value(), gradient, output: value_gradient});

                    next_nodes.push_back(value.into());
                }
            },
            Op::Tanh{value, output} =>
            {
                // 1 - tanh^2(x)
                let gradient = gradient.as_tensor();

                if let Some(value_gradient) = value.as_gradient()
                {
                    add_gradient_operation(self, value_gradient.into(), GradientOp::TanhDiff{value: output.as_value(), gradient, output: value_gradient});

                    next_nodes.push_back(value.into());
                }
            },
            Op::LeakyRelu{value, output: _} =>
            {
                let gradient = gradient.as_tensor();

                if let Some(value_gradient) = value.as_gradient()
                {
                    add_gradient_operation(self, value_gradient.into(), GradientOp::LeakyReluDiff{value: value.as_value(), gradient, output: value_gradient});

                    next_nodes.push_back(value.into());
                }
            },
            Op::SoftmaxCrossEntropy{values, targets, softmaxed_output, output: _} =>
            {
                let gradient = gradient.as_value();

                if let Some(values_gradient) = values.as_gradient()
                {
                    add_gradient_operation(self, values_gradient.into(), GradientOp::SoftmaxCrossEntropyDiff{
                        softmaxed_values: softmaxed_output.as_value(),
                        gradient,
                        targets: targets.clone(),
                        output: values_gradient
                    });

                    next_nodes.push_back(values.into());
                }
            },
            Op::Matmulv{lhs, rhs, output: _} =>
            {
                let gradient = gradient.as_tensor();

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, lhs_gradient.into(), GradientOp::OuterProduct{lhs: gradient, rhs: rhs.as_value(), output: lhs_gradient});
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    add_gradient_operation(self, rhs_gradient.into(), GradientOp::MatmulvTransposed{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                }

                next_nodes.push_back(lhs.into());
                next_nodes.push_back(rhs.into());
            },
            Op::MatmulvAdd{lhs, rhs, added, output: _} =>
            {
                let gradient = gradient.as_tensor();

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, lhs_gradient.into(), GradientOp::OuterProduct{lhs: gradient, rhs: rhs.as_value(), output: lhs_gradient});
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    add_gradient_operation(self, rhs_gradient.into(), GradientOp::MatmulvTransposed{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                }

                if let Some(added_gradient) = added.as_gradient()
                {
                    add_gradient_operation(self, added_gradient.into(), GradientOp::Copy{src: gradient, dst: added_gradient});
                }

                next_nodes.push_back(lhs.into());
                next_nodes.push_back(rhs.into());
                next_nodes.push_back(added.into());
            },
            Op::MatmulOneHotvAdd{lhs, rhs, added, output: _} =>
            {
                let gradient = gradient.as_tensor();

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, lhs_gradient.into(), GradientOp::OuterProductOneHot{lhs: gradient, rhs: rhs, output: lhs_gradient});
                }

                if let Some(added_gradient) = added.as_gradient()
                {
                    add_gradient_operation(self, added_gradient.into(), GradientOp::Copy{src: gradient, dst: added_gradient});
                }

                next_nodes.push_back(lhs.into());
                next_nodes.push_back(added.into());
            },
            Op::Loop{index, inputs, ..} =>
            {
                self.gradient_operations.push(GradientOp::Jump(JumpInfo::Target{inputs, index}));

                debug_assert!(source.1.is_some());

                //let previous_assigned = assigned_gradients.clone();

                let mut loop_nodes = VecDeque::new();

                self.calculate_gradient_op(assigned_gradients, &mut loop_nodes, gradient, source, true);

                while let Some(loop_node) = loop_nodes.pop_front()
                {
                    if let Some((loop_node_gradient, loop_node_source)) = loop_node.destructure()
                    {
                        if let Some(loop_node_source@OperationIndex(_, Some(_))) = loop_node_source
                        {
                            self.calculate_gradient_op(assigned_gradients, &mut loop_nodes, loop_node_gradient, loop_node_source, true);
                        } else
                        {
                            next_nodes.push_back(loop_node);
                        }
                    }
                }

                // new gradients created inside the loop which will get overriden with each new run
                // could be useful to check that theyre not important!
                /*assigned_gradients.iter().filter(|(value, _)| !previous_assigned.iter().any(|(x, _)| x == value))
                    .for_each(|(new_value, new_position)|
                    {
                        eprintln!("{new_value:?} {new_position:?}");
                    });*/

                self.gradient_operations.push(GradientOp::Jump(JumpInfo::Source(index)));
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorPtr(usize);

impl TensorPtr
{
    pub fn from_raw(x: usize) -> Self { Self(x) }

    pub fn undefined() -> Self { Self(usize::MAX) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorIndex(usize);

impl TensorIndex
{
    pub fn from_raw(x: usize) -> Self { Self(x) }

    pub fn undefined() -> Self { Self(usize::MAX) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct TensorIndexRaw(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
struct OperationIndex(usize, Option<usize>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct LoopOperationIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct GradientOperationIndex(usize);

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

    pub fn clear_gradient(&mut self)
    {
        self.gradient = None;
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

impl DiffWrapper
{
    fn destructure(self) -> Option<(DiffValue, Option<OperationIndex>)>
    {
        match self
        {
            DiffWrapper::Tensor(DiffTensorPtr{index: _, gradient, source}) =>
            {
                let gradient = if let Some(x) = gradient { x } else { return None; };

                Some((gradient.into(), source))
            },
            DiffWrapper::Value(DiffScalar{index: _, gradient, source}) =>
            {
                let gradient = if let Some(x) = gradient { x } else { return None; };

                Some((gradient.into(), source))
            }
        }
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
enum JumpInfo
{
    Target{inputs: Vec<InputTypePtr>, index: LoopIndex},
    Source(LoopIndex)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientOp<T, J>
{
    None,
    Jump(J),
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
    MulComponentwiseAdd{lhs: T, rhs: T, added: T, output: T},
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

impl<T, J> GradientOp<T, J>
{
    fn map_tensors<U, K>(self, mut f: impl FnMut(T) -> U, jump_f: impl FnOnce(J) -> K) -> GradientOp<U, K>
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
            Self::MulComponentwiseAdd{lhs, rhs, added, output} =>
            {
                GradientOp::MulComponentwiseAdd{lhs: f(lhs), rhs: f(rhs), added: f(added), output: f(output)}
            },
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
            Self::Jump(x) => GradientOp::Jump(jump_f(x)),
            Self::AddInplace{..}
            | Self::SoftmaxCrossEntropyNoSoftmaxed{..} => unreachable!()
        }
    }
}

impl<T: Clone, J: Clone> GradientOp<T, J>
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
            Self::MulComponentwiseAdd{output, lhs, rhs, added} => Self::MulComponentwiseAdd{output: tf(output), lhs, rhs, added},
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
            Self::Jump(x) => Self::Jump(x),
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
            Self::MulComponentwiseAdd{lhs, rhs, added, output} => Self::MulComponentwiseAdd{lhs: tf(lhs), rhs: tf(rhs), added: tf(added), output},
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
            Self::Jump(x) => Self::Jump(x),
            Self::AddInplace{..} => unreachable!()
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
    MatmulOneHotvAdd{lhs: DiffTensorPtr, rhs: OneHotIndex, added: DiffTensorPtr, output: DiffTensorPtr},
    Loop{index: LoopIndex, inputs: Vec<InputTypePtr>, ops: Vec<Op>}
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InputTypePtr
{
    Normal(TensorPtr),
    OneHot(OneHotIndex)
}

impl From<TensorPtr> for InputTypePtr
{
    fn from(index: TensorPtr) -> Self
    {
        Self::Normal(index)
    }
}

impl From<OneHotIndex> for InputTypePtr
{
    fn from(index: OneHotIndex) -> Self
    {
        Self::OneHot(index)
    }
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

        recorder.gradient_with_respect(out.into());

        recorder.resolve_memory();

        let a_gradient = recorder.resolve_tensor_ptr(a_gradient);
        let b_gradient = recorder.resolve_tensor_ptr(b_gradient);

        recorder.calculate();

        let a_g = recorder.get_tensor(a_gradient).clone_owned();
        let b_g = recorder.get_tensor(b_gradient).clone_owned();

        let mut vals = |a: LayerType, b: LayerType|
        {
            let mut new_recorder = OperationsRecorder::new();

            let new_a = new_recorder.set_new_tensor(a);
            let new_b = new_recorder.set_new_tensor(b);

            let output = f(&mut new_recorder, new_a, new_b);

            let output_value = output.as_value();

            new_recorder.store_tensor_until_end(output_value);

            new_recorder.finish();
            new_recorder.gradient_with_respect(output.into());

            new_recorder.resolve_memory();

            let output_value = new_recorder.resolve_tensor_ptr(output_value);

            new_recorder.calculate();

            new_recorder.get_tensor(output_value).clone_owned()
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

    #[test]
    fn loops()
    {
        let loops_count = 3;
        let is: Vec<LayerType> = (0..loops_count).map(|_| LayerType::new_with(LAYER_CURR, LAYER_PREV, random_value)).collect();

        check_tensor(|recorder, a, b|
        {
            let ab = recorder.mul_componentwise(a, b);

            let (rows, columns) = recorder.tensor_shape(a.as_value());
            let i = recorder.new_tensor_no_gradient(rows, columns).as_value();

            let loops_value = recorder.set_new_value(loops_count as f32 * 20.0);
            let loop_index = recorder.begin_loop(vec![i.into()]);

            let abb = recorder.mul_scalar(ab, loops_value);
            let s = recorder.tanh(abb);
            let si = recorder.mul_componentwise(s, DiffTensorPtr::no_gradient(i));

            recorder.end_loop(loop_index);

            recorder.set_loop_times(loop_index, loops_count);
            recorder.set_loop_inputs(loop_index, is.clone());

            si
        })
    }
}
