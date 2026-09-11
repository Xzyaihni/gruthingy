use std::{
    f32,
    mem,
    convert,
    debug_assert_matches,
    fmt::{self, Debug},
    borrow::Borrow,
    collections::HashSet,
    ops::Range
};

#[allow(unused_imports)]
use std::{iter, cell::RefCell, marker::PhantomData, cmp::Ordering, collections::HashMap};

use serde::{Serialize, Deserialize};

//use matrix_wrapper::{MatrixWrapper, MatrixWrapperRef, MatrixWrapperMut, VectorWrapper, VectorWrapperMut};
use ywrapper::*;

//mod matrix_wrapper;
mod ywrapper;


pub type LayerType = YWrapper;
pub type LayerTypeRef<'a> = YWrapperRef<'a>;
pub type LayerTypeMut<'a> = YWrapperMut<'a>;

pub type LayerTypeVectorRef<'a> = YVectorWrapperRef<'a>;
pub type LayerTypeVectorMut<'a> = YVectorWrapperMut<'a>;

pub const LEAKY_SLOPE: f32 = 0.01;

const OPT_INFO: bool = false;
const NO_COLORING: bool = false;
const _REASSIGN_CHECKS: bool = true;

#[allow(dead_code)]
const PRINT_CALCULATE_VALUES: bool = false;


macro_rules! get_disjoint_mut_with
{
    ($this:expr, $(($target_type:ident, $name:expr, $tmp_name:ident)),+$(,)?) =>
    {
        {
            let indices = [$($name.range(),)+];

            let [$($tmp_name,)+] = {
                #[cfg(debug_assertions)]
                {
                    $this.memory.tensors_raw_data.get_disjoint_mut(indices).unwrap()
                }

                #[cfg(not(debug_assertions))]
                {
                    unsafe{ $this.memory.tensors_raw_data.get_disjoint_unchecked_mut(indices) }
                }
            };

            ($($target_type::from_data($tmp_name, $name.rows, $name.columns),)+)
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
{
    fn exp_inplace(&mut self);
    fn sum(&self) -> f32;

    fn mul_scalar_inplace(&mut self, value: f32);
}

#[derive(Debug)]
pub struct Softmaxer;

impl Softmaxer
{
    #[allow(dead_code)]
    pub fn softmax_temperature(layer: &mut LayerType, temperature: f32)
    {
        layer.mul_scalar_inplace(temperature.recip());

        Self::softmax(layer)
    }

    pub fn softmax(layer: &mut impl Softmaxable)
    {
        layer.exp_inplace();
        let s = layer.sum();

        layer.mul_scalar_inplace(s.recip());
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

#[derive(Debug, Default, Clone, PartialEq)]
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
        debug_assert!(self.valid_range(), "{self:?} is an invalid range");
        debug_assert!(other.valid_range(), "{other:?} is an invalid range");

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
    times_total: usize,
    current_index: usize,
    reversed: bool,
    live_range: LiveRange,
    loops_gradient: Option<LoopIndex>,
    gradient_of_loop: Option<LoopIndex>,
    kept_inside: Vec<DiffValue>,
    defined_values: Vec<DiffValue>,
    used_values: Vec<DiffValue>,
    input_values: LoopValuesIndex,
    stack_values: LoopStackIndex,
    inputs: Vec<Option<InputType>>,
    #[cfg(debug_assertions)]
    expected_pairs: Vec<(DiffValue, DiffValue)>
}

#[allow(dead_code)]
struct LoopInfoDebug<'a>
{
    memory: &'a OperationsRecorderMemory,
    info: &'a LoopInfo
}

impl Debug for LoopInfoDebug<'_>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let info = self.info;

        let fmd = |values: &[DiffValue]| -> ForceNoPretty<Vec<DebugStringRaw>>
        {
            ForceNoPretty(values.iter().map(|x| DebugStringRaw(self.memory.format_variable(*x))).collect::<Vec<_>>())
        };

        let kept_inside = fmd(&info.kept_inside);
        let defined_values = fmd(&info.defined_values);
        let used_values = fmd(&info.used_values);

        let mut struct_info = f.debug_struct("LoopInfo");

        struct_info.field("times", &DebugStringRaw(format!("{}/{}", info.times, info.times_total)))
            .field("current_index", &info.current_index)
            .field("reversed", &info.reversed)
            .field("live_range", &ForceNoPretty(&info.live_range))
            .field("loops_gradient", &ForceNoPretty(&info.loops_gradient))
            .field("gradient_of_loop", &ForceNoPretty(&info.gradient_of_loop))
            .field("kept_inside", &kept_inside)
            .field("defined_values", &defined_values)
            .field("used_values", &used_values)
            .field("input_values", &ForceNoPretty(&info.input_values))
            .field("inputs", &info.inputs);

        #[cfg(debug_assertions)]
        {
            let expected_pairs = info.expected_pairs.iter().map(|(source, target)|
            {
                ForceNoPretty((DebugStringRaw(self.memory.format_variable(*source)), DebugStringRaw(self.memory.format_variable(*target))))
            }).collect::<Vec<_>>();

            struct_info.field("expected_pairs", &expected_pairs);
        }

        struct_info.finish()
    }
}

#[cfg(debug_assertions)]
#[derive(Debug, Default, Clone)]
struct LoopStackValue<T, TargetType>
{
    value: T,
    source: Option<TargetType>
}

#[cfg(debug_assertions)]
impl<T, TargetType> From<T> for LoopStackValue<T, TargetType>
{
    fn from(value: T) -> Self
    {
        Self{value, source: None}
    }
}

#[allow(dead_code)]
trait Targettable
{
    fn convert(self, recorder: &OperationsRecorder) -> Option<DiffValue>;
}

impl Targettable for ValueIndex
{
    fn convert(self, _recorder: &OperationsRecorder) -> Option<DiffValue> { Some(DiffValue::Value(self)) }
}

impl Targettable for TensorRawDataPointer
{
    fn convert(self, recorder: &OperationsRecorder) -> Option<DiffValue>
    {
        recorder.memory.raw_ptr_to_ptr(self).map(DiffValue::Tensor)
    }
}

#[cfg(debug_assertions)]
impl<T, TargetType: Targettable> LoopStackValue<T, TargetType>
{
    fn get_stack_value_for(
        self,
        recorder: &OperationsRecorder,
        loop_index: LoopIndex,
        target: TargetType
    ) -> T
    {
        let expected_pairs = &recorder.loops[loop_index.0].expected_pairs;

        let source = self.source.expect("source must be initialized").convert(recorder);
        let target = target.convert(recorder);

        if let (Some(source), Some(target)) = (source, target)
        {
            let target_fmt = recorder.memory.format_variable(target);

            let (expected_source, _target) = expected_pairs.iter().find(|(_, expected_target)|
            {
                *expected_target == target
            }).expect("pop target must exist").clone();

            assert_eq!(
                expected_source, source,
                "stack operation expected to push from {} and pop to {}, instead pushed from {} and popped to {}",
                recorder.memory.format_variable(expected_source),
                target_fmt,
                recorder.memory.format_variable(source),
                target_fmt
            );
        }

        self.value
    }

    fn set_source(&mut self, source: TargetType)
    {
        self.source = Some(source);
    }
}

#[cfg(not(debug_assertions))]
#[derive(Debug, Default, Clone)]
struct LoopStackValue<T, TargetType>
{
    value: T,
    target_type: PhantomData<TargetType>
}

#[cfg(not(debug_assertions))]
impl<T, TargetType> From<T> for LoopStackValue<T, TargetType>
{
    fn from(value: T) -> Self
    {
        Self{value, target_type: PhantomData}
    }
}

#[cfg(not(debug_assertions))]
impl<T, TargetType> LoopStackValue<T, TargetType>
{
    fn get_stack_value_for(
        self,
        _recorder: &OperationsRecorder,
        _loop_index: LoopIndex,
        _target: TargetType
    ) -> T
    {
        self.value
    }

    fn set_source(&self, _x: TargetType) {}
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum LoopInputs
{
    Values(Vec<OwnedInputType>),
    Dependent(usize)
}

impl From<Vec<OwnedInputType>> for LoopInputs
{
    fn from(inputs: Vec<OwnedInputType>) -> Self
    {
        Self::Values(inputs)
    }
}

#[derive(Debug, Clone)]
struct LoopValues
{
    input_values: LoopInputs
}

impl Default for LoopValues
{
    fn default() -> Self
    {
        Self{
            input_values: LoopInputs::Values(Vec::new())
        }
    }
}

#[derive(Debug, Default, Clone)]
struct LoopStack
{
    values_stack: Vec<LoopStackValue<f32, ValueIndex>>,
    tensors_stack: Vec<LoopStackValue<LayerType, TensorRawDataPointer>>
}

#[derive(Debug, Clone)]
struct RawJumpInfo
{
    loop_index: LoopIndex,
    operation_index: GradientOperationIndex
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct LoopValuesIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct LoopStackIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopIndex(usize);

#[derive(Debug, Clone)]
enum OperationsTarget
{
    Normal,
    Loop(LoopOperationIndex)
}

#[allow(dead_code)]
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

#[derive(Debug, Clone)]
struct PhiOtherSelectorRecording
{
    first: DiffWrapper,
    other: Option<DiffWrapper>,
    value_index: Option<PhiOtherSelectorIndex>,
    gradient_index: Option<PhiOtherSelectorIndex>
}

#[derive(Debug, Clone)]
struct PhiOtherSelectorValue
{
    loop_index: LoopIndex,
    is_set: bool
}

#[derive(Debug)]
struct AssignedInfo
{
    value: DiffValue,
    operation_index: GradientOperationIndex,
    extra_operation_index: Option<GradientOperationIndex>,
    loop_selected: Option<LoopIndex>
}

#[derive(Debug)]
struct LoopSelectorInfo
{
    next_total: DiffValue,
    next_previous: DiffValue,
    other: DiffValue
}

#[derive(Debug, Clone)]
struct UsedStackValueInfo
{
    loop_index: LoopIndex,
    source: DiffValue,
    target: DiffValue
}

#[cfg(debug_assertions)]
#[derive(Debug, Clone)]
struct VariableNames(HashMap<DiffValue, String>);

#[cfg(debug_assertions)]
impl VariableNames
{
    fn new() -> Self
    {
        Self(HashMap::new())
    }

    fn format_variable<V: Into<DiffValue> + Clone + Debug>(&self, variable: V) -> String
    {
        let value: DiffValue = variable.clone().into();

        self.0.get(&value).cloned().unwrap_or_else(|| format!("{variable:?}"))
    }
}

#[cfg(not(debug_assertions))]
#[derive(Debug, Clone)]
struct VariableNames;

#[cfg(not(debug_assertions))]
impl VariableNames
{
    fn new() -> Self { Self }

    fn format_variable(&self, variable: impl Debug) -> String
    {
        format!("{variable:?}")
    }
}

#[cfg(debug_assertions)]
#[derive(Clone)]
struct SetTensorMemoryChecks
{
    set_ptrs: Vec<TensorPtr>,
    read_memory: Vec<TensorIndex>,
    set_memory: Vec<TensorIndex>
}

#[cfg(debug_assertions)]
struct SetTensorMemoryChecksDebug<'a>
{
    memory: &'a OperationsRecorderMemory,
    info: &'a SetTensorMemoryChecks
}

#[cfg(debug_assertions)]
impl Debug for SetTensorMemoryChecksDebug<'_>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let fm = |x: &TensorIndex|
        {
            DebugStringRaw(self.memory.format_tensor_index(*x))
        };

        let fv = |x: &[TensorIndex]| -> Vec<_>
        {
            x.iter().map(fm).collect::<Vec<_>>()
        };

        f.debug_struct("SetTensorMemoryChecks")
            .field("set_ptrs", &self.info.set_ptrs.iter().map(|x| self.memory.format_variable(*x)).collect::<Vec<_>>())
            .field("read_memory", &fv(&self.info.read_memory))
            .field("set_memory", &fv(&self.info.set_memory))
            .finish()
    }
}

#[cfg(debug_assertions)]
impl SetTensorMemoryChecks
{
    fn new() -> Self
    {
        Self{
            set_ptrs: Vec::new(),
            read_memory: Vec::new(),
            set_memory: Vec::new()
        }
    }
}

#[derive(Clone)]
pub struct OperationsRecorderMemory
{
    values: Vec<f32>,
    tensors: Vec<TensorRawDataPointer>,
    value_live_ranges: Vec<LiveRange>,
    tensor_live_ranges: Vec<LiveRange>,
    tensors_memory: Vec<TensorMemorySlot>,
    tensors_raw_data: Vec<f32>,
    one_hot_layers: Vec<OneHotLayer>,
    phi_other_selectors_values: Vec<PhiOtherSelectorValue>,
    variable_names: VariableNames,
    #[cfg(debug_assertions)]
    set_tensor_memory: RefCell<SetTensorMemoryChecks>,
    #[cfg(debug_assertions)]
    tensor_inputs: Vec<TensorPtr>,
    #[cfg(debug_assertions)]
    allow_discard: Vec<TensorPtr>,
    #[cfg(debug_assertions)]
    set_tensors_check: Vec<InputCheckType>,
    #[cfg(debug_assertions)]
    store_tensors_check: Vec<StoreCheckKey<TensorPtr, TensorIndex>>,
    #[cfg(debug_assertions)]
    store_values_check: Vec<StoreCheckKey<ValueIndex, ValueIndex>>
}

impl Debug for OperationsRecorderMemory
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let max_length = 50;

        let mut s = f.debug_struct("OperationsRecorderMemory");

        s.field("value_live_ranges", &self.value_live_ranges.iter().map(ForceNoPretty).collect::<Vec<_>>())
            .field("tensor_live_ranges", &self.tensor_live_ranges.iter().map(ForceNoPretty).collect::<Vec<_>>())
            .field("tensors_memory", &self.tensors_memory.iter().map(|x| ForceNoPretty(SlotNoLong(max_length, x))).collect::<Vec<_>>())
            .field("values", &ForceNoPretty(&self.values))
            .field("tensors", &self.tensors.iter().map(ForceNoPretty).collect::<Vec<_>>())
            .field("tensors_raw_data", &DebugStringRaw(format!("{} values", self.tensors_raw_data.len())))
            .field("one_hot_layers", &self.one_hot_layers)
            .field("phi_other_selectors_values", &self.phi_other_selectors_values);

        #[cfg(debug_assertions)]
        {
            let mut variable_names = self.variable_names.0.iter().collect::<Vec<_>>();

            variable_names.sort_by(|a, b|
            {
                match (&a.0, &b.0)
                {
                    (DiffValue::Value(_), DiffValue::Tensor(_)) => Ordering::Less,
                    (DiffValue::Value(_), DiffValue::OneHot(_)) => Ordering::Less,
                    (DiffValue::OneHot(_), DiffValue::Tensor(_)) => Ordering::Less,
                    (DiffValue::OneHot(_), DiffValue::Value(_)) => Ordering::Greater,
                    (DiffValue::Tensor(_), DiffValue::Value(_)) => Ordering::Greater,
                    (DiffValue::Tensor(_), DiffValue::OneHot(_)) => Ordering::Greater,
                    (DiffValue::Value(a), DiffValue::Value(b)) => a.0.cmp(&b.0),
                    (DiffValue::OneHot(a), DiffValue::OneHot(b)) => a.0.cmp(&b.0),
                    (DiffValue::Tensor(a), DiffValue::Tensor(b)) => a.0.cmp(&b.0)
                }
            });

            let variable_names = variable_names.into_iter().map(|(key, value)|
            {
                let key = match key
                {
                    DiffValue::Tensor(x) => format!("{x:?}"),
                    DiffValue::OneHot(x) => format!("{x:?}"),
                    DiffValue::Value(x) => format!("{x:?}")
                };

                DebugStringRaw(format!("{key}: {value}"))
            }).collect::<Vec<_>>();

            s.field("set_tensor_memory", &SetTensorMemoryChecksDebug{info: &self.set_tensor_memory.borrow(), memory: self})
                .field("variable_names", &variable_names)
                .field("tensor_inputs", &self.tensor_inputs.iter().map(ForceNoPretty).collect::<Vec<_>>())
                .field("allow_discard", &self.allow_discard.iter().map(ForceNoPretty).collect::<Vec<_>>())
                .field("set_tensors_check", &self.set_tensors_check.iter().map(ForceNoPretty).collect::<Vec<_>>())
                .field("store_tensors_check", &self.store_tensors_check.iter().map(ForceNoPretty).collect::<Vec<_>>())
                .field("store_values_check", &self.store_values_check.iter().map(ForceNoPretty).collect::<Vec<_>>());
        }

        s.finish()
    }
}

#[allow(dead_code)]
impl OperationsRecorderMemory
{
    pub fn new() -> Self
    {
        Self{
            value_live_ranges: Vec::new(),
            tensor_live_ranges: Vec::new(),
            tensors_memory: Vec::new(),
            values: Vec::new(),
            tensors: Vec::new(),
            tensors_raw_data: Vec::new(),
            one_hot_layers: Vec::new(),
            phi_other_selectors_values: Vec::new(),
            variable_names: VariableNames::new(),
            #[cfg(debug_assertions)]
            set_tensor_memory: RefCell::new(SetTensorMemoryChecks::new()),
            #[cfg(debug_assertions)]
            tensor_inputs: Vec::new(),
            #[cfg(debug_assertions)]
            allow_discard: Vec::new(),
            #[cfg(debug_assertions)]
            set_tensors_check: Vec::new(),
            #[cfg(debug_assertions)]
            store_tensors_check: Vec::new(),
            #[cfg(debug_assertions)]
            store_values_check: Vec::new()
        }
    }

    fn new_tensor_index(&mut self, value: TensorMemoryValue) -> TensorPtr
    {
        let id = self.tensors_memory.len();

        self.tensor_live_ranges.push(LiveRange::default());
        self.tensors_memory.push(TensorMemorySlot{value, memory: None});

        TensorPtr(id)
    }

    fn new_value_index(&mut self) -> ValueIndex
    {
        let id = self.values.len();

        self.value_live_ranges.push(LiveRange::default());
        self.values.push(0.0);

        ValueIndex(id)
    }

    fn new_diff_intermediate(
        &mut self,
        other: DiffValue,
        f: impl FnOnce((DiffValue, DiffValue))
    ) -> DiffValue
    {
        match other
        {
            DiffValue::Value(other) =>
            {
                let next = self.new_value_index();
                f((next.into(), other.into()));

                DiffValue::Value(next)
            },
            DiffValue::Tensor(other) =>
            {
                let shape = self.tensor_shape_value(other);

                let next = self.new_tensor_index(shape);
                f((next.into(), other.into()));

                DiffValue::Tensor(next)
            },
            DiffValue::OneHot(_) => unimplemented!()
        }
    }

    fn new_tensor(&mut self, has_gradient: bool, value: TensorMemoryValue) -> DiffTensorPtr
    {
        let (rows, columns) = value.tensor_shape();

        let size_value = TensorMemoryValue::Size{rows, columns};

        DiffTensorPtr{
            index: self.new_tensor_index(value),
            gradient: has_gradient.then(|| self.new_tensor_index(size_value))
        }
    }

    fn new_value(&mut self, has_gradient: bool) -> DiffScalar
    {
        DiffScalar{
            index: self.new_value_index(),
            gradient: has_gradient.then(|| self.new_value_index())
        }
    }

    fn set_input(&mut self, input: InputType, value: OwnedInputType)
    {
        match input
        {
            InputType::Normal(input) => self.set_tensor(input, value.into_ref_normal()),
            InputType::OneHot(input) => self.set_one_hot(input, value.into_one_hot())
        }
    }

    fn set_tensor(&mut self, index: TensorIndex, value: &LayerType)
    {
        #[cfg(debug_assertions)]
        {
            self.set_tensors_check.push(index.into());

            self.verify_raw_ptr_assign_index(index);
        }

        self.set_tensor_raw(self.tensors[index.0], value);
    }

    fn set_tensor_raw(&mut self, index: TensorRawDataPointer, value: &LayerType)
    {
        let dst = LayerTypeMut::from_data_with_start(&mut self.tensors_raw_data, index);

        let src = LayerTypeRef::from(value);

        dst.copy_from(src);
    }

    fn set_one_hot(&mut self, index: OneHotIndex, value: OneHotLayer)
    {
        self.one_hot_layers[index.0] = value;
    }

    pub fn get_tensor(&self, index: TensorIndex) -> LayerTypeRef<'_>
    {
        debug_assert_ne!(index, TensorIndex::undefined());

        #[cfg(debug_assertions)]
        {
            verify_store_check(&self.store_tensors_check, index, "tensor");

            self.verify_raw_ptr_use_index(index);
        }

        let info = self.tensors[index.0];
        debug_assert_ne!(info, TensorRawDataPointer::undefined(), "{index:?} location is undefined");

        LayerTypeRef::from_data_with_start(&self.tensors_raw_data, info)
    }

    fn tensor_shape(&self, tensor: TensorPtr) -> (usize, usize)
    {
        self.tensors_memory[tensor.0].value.tensor_shape()
    }

    fn tensor_shape_value(&self, tensor: TensorPtr) -> TensorMemoryValue
    {
        let (rows, columns) = self.tensor_shape(tensor);

        TensorMemoryValue::Size{rows, columns}
    }

    fn raw_ptr_to_memory(&self, raw_ptr: TensorRawDataPointer) -> TensorIndex
    {
        TensorIndex(self.tensors.iter().position(|x| *x == raw_ptr).expect("must be a real ptr"))
    }

    fn raw_ptr_to_ptr(&self, raw_ptr: TensorRawDataPointer) -> Option<TensorPtr>
    {
        let tensor_index = self.raw_ptr_to_memory(raw_ptr);

        self.memory_to_ptr(tensor_index)
    }

    fn memory_to_ptr(&self, tensor_index: TensorIndex) -> Option<TensorPtr>
    {
        let possible_tensor_ptrs: Vec<_> = self.tensors_memory.iter().enumerate().filter(|(_, x)|
        {
            x.memory == Some(tensor_index)
        }).collect();

        (possible_tensor_ptrs.len() == 1).then(||
        {
            let (tensor_ptr, _spot) = possible_tensor_ptrs.into_iter().next().unwrap();

            TensorPtr(tensor_ptr)
        })
    }

    fn format_variable<V: Into<DiffValue> + Clone + Debug>(&self, variable: V) -> String
    {
        self.variable_names.format_variable(variable)
    }

    fn format_tensor_index(&self, tensor_index: TensorIndex) -> String
    {
        if let Some(tensor_ptr) = self.memory_to_ptr(tensor_index)
        {
            self.format_variable(tensor_ptr)
        } else
        {
            format!("{tensor_index:?}")
        }
    }

    fn format_tensor_raw_ptr(&self, raw_ptr: TensorRawDataPointer) -> String
    {
        if let Some(tensor_ptr) = self.raw_ptr_to_ptr(raw_ptr)
        {
            self.format_variable(tensor_ptr)
        } else
        {
            format!("{raw_ptr:?}")
        }
    }

    #[cfg(debug_assertions)]
    fn verify_raw_ptr_use(&self, raw_ptr: TensorRawDataPointer)
    {
        let memory_index = self.raw_ptr_to_memory(raw_ptr);

        self.verify_raw_ptr_use_index(memory_index)
    }

    #[cfg(debug_assertions)]
    fn verify_raw_ptr_use_index(&self, memory_index: TensorIndex)
    {
        let mut set_tensor_memory = self.set_tensor_memory.borrow_mut();

        debug_assert!(
            set_tensor_memory.set_memory.contains(&memory_index),
            "{} was used without being set",
            self.format_tensor_index(memory_index)
        );

        if !set_tensor_memory.read_memory.contains(&memory_index)
        {
            set_tensor_memory.read_memory.push(memory_index);
        }
    }

    #[cfg(debug_assertions)]
    fn verify_raw_ptr_assign(&mut self, raw_ptr: TensorRawDataPointer)
    {
        let memory_index = self.raw_ptr_to_memory(raw_ptr);

        self.verify_raw_ptr_assign_index(memory_index)
    }

    #[cfg(debug_assertions)]
    fn verify_raw_ptr_assign_index(&mut self, memory_index: TensorIndex)
    {
        let mut set_tensor_memory = self.set_tensor_memory.borrow_mut();

        if let Some(read_index) = set_tensor_memory.read_memory.iter().position(|x| *x == memory_index)
        {
            set_tensor_memory.read_memory.remove(read_index);
        } else
        {
            let is_no_read_reassigned = set_tensor_memory.set_memory.contains(&memory_index);

            if _REASSIGN_CHECKS
            {
                if is_no_read_reassigned
                {
                    eprintln!("{} was reassigned without being read", self.format_tensor_index(memory_index));
                }
            }
        }

        set_tensor_memory.set_memory.push(memory_index);
    }
}

#[derive(Debug, Clone)]
struct LoopsMemory
{
    loops_values: Vec<LoopValues>,
    loops_stack: Vec<LoopStack>
}

impl LoopsMemory
{
    fn new() -> Self
    {
        Self{
            loops_values: Vec::new(),
            loops_stack: Vec::new()
        }
    }
}

#[derive(Clone)]
pub struct OperationsRecorder
{
    memory: OperationsRecorderMemory,
    state: RecorderState,
    operations_target: OperationsTarget,
    loops: Vec<LoopInfo>,
    loops_memory: LoopsMemory,
    phi_other_selectors_recording: Vec<PhiOtherSelectorRecording>,
    recording_operations: Vec<Op>,
    gradient_operations: Vec<StandardGradientOp>,
    raw_operations: Vec<RawGradientOp>,
    feedforward_operations_count: usize
}

impl Debug for OperationsRecorder
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let gradient_operations;

        #[cfg(debug_assertions)]
        {
            gradient_operations = self.gradient_operations.iter().map(|op|
            {
                NotationGradientOp::from_nameable(&self.memory, op.clone())
            }).collect::<Vec<_>>();
        }

        #[cfg(not(debug_assertions))]
        {
            gradient_operations = self.gradient_operations.iter().map(ForceNoPretty).collect::<Vec<_>>();
        }

        let raw_operations;
        let loops;

        #[cfg(debug_assertions)]
        {
            raw_operations = self.raw_operations.iter().map(|op|
            {
                NotationGradientOp(op.clone().map(|t|
                {
                    DebugStringRaw(self.memory.format_tensor_raw_ptr(t))
                }, |v|
                {
                    DebugStringRaw(self.memory.format_variable(v))
                }, convert::identity, convert::identity))
            }).collect::<Vec<_>>();

            loops = self.loops.iter().map(|info| LoopInfoDebug{memory: &self.memory, info}).collect::<Vec<_>>();
        }

        #[cfg(not(debug_assertions))]
        {
            raw_operations = self.raw_operations.iter().map(ForceNoPretty).collect::<Vec<_>>();
            loops = self.loops.clone();
        }

        f.debug_struct("OperationsRecorder")
            .field("memory", &self.memory)
            .field("state", &self.state)
            .field("operations_target", &self.operations_target)
            .field("loops", &loops)
            .field("loops_memory", &self.loops_memory)
            .field("phi_other_selectors_recording", &self.phi_other_selectors_recording)
            .field("recording_operations", &self.recording_operations.iter().map(ForceNoPretty).collect::<Vec<_>>())
            .field("gradient_operations", &gradient_operations)
            .field("raw_operations", &raw_operations)
            .field("feedforward_operations_count", &self.feedforward_operations_count)
            .finish()
    }
}

macro_rules! impl_pair_tensor_op
{
    ($this:expr, $a:expr, $b:expr, $name:ident) =>
    {
        {
            let a_shape@(a_rows, a_columns) = $this.tensor_shape($a.as_value());
            let b_shape = $this.tensor_shape($b.as_value());

            debug_assert_eq!(a_shape, b_shape);

            let output = $this.new_tensor_op(a_rows, a_columns);

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
            let (rows, columns) = $this.tensor_shape($a.as_value());

            let output = $this.new_tensor_op(rows, columns);

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
            memory: OperationsRecorderMemory::new(),
            state: RecorderState::Recording,
            operations_target: OperationsTarget::Normal,
            loops: Vec::new(),
            loops_memory: LoopsMemory::new(),
            phi_other_selectors_recording: Vec::new(),
            recording_operations: Vec::new(),
            gradient_operations: Vec::new(),
            raw_operations: Vec::new(),
            feedforward_operations_count: 0
        }
    }

    pub fn new_tensor(&mut self, rows: usize, columns: usize) -> DiffTensorPtr
    {
        let input = self.memory.new_tensor(true, TensorMemoryValue::Size{rows, columns});
        self.memory.tensor_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn new_tensor_no_gradient(&mut self, rows: usize, columns: usize) -> DiffTensorPtr
    {
        let input = self.memory.new_tensor(false, TensorMemoryValue::Size{rows, columns});
        self.memory.tensor_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn new_value(&mut self) -> DiffScalar
    {
        let input = self.memory.new_value(true);
        self.memory.value_live_ranges[input.as_value().0].start = Some(-1);

        input
    }

    pub fn new_one_hot(&mut self) -> OneHotIndex
    {
        let id = self.memory.one_hot_layers.len();

        self.memory.one_hot_layers.push(OneHotLayer::empty());

        OneHotIndex(id)
    }

    fn new_tensor_op(
        &mut self,
        rows: usize,
        columns: usize
    ) -> DiffTensorPtr
    {
        self.memory.new_tensor(true, TensorMemoryValue::Size{rows, columns})
    }

    fn new_value_op(&mut self) -> DiffScalar
    {
        self.memory.new_value(true)
    }

    pub fn set_tensor(&mut self, index: TensorIndex, value: &LayerType)
    {
        debug_assert_eq!(self.state, RecorderState::Ready);

        self.memory.set_tensor(index, value)
    }

    pub fn set_tensor_ptr_zeroed(&mut self, _index: TensorPtr)
    {
        #[cfg(debug_assertions)]
        {
            self.memory.set_tensors_check.push(_index.into());

            self.memory.set_tensor_memory.borrow_mut().set_ptrs.push(_index);
        }
    }

    pub fn allow_discard(&mut self, _index: TensorPtr)
    {
        #[cfg(debug_assertions)]
        {
            self.memory.allow_discard.push(_index);
        }
    }

    pub fn set_tensor_from(&mut self, index: TensorIndex, src: TensorIndex)
    {
        debug_assert_eq!(self.state, RecorderState::Ready);

        #[cfg(debug_assertions)]
        {
            self.memory.set_tensors_check.push(index.into());
        }

        let dst = self.memory.tensors[index.0];
        let src = self.memory.tensors[src.0];

        let (dst, src) = get_disjoint_mut_with!(self, (LayerTypeMut, dst, x0), (LayerTypeRef, src, x1));

        dst.copy_from(src);
    }

    pub fn set_value(&mut self, index: ValueIndex, value: f32)
    {
        self.memory.values[index.0] = value;
    }

    pub fn set_one_hot(&mut self, index: OneHotIndex, value: OneHotLayer)
    {
        self.memory.set_one_hot(index, value)
    }

    pub fn set_input(&mut self, input: InputType, value: OwnedInputType)
    {
        self.memory.set_input(input, value)
    }

    pub fn set_new_tensor_gradientable(&mut self, value: LayerType) -> DiffTensorPtr
    {
        let input = self.memory.new_tensor(true, TensorMemoryValue::Value(value));

        self.set_new_tensor_common(&input);

        input
    }

    pub fn set_new_tensor(&mut self, value: LayerType) -> DiffTensorPtr
    {
        let input = self.memory.new_tensor(false, TensorMemoryValue::Value(value));

        self.set_new_tensor_common(&input);

        input
    }

    fn set_new_tensor_common(&mut self, input: &DiffTensorPtr)
    {
        #[cfg(debug_assertions)]
        {
            self.memory.set_tensors_check.push(input.as_value().into());
        }

        self.memory.tensor_live_ranges[input.as_value().0].start = Some(-1);
    }

    pub fn set_new_value(&mut self, value: f32) -> DiffScalar
    {
        let scalar = self.memory.new_value(false);
        self.set_value(scalar.as_value(), value);

        self.memory.value_live_ranges[scalar.as_value().0].start = Some(-1);

        scalar
    }

    pub fn phi_other_selector(&mut self, first: impl Into<DiffWrapper>) -> PhiOtherSelectorRecordingIndex
    {
        debug_assert_eq!(self.state, RecorderState::Recording);

        let id = self.phi_other_selectors_recording.len();

        self.phi_other_selectors_recording.push(PhiOtherSelectorRecording{
            first: first.into(),
            other: None,
            value_index: None,
            gradient_index: None
        });

        PhiOtherSelectorRecordingIndex(id)
    }

    pub fn set_phi_other_selector(&mut self, index: PhiOtherSelectorRecordingIndex, value: impl Into<DiffWrapper>)
    {
        debug_assert_eq!(self.state, RecorderState::Recording);

        let selector = &mut self.phi_other_selectors_recording[index.0];

        let new_other = value.into();

        match (&selector.first, &new_other)
        {
            (DiffWrapper::Value(_), DiffWrapper::Value(_))
            | (DiffWrapper::Tensor(_), DiffWrapper::Tensor(_)) => (),
            x => panic!("phi selector type mismatch: {x:#?}")
        }

        debug_assert!(selector.other.is_none());

        selector.other = Some(new_other);

        self.add_recording_operation(Op::SetOtherSelector(index));
    }

    pub fn select_value(&mut self, index: PhiOtherSelectorRecordingIndex) -> DiffScalar
    {
        debug_assert_eq!(self.state, RecorderState::Recording);

        let this_selector = &self.phi_other_selectors_recording[index.0];

        let output = self.memory.new_value(true);

        if let DiffWrapper::Value(_) = this_selector.first
        {
            self.add_recording_operation(Op::GetOtherSelectorValue{index, output});

            output
        } else
        {
            panic!("called select_value on a tensor selector");
        }
    }

    pub fn select_tensor(&mut self, index: PhiOtherSelectorRecordingIndex) -> DiffTensorPtr
    {
        debug_assert_eq!(self.state, RecorderState::Recording);

        let this_selector = &self.phi_other_selectors_recording[index.0];

        if let DiffWrapper::Tensor(first_tensor) = this_selector.first
        {
            let (rows, columns) = self.tensor_shape(first_tensor.as_value());
            let output = self.memory.new_tensor(true, TensorMemoryValue::Size{rows, columns});

            self.add_recording_operation(Op::GetOtherSelectorTensor{index, output});

            output
        } else
        {
            panic!("called select_tensor on a value selector");
        }
    }

    fn set_ones(&mut self, wrapper: DiffWrapper)
    {
        match wrapper
        {
            DiffWrapper::Tensor(DiffTensorPtr{index, gradient, ..}) =>
            {
                let (rows, columns) = self.memory.tensor_shape(index);

                let new_value = TensorMemoryValue::Value(LayerType::repeat(rows, columns, 1.0));

                let gradient_ptr: TensorPtr = gradient.expect("gradient must exist");

                #[cfg(debug_assertions)]
                {
                    self.memory.set_tensors_check.push(gradient_ptr.into());

                    self.allow_discard(gradient_ptr);
                }

                self.memory.tensor_live_ranges[gradient_ptr.0].start = Some(-1);
                self.memory.tensors_memory[gradient_ptr.0].value = new_value;
            },
            DiffWrapper::Value(DiffScalar{gradient, ..}) =>
            {
                let gradient = gradient.expect("gradient must exist");

                self.memory.value_live_ranges[gradient.0].start = Some(-1);

                self.set_value(gradient, 1.0)
            }
        }
    }

    pub fn get_tensor_memory_value(&self, index: TensorPtr) -> LayerTypeRef<'_>
    {
        debug_assert_eq!(self.state, RecorderState::Ready);

        if let TensorMemoryValue::Value(x) = &self.memory.tensors_memory[index.0].value
        {
            LayerTypeRef::from(x)
        } else
        {
            panic!("{index:?} has no memory value");
        }
    }

    pub fn is_undefined_location(&self, index: TensorIndex) -> bool
    {
        self.memory.tensors[index.0] == TensorRawDataPointer::undefined()
    }

    pub fn get_tensor(&self, index: TensorIndex) -> LayerTypeRef<'_>
    {
        debug_assert_eq!(self.state, RecorderState::Ready);

        self.memory.get_tensor(index)
    }

    pub fn get_tensor_mut<const USES_VALUE: bool>(&mut self, index: TensorIndex) -> LayerTypeMut<'_>
    {
        debug_assert_eq!(self.state, RecorderState::Ready);
        debug_assert_ne!(index, TensorIndex::undefined());

        #[cfg(debug_assertions)]
        {
            if USES_VALUE
            {
                verify_store_check(&self.memory.store_tensors_check, index, "tensor");

                self.memory.verify_raw_ptr_use_index(index);
            } else
            {
                self.memory.set_tensors_check.push(index.into());

                self.memory.verify_raw_ptr_assign_index(index);
            }
        }

        let info = self.memory.tensors[index.0];
        debug_assert_ne!(info, TensorRawDataPointer::undefined(), "{index:?} location is undefined");

        LayerTypeMut::from_data_with_start(&mut self.memory.tensors_raw_data, info)
    }

    pub fn get_value(&self, index: ValueIndex) -> f32
    {
        debug_assert_eq!(self.state, RecorderState::Ready);
        debug_assert_ne!(index, ValueIndex::undefined());

        #[cfg(debug_assertions)]
        {
            verify_store_check(&self.memory.store_values_check, index, "value");
        }

        self.memory.values[index.0]
    }

    pub fn get_one_hot(&self, index: OneHotIndex) -> &OneHotLayer
    {
        debug_assert_eq!(self.state, RecorderState::Ready);
        debug_assert_ne!(index, OneHotIndex::undefined());

        &self.memory.one_hot_layers[index.0]
    }

    fn name_diff_value(&mut self, _value: DiffValue, _name: String)
    {
        #[cfg(debug_assertions)]
        {
            if self.memory.variable_names.0.values().any(|x| *x == _name)
            {
                let name_chars: Vec<char> = _name.chars().collect();

                let mut count = 0;

                let end_number: String = name_chars.iter().rev().copied().take_while(|c: &char|
                {
                    let is_digit = c.is_ascii_digit();

                    if is_digit
                    {
                        count += 1;
                    }

                    is_digit
                }).collect();

                let new_name = if end_number.is_empty()
                {
                    _name + "1"
                } else
                {
                    let new_end_number = (end_number.parse::<u32>().expect("must be valid") + 1).to_string();

                    let total_chars = name_chars.len();
                    name_chars.into_iter().take(total_chars - count).collect::<String>() + &new_end_number
                };

                self.name_diff_value(_value, new_name);
            } else
            {
                self.memory.variable_names.0.insert(_value, _name);
            }
        }
    }

    pub fn name_tensor(&mut self, tensor: TensorPtr, name: impl Into<String>)
    {
        self.name_diff_value(DiffValue::Tensor(tensor), name.into().to_uppercase());
    }

    pub fn name_value(&mut self, value: ValueIndex, name: impl Into<String>)
    {
        self.name_diff_value(DiffValue::Value(value), name.into().to_lowercase());
    }

    pub fn name_one_hot(&mut self, value: OneHotIndex, name: impl Into<String>)
    {
        self.name_diff_value(value.into(), name.into().to_uppercase());
    }

    pub fn name_suffix_generic(&mut self, _value: DiffValue, _inherit: DiffValue, _suffix: &str)
    {
        #[cfg(debug_assertions)]
        {
            if let Some(inherit_name) = self.memory.variable_names.0.get(&_inherit)
            {
                let new_name = inherit_name.to_owned() + _suffix;

                match _value
                {
                    DiffValue::Value(value) => self.name_value(value, new_name),
                    DiffValue::Tensor(tensor) => self.name_tensor(tensor, new_name),
                    DiffValue::OneHot(one_hot) => self.name_one_hot(one_hot, new_name)
                }
            }
        }
    }

    pub fn name_value_suffix(&mut self, value: ValueIndex, inherit: ValueIndex, suffix: &str)
    {
        self.name_suffix_generic(value.into(), inherit.into(), suffix)
    }

    pub fn name_tensor_suffix(&mut self, tensor: TensorPtr, inherit: TensorPtr, suffix: &str)
    {
        self.name_suffix_generic(tensor.into(), inherit.into(), suffix)
    }

    pub fn name_input(&mut self, value: InputTypePtr, name: impl Into<String>)
    {
        self.name_diff_value(value.into(), name.into().to_uppercase());
    }

    pub fn name_diff_tensor(&mut self, tensor: DiffTensorPtr, name: impl Into<String>)
    {
        let name = name.into();

        self.name_tensor(tensor.as_value(), name.clone());

        if let Some(gradient) = tensor.as_gradient()
        {
            self.name_tensor(gradient, "∇".to_owned() + &name);
        }
    }

    pub fn name_diff_scalar(&mut self, scalar: DiffScalar, name: impl Into<String>)
    {
        let name = name.into();

        self.name_value(scalar.as_value(), name.clone());

        if let Some(gradient) = scalar.as_gradient()
        {
            self.name_value(gradient, "∇".to_owned() + &name);
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

    pub fn copy_scalar(&mut self, src: DiffScalar) -> DiffScalar
    {
        let dst = self.new_value_op();

        self.add_recording_operation(Op::CopyScalar{src, dst});

        dst
    }

    pub fn copy(&mut self, src: DiffTensorPtr) -> DiffTensorPtr
    {
        let (rows, columns) = self.tensor_shape(src.as_value());
        let dst = self.new_tensor_op(rows, columns);

        self.add_recording_operation(Op::Copy{src, dst});

        dst
    }

    pub fn add_scalars(&mut self, a: DiffScalar, b: DiffScalar) -> DiffScalar
    {
        let output = self.new_value_op();

        self.add_recording_operation(Op::AddScalars{lhs: a, rhs: b, output});

        output
    }

    pub fn add_scalar(&mut self, a: DiffTensorPtr, b: DiffScalar) -> DiffTensorPtr
    {
        let (a_rows, a_columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(a_rows, a_columns);

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
        let (a_rows, a_columns) = self.tensor_shape(b.as_value());

        let output = self.new_tensor_op(a_rows, a_columns);

        self.add_recording_operation(Op::SubFromScalar{lhs: a, rhs: b, output});

        output
    }

    pub fn mul_scalars(&mut self, a: DiffScalar, b: DiffScalar) -> DiffScalar
    {
        let output = self.new_value_op();

        self.add_recording_operation(Op::MulScalars{lhs: a, rhs: b, output});

        output
    }

    pub fn mul_scalar(&mut self, a: DiffTensorPtr, b: DiffScalar) -> DiffTensorPtr
    {
        let (a_rows, a_columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(a_rows, a_columns);

        self.add_recording_operation(Op::MulScalar{lhs: a, rhs: b, output});

        output
    }

    pub fn mul_componentwise(&mut self, a: DiffTensorPtr, b: DiffTensorPtr) -> DiffTensorPtr
    {
        impl_pair_tensor_op!(self, a, b, MulComponentwise)
    }

    pub fn matmulv(&mut self, a: DiffTensorPtr, b: DiffTensorPtr) -> DiffTensorPtr
    {
        let (a_rows, a_columns) = self.tensor_shape(a.as_value());
        let (b_rows, b_columns) = self.tensor_shape(b.as_value());

        debug_assert_eq!(a_columns, b_rows);

        let output = self.new_tensor_op(a_rows, b_columns);

        self.add_recording_operation(Op::Matmulv{lhs: a, rhs: b, output});

        output
    }

    pub fn matmulv_add(&mut self, a: DiffTensorPtr, b: DiffTensorPtr, added: DiffTensorPtr) -> DiffTensorPtr
    {
        let (a_rows, a_columns) = self.tensor_shape(a.as_value());
        let (b_rows, b_columns) = self.tensor_shape(b.as_value());

        let (rows, columns) = self.tensor_shape(added.as_value());

        debug_assert_eq!(a_columns, b_rows);

        debug_assert_eq!(a_rows, rows);
        debug_assert_eq!(b_columns, columns);

        let output = self.new_tensor_op(rows, columns);

        self.add_recording_operation(Op::MatmulvAdd{lhs: a, rhs: b, added, output});

        output
    }

    pub fn matmul_onehotv_add(&mut self, a: DiffTensorPtr, b: OneHotIndex, added: DiffTensorPtr) -> DiffTensorPtr
    {
        let (a_rows, _a_columns) = self.tensor_shape(a.as_value());
        let b_columns = 1;

        let (rows, columns) = self.tensor_shape(added.as_value());

        debug_assert_eq!(a_rows, rows);
        debug_assert_eq!(b_columns, columns);

        let output = self.new_tensor_op(rows, columns);

        self.add_recording_operation(Op::MatmulOneHotvAdd{lhs: a, rhs: b, added, output});

        output
    }

    pub fn sum_tensor(&mut self, a: DiffTensorPtr) -> DiffScalar
    {
        let output = self.new_value_op();

        self.add_recording_operation(Op::SumTensor{value: a, output});

        output
    }

    pub fn dot(&mut self, a: DiffTensorPtr, b: DiffTensorPtr) -> DiffScalar
    {
        debug_assert_eq!(self.tensor_shape(a.as_value()), self.tensor_shape(b.as_value()));

        let output = self.new_value_op();

        self.add_recording_operation(Op::Dot{lhs: a, rhs: b, output});

        output
    }

    pub fn pow(&mut self, a: DiffTensorPtr, power: i32) -> DiffTensorPtr
    {
        let (rows, columns) = self.tensor_shape(a.as_value());

        let output = self.new_tensor_op(rows, columns);

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
        let (rows, columns) = self.tensor_shape(values.as_value());

        debug_assert_eq!(columns, 1);

        let softmaxed_output = self.memory.new_tensor(false, TensorMemoryValue::Size{rows, columns});
        let output = self.new_value_op();

        self.add_recording_operation(Op::SoftmaxCrossEntropy{values, targets, softmaxed_output, output});

        (softmaxed_output, output)
    }

    pub fn tensor_shape(&self, tensor: TensorPtr) -> (usize, usize)
    {
        self.memory.tensor_shape(tensor)
    }

    pub fn resolve_tensor_ptr(&self, index_ptr: TensorPtr) -> TensorIndex
    {
        debug_assert_eq!(self.state, RecorderState::Ready);

        self.memory.tensors_memory[index_ptr.0].memory.unwrap_or_else(||
        {
            panic!("{} must be resolved", self.memory.format_variable(index_ptr))
        })
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

            if !self.memory.store_tensors_check.contains(&key)
            {
                self.memory.store_tensors_check.push(key);
            }
        }

        self.memory.tensor_live_ranges[index_ptr.0].end = Some(i32::MAX);
    }

    pub fn store_value_until_end(&mut self, index: ValueIndex)
    {
        debug_assert!(self.state.before_or_at(RecorderState::AwaitingGradient));

        #[cfg(debug_assertions)]
        {
            let key = StoreCheckKey::PreResolve(index);

            if !self.memory.store_values_check.contains(&key)
            {
                self.memory.store_values_check.push(key);
            }
        }

        self.memory.value_live_ranges[index.0].end = Some(i32::MAX);
    }

    pub fn begin_loop(&mut self, inputs: Vec<InputTypePtr>) -> LoopIndex
    {
        debug_assert_eq!(self.state, RecorderState::Recording);
        debug_assert_matches!(self.operations_target, OperationsTarget::Normal);

        #[cfg(debug_assertions)]
        {
            inputs.iter().for_each(|input|
            {
                if let InputTypePtr::Normal(tensor) = input
                {
                    self.memory.set_tensors_check.push((*tensor).into());
                }
            });
        }

        let id = LoopIndex(self.loops.len());
        let operation_index = LoopOperationIndex(self.recording_operations.len());

        let loops_values_index = LoopValuesIndex(self.loops_memory.loops_values.len());
        self.loops_memory.loops_values.push(LoopValues::default());

        let loops_stack_index = LoopStackIndex(self.loops_memory.loops_stack.len());
        self.loops_memory.loops_stack.push(LoopStack::default());

        self.loops.push(LoopInfo{
            times: 0,
            times_total: 0,
            current_index: 0,
            reversed: false,
            live_range: LiveRange::default(),
            loops_gradient: None,
            gradient_of_loop: None,
            kept_inside: Vec::new(),
            defined_values: Vec::new(),
            used_values: Vec::new(),
            input_values: loops_values_index,
            stack_values: loops_stack_index,
            inputs: Vec::new(),
            #[cfg(debug_assertions)]
            expected_pairs: Vec::new()
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
        let this_loop = &mut self.loops[index.0];

        this_loop.times_total = times;

        if let Some(loops_gradient) = this_loop.loops_gradient
        {
            self.set_loop_times(loops_gradient, times);
        }
    }

    pub fn set_loop_inputs(&mut self, index: LoopIndex, inputs: impl Into<LoopInputs>)
    {
        let inputs = inputs.into();

        self.loops_memory.loops_values[self.loops[index.0].input_values.0].input_values = inputs;
    }

    pub fn calculate_feedforward(&mut self)
    {
        let count = self.feedforward_operations_count;

        self.calculate_steps(|_, _| { unimplemented!() }, 0, count);
    }

    pub fn calculate_feedforward_with_dependent(
        &mut self,
        dependent_inputs: impl FnMut(usize, &OperationsRecorderMemory) -> OwnedInputType
    )
    {
        let count = self.feedforward_operations_count;

        self.calculate_steps(dependent_inputs, 0, count);
    }

    pub fn calculate_backpropagate(&mut self)
    {
        let total = self.raw_operations.len();
        let count = self.feedforward_operations_count;

        self.calculate_steps(|_, _| { unimplemented!() }, count, total);
    }

    pub fn calculate(&mut self)
    {
        let total = self.raw_operations.len();

        self.calculate_steps(|_, _| { unimplemented!() }, 0, total);
    }

    fn calculate_steps<DependentInputs: FnMut(usize, &OperationsRecorderMemory) -> OwnedInputType>(
        &mut self,
        mut dependent_inputs: DependentInputs,
        start: usize,
        end: usize
    )
    {
        debug_assert_eq!(self.state, RecorderState::Ready);

        self.memory.phi_other_selectors_values.iter_mut().for_each(|selector| selector.is_set = false);

        for loop_index in 0..self.loops.len()
        {
            let loop_info = &mut self.loops[loop_index];

            let inputs_count = loop_info.inputs.len();

            debug_assert!(loop_info.times_total > 0, "LoopIndex({loop_index}) was uninitialized or has 0 iterations");

            if inputs_count > 0
            {
                loop_info.current_index = match &self.loops_memory.loops_values[loop_info.input_values.0].input_values
                {
                    LoopInputs::Values(input_values) =>
                    {
                        let input_values_amount = input_values.len();

                        debug_assert_eq!(loop_info.times_total * loop_info.inputs.len(), input_values_amount);

                        if loop_info.reversed
                        {
                            loop_info.times_total - 1
                        } else
                        {
                            0
                        }
                    },
                    LoopInputs::Dependent(_) =>
                    {
                        debug_assert!(!loop_info.reversed);

                        0
                    }
                };
            }

            loop_info.times = loop_info.times_total;
        }

        #[cfg(debug_assertions)]
        {
            self.memory.tensor_inputs.iter().for_each(|input_tensor_ptr|
            {
                let contains_ptr = self.memory.set_tensors_check.iter()
                    .filter_map(|x| if let InputCheckType::Ptr(x) = x { Some(x) } else { None })
                    .any(|x| x == input_tensor_ptr);

                if contains_ptr
                {
                    return;
                }

                if let Some(input_memory_index) = self.memory.tensors_memory[input_tensor_ptr.0].memory
                {
                    let contains_index = self.memory.set_tensors_check.iter()
                        .filter_map(|x| if let InputCheckType::Index(x) = x { Some(x) } else { None })
                        .any(|x| *x == input_memory_index);

                    assert!(contains_index, "{} ({input_memory_index:?}) wasnt set", self.memory.format_variable(*input_tensor_ptr));
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
                    self.memory.tensors_raw_data.copy_within($src.range(), $dst.raw_index.0)
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
                            #[allow(unused_variables, unused_mut)]
                            let mut is_first = true;

                            $(
                                if !is_first { eprint!(", "); }

                                let var_name = self.memory.format_tensor_raw_ptr(*$t_name);
                                eprint!("{}: {} = {:?}", stringify!($t_name), var_name, &self.memory.tensors_raw_data[$t_name.range()]);

                                is_first = false;
                            )*

                            $(
                                if !is_first { eprint!(", "); }

                                let var_name = self.memory.format_variable(*$v_name);
                                eprint!("{}: {} = {:?}", stringify!($v_name), var_name, &self.memory.values[$v_name.0]);

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
                    let _counter: [(); _] = [$({ let _ = stringify!($t_name); () },)* $({ let _ = stringify!($v_name); () },)*];

                    #[cfg(debug_assertions)]
                    {
                        $(
                            self.memory.verify_raw_ptr_use(*$t_name);
                        )*

                        if PRINT_CALCULATE_VALUES
                        {
                            eprint!("{}", stringify!($name));

                            if _counter.len() > 0
                            {
                                eprint!(" (BEFORE ");
                            }
                        }
                    }

                    {
                        debug_calculate_common!(($($t_name,)*),($($v_name,)*));
                    }

                    #[cfg(debug_assertions)]
                    {
                        if PRINT_CALCULATE_VALUES
                        {
                            if _counter.len() > 0
                            {
                                eprint!(") (AFTER ");
                            } else
                            {
                                eprint!(" (");
                            }
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

                        $(
                            self.memory.verify_raw_ptr_assign(*$t_name);
                        )*
                    }
                }
            }

            macro_rules! debug_print_op
            {
                ($x:expr) =>
                {
                    #[cfg(debug_assertions)]
                    {
                        if PRINT_CALCULATE_VALUES
                        {
                            eprintln!("{:?}", $x);
                        }
                    }
                }
            }

            //let before_instant = std::time::Instant::now();
            match gradient_op
            {
                GradientOp::None
                | GradientOp::FeedforwardEndMarker => unreachable!(),
                GradientOp::ZeroValue(dst) =>
                {
                    debug_calculate_values!(ZeroValue, (),());

                    self.memory.values[dst.0] = 0.0;

                    debug_calculate_values_result!((),(dst));
                },
                GradientOp::ZeroTensor(dst) =>
                {
                    debug_calculate_values!(ZeroTensor, (),());

                    self.memory.tensors_raw_data[dst.range()].fill(0.0);

                    debug_calculate_values_result!((dst),());
                },
                GradientOp::PushStackValue{loop_index, value} =>
                {
                    debug_calculate_values!(PushStackValue, (),(value));

                    let loop_stack_index = self.loops[loop_index.0].stack_values;

                    #[allow(unused_mut)]
                    let mut stack_value: LoopStackValue<_, _> = self.memory.values[value.0].into();
                    stack_value.set_source(*value);

                    self.loops_memory.loops_stack[loop_stack_index.0].values_stack.push(stack_value);

                    debug_calculate_values_result!((),());
                },
                GradientOp::PushStackTensor{loop_index, tensor} =>
                {
                    debug_calculate_values!(PushStackTensor, (tensor),());

                    let loop_stack_index = self.loops[loop_index.0].stack_values;

                    let tensor_ref = LayerTypeRef::from_data_with_start(&self.memory.tensors_raw_data, *tensor);

                    #[allow(unused_mut)]
                    let mut stack_value: LoopStackValue<_, _> = tensor_ref.clone_owned().into();
                    stack_value.set_source(*tensor);

                    self.loops_memory.loops_stack[loop_stack_index.0].tensors_stack.push(stack_value);

                    debug_calculate_values_result!((),());
                },
                GradientOp::PopStackValue{loop_index, output} =>
                {
                    debug_calculate_values!(PopStackValue, (),());

                    let loop_stack_index = self.loops[loop_index.0].stack_values;

                    let stack_value = self.loops_memory.loops_stack[loop_stack_index.0].values_stack.pop()
                        .expect("stack must not be empty");

                    self.memory.values[output.0] = stack_value.get_stack_value_for(self, *loop_index, *output);

                    debug_calculate_values_result!((),(output));
                },
                GradientOp::PopStackTensor{loop_index, output} =>
                {
                    debug_calculate_values!(PopStackTensor, (),());

                    let loop_stack_index = self.loops[loop_index.0].stack_values;

                    let stack_value = self.loops_memory.loops_stack[loop_stack_index.0].tensors_stack.pop()
                        .expect("stack must not be empty");

                    let tensor = stack_value.get_stack_value_for(self, *loop_index, *output);

                    self.memory.set_tensor_raw(*output, &tensor);

                    debug_calculate_values_result!((output),());
                },
                GradientOp::SetInputs(loop_index) =>
                {
                    debug_print_op!(gradient_op);

                    let loop_info = &self.loops[loop_index.0];

                    let inputs_count = loop_info.inputs.len();

                    (0..inputs_count).for_each(|input_index|
                    {
                        if let Some(input_ptr) = loop_info.inputs[input_index]
                        {
                            let values = &mut self.loops_memory.loops_values[loop_info.input_values.0].input_values;
                            let value = match values
                            {
                                LoopInputs::Values(values) => values[loop_info.current_index * inputs_count + input_index].clone(),
                                LoopInputs::Dependent(i) => (dependent_inputs)(*i, &self.memory)
                            };

                            self.memory.set_input(input_ptr, value);
                        }
                    });
                },
                GradientOp::SetOtherSelector(index) =>
                {
                    debug_print_op!(gradient_op);

                    self.memory.phi_other_selectors_values[index.0].is_set = true;
                },
                GradientOp::GetOtherSelectorValue{info: index, first, other, output} =>
                {
                    let this_selector = &mut self.memory.phi_other_selectors_values[index.0];

                    let src = if this_selector.is_set
                    {
                        debug_calculate_values!(GetOtherSelectorValue, (),(other));

                        other
                    } else
                    {
                        debug_calculate_values!(GetOtherSelectorValue, (),(first));

                        first
                    };

                    self.memory.values[output.0] = self.memory.values[src.0];

                    debug_calculate_values_result!((),(output));
                },
                GradientOp::GetOtherSelectorTensor{info: index, first, other, output} =>
                {
                    let this_selector = &mut self.memory.phi_other_selectors_values[index.0];

                    let src = if this_selector.is_set
                    {
                        debug_calculate_values!(GetOtherSelectorTensor, (other),());

                        other
                    } else
                    {
                        debug_calculate_values!(GetOtherSelectorTensor, (first),());

                        first
                    };

                    copy_tensor!(src, output);

                    debug_calculate_values_result!((output),());
                },
                GradientOp::OtherSelectorValueGradient{index, first, other, src} =>
                {
                    debug_calculate_values!(OtherSelectorValueGradient, (),(src));

                    let this_selector = &mut self.memory.phi_other_selectors_values[index.0];

                    let dst = if this_selector.is_set
                    {
                        first
                    } else
                    {
                        other
                    };

                    self.memory.values[dst.0] = self.memory.values[src.0];

                    if this_selector.is_set
                    {
                        debug_calculate_values_result!((),(first));
                    } else
                    {
                        debug_calculate_values_result!((),(other));
                    }
                },
                GradientOp::OtherSelectorTensorGradient{index, first, other, src} =>
                {
                    debug_calculate_values!(OtherSelectorTensorGradient, (src),());

                    let this_selector = &mut self.memory.phi_other_selectors_values[index.0];

                    let dst = if this_selector.is_set
                    {
                        first
                    } else
                    {
                        other
                    };

                    copy_tensor!(src, dst);

                    if this_selector.is_set
                    {
                        debug_calculate_values_result!((first),());
                    } else
                    {
                        debug_calculate_values_result!((other),());
                    }
                },
                GradientOp::GradientSelectAddValue{index, first, other, src, added} =>
                {
                    let this_selector = &mut self.memory.phi_other_selectors_values[index.0];

                    let src_value = self.memory.values[src.0];

                    let is_set = this_selector.is_set;

                    if is_set
                    {
                        debug_calculate_values!(GradientSelectAddValue, (),(src, added));

                        self.memory.values[other.0] = src_value + self.memory.values[added.0];
                    } else
                    {
                        debug_calculate_values!(GradientSelectAddValue, (),(src));

                        self.memory.values[first.0] = src_value;
                    }

                    if is_set
                    {
                        debug_calculate_values_result!((),(other));
                    } else
                    {
                        debug_calculate_values_result!((),(first));
                    }
                },
                GradientOp::GradientSelectAddTensor{index, first, other, src, added} =>
                {
                    let this_selector = &mut self.memory.phi_other_selectors_values[index.0];

                    let is_set = this_selector.is_set;

                    if is_set
                    {
                        debug_calculate_values!(GradientSelectAddTensor, (src, added),());

                        let (other, src, added) = get_disjoint_mut!(
                            (LayerTypeMut, other, x0),
                            (LayerTypeRef, src, x1),
                            (LayerTypeRef, added, x2)
                        );

                        other.add_to(src, added);
                    } else
                    {
                        debug_calculate_values!(GradientSelectAddTensor, (src),());

                        copy_tensor!(src, first);
                    }

                    if is_set
                    {
                        debug_calculate_values_result!((other),());
                    } else
                    {
                        debug_calculate_values_result!((first),());
                    }
                },
                GradientOp::IfSetTensor{index, dst, src} =>
                {
                    if self.memory.phi_other_selectors_values[index.0].is_set
                    {
                        debug_calculate_values!(IfSetTensor, (src),());

                        copy_tensor!(src, dst);

                        debug_calculate_values_result!((dst),());
                    }
                },
                GradientOp::IfNotSetTensor{index, dst, src} =>
                {
                    if !self.memory.phi_other_selectors_values[index.0].is_set
                    {
                        debug_calculate_values!(IfNotSetTensor, (src),());

                        copy_tensor!(src, dst);

                        debug_calculate_values_result!((dst),());
                    }
                },
                GradientOp::SetOtherSelectorGradient{loop_index, selector_index} =>
                {
                    debug_print_op!(gradient_op);

                    if self.loops[loop_index.0].times <= 1
                    {
                        debug_assert!(!self.memory.phi_other_selectors_values[selector_index.0].is_set);

                        self.memory.phi_other_selectors_values[selector_index.0].is_set = true;
                    }
                },
                GradientOp::Jump(RawJumpInfo{loop_index, operation_index}) =>
                {
                    debug_print_op!(gradient_op);

                    let loop_info = &mut self.loops[loop_index.0];
                    if loop_info.times > 1
                    {
                        current_index = operation_index.0;

                        loop_info.times -= 1;

                        if loop_info.reversed
                        {
                            loop_info.current_index = loop_info.current_index.saturating_sub(1);
                        } else
                        {
                            loop_info.current_index += 1;
                        }

                        continue;
                    }
                },
                GradientOp::Copy{src, dst} =>
                {
                    debug_calculate_values!(Copy, (src),());
                    copy_tensor!(src, dst);
                    debug_calculate_values_result!((dst),());
                },
                GradientOp::CopyScalar{src, dst} =>
                {
                    debug_calculate_values!(CopyScalar, (),(src));
                    self.memory.values[dst.0] = self.memory.values[src.0];
                    debug_calculate_values_result!((),(dst));
                },
                GradientOp::AddScalars{lhs, rhs, output} =>
                {
                    debug_calculate_values!(AddScalars, (),(lhs, rhs));
                    self.memory.values[output.0] = self.memory.values[lhs.0] + self.memory.values[rhs.0];
                    debug_calculate_values_result!((),(output));
                },
                GradientOp::AddScalar{lhs, rhs, output} =>
                {
                    debug_calculate_values!(AddScalar, (lhs),(rhs));
                    copy_tensor!(lhs, output);

                    LayerTypeMut::from_data_with_start(&mut self.memory.tensors_raw_data, *output).add_scalar_inplace(self.memory.values[rhs.0]);
                    debug_calculate_values_result!((output),());
                },
                GradientOp::Add{lhs, rhs, output} =>
                {
                    debug_calculate_values!(Add, (lhs, rhs),());

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
                GradientOp::Sub{lhs, rhs, output} =>
                {
                    debug_calculate_values!(Sub, (lhs, rhs),());

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
                    debug_calculate_values!(SubFromScalar, (rhs),(lhs));

                    {
                        let (output, rhs) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, rhs, x1)
                        );

                        output.sub_from_scalar(self.memory.values[lhs.0], rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::MulScalar{lhs, rhs, output} =>
                {
                    debug_calculate_values!(MulScalar, (lhs),(rhs));
                    copy_tensor!(lhs, output);

                    LayerTypeMut::from_data_with_start(&mut self.memory.tensors_raw_data, *output).mul_scalar_inplace(self.memory.values[rhs.0]);
                    debug_calculate_values_result!((output),());
                },
                GradientOp::MulScalars{lhs, rhs, output} =>
                {
                    debug_calculate_values!(MulScalars, (),(lhs, rhs));
                    self.memory.values[output.0] = self.memory.values[lhs.0] * self.memory.values[rhs.0];
                    debug_calculate_values_result!((),(output));
                },
                GradientOp::MulComponentwise{lhs, rhs, output} =>
                {
                    debug_calculate_values!(MulComponentwise, (lhs, rhs),());

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
                    debug_calculate_values!(MulComponentwise, (lhs, rhs, added),());

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
                    debug_calculate_values!(SumTensor, (value),());
                    self.memory.values[output.0] = self.memory.tensors_raw_data[value.range()].iter().sum();
                    debug_calculate_values_result!((),(output));
                },
                GradientOp::Dot{lhs, rhs, output} =>
                {
                    debug_calculate_values!(Dot, (lhs, rhs),());
                    let lhs = LayerTypeRef::from_data_with_start(&self.memory.tensors_raw_data, *lhs);
                    let rhs = LayerTypeRef::from_data_with_start(&self.memory.tensors_raw_data, *rhs);

                    self.memory.values[output.0] = lhs.dot(rhs);
                    debug_calculate_values_result!((),(output));
                },
                GradientOp::Fill{value, output} =>
                {
                    debug_calculate_values!(Fill, (),(value));
                    self.memory.tensors_raw_data[output.range()].fill(self.memory.values[value.0]);
                    debug_calculate_values_result!((output),());
                },
                GradientOp::Pow{lhs, power, output} =>
                {
                    debug_calculate_values!(Pow, (lhs),());
                    copy_tensor!(lhs, output);

                    LayerTypeMut::from_data_with_start(&mut self.memory.tensors_raw_data, *output).pow_inplace(*power);
                    debug_calculate_values_result!((output),());
                },
                GradientOp::Sigmoid{value, output} =>
                {
                    debug_calculate_values!(Sigmoid, (value),());
                    copy_tensor!(value, output);

                    LayerTypeMut::from_data_with_start(&mut self.memory.tensors_raw_data, *output).sigmoid_inplace();
                    debug_calculate_values_result!((output),());
                },
                GradientOp::SigmoidDiff{value, gradient, output} =>
                {
                    debug_calculate_values!(SigmoidDiff, (value, gradient),());

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
                    debug_calculate_values!(Tanh, (value),());
                    copy_tensor!(value, output);

                    LayerTypeMut::from_data_with_start(&mut self.memory.tensors_raw_data, *output).tanh_inplace();
                    debug_calculate_values_result!((output),());
                },
                GradientOp::TanhDiff{value, gradient, output} =>
                {
                    debug_calculate_values!(TanhDiff, (value, gradient),());

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
                    debug_calculate_values!(LeakyRelu, (value),());
                    copy_tensor!(value, output);

                    LayerTypeMut::from_data_with_start(&mut self.memory.tensors_raw_data, *output).leaky_relu_inplace();
                    debug_calculate_values_result!((output),());
                },
                GradientOp::LeakyReluDiff{value, gradient, output} =>
                {
                    debug_calculate_values!(LeakyReluDiff, (value, gradient),());

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
                    debug_calculate_values!(SoftmaxCrossEntropy, (values),());

                    {
                        copy_tensor!(values, softmaxed_output);

                        let mut softmaxed_output = LayerTypeMut::from_data_with_start(&mut self.memory.tensors_raw_data, *softmaxed_output);

                        self.memory.values[output.0] = softmaxed_output.softmax_cross_entropy_inplace(&self.memory.one_hot_layers[targets.0]);
                    }

                    debug_calculate_values_result!((softmaxed_output),(output));
                },
                GradientOp::SoftmaxCrossEntropyNoSoftmaxed{values, targets, output} =>
                {
                    debug_calculate_values!(SoftmaxCrossEntropyNoSoftmaxed, (values),());
                    let values = LayerTypeRef::from_data_with_start(&self.memory.tensors_raw_data, *values);

                    self.memory.values[output.0] = values.softmax_cross_entropy(&self.memory.one_hot_layers[targets.0]);
                    debug_calculate_values_result!((),(output));
                },
                GradientOp::SoftmaxCrossEntropyDiff{softmaxed_values, gradient, targets, output} =>
                {
                    debug_calculate_values!(SoftmaxCrossEntropyDiff, (softmaxed_values),(gradient));

                    {
                        debug_assert_eq!(
                            (softmaxed_values.rows, softmaxed_values.columns), (self.memory.one_hot_layers[targets.0].size, 1),
                            "softmaxed: {softmaxed_values:?}, targets: {targets:?}"
                        );

                        let (mut output, softmaxed_values) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeRef, softmaxed_values, x1)
                        );

                        output.sub_to(softmaxed_values, LayerTypeRef::from(&self.memory.one_hot_layers[targets.0].clone().into_layer()));

                        output.mul_scalar_inplace(self.memory.values[gradient.0]);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::Matmulv{lhs, rhs, output} =>
                {
                    debug_calculate_values!(Matmulv, (lhs, rhs),());

                    {
                        let (output, lhs, rhs) = get_disjoint_mut!(
                            (LayerTypeVectorMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeVectorRef, rhs, x2)
                        );

                        output.matmulv_into(lhs, rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::MatmulvAdd{lhs, rhs, added, output} =>
                {
                    debug_calculate_values!(MatmulvAdd, (lhs, rhs, added),());

                    {
                        let (output, lhs, rhs, added) = get_disjoint_mut!(
                            (LayerTypeVectorMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeVectorRef, rhs, x2),
                            (LayerTypeVectorRef, added, x3)
                        );

                        output.matmulv_add_into(lhs, rhs, added);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::MatmulOneHotvAdd{lhs, rhs, added, output} =>
                {
                    debug_calculate_values!(MatmulOneHotvAdd, (lhs, added),());

                    {
                        let (output, lhs, added) = get_disjoint_mut!(
                            (LayerTypeVectorMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeVectorRef, added, x2)
                        );

                        output.matmul_onehotv_add_into(lhs, &self.memory.one_hot_layers[rhs.0], added);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::MatmulvTransposed{lhs, rhs, output} =>
                {
                    debug_calculate_values!(MatmulvTransposed, (lhs, rhs),());

                    {
                        let (output, lhs, rhs) = get_disjoint_mut!(
                            (LayerTypeVectorMut, output, x0),
                            (LayerTypeRef, lhs, x1),
                            (LayerTypeVectorRef, rhs, x2)
                        );

                        output.matmulv_transposed_into(lhs, rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::OuterProduct{lhs, rhs, output} =>
                {
                    debug_calculate_values!(OuterProduct, (lhs, rhs),());

                    {
                        let (output, lhs, rhs) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeVectorRef, lhs, x1),
                            (LayerTypeVectorRef, rhs, x2)
                        );

                        output.outer_product_into(lhs, rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::OuterProductAdd{lhs, rhs, added, output} =>
                {
                    debug_calculate_values!(OuterProductAdd, (lhs, rhs, added),());

                    debug_assert_eq!(added, output);

                    {
                        let (output, lhs, rhs) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeVectorRef, lhs, x1),
                            (LayerTypeVectorRef, rhs, x2)
                        );

                        output.outer_product_add_inplace(lhs, rhs);
                    }

                    debug_calculate_values_result!((output),());
                },
                GradientOp::OuterProductOneHot{lhs, rhs, output} =>
                {
                    debug_calculate_values!(OuterProductOneHot, (lhs),());

                    {
                        let (output, lhs) = get_disjoint_mut!(
                            (LayerTypeMut, output, x0),
                            (LayerTypeVectorRef, lhs, x1)
                        );

                        output.outer_product_one_hot_into(lhs, &self.memory.one_hot_layers[rhs.0]);
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

        fn handle_op(
            target: &mut Vec<StandardGradientOp>,
            phi_other_selectors_values: &mut Vec<PhiOtherSelectorValue>,
            phi_other_selectors_recording: &mut [PhiOtherSelectorRecording],
            loops: &mut [LoopInfo],
            op: &Op
        )
        {
            let new_op = match op
            {
                Op::CopyScalar{src, dst} =>
                {
                    GradientOp::CopyScalar{src: src.as_value(), dst: dst.as_value()}
                },
                Op::Copy{src, dst} =>
                {
                    GradientOp::Copy{src: src.as_value(), dst: dst.as_value()}
                },
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
                Op::SetOtherSelector(index) =>
                {
                    GradientOp::SetOtherSelector(phi_other_selectors_recording[index.0].value_index.unwrap())
                },
                Op::GetOtherSelectorValue{index, ..}
                | Op::GetOtherSelectorTensor{index, ..} =>
                {
                    let this_selector = &mut phi_other_selectors_recording[index.0];

                    let value_index = this_selector.value_index.expect("must be initialized");

                    let first = this_selector.first.as_value();
                    let other = this_selector.other.expect("must be initialized").as_value();

                    match op
                    {
                        Op::GetOtherSelectorValue{output, ..} =>
                        {
                            GradientOp::GetOtherSelectorValue{
                                info: value_index,
                                first: first.into_value(),
                                other: other.into_value(),
                                output: output.as_value()
                            }
                        },
                        Op::GetOtherSelectorTensor{output, ..} =>
                        {
                            GradientOp::GetOtherSelectorTensor{
                                info: value_index,
                                first: first.into_tensor(),
                                other: other.into_tensor(),
                                output: output.as_value()
                            }
                        },
                        _ => unreachable!()
                    }
                },
                Op::Loop{index, inputs, ops} =>
                {
                    target.push(GradientOp::Jump(JumpInfo::JumpTo{inputs: inputs.clone(), index: *index}));

                    ops.iter().for_each(|op|
                    {
                        if let Op::SetOtherSelector(phi_selector_index) = op
                        {
                            let this_selector = &mut phi_other_selectors_recording[phi_selector_index.0];

                            let new_index = PhiOtherSelectorIndex(phi_other_selectors_values.len());
                            phi_other_selectors_values.push(PhiOtherSelectorValue{
                                loop_index: *index,
                                is_set: false
                            });

                            this_selector.value_index = Some(new_index);
                        }
                    });

                    ops.iter().for_each(|inner_op|
                    {
                        handle_op(
                            target,
                            phi_other_selectors_values,
                            phi_other_selectors_recording,
                            loops,
                            inner_op
                        );

                        inner_op.for_outputs(|output|
                        {
                            let output = output.as_value();

                            loops[index.0].defined_values.push(output);

                            let stack_op = match output
                            {
                                DiffValue::Value(value) => GradientOp::PushStackValue{loop_index: *index, value},
                                DiffValue::Tensor(tensor) => GradientOp::PushStackTensor{loop_index: *index, tensor},
                                DiffValue::OneHot(_) => unimplemented!()
                            };

                            target.push(stack_op);
                        });
                    });

                    target.push(GradientOp::Jump(JumpInfo::JumpFrom(*index)));

                    return;
                }
            };

            target.push(new_op);
        }

        self.recording_operations.iter().for_each(|op|
        {
            handle_op(
                &mut self.gradient_operations,
                &mut self.memory.phi_other_selectors_values,
                &mut self.phi_other_selectors_recording,
                &mut self.loops,
                op
            )
        });

        self.gradient_operations.push(GradientOp::FeedforwardEndMarker);

        self.state = RecorderState::AwaitingGradient;
    }

    fn is_ptr_output(&self, ptr: TensorPtr) -> bool
    {
        self.memory.tensor_live_ranges[ptr.0].end == Some(i32::MAX)
    }

    fn remove_unused_pushes(&mut self)
    {
        self.gradient_operations.retain(|op|
        {
            let is_stack_used = |loop_index: LoopIndex, value: DiffValue|
            {
                self.loops[loop_index.0].used_values.contains(&value)
            };

            match op
            {
                GradientOp::PushStackValue{loop_index, value} => is_stack_used(*loop_index, (*value).into()),
                GradientOp::PushStackTensor{loop_index, tensor} => is_stack_used(*loop_index, (*tensor).into()),
                _ => true
            }
        });
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

                    for (_, check_op) in self.gradient_operations.iter().enumerate().filter(|(x_index, _)| *x_index != i)
                    {
                        let mut any_is_src = false;
                        let mut any_is_dst = false;

                        let mut f = |v|
                        {
                            if v == src { any_is_src = true }
                            if v == dst { any_is_dst = true }
                        };

                        check_op.for_args(|arg| if let DiffValue::Tensor(t_arg) = arg { f(t_arg) });
                        check_op.for_outputs(|out| if let DiffValue::Tensor(t_out) = out { f(t_out) });

                        overlaps_args = any_is_src && any_is_dst;

                        if overlaps_args
                        {
                            break;
                        }
                    }

                    if !overlaps_args
                    {
                        for check_index in 0..self.gradient_operations.len()
                        {
                            if check_index != i
                            {
                                self.replace_op_args(check_index, dst.into(), src.into())
                            }
                        }

                        self.gradient_operations.remove(i);

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

    pub fn tensors_raw_data(&self) -> &[f32]
    {
        &self.memory.tensors_raw_data
    }

    fn calculate_live_ranges_once(&mut self) -> bool
    {
        let mut inside_loop_values = Vec::new();

        {
            let mut inside_loop: Option<LoopIndex> = None;

            self.gradient_operations.iter().for_each(|op|
            {
                let memory = &mut self.memory;

                let mut set_loop_lifetime = |inside_loop: Option<LoopIndex>, live_range: &mut LiveRange, value: DiffValue|
                {
                    if let Some(loop_index) = inside_loop
                    {
                        let loop_range = self.loops[loop_index.0].live_range.clone();

                        *live_range = loop_range;

                        inside_loop_values.push(value);
                    }
                };

                match op
                {
                    GradientOp::Jump(JumpInfo::JumpTo{index, ..}) =>
                    {
                        inside_loop = Some(*index);
                    },
                    GradientOp::Jump(JumpInfo::JumpFrom(_)) =>
                    {
                        debug_assert!(inside_loop.is_some());

                        inside_loop = None;
                    },
                    GradientOp::GetOtherSelectorValue{..}
                    | GradientOp::GetOtherSelectorTensor{..} =>
                    {
                        let (live_range, value): (_, DiffValue) = match op
                        {
                            GradientOp::GetOtherSelectorValue{other, ..} => (&mut memory.value_live_ranges[other.0], (*other).into()),
                            GradientOp::GetOtherSelectorTensor{other, ..} => (&mut memory.tensor_live_ranges[other.0], (*other).into()),
                            _ => unreachable!()
                        };

                        set_loop_lifetime(inside_loop, live_range, value);
                    },
                    GradientOp::OtherSelectorValueGradient{..}
                    | GradientOp::OtherSelectorTensorGradient{..}
                    | GradientOp::IfNotSetTensor{..} =>
                    {
                        let (live_range, value): (_, DiffValue) = match op
                        {
                            GradientOp::OtherSelectorValueGradient{other, ..} => (&mut memory.value_live_ranges[other.0], (*other).into()),
                            GradientOp::OtherSelectorTensorGradient{other, ..}
                            | GradientOp::IfNotSetTensor{dst: other, ..} => (&mut memory.tensor_live_ranges[other.0], (*other).into()),
                            _ => unreachable!()
                        };

                        set_loop_lifetime(inside_loop, live_range, value);
                    },
                    _ => ()
                }
            });
        }

        self.gradient_operations.iter().enumerate().for_each(|(op_index, op)|
        {
            let handle_output = |live_range: &mut LiveRange, allow_reuse: bool, err_name: String|
            {
                let start = &mut live_range.start;

                let new_start = op_index as i32;

                if allow_reuse
                {
                    *start = Some(start.map(|start| start.min(new_start)).unwrap_or(new_start));
                } else
                {
                    debug_assert!(start.is_none(), "{err_name} was reused at operation {} and {op_index}", start.unwrap());

                    *start = Some(new_start);
                }
            };

            let is_allow_reuse = |value: DiffValue| -> bool
            {
                inside_loop_values.contains(&value)
            };

            if !matches!(op, GradientOp::OtherSelectorValueGradient{..})
            {
                op.for_outputs(|output|
                {
                    let allow_reuse = is_allow_reuse(output);
                    let name = self.memory.format_variable(output);

                    match output
                    {
                        DiffValue::Tensor(tensor_ptr) => handle_output(&mut self.memory.tensor_live_ranges[tensor_ptr.0], allow_reuse, name),
                        DiffValue::Value(value_index) => handle_output(&mut self.memory.value_live_ranges[value_index.0], allow_reuse, name),
                        DiffValue::OneHot(_) => unimplemented!()
                    }
                });
            }
        });

        self.gradient_operations.iter().enumerate().rev().for_each(|(op_index, op)|
        {
            let allow_end_before = match op
            {
                GradientOp::GetOtherSelectorValue{..}
                | GradientOp::GetOtherSelectorTensor{..}
                | GradientOp::GradientSelectAddValue{..}
                | GradientOp::GradientSelectAddTensor{..} => true,
                _ => false
            };

            let handle_arg = |live_range: &mut LiveRange, err_name: String|
            {
                if let Some(start) = live_range.start
                {
                    if !allow_end_before && start >= op_index as i32
                    {
                        panic!("{err_name} was defined at {start} after being used at {op_index}");
                    }
                }

                let new_end = op_index as i32;
                live_range.end = Some(live_range.end.map(|previous_end| previous_end.max(new_end)).unwrap_or(new_end));
            };

            op.for_args(|arg|
            {
                let name = self.memory.format_variable(arg);

                let live_range = match arg
                {
                    DiffValue::Tensor(tensor_ptr) => &mut self.memory.tensor_live_ranges[tensor_ptr.0],
                    DiffValue::Value(value_index) => &mut self.memory.value_live_ranges[value_index.0],
                    DiffValue::OneHot(_) => return
                };

                handle_arg(live_range, name);
            });
        });

        self.loops.iter().for_each(|loop_info|
        {
            loop_info.kept_inside.iter().for_each(|kept_inside|
            {
                let live_range = match kept_inside
                {
                    DiffValue::Tensor(tensor) => &mut self.memory.tensor_live_ranges[tensor.0],
                    DiffValue::Value(value) => &mut self.memory.value_live_ranges[value.0],
                    DiffValue::OneHot(_) => return
                };

                if live_range.end.map(|x| x < loop_info.live_range.end.unwrap()).unwrap_or(false)
                {
                    live_range.end = loop_info.live_range.end;
                }
            });
        });

        let mut any_unused = false;
        self.gradient_operations.iter_mut().for_each(|op|
        {
            let mut any_unused_single: bool = false;
            let mut all_unused: Option<bool> = None;

            let mut handle_output = |live_range: &mut LiveRange, err_name: String|
            {
                let is_unused = live_range.end.is_none();
                any_unused_single |= is_unused;

                debug_assert!(live_range.start != Some(-1), "{err_name} is an unused input");

                if let Some(all_unused) = all_unused.as_mut()
                {
                    *all_unused &= is_unused;
                } else
                {
                    all_unused = Some(is_unused);
                }
            };

            op.for_outputs(|output|
            {
                match output
                {
                    DiffValue::Tensor(tensor_ptr) => handle_output(&mut self.memory.tensor_live_ranges[tensor_ptr.0], format!("{tensor_ptr:?}")),
                    DiffValue::Value(value_index) => handle_output(&mut self.memory.value_live_ranges[value_index.0], format!("{value_index:?}")),
                    DiffValue::OneHot(_) => unimplemented!()
                }
            });

            let is_unused = all_unused.unwrap_or(false);

            if is_unused
            {
                *op = GradientOp::None;
                any_unused = true;
            } else if any_unused_single
            {
                /*let is_value_unused = |value_index: ValueIndex| -> bool
                {
                    self.memory.value_live_ranges[value_index.0].end.is_none()
                };*/

                let is_tensor_unused = |tensor_ptr: TensorPtr| -> bool
                {
                    self.memory.tensor_live_ranges[tensor_ptr.0].end.is_none()
                };

                let op_cloned = op.clone();

                *op = match op_cloned
                {
                    GradientOp::SoftmaxCrossEntropy{
                        values,
                        targets,
                        softmaxed_output,
                        output
                    } if is_tensor_unused(softmaxed_output) =>
                    {
                        GradientOp::SoftmaxCrossEntropyNoSoftmaxed{values, targets, output}
                    },
                    GradientOp::SoftmaxCrossEntropy{..} => op_cloned,
                    GradientOp::OtherSelectorValueGradient{..} => op_cloned,
                    GradientOp::OtherSelectorTensorGradient{
                        index,
                        first,
                        other,
                        src
                    } if is_tensor_unused(other) =>
                    {
                        GradientOp::IfSetTensor{index, dst: first, src}
                    },
                    GradientOp::OtherSelectorTensorGradient{
                        index,
                        first,
                        other,
                        src
                    } if is_tensor_unused(first) =>
                    {
                        GradientOp::IfNotSetTensor{index, dst: other, src}
                    },
                    x => unimplemented!("{x:?}")
                };
            }
        });

        any_unused
    }

    fn calculate_loop_live_ranges(&mut self)
    {
        self.gradient_operations.iter().enumerate().for_each(|(operation_index, op)|
        {
            match op
            {
                GradientOp::Jump(JumpInfo::JumpTo{index, ..}) =>
                {
                    self.loops[index.0].live_range.start = Some(operation_index as i32);
                },
                GradientOp::Jump(JumpInfo::JumpFrom(index)) =>
                {
                    self.loops[index.0].live_range.end = Some(operation_index as i32 - 1);
                },
                _ => ()
            }
        });

        self.gradient_operations.iter().for_each(|op|
        {
            match op
            {
                GradientOp::Jump(JumpInfo::JumpTo{index, inputs}) =>
                {
                    inputs.iter().for_each(|input|
                    {
                        if let InputTypePtr::Normal(tensor) = input
                        {
                            self.memory.tensor_live_ranges[tensor.0] = self.loops[index.0].live_range.clone();
                        }
                    });
                },
                _ => ()
            }
        });
    }

    fn calculate_live_ranges(&mut self)
    {
        self.calculate_loop_live_ranges();

        let tensor_live_ranges = self.memory.tensor_live_ranges.clone();
        let value_live_ranges = self.memory.value_live_ranges.clone();

        loop
        {
            let any_unused = self.calculate_live_ranges_once();

            if !any_unused
            {
                break;
            }

            self.memory.tensor_live_ranges = tensor_live_ranges.clone();
            self.memory.value_live_ranges = value_live_ranges.clone();
        }
    }

    fn greedy_graph_color(&mut self, memory_assignments: &mut Vec<TensorMemoryValue>)
    {
        let nodes_count = self.memory.tensor_live_ranges.len();

        let mut graph_connections: Vec<Vec<usize>> = iter::from_fn(|| Some(Vec::new()))
            .take(nodes_count)
            .collect();

        let verify_range = |range: &LiveRange, index|
        {
            if let Some(end) = range.end
            {
                if range.start.is_none()
                {
                    panic!(
                        "{} was used at OperationIndex({end}) but never set",
                        self.memory.format_variable(TensorPtr(index))
                    );
                }
            }

            if range.start.is_some() && range.end.is_some()
            {
                debug_assert!(range.valid_range(), "{} has an invalid range: {range:?}", self.memory.format_variable(TensorPtr(index)));
            }
        };

        (0..nodes_count).for_each(|node_index|
        {
            let this_range = &self.memory.tensor_live_ranges[node_index];

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
                    let other_range = &self.memory.tensor_live_ranges[check_index];

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
            if self.memory.tensors_memory[node_index].memory.is_some()
            {
                return;
            }

            if self.memory.tensor_live_ranges[node_index].end.is_none()
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
                        let connected_node_color: Option<usize> = self.memory.tensors_memory[*connected_node_index].memory.map(|x| x.0);

                        connected_node_color != Some(*color)
                    });

                    let spot_size_matches = memory_assignments.get(*color).map(|spot_tensor|
                    {
                        spot_tensor.tensor_shape() == self.memory.tensors_memory[node_index].value.tensor_shape()
                    }).unwrap_or(true);

                    all_connected_unconflicted && spot_size_matches
                }).unwrap()
            };

            if this_color == memory_assignments.len()
            {
                memory_assignments.push(self.memory.tensors_memory[node_index].value.clone());
            } else if let TensorMemoryValue::Value(x) = &self.memory.tensors_memory[node_index].value
            {
                memory_assignments[this_color] = TensorMemoryValue::Value(x.clone());
            }

            debug_assert!(self.memory.tensors_memory[node_index].memory.is_none(), "tried to replace slot of TensorPtr({node_index})");
            self.memory.tensors_memory[node_index].memory = Some(TensorIndex(this_color));
        });
    }

    fn operations_to_raw(&mut self, memory_assignments: &mut Vec<TensorMemoryValue>)
    {
        let mut usage_counts: Vec<(TensorIndex, usize, Vec<(TensorIndex, usize)>)> = (0..self.memory.tensors.len())
            .map(|x| (TensorIndex(x), 0, Vec::new()))
            .collect();

        self.gradient_operations.iter().for_each(|op|
        {
            let mut local = Vec::new();

            let mut f_local = |ptr: TensorPtr|
            {
                if let Some(this_index) = self.memory.tensors_memory[ptr.0].memory
                {
                    local.push(this_index);
                }
            };

            op.for_args(|arg| if let DiffValue::Tensor(t_arg) = arg { f_local(t_arg) });
            op.for_outputs(|out| if let DiffValue::Tensor(t_out) = out { f_local(t_out) });

            let mut f = |ptr: TensorPtr|
            {
                if let Some(this_index) = self.memory.tensors_memory[ptr.0].memory
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

            op.for_args(|arg| if let DiffValue::Tensor(t_arg) = arg { f(t_arg) });
            op.for_outputs(|out| if let DiffValue::Tensor(t_out) = out { f(t_out) });
        });

        usage_counts.sort_by_key(|x| x.1);

        let mut create_tensor = |this_index: TensorIndex|
        {
            if self.memory.tensors[this_index.0] != TensorRawDataPointer::undefined()
            {
                return;
            }

            let this_tensor: &TensorMemoryValue = &memory_assignments[this_index.0];

            let (rows, columns) = this_tensor.tensor_shape();
            let size = rows * columns;

            let id = TensorIndexRaw(self.memory.tensors_raw_data.len());

            let this_tensor_raw_ptr = TensorRawDataPointer{
                raw_index: id,
                rows,
                columns
            };

            debug_assert_eq!(self.memory.tensors[this_index.0], TensorRawDataPointer::undefined());
            self.memory.tensors[this_index.0] = this_tensor_raw_ptr;

            match this_tensor
            {
                TensorMemoryValue::Value(x) =>
                {
                    #[cfg(debug_assertions)]
                    {
                        self.memory.verify_raw_ptr_assign(this_tensor_raw_ptr);
                    }

                    self.memory.tensors_raw_data.extend(x.as_slice())
                },
                TensorMemoryValue::Size{..} => self.memory.tensors_raw_data.resize(self.memory.tensors_raw_data.len() + size, 0.0)
            }
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
            let this_index: TensorIndex = self.memory.tensors_memory[ptr.0].memory.unwrap_or_else(||
            {
                panic!("{} was used but not resolved", self.memory.variable_names.format_variable(ptr))
            });

            let current_value = self.memory.tensors[this_index.0];

            debug_assert_ne!(current_value, TensorRawDataPointer::undefined());

            current_value
        };

        let mut loops_labels: Vec<(LoopIndex, GradientOperationIndex)> = Vec::new();

        self.raw_operations.reserve_exact(self.gradient_operations.len());

        for gradient_op in mem::take(&mut self.gradient_operations).into_iter()
        {
            let operation_index = GradientOperationIndex(self.raw_operations.len());

            let map_push_stack = |loop_index: LoopIndex, value: DiffValue, gradient_op: StandardGradientOp|
            {
                debug_assert!(&self.loops[loop_index.0].used_values.contains(&value));

                gradient_op.map(&access_tensor, convert::identity, |_| unreachable!(), convert::identity)
            };

            let new_op = match gradient_op
            {
                GradientOp::None => None,
                GradientOp::FeedforwardEndMarker =>
                {
                    self.feedforward_operations_count = self.raw_operations.len();

                    None
                },
                GradientOp::PushStackValue{loop_index, value} => Some(map_push_stack(loop_index, value.into(), gradient_op)),
                GradientOp::PushStackTensor{loop_index, tensor} => Some(map_push_stack(loop_index, tensor.into(), gradient_op)),
                x =>
                {
                    let mut ignore_output = false;

                    let loops = &mut self.loops;
                    let output = x.map(&access_tensor, convert::identity, |jump_info|
                    {
                        match jump_info
                        {
                            JumpInfo::JumpTo{inputs, index} =>
                            {
                                loops[index.0].inputs = inputs.into_iter().map(|x|
                                {
                                    match x
                                    {
                                        InputTypePtr::Normal(x) =>
                                        {
                                            self.memory.tensors_memory[x.0].memory.map(InputType::Normal)
                                        },
                                        InputTypePtr::OneHot(x) => Some(InputType::OneHot(x))
                                    }
                                }).collect();

                                debug_assert!(!loops_labels.iter().any(|x| x.0 == index));

                                loops_labels.push((index, GradientOperationIndex(operation_index.0)));

                                ignore_output = true;

                                self.raw_operations.push(GradientOp::SetInputs(index));

                                RawJumpInfo{loop_index: LoopIndex(usize::MAX), operation_index: GradientOperationIndex(usize::MAX)}
                            },
                            JumpInfo::JumpFrom(index) =>
                            {
                                let target_index: GradientOperationIndex = loops_labels.iter().find(|(loop_index, _)| index == *loop_index)
                                    .expect("loop must be defined before being used")
                                    .1;

                                RawJumpInfo{loop_index: index, operation_index: target_index}
                            }
                        }
                    }, convert::identity);

                    (!ignore_output).then_some(output)
                }
            };

            if let Some(raw_op) = new_op
            {
                self.raw_operations.push(raw_op);
            }
        }
    }

    fn replace_op_args(&mut self, i: usize, src: DiffValue, dst: DiffValue)
    {
        let op = &mut self.gradient_operations[i];

        let mut replace_stack_push = |loop_index: LoopIndex|
        {
            let this_loop = &mut self.loops[loop_index.0];

            if let Some(used_value) = this_loop.used_values.iter_mut().find(|x| **x == src)
            {
                *used_value = dst;

                this_loop.used_values.dedup();
            }

            if let Some(defined_value) = this_loop.defined_values.iter_mut().find(|x| **x == src)
            {
                *defined_value = dst;

                this_loop.defined_values.dedup();
            }
        };

        match op
        {
            GradientOp::PushStackTensor{loop_index, ..} => replace_stack_push(*loop_index),
            GradientOp::PushStackValue{loop_index, ..} => replace_stack_push(*loop_index),
            _ => ()
        }

        *op = op.clone().map_args(|arg|
        {
            if arg == src
            {
                dst
            } else
            {
                arg
            }
        });
    }

    fn swap_assignment(&mut self, before: usize, src: DiffValue, dst: DiffValue)
    {
        for i in 0..before
        {
            self.replace_op_args(i, src, dst);

            let op = &mut self.gradient_operations[i];

            *op = op.clone().map_outputs(|output|
            {
                if output == src
                {
                    dst
                } else
                {
                    output
                }
            });
        }
    }

    fn combine_inplace_assignments(&mut self)
    {
        for i in (0..self.gradient_operations.len()).rev()
        {
            match self.gradient_operations[i]
            {
                GradientOp::OuterProductAdd{lhs, rhs, added, output} =>
                {
                    let combined_output_add = output;

                    self.memory.tensor_live_ranges[combined_output_add.0].start = self.memory.tensor_live_ranges[added.0].start;

                    self.gradient_operations[i] = GradientOp::OuterProductAdd{
                        lhs,
                        rhs,
                        added: combined_output_add,
                        output: combined_output_add
                    };

                    self.swap_assignment(i, added.into(), output.into());
                },
                _ => ()
            }
        }
    }

    pub fn resolve_memory(&mut self)
    {
        debug_assert_eq!(self.state, RecorderState::AwaitingResolve);

        self.calculate_live_ranges();

        self.combine_inplace_assignments();

        let mut memory_assignments = Vec::new();
        self.greedy_graph_color(&mut memory_assignments);

        self.memory.tensors.resize(memory_assignments.len(), TensorRawDataPointer::undefined());

        if OPT_INFO
        {
            self.memory.tensor_live_ranges.iter().enumerate().for_each(|(tensor_ptr_index, live_range)|
            {
                let tensor_ptr = TensorPtr(tensor_ptr_index);

                let f_live = |x: Option<i32>| -> String
                {
                    x.map(|x|
                    {
                        match x
                        {
                            -1 => "start".to_owned(),
                            i32::MAX => "end".to_owned(),
                            x => x.to_string()
                        }
                    }).unwrap()
                };

                let tensor_name = self.memory.format_variable(tensor_ptr);

                if live_range.start.is_none() && live_range.end.is_some()
                {
                    eprintln!("{tensor_name} is malformed, no start but ends at {}", f_live(live_range.end));
                } else if live_range.end.is_none()
                {
                    eprintln!("{tensor_name} is unused");
                } else
                {
                    eprintln!("{tensor_name} has live range {} to {}", f_live(live_range.start), f_live(live_range.end));
                }
            });

            (0..self.memory.tensors.len()).for_each(|tensor_index|
            {
                let used_ptrs: Vec<_> = self.memory.tensors_memory.iter().enumerate().filter_map(|(tensor_ptr, slot)|
                {
                    (slot.memory == Some(TensorIndex(tensor_index))).then(||
                    {
                        DebugStringRaw(self.memory.format_variable(TensorPtr(tensor_ptr)))
                    })
                }).collect();

                eprintln!("memory spot {tensor_index} used by: {used_ptrs:?}");
            });

            eprintln!("using {} memory spots", self.memory.tensors.len());
        }

        self.operations_to_raw(&mut memory_assignments);

        #[cfg(debug_assertions)]
        {
            self.raw_operations.iter().for_each(|op|
            {
                let allow_overlap = matches!(
                    op,
                    GradientOp::GetOtherSelectorValue{..}
                    | GradientOp::GetOtherSelectorTensor{..}
                    | GradientOp::OuterProductAdd{..}
                );

                if allow_overlap
                {
                    return;
                }

                let mut tensor_args = Vec::new();
                let mut value_args = Vec::new();

                op.clone().map_args_with_state((), |_s, t_arg|
                {
                    tensor_args.push(t_arg);
                    t_arg
                }, |_s, v_arg|
                {
                    value_args.push(v_arg);
                    v_arg
                }, |_s, o_arg| o_arg);

                fn any_duplicates<T: Eq>(values: &[T]) -> bool
                {
                    values.iter().enumerate().any(|(index_x, x)|
                    {
                        values.iter().enumerate()
                            .filter(|(index_y, _y)| *index_y != index_x)
                            .any(|(_, y)| x == y)
                    })
                }

                debug_assert!(!any_duplicates(&tensor_args), "{op:?} has duplicate arguments");
                debug_assert!(!any_duplicates(&value_args), "{op:?} has duplicate arguments");

                op.clone().map_outputs_with_state((), |_s, t_out|
                {
                    let t_out_index = self.memory.raw_ptr_to_memory(t_out);

                    debug_assert!(
                        !tensor_args.contains(&t_out),
                        "{op:?} has overlap between args and outputs ({t_out:?} which is {})",
                        self.memory.format_tensor_index(t_out_index)
                    );

                    t_out
                }, |_s, v_out|
                {
                    debug_assert!(
                        !value_args.contains(&v_out),
                        "{op:?} has overlap between args and outputs ({v_out:?} which is {})",
                        self.memory.format_variable(v_out)
                    );

                    v_out
                });
            });

            self.memory.tensor_inputs = self.memory.tensor_live_ranges.iter().enumerate()
                .filter(|(_, x)| x.start == Some(-1))
                .map(|(index, _)| TensorPtr(index))
                .collect();
        }

        self.state = RecorderState::Ready;

        #[cfg(debug_assertions)]
        {
            let mut new_store_tensors_check = Vec::new();

            self.memory.store_tensors_check.iter().for_each(|k|
            {
                let this_index: TensorIndex = match *k
                {
                    StoreCheckKey::PreResolve(ptr) => self.memory.tensors_memory[ptr.0].memory.expect("must be resolved"),
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

            self.memory.store_tensors_check = new_store_tensors_check;

            self.memory.store_values_check = self.memory.store_values_check.iter().map(|k|
            {
                match *k
                {
                    StoreCheckKey::PreResolve(x) => StoreCheckKey::Resolved(x),
                    StoreCheckKey::Resolved(_) => unreachable!()
                }
            }).collect();

            if self.raw_operations.len() > 0
            {
                self.memory.tensor_live_ranges.iter().enumerate().for_each(|(tensor_ptr_index, live_range)|
                {
                    let tensor_ptr = TensorPtr(tensor_ptr_index);

                    if live_range.start == Some(-1) && live_range.end != Some(i32::MAX)
                    {
                        debug_assert!(
                            self.memory.allow_discard.contains(&tensor_ptr),
                            "{} will be discarded after running calculate once, either store the tensor or call allow_discard on it",
                            self.memory.format_variable(tensor_ptr)
                        );
                    }
                });
            }

            {
                let resolved_set_ptrs: Vec<_> = {
                    self.memory.set_tensor_memory.borrow().set_ptrs.iter().map(|x| self.resolve_tensor_ptr(*x)).collect()
                };

                let mut set_tensor_memory = self.memory.set_tensor_memory.borrow_mut();

                set_tensor_memory.set_memory.extend(resolved_set_ptrs.into_iter());
            }
        }

        self.memory.value_live_ranges = Vec::new();
        self.memory.tensor_live_ranges = Vec::new();
    }

    pub fn no_gradient(&mut self)
    {
        debug_assert_eq!(self.state, RecorderState::AwaitingGradient);

        self.calculate_kept_inside();

        self.remove_unused_pushes();

        self.copy_coalesce();

        self.recording_operations = Vec::new();

        self.state = RecorderState::AwaitingResolve;
    }

    fn calculate_kept_inside(&mut self)
    {
        self.recording_operations.iter().for_each(|op|
        {
            if let Op::Loop{ops, index: loop_index, inputs} = op
            {
                let mut defined_values: Vec<_> = inputs.iter().map(|input|
                {
                    DiffValue::from(*input)
                }).collect();

                ops.iter().for_each(|op|
                {
                    op.for_outputs(|output|
                    {
                        defined_values.push(output.as_value());
                    });
                });

                ops.iter().for_each(|op|
                {
                    op.for_args(|arg|
                    {
                        let arg = arg.as_value();

                        if !defined_values.contains(&arg)
                        {
                            self.loops[loop_index.0].kept_inside.push(arg);
                        }
                    });
                });
            }
        });
    }

    fn scan_out_loop_gradients(
        &mut self,
        assigned_gradients: &mut Vec<AssignedInfo>,
        selectors: &mut Vec<LoopSelectorInfo>
    )
    {
        let mut new_names: Vec<(DiffValue, DiffValue, &str)> = Vec::new();

        self.recording_operations.iter().for_each(|op|
        {
            if let Op::Loop{ops, index: loop_index, ..} = op
            {
                let mut reachable_next_loop: HashSet<DiffValue> = HashSet::new();

                loop
                {
                    let reachable_count = reachable_next_loop.len();

                    ops.iter().for_each(|op|
                    {
                        match op
                        {
                            Op::GetOtherSelectorValue{output, ..} =>
                            {
                                reachable_next_loop.insert(output.as_value().into());
                            }
                            Op::GetOtherSelectorTensor{output, ..} =>
                            {
                                reachable_next_loop.insert(output.as_value().into());
                            },
                            x =>
                            {
                                let mut influenced = Vec::new();
                                x.for_args(|arg| influenced.push(arg.as_value()));
                                x.for_outputs(|output| influenced.push(output.as_value()));

                                if reachable_next_loop.iter().any(|reachable| influenced.contains(reachable))
                                {
                                    reachable_next_loop.extend(influenced);
                                }
                            }
                        }
                    });

                    if reachable_count == reachable_next_loop.len()
                    {
                        break;
                    }
                }

                let mut zero_out = |gradient_operations: &mut Vec<_>, output: DiffValue|
                {
                    if assigned_gradients.iter().any(|x| x.value == output)
                    {
                        return;
                    }

                    let id = GradientOperationIndex(gradient_operations.len());

                    let op = match output
                    {
                        DiffValue::Tensor(t_output) => GradientOp::ZeroTensor(t_output),
                        DiffValue::Value(v_output) => GradientOp::ZeroValue(v_output),
                        DiffValue::OneHot(_) => unreachable!()
                    };

                    gradient_operations.push(op);

                    assigned_gradients.push(AssignedInfo{
                        value: output,
                        operation_index: id,
                        extra_operation_index: None,
                        loop_selected: Some(*loop_index)
                    });
                };

                let mut defined_inside = Vec::new();

                ops.iter().for_each(|op|
                {
                    if let Op::GetOtherSelectorValue{index, ..} | Op::GetOtherSelectorTensor{index, ..} = op
                    {
                        let other = self.phi_other_selectors_recording[index.0].other.expect("must be initialized");
                        let other = other.as_gradient().expect("must have a gradient");

                        let next = self.memory.new_diff_intermediate(other, |(a, b)| new_names.push((a, b, "_next")));

                        selectors.push(LoopSelectorInfo{
                            other,
                            next_total: next,
                            next_previous: next
                        });
                    }

                    op.for_outputs(|output|
                    {
                        if let Some(output) = output.as_gradient()
                        {
                            defined_inside.push(output);
                        }
                    });
                });

                ops.iter().for_each(|op|
                {
                    op.for_args(|arg_diff|
                    {
                        if let Some(arg) = arg_diff.as_gradient()
                        {
                            if !defined_inside.contains(&arg)
                            {
                                debug_assert!(
                                    reachable_next_loop.contains(&arg_diff.as_value()),
                                    "{} is not reachable: {:#?}",
                                    self.memory.format_variable(arg_diff.as_value()),
                                    reachable_next_loop.iter().map(|x| DebugStringRaw(self.memory.format_variable(*x))).collect::<Vec<_>>()
                                );

                                zero_out(&mut self.gradient_operations, arg);
                            }
                        }
                    });
                });
            }
        });

        new_names.into_iter().for_each(|(value, inherit, suffix)|
        {
            match value
            {
                DiffValue::Value(value) => self.name_value_suffix(value, inherit.into_value(), suffix),
                DiffValue::Tensor(value) => self.name_tensor_suffix(value, inherit.into_tensor(), suffix),
                DiffValue::OneHot(_) => unreachable!()
            }
        });
    }

    fn resolve_stack_values(&mut self, used_stack_values_unsorted: Vec<UsedStackValueInfo>)
    {
        fn position_of_target(gradient_operations: &[StandardGradientOp], target: DiffValue) -> GradientOperationIndex
        {
            GradientOperationIndex(gradient_operations.iter().position(|op|
            {
                let mut is_found = false;
                op.for_args(|arg| if arg == target { is_found = true });

                is_found
            }).expect("pop argument must exist"))
        }

        let used_stack_values = {
            let mut used_stack_values = used_stack_values_unsorted;
            used_stack_values.sort_by_key(|x| position_of_target(&self.gradient_operations, x.target).0);

            used_stack_values
        };

        used_stack_values.iter().cloned().for_each(|UsedStackValueInfo{loop_index, source, ..}|
        {
            let loop_info = &mut self.loops[loop_index.0];
            loop_info.used_values.push(source);

            if let Some(source_loop) = loop_info.gradient_of_loop
            {
                self.loops[source_loop.0].used_values.push(source);
            }
        });

        let pushed_ordered: Vec<Vec<_>> = {
            (0..self.loops.len()).map(|current_loop_index| -> Vec<_>
            {
                self.loops[current_loop_index].defined_values.iter().filter(|defined_value|
                {
                    used_stack_values.iter().any(|info|
                    {
                        info.loop_index == LoopIndex(current_loop_index) && info.source == **defined_value
                    })
                }).cloned().collect()
            }).collect()
        };

        let mut popped = Vec::new();

        fn pop_stack_value(
            this: &mut OperationsRecorder,
            popped: &mut Vec<DiffValue>,
            used_stack_values: &[UsedStackValueInfo],
            selected_stack_value: usize,
            pop_op_index: GradientOperationIndex
        )
        {
            let UsedStackValueInfo{loop_index, source, target} = used_stack_values[selected_stack_value];

            let pop_op: StandardGradientOp = match target
            {
                DiffValue::Value(output) =>
                {
                    this.name_value_suffix(output, source.into_value(), "_loop");

                    GradientOp::PopStackValue{loop_index, output}
                },
                DiffValue::Tensor(output) =>
                {
                    this.name_tensor_suffix(output, source.into_tensor(), "_loop");

                    GradientOp::PopStackTensor{loop_index, output}
                },
                DiffValue::OneHot(_) => unimplemented!()
            };

            this.gradient_operations.insert(pop_op_index.0, pop_op);

            popped.push(source);

            #[cfg(debug_assertions)]
            {
                let this_pair = (source, target);

                this.loops[loop_index.0].expected_pairs.push(this_pair);
            }
        }

        let mut current_used_index = 0;
        pushed_ordered.iter().enumerate().rev().for_each(|(current_loop_index, pushes)|
        {
            pushes.iter().rev().for_each(|defined_value|
            {
                debug_assert!(!popped.contains(defined_value));

                let index = used_stack_values.iter().position(|info|
                {
                    info.loop_index == LoopIndex(current_loop_index) && info.source == *defined_value
                }).expect("must be a defined value");

                let pop_op_index = position_of_target(&self.gradient_operations, used_stack_values[current_used_index].target);

                pop_stack_value(self, &mut popped, &used_stack_values, index, pop_op_index);

                if index == current_used_index
                {
                    while popped.contains(&used_stack_values[current_used_index].source)
                    {
                        current_used_index += 1;

                        if current_used_index == used_stack_values.len()
                        {
                            return;
                        }
                    }
                } else
                {
                    debug_assert!(index > current_used_index)
                }
            });
        });

        debug_assert_eq!(popped.len(), popped.iter().cloned().collect::<HashSet<_>>().len());

        debug_assert_eq!(popped.len(), used_stack_values.len());
    }

    pub fn gradient(&mut self, respect: DiffWrapper)
    {
        debug_assert_eq!(self.state, RecorderState::AwaitingGradient);

        self.set_ones(respect);

        {
            let mut assigned_gradients = Vec::new();
            let mut selectors = Vec::new();

            self.scan_out_loop_gradients(&mut assigned_gradients, &mut selectors);
            self.calculate_kept_inside();

            let mut used_stack_values = Vec::new();

            for op_index in (0..self.recording_operations.len()).rev()
            {
                let op = self.recording_operations[op_index].clone();

                self.calculate_gradient(
                    &mut assigned_gradients,
                    &mut selectors,
                    &mut None,
                    &mut used_stack_values,
                    &[],
                    op,
                    None
                );
            }

            self.resolve_stack_values(used_stack_values);
        }

        self.remove_unused_pushes();

        self.copy_coalesce();

        self.recording_operations = Vec::new();

        self.state = RecorderState::AwaitingResolve;
    }

    fn calculate_gradient(
        &mut self,
        assigned_gradients: &mut Vec<AssignedInfo>,
        selectors: &mut Vec<LoopSelectorInfo>,
        intermediate_selector: &mut Option<PhiOtherSelectorIndex>,
        used_stack_values: &mut Vec<UsedStackValueInfo>,
        loop_inputs: &[(InputTypePtr, InputTypePtr)],
        op: Op,
        inside_loop: Option<LoopIndex>
    )
    {
        let mut add_gradient_operation = |this: &mut Self, selectors: &mut Vec<LoopSelectorInfo>, gradient_op: StandardGradientOp|
        {
            let make_new = |this: &mut Self, x: DiffValue| -> DiffValue
            {
                if let DiffValue::Tensor(x) = x
                {
                    let shape = this.memory.tensor_shape_value(x);

                    DiffValue::Tensor(this.memory.new_tensor_index(shape))
                } else
                {
                    DiffValue::Value(this.memory.new_value_index())
                }
            };

            let gradient_op = if let Some(inside_loop) = inside_loop
            {
                gradient_op.map_args(|arg|
                {
                    if let Some((_source_input, gradient_input)) = loop_inputs.iter().find(|(source_input, _)|
                    {
                        DiffValue::from(*source_input) == arg
                    })
                    {
                        return DiffValue::from(*gradient_input);
                    }

                    if this.loops[inside_loop.0].defined_values.contains(&arg)
                    {
                        if let Some(info) = used_stack_values.iter().find(|x| x.source == arg && x.loop_index == inside_loop)
                        {
                            info.target
                        } else
                        {
                            let loop_intermediate = make_new(this, arg);

                            used_stack_values.push(UsedStackValueInfo{
                                loop_index: inside_loop,
                                source: arg,
                                target: loop_intermediate
                            });

                            loop_intermediate
                        }
                    } else
                    {
                        arg
                    }
                })
            } else
            {
                gradient_op
            };

            let handle_output = |this: &mut Self, new_out: &mut Option<DiffValue>, x: DiffValue| -> DiffValue
            {
                let new_value = make_new(this, x);

                *new_out = Some(new_value);

                new_value
            };

            let assigned_insert = |
                assigned_gradients: &mut Vec<AssignedInfo>,
                operation_index: GradientOperationIndex,
                value: DiffValue
            |
            {
                assigned_gradients.push(AssignedInfo{
                    value,
                    operation_index,
                    extra_operation_index: None,
                    loop_selected: None
                });
            };

            let simple_insert = |this: &mut Self, assigned_gradients: &mut Vec<AssignedInfo>, op: StandardGradientOp|
            {
                let this_op_index = GradientOperationIndex(this.gradient_operations.len());
                this.gradient_operations.push(op.clone());

                op.for_outputs(|output|
                {
                    assigned_insert(assigned_gradients, this_op_index, output);
                });
            };

            let mut add_selector = |this: &mut Self| -> PhiOtherSelectorIndex
            {
                intermediate_selector.unwrap_or_else(||
                {
                    let selector_index = PhiOtherSelectorIndex(this.memory.phi_other_selectors_values.len());
                    this.memory.phi_other_selectors_values.push(PhiOtherSelectorValue{
                        loop_index: inside_loop.expect("must only be called inside a loop"),
                        is_set: false
                    });

                    *intermediate_selector = Some(selector_index);

                    selector_index
                })
            };

            let gradient_op = if let Some(this_selector) = selectors.iter().find(|info|
            {
                let mut any_args_match = false;
                gradient_op.for_args(|arg| if arg == info.other { any_args_match = true; });

                any_args_match
            })
            {
                let selector_index = add_selector(this);

                let new_gradient_op = match gradient_op
                {
                    GradientOp::Copy{src, dst} =>
                    {
                        GradientOp::GetOtherSelectorTensor{
                            info: selector_index,
                            first: src,
                            other: this_selector.next_total.into_tensor(),
                            output: dst
                        }
                    },
                    GradientOp::CopyScalar{src, dst} =>
                    {
                        GradientOp::GetOtherSelectorValue{
                            info: selector_index,
                            first: src,
                            other: this_selector.next_total.into_value(),
                            output: dst
                        }
                    },
                    _ => unimplemented!("only add selectors work rn")
                };

                let mut outputs_count = 0;
                gradient_op.for_outputs(|_output| outputs_count += 1);

                debug_assert_eq!(outputs_count, 1);

                new_gradient_op
            } else
            {
                gradient_op
            };

            let mut sum_gradients = |
                info: &mut AssignedInfo,
                gradient_op: StandardGradientOp,
                output: DiffValue
            |
            {
                let previous_op = this.gradient_operations[info.operation_index.0].clone();

                let mut old_previous_op_output = None;

                let mut lhs = None;

                let new_previous_op = previous_op.map_outputs(|x|
                {
                    if x == output
                    {
                        debug_assert!(lhs.is_none());

                        old_previous_op_output = Some(x);

                        handle_output(this, &mut lhs, x)
                    } else
                    {
                        x
                    }
                });

                debug_assert!(lhs.is_some());

                let mut rhs = None;
                let mut selector_rhs = None;

                let mut new_extra_operation_index = None;

                let is_inside_selected_loop = inside_loop.map(|inside_loop|
                {
                    info.loop_selected == this.loops[inside_loop.0].gradient_of_loop
                }).unwrap_or(false);

                let needs_to_add_selector = is_inside_selected_loop && info.extra_operation_index.is_none();

                let new_op = gradient_op.map_outputs(|x|
                {
                    if x != output
                    {
                        return x;
                    }

                    debug_assert!(rhs.is_none());

                    let new_output = handle_output(this, &mut rhs, x);

                    if needs_to_add_selector
                    {
                        let selector_index = add_selector(this);

                        let state = make_new(this, x);

                        let op = if matches!(x, DiffValue::Tensor(_))
                        {
                            GradientOp::GetOtherSelectorTensor{
                                info: selector_index,
                                first: lhs.unwrap().into_tensor(),
                                other: output.into_tensor(),
                                output: state.into_tensor()
                            }
                        } else
                        {
                            GradientOp::GetOtherSelectorValue{
                                info: selector_index,
                                first: lhs.unwrap().into_value(),
                                other: output.into_value(),
                                output: state.into_value()
                            }
                        };

                        let selector_operation_index = GradientOperationIndex(this.gradient_operations.len());
                        this.gradient_operations.push(op);

                        new_extra_operation_index = Some(selector_operation_index);

                        selector_rhs = Some(state);
                    }

                    new_output
                });

                let add_lhs = if needs_to_add_selector
                {
                    selector_rhs.unwrap()
                } else
                {
                    lhs.unwrap()
                };

                let add_rhs = rhs.unwrap();

                if let Some(extra_operation_index) = info.extra_operation_index
                {
                    if !is_inside_selected_loop
                    {
                        // for selector args
                        let new_extra = this.gradient_operations[extra_operation_index.0].clone().map_args(|arg|
                        {
                            if arg == old_previous_op_output.unwrap()
                            {
                                lhs.unwrap()
                            } else
                            {
                                arg
                            }
                        });

                        this.gradient_operations[extra_operation_index.0] = new_extra;
                    }
                }

                this.gradient_operations[info.operation_index.0] = new_previous_op;

                let new_id = this.gradient_operations.len();

                let new_id = match new_op
                {
                    GradientOp::OuterProduct{lhs, rhs, output: output_temp} =>
                    {
                        debug_assert_eq!(output_temp, add_rhs.into_tensor());

                        this.gradient_operations.push(GradientOp::OuterProductAdd{
                            lhs,
                            rhs,
                            added: add_lhs.into_tensor(),
                            output: output.into_tensor()
                        });

                        new_id
                    },
                    new_op => {
                        this.gradient_operations.push(new_op);

                        let new_id = this.gradient_operations.len();

                        let add_op = match output
                        {
                            DiffValue::Tensor(output) =>
                            {
                                GradientOp::Add{lhs: add_lhs.into_tensor(), rhs: add_rhs.into_tensor(), output}
                            },
                            DiffValue::Value(output) =>
                            {
                                GradientOp::AddScalars{lhs: add_lhs.into_value(), rhs: add_rhs.into_value(), output}
                            },
                            DiffValue::OneHot(_) => unimplemented!()
                        };

                        this.gradient_operations.push(add_op);

                        new_id
                    }
                };

                info.operation_index = GradientOperationIndex(new_id);
                info.extra_operation_index = new_extra_operation_index;
            };

            let last_op_index = |this: &mut Self|
            {
                GradientOperationIndex(this.gradient_operations.len() - 1)
            };

            let mut overlaps = false;
            let mut outputs_count = 0;

            if let Some((info, output)) = assigned_gradients.iter_mut().find_map(|info|
            {
                outputs_count = 0;

                let mut is_assigned = None;
                gradient_op.for_outputs(|output|
                {
                    outputs_count += 1;

                    if info.value == output
                    {
                        debug_assert!(is_assigned.is_none());

                        is_assigned = Some(output);
                    }
                });

                is_assigned.map(|output| (info, output))
            })
            {
                overlaps = true;

                sum_gradients(info, gradient_op.clone(), output);

                if outputs_count > 1
                {
                    let operation_index = last_op_index(this);

                    gradient_op.for_outputs(|this_output|
                    {
                        if this_output != output
                        {
                            assigned_insert(assigned_gradients, operation_index, this_output);
                        }
                    });
                }
            } else
            {
                simple_insert(this, assigned_gradients, gradient_op.clone());
            }

            if inside_loop.is_some()
            {
                if let Some(this_selector) = selectors.iter_mut().find(|info|
                {
                    let mut any_outputs_match = false;
                    gradient_op.for_outputs(|output| if output == info.other { any_outputs_match = true; });

                    any_outputs_match
                })
                {
                    debug_assert!(!overlaps);

                    let last_operation_index = last_op_index(this);

                    let selector_index = add_selector(this);

                    let mut new_names = Vec::new();

                    let other = this_selector.other;
                    let total = this.memory.new_diff_intermediate(other, |(a, b)| new_names.push((a, b, "_total")));

                    let next = this_selector.next_previous;
                    this_selector.next_total = total;

                    let temp = this.memory.new_diff_intermediate(other, |(a, b)| new_names.push((a, b, "_temp")));

                    let add_op = match other
                    {
                        DiffValue::Value(_) =>
                        {
                            GradientOp::GradientSelectAddValue{
                                index: selector_index,
                                first: other.into_value(),
                                other: total.into_value(),
                                src: temp.into_value(),
                                added: next.into_value()
                            }
                        },
                        DiffValue::Tensor(_) =>
                        {
                            GradientOp::GradientSelectAddTensor{
                                index: selector_index,
                                first: other.into_tensor(),
                                other: total.into_tensor(),
                                src: temp.into_tensor(),
                                added: next.into_tensor()
                            }
                        },
                        DiffValue::OneHot(_) => unimplemented!()
                    };

                    this.gradient_operations[last_operation_index.0] = gradient_op.map_outputs(|output|
                    {
                        if output == other
                        {
                            temp
                        } else
                        {
                            output
                        }
                    });

                    this.gradient_operations.push(add_op);

                    new_names.into_iter().for_each(|(value, inherit, suffix): (DiffValue, DiffValue, &str)|
                    {
                        match value
                        {
                            DiffValue::Value(value) => this.name_value_suffix(value, inherit.into_value(), suffix),
                            DiffValue::Tensor(value) => this.name_tensor_suffix(value, inherit.into_tensor(), suffix),
                            DiffValue::OneHot(_) => unimplemented!()
                        }
                    });
                }
            }
        };

        macro_rules! gradient_or_return
        {
            ($value:expr) =>
            {
                if let Some(gradient) = $value.gradient { gradient } else { return; }
            }
        }

        match op
        {
            Op::CopyScalar{src, dst} =>
            {
                let gradient = gradient_or_return!(dst);

                if let Some(src_gradient) = src.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::CopyScalar{src: gradient, dst: src_gradient});
                }
            },
            Op::Copy{src, dst} =>
            {
                let gradient = gradient_or_return!(dst);

                if let Some(src_gradient) = src.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::Copy{src: gradient, dst: src_gradient});
                }
            },
            Op::Add{lhs, output, ..}
            | Op::AddScalar{lhs, output, ..} =>
            {
                let gradient = gradient_or_return!(output);

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::Copy{src: gradient, dst: lhs_gradient});
                }

                if let Op::Add{rhs, ..} = op
                {
                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        add_gradient_operation(self, selectors, GradientOp::Copy{src: gradient, dst: rhs_gradient});
                    }
                } else if let Op::AddScalar{rhs, ..} = op
                {
                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        add_gradient_operation(self, selectors, GradientOp::SumTensor{value: gradient, output: rhs_gradient});
                    }
                } else
                {
                    unreachable!()
                }
            },
            Op::AddScalars{lhs, rhs, output} =>
            {
                let gradient = gradient_or_return!(output);

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::CopyScalar{src: gradient, dst: lhs_gradient});
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::CopyScalar{src: gradient, dst: rhs_gradient});
                }
            },
            Op::Sub{rhs, output, ..}
            | Op::SubFromScalar{rhs, output, ..} =>
            {
                let gradient = gradient_or_return!(output);

                if let Op::Sub{lhs, ..} = op
                {
                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        add_gradient_operation(self, selectors, GradientOp::Copy{src: gradient, dst: lhs_gradient});
                    }
                } else if let Op::SubFromScalar{lhs, ..} = op
                {
                    if let Some(lhs_gradient) = lhs.as_gradient()
                    {
                        add_gradient_operation(self, selectors, GradientOp::SumTensor{value: gradient, output: lhs_gradient});
                    }
                } else
                {
                    unreachable!()
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    let m1_index = self.memory.new_value_index();
                    self.memory.values[m1_index.0] = -1.0;

                    add_gradient_operation(self, selectors, GradientOp::MulScalar{lhs: gradient, rhs: m1_index, output: rhs_gradient});
                }
            },
            Op::MulScalars{lhs, rhs, output} =>
            {
                let gradient = gradient_or_return!(output);

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::MulScalars{lhs: rhs.as_value(), rhs: gradient, output: lhs_gradient});
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::MulScalars{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                }
            },
            Op::MulComponentwise{lhs, output, ..}
            | Op::MulScalar{lhs, output, ..} =>
            {
                let gradient = gradient_or_return!(output);

                let shape = self.memory.tensor_shape_value(gradient);

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    if let Op::MulComponentwise{rhs, ..} = op
                    {
                        add_gradient_operation(self, selectors, GradientOp::MulComponentwise{lhs: rhs.as_value(), rhs: gradient, output: lhs_gradient});
                    } else if let Op::MulScalar{rhs, ..} = op
                    {
                        add_gradient_operation(self, selectors, GradientOp::MulScalar{lhs: gradient, rhs: rhs.as_value(), output: lhs_gradient});
                    } else
                    {
                        unreachable!()
                    }
                }

                if let Op::MulComponentwise{rhs, ..} = op
                {
                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        add_gradient_operation(self, selectors, GradientOp::MulComponentwise{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                    }
                } else if let Op::MulScalar{rhs, ..} = op
                {
                    if let Some(rhs_gradient) = rhs.as_gradient()
                    {
                        let pre_fold = self.memory.new_tensor_index(shape);
                        self.gradient_operations.push(GradientOp::MulComponentwise{lhs: lhs.as_value(), rhs: gradient, output: pre_fold});

                        add_gradient_operation(self, selectors, GradientOp::SumTensor{value: pre_fold, output: rhs_gradient});
                    }
                } else
                {
                    unreachable!()
                }
            },
            Op::SumTensor{value, output} =>
            {
                let gradient = gradient_or_return!(output);

                if let Some(value_gradient) = value.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::Fill{value: gradient, output: value_gradient});
                }
            },
            Op::Dot{lhs, rhs, output} =>
            {
                let gradient = gradient_or_return!(output);

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::MulScalar{lhs: rhs.as_value(), rhs: gradient, output: lhs_gradient});
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::MulScalar{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                }
            },
            Op::Pow{lhs, power, output} =>
            {
                let gradient = gradient_or_return!(output);

                let shape = self.memory.tensor_shape_value(lhs.as_value());

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    let power_index = self.memory.new_value_index();
                    self.memory.values[power_index.0] = power as f32;

                    let pow_d_lhs = self.memory.new_tensor_index(shape.clone());
                    self.gradient_operations.push(GradientOp::Pow{lhs: lhs.as_value(), power: (power - 1) as u32, output: pow_d_lhs});

                    let pow_d = self.memory.new_tensor_index(shape);
                    self.gradient_operations.push(GradientOp::MulScalar{lhs: pow_d_lhs, rhs: power_index.into(), output: pow_d});

                    add_gradient_operation(self, selectors, GradientOp::MulComponentwise{lhs: pow_d, rhs: gradient, output: lhs_gradient});
                }
            },
            Op::Sigmoid{value, output} =>
            {
                // sigmoid(x) * (1.0 - sigmoid(x))
                let gradient = gradient_or_return!(output);

                if let Some(value_gradient) = value.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::SigmoidDiff{value: output.as_value(), gradient, output: value_gradient});
                }
            },
            Op::Tanh{value, output} =>
            {
                // 1 - tanh^2(x)
                let gradient = gradient_or_return!(output);

                if let Some(value_gradient) = value.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::TanhDiff{value: output.as_value(), gradient, output: value_gradient});
                }
            },
            Op::LeakyRelu{value, output} =>
            {
                let gradient = gradient_or_return!(output);

                if let Some(value_gradient) = value.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::LeakyReluDiff{value: value.as_value(), gradient, output: value_gradient});
                }
            },
            Op::SoftmaxCrossEntropy{values, targets, softmaxed_output, output} =>
            {
                let gradient = gradient_or_return!(output);

                if let Some(values_gradient) = values.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::SoftmaxCrossEntropyDiff{
                        softmaxed_values: softmaxed_output.as_value(),
                        gradient,
                        targets: targets.clone(),
                        output: values_gradient
                    });
                }
            },
            Op::Matmulv{lhs, rhs, output} =>
            {
                let gradient = gradient_or_return!(output);

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::OuterProduct{lhs: gradient, rhs: rhs.as_value(), output: lhs_gradient});
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::MatmulvTransposed{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                }
            },
            Op::MatmulvAdd{lhs, rhs, added, output} =>
            {
                let gradient = gradient_or_return!(output);

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::OuterProduct{lhs: gradient, rhs: rhs.as_value(), output: lhs_gradient});
                }

                if let Some(rhs_gradient) = rhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::MatmulvTransposed{lhs: lhs.as_value(), rhs: gradient, output: rhs_gradient});
                }

                if let Some(added_gradient) = added.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::Copy{src: gradient, dst: added_gradient});
                }
            },
            Op::MatmulOneHotvAdd{lhs, rhs, added, output} =>
            {
                let gradient = gradient_or_return!(output);

                if let Some(lhs_gradient) = lhs.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::OuterProductOneHot{lhs: gradient, rhs: rhs, output: lhs_gradient});
                }

                if let Some(added_gradient) = added.as_gradient()
                {
                    add_gradient_operation(self, selectors, GradientOp::Copy{src: gradient, dst: added_gradient});
                }
            },
            Op::SetOtherSelector(index) =>
            {
                let gradient_index = self.phi_other_selectors_recording[index.0].gradient_index.expect("must be initialized");

                self.gradient_operations.push(GradientOp::SetOtherSelectorGradient{
                    loop_index: self.memory.phi_other_selectors_values[gradient_index.0].loop_index,
                    selector_index: gradient_index
                });
            },
            Op::GetOtherSelectorValue{index, ..}
            | Op::GetOtherSelectorTensor{index, ..} =>
            {
                let this_selector = &self.phi_other_selectors_recording[index.0];
                let gradient_index = this_selector.gradient_index.expect("must be initialized");

                let output: DiffWrapper = match op
                {
                    Op::GetOtherSelectorValue{output, ..} => output.into(),
                    Op::GetOtherSelectorTensor{output, ..} => output.into(),
                    _ => unreachable!()
                };

                if let Some(output_gradient) = output.as_gradient()
                {
                    let first = {
                        let first = this_selector.first;

                        first.as_gradient().unwrap_or_else(||
                        {
                            panic!("selector first ({}) must have a gradient", self.memory.format_variable(first.as_value()))
                        })
                    };

                    let other = this_selector.other.expect("must be initialized").as_gradient().expect("selectors must have a gradient");

                    let next_output = selectors.iter()
                        .find(|x| x.other == other)
                        .expect("must be a registered selector")
                        .next_previous.clone();

                    let gradient_op = match op
                    {
                        Op::GetOtherSelectorValue{..} =>
                        {
                            GradientOp::OtherSelectorValueGradient{
                                index: gradient_index,
                                first: first.into_value(),
                                other: next_output.into_value(),
                                src: output_gradient.into_value()
                            }
                        },
                        Op::GetOtherSelectorTensor{..} =>
                        {
                            GradientOp::OtherSelectorTensorGradient{
                                index: gradient_index,
                                first: first.into_tensor(),
                                other: next_output.into_tensor(),
                                src: output_gradient.into_tensor()
                            }
                        },
                        _ => unreachable!()
                    };

                    add_gradient_operation(self, selectors, gradient_op);
                }
            },
            Op::Loop{index, inputs, ops} =>
            {
                let gradient_loop_index = LoopIndex(self.loops.len());

                self.loops.push(LoopInfo{
                    reversed: true,
                    gradient_of_loop: Some(index),
                    kept_inside: Vec::new(),
                    ..self.loops[index.0].clone()
                });

                self.loops[index.0].loops_gradient = Some(gradient_loop_index);

                ops.iter().for_each(|op|
                {
                    if let Op::SetOtherSelector(phi_selector_index) = op
                    {
                        let this_selector = &mut self.phi_other_selectors_recording[phi_selector_index.0];

                        let new_index = PhiOtherSelectorIndex(self.memory.phi_other_selectors_values.len());
                        self.memory.phi_other_selectors_values.push(PhiOtherSelectorValue{
                            loop_index: gradient_loop_index,
                            is_set: false
                        });

                        this_selector.gradient_index = Some(new_index);
                    }
                });

                let gradient_inputs: Vec<_> = inputs.iter().map(|x|
                {
                    match x
                    {
                        InputTypePtr::Normal(tensor) =>
                        {
                            let (rows, columns) = self.memory.tensor_shape(*tensor);

                            let gradient_input = self.new_tensor_no_gradient(rows, columns).as_value();
                            self.name_tensor_suffix(gradient_input, *tensor, "_grad");

                            #[cfg(debug_assertions)]
                            {
                                self.memory.set_tensors_check.push(gradient_input.into());
                            }

                            InputTypePtr::Normal(gradient_input)
                        },
                        InputTypePtr::OneHot(_) => InputTypePtr::OneHot(self.new_one_hot())
                    }
                }).collect();

                let input_replacements: Vec<(_, _)> = inputs.iter().cloned().zip(gradient_inputs.iter().cloned()).collect();

                self.gradient_operations.push(GradientOp::Jump(JumpInfo::JumpTo{inputs: gradient_inputs.clone(), index: gradient_loop_index}));

                let gradient_operations_start = self.gradient_operations.len();
                ops.into_iter().rev().for_each(|op|
                {
                    self.calculate_gradient(
                        assigned_gradients,
                        selectors,
                        intermediate_selector,
                        used_stack_values,
                        &input_replacements,
                        op,
                        Some(gradient_loop_index)
                    );
                });

                {
                    let mut defined_values: Vec<_> = gradient_inputs.iter().map(|input|
                    {
                        DiffValue::from(*input)
                    }).collect();

                    let gradient_ops = &self.gradient_operations[gradient_operations_start..];
                    gradient_ops.iter().for_each(|op|
                    {
                        op.for_outputs(|out| defined_values.push(out));
                    });

                    gradient_ops.iter().for_each(|op|
                    {
                        op.for_args(|arg|
                        {
                            if !defined_values.contains(&arg)
                            {
                                let kept_inside = &mut self.loops[gradient_loop_index.0].kept_inside;

                                if !kept_inside.contains(&arg)
                                {
                                    kept_inside.push(arg);
                                }
                            }
                        });
                    });
                }

                if let Some(intermediate_selector_index) = intermediate_selector.take()
                {
                    self.gradient_operations.push(GradientOp::SetOtherSelector(intermediate_selector_index));
                }

                self.gradient_operations.push(GradientOp::Jump(JumpInfo::JumpFrom(gradient_loop_index)));
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OneHotIndex(usize);

#[allow(dead_code)]
impl OneHotIndex
{
    pub fn undefined() -> Self { Self(usize::MAX) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorPtr(usize);

#[allow(dead_code)]
impl TensorPtr
{
    pub fn from_raw(x: usize) -> Self { Self(x) }

    pub fn undefined() -> Self { Self(usize::MAX) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorIndex(usize);

#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffValueRaw
{
    Tensor(TensorRawDataPointer),
    Value(ValueIndex)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiffValue
{
    Tensor(TensorPtr),
    Value(ValueIndex),
    OneHot(OneHotIndex)
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

impl From<OneHotIndex> for DiffValue
{
    fn from(index: OneHotIndex) -> Self
    {
        Self::OneHot(index)
    }
}

impl From<InputTypePtr> for DiffValue
{
    fn from(input: InputTypePtr) -> Self
    {
        match input
        {
            InputTypePtr::Normal(x) => x.into(),
            InputTypePtr::OneHot(x) => x.into()
        }
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

#[allow(dead_code)]
impl DiffValue
{
    fn into_tensor(self) -> TensorPtr
    {
        if let Self::Tensor(x) = self { x } else { panic!("into_tensor must be called on a tensor") }
    }

    fn into_value(self) -> ValueIndex
    {
        if let Self::Value(x) = self { x } else { panic!("into_value must be called on a value") }
    }

    fn into_one_hot(self) -> OneHotIndex
    {
        if let Self::OneHot(x) = self { x } else { panic!("into_one_hot must be called on a onehot") }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct LoopOperationIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct GradientOperationIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhiOtherSelectorIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhiOtherSelectorRecordingIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffTensorPtr
{
    index: TensorPtr,
    gradient: Option<TensorPtr>
}

#[allow(dead_code)]
impl DiffTensorPtr
{
    pub fn no_gradient(index: TensorPtr) -> Self
    {
        Self{
            index,
            gradient: None
        }
    }

    pub fn undefined() -> Self
    {
        Self{
            index: TensorPtr::undefined(),
            gradient: None
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
    gradient: Option<ValueIndex>
}

#[allow(dead_code)]
impl DiffScalar
{
    pub fn undefined() -> Self
    {
        Self{
            index: ValueIndex(usize::MAX),
            gradient: None
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

#[allow(dead_code)]
impl DiffWrapper
{
    fn into_value(self) -> DiffScalar
    {
        if let Self::Value(x) = self { x } else { panic!("called into_value on a non-value DiffWrapper") }
    }

    fn into_tensor(self) -> DiffTensorPtr
    {
        if let Self::Tensor(x) = self { x } else { panic!("called into_tensor on a non-tensor DiffWrapper") }
    }

    fn as_value(self) -> DiffValue
    {
        match self
        {
            Self::Value(x) => DiffValue::Value(x.as_value()),
            Self::Tensor(x) => DiffValue::Tensor(x.as_value())
        }
    }

    fn as_gradient(self) -> Option<DiffValue>
    {
        match self
        {
            Self::Value(x) => x.as_gradient().map(DiffValue::Value),
            Self::Tensor(x) => x.as_gradient().map(DiffValue::Tensor)
        }
    }
}

#[allow(dead_code)]
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

#[derive(Clone)]
enum JumpInfo<I=InputTypePtr>
{
    JumpTo{inputs: Vec<I>, index: LoopIndex},
    JumpFrom(LoopIndex)
}

impl<I: Debug> Debug for JumpInfo<I>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        match self
        {
            Self::JumpTo{inputs, index} => write!(f, "id {}: {inputs:?}", index.0),
            Self::JumpFrom(index) => write!(f, "jump to id {}", index.0)
        }
    }
}

impl<I> JumpInfo<I>
{
    fn map_inputs<F: FnMut(I) -> U, U>(self, f: F) -> JumpInfo<U>
    {
        match self
        {
            Self::JumpTo{inputs, index} => JumpInfo::JumpTo{inputs: inputs.into_iter().map(f).collect(), index},
            Self::JumpFrom(x) => JumpInfo::JumpFrom(x)
        }
    }
}

struct NotationGradientOp<T, V, J, S>(GradientOp<T, V, J, S>);

impl<S> NotationGradientOp<DebugStringRaw, DebugStringRaw, JumpInfo<DebugStringRaw>, S>
{
    #[allow(dead_code)]
    fn from_nameable(memory: &OperationsRecorderMemory, op: GradientOp<TensorPtr, ValueIndex, JumpInfo, S>) -> Self
    {
        Self(op.map(|t|
        {
            DebugStringRaw(memory.format_variable(t))
        }, |v|
        {
            DebugStringRaw(memory.format_variable(v))
        }, |jump_info|
        {
            jump_info.map_inputs(|input| DebugStringRaw(memory.format_variable(input)))
        }, convert::identity))
    }
}

impl<T: Debug, V: Debug, J: Debug> Debug for NotationGradientOp<T, V, J, PhiOtherSelectorIndex>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        match &self.0
        {
            GradientOp::ZeroValue(dst) => write!(f, "{dst:?} ← 0"),
            GradientOp::ZeroTensor(dst) => write!(f, "{dst:?} ← 0"),
            GradientOp::Copy{src, dst} => write!(f, "{dst:?} ← {src:?}"),
            GradientOp::CopyScalar{src, dst} => write!(f, "{dst:?} ← {src:?}"),
            GradientOp::Add{lhs, rhs, output} => write!(f, "{output:?} ← {lhs:?} + {rhs:?}"),
            GradientOp::AddScalar{lhs, rhs, output} => write!(f, "{output:?} ← {lhs:?} + {rhs:?}"),
            GradientOp::AddScalars{lhs, rhs, output} => write!(f, "{output:?} ← {lhs:?} + {rhs:?}"),
            GradientOp::MulComponentwise{lhs, rhs, output} => write!(f, "{output:?} ← {lhs:?} ⊙ {rhs:?}"),
            GradientOp::MulComponentwiseAdd{lhs, rhs, added, output} => write!(f, "{output:?} ← {lhs:?} ⊙ {rhs:?} + {added:?}"),
            GradientOp::Tanh{value, output} => write!(f, "{output:?} ← tanh({value:?})"),
            GradientOp::TanhDiff{value, gradient, output} => write!(f, "{output:?} ← tanh'({value:?}) ⊙ {gradient:?}"),
            GradientOp::MulScalar{lhs, rhs, output} => write!(f, "{output:?} ← {lhs:?} ⋅ {rhs:?}"),
            GradientOp::Matmulv{lhs, rhs, output} => write!(f, "{output:?} ← {lhs:?} ⋅ {rhs:?}"),
            GradientOp::MatmulvTransposed{lhs, rhs, output} => write!(f, "{output:?} ← {lhs:?}ᵀ ⋅ {rhs:?}"),
            GradientOp::MatmulvAdd{lhs, rhs, added, output} => write!(f, "{output:?} ← {lhs:?} ⋅ {rhs:?} + {added:?}"),
            GradientOp::MatmulOneHotvAdd{lhs, rhs, added, output} => write!(f, "{output:?} ← {lhs:?} ⋅ {rhs:?} + {added:?}"),
            GradientOp::OuterProduct{lhs, rhs, output} => write!(f, "{output:?} ← {lhs:?} ⊗ {rhs:?}"),
            GradientOp::OuterProductOneHot{lhs, rhs, output} => write!(f, "{output:?} ← {lhs:?} ⊗ {rhs:?}"),
            GradientOp::SumTensor{value, output} => write!(f, "{output:?} ← ∑{value:?}"),
            GradientOp::GetOtherSelectorValue{info, first, other, output} => write!(f, "{output:?} ← {first:?} | {other:?} (id {})", info.0),
            GradientOp::GetOtherSelectorTensor{info, first, other, output} => write!(f, "{output:?} ← {first:?} | {other:?} (id {})", info.0),
            GradientOp::SetOtherSelector(info) => write!(f, "SetOtherSelector(id {})", info.0),
            GradientOp::OtherSelectorValueGradient{index, first, other, src} => write!(f, "{first:?} | {other:?} ← {src:?} (id {})", index.0),
            GradientOp::OtherSelectorTensorGradient{index, first, other, src} => write!(f, "{first:?} | {other:?} ← {src:?} (id {})", index.0),
            x => write!(f, "{x:?}")
        }
    }
}

type RawGradientOp = GradientOp<TensorRawDataPointer, ValueIndex, RawJumpInfo, PhiOtherSelectorIndex>;
type StandardGradientOp = GradientOp<TensorPtr, ValueIndex, JumpInfo, PhiOtherSelectorIndex>;

#[derive(Debug, Clone)]
pub enum GradientOp<T, V, J, S>
{
    None,
    FeedforwardEndMarker,
    SetInputs(LoopIndex),
    Jump(J),
    SetOtherSelector(S),
    GetOtherSelectorValue{info: S, first: V, other: V, output: V},
    GetOtherSelectorTensor{info: S, first: T, other: T, output: T},
    OtherSelectorValueGradient{index: PhiOtherSelectorIndex, first: V, other: V, src: V},
    OtherSelectorTensorGradient{index: PhiOtherSelectorIndex, first: T, other: T, src: T},
    GradientSelectAddValue{index: PhiOtherSelectorIndex, first: V, other: V, src: V, added: V},
    GradientSelectAddTensor{index: PhiOtherSelectorIndex, first: T, other: T, src: T, added: T},
    IfSetTensor{index: PhiOtherSelectorIndex, dst: T, src: T},
    IfNotSetTensor{index: PhiOtherSelectorIndex, dst: T, src: T},
    SetOtherSelectorGradient{loop_index: LoopIndex, selector_index: PhiOtherSelectorIndex},
    PushStackValue{loop_index: LoopIndex, value: V},
    PushStackTensor{loop_index: LoopIndex, tensor: T},
    PopStackValue{loop_index: LoopIndex, output: V},
    PopStackTensor{loop_index: LoopIndex, output: T},
    ZeroValue(V),
    ZeroTensor(T),
    Copy{src: T, dst: T},
    CopyScalar{src: V, dst: V},
    AddScalar{lhs: T, rhs: V, output: T},
    AddScalars{lhs: V, rhs: V, output: V},
    Add{lhs: T, rhs: T, output: T},
    Sub{lhs: T, rhs: T, output: T},
    SubFromScalar{lhs: V, rhs: T, output: T},
    MulScalar{lhs: T, rhs: V, output: T},
    MulScalars{lhs: V, rhs: V, output: V},
    MulComponentwise{lhs: T, rhs: T, output: T},
    MulComponentwiseAdd{lhs: T, rhs: T, added: T, output: T},
    SumTensor{value: T, output: V},
    Fill{value: V, output: T},
    Pow{lhs: T, power: u32, output: T},
    LeakyRelu{value: T, output: T},
    LeakyReluDiff{value: T, gradient: T, output: T},
    Sigmoid{value: T, output: T},
    SigmoidDiff{value: T, gradient: T, output: T},
    Tanh{value: T, output: T},
    TanhDiff{value: T, gradient: T, output: T},
    Dot{lhs: T, rhs: T, output: V},
    SoftmaxCrossEntropy{values: T, targets: OneHotIndex, softmaxed_output: T, output: V},
    SoftmaxCrossEntropyNoSoftmaxed{values: T, targets: OneHotIndex, output: V},
    SoftmaxCrossEntropyDiff{softmaxed_values: T, gradient: V, targets: OneHotIndex, output: T},
    Matmulv{lhs: T, rhs: T, output: T},
    MatmulvAdd{lhs: T, rhs: T, added: T, output: T},
    MatmulOneHotvAdd{lhs: T, rhs: OneHotIndex, added: T, output: T},
    MatmulvTransposed{lhs: T, rhs: T, output: T},
    OuterProduct{lhs: T, rhs: T, output: T},
    OuterProductAdd{lhs: T, rhs: T, added: T, output: T},
    OuterProductOneHot{lhs: T, rhs: OneHotIndex, output: T}
}

impl<T, V, J, S> GradientOp<T, V, J, S>
{
    fn map<U, W, K, SU>(
        self,
        mut tf: impl FnMut(T) -> U,
        mut vf: impl FnMut(V) -> W,
        jump_f: impl FnOnce(J) -> K,
        select_f: impl FnOnce(S) -> SU
    ) -> GradientOp<U, W, K, SU>
    {
        match self
        {
            Self::None => GradientOp::None,
            Self::FeedforwardEndMarker => GradientOp::FeedforwardEndMarker,
            Self::SetOtherSelector(info) => GradientOp::SetOtherSelector(select_f(info)),
            Self::GetOtherSelectorValue{info, first, other, output}  =>
            {
                GradientOp::GetOtherSelectorValue{info: select_f(info), first: vf(first), other: vf(other), output: vf(output)}
            },
            Self::GetOtherSelectorTensor{info, first, other, output}  =>
            {
                GradientOp::GetOtherSelectorTensor{info: select_f(info), first: tf(first), other: tf(other), output: tf(output)}
            },
            Self::OtherSelectorValueGradient{index, first, other, src} =>
            {
                GradientOp::OtherSelectorValueGradient{index, first: vf(first), other: vf(other), src: vf(src)}
            },
            Self::OtherSelectorTensorGradient{index, first, other, src} =>
            {
                GradientOp::OtherSelectorTensorGradient{index, first: tf(first), other: tf(other), src: tf(src)}
            },
            Self::GradientSelectAddValue{index, first, other, src, added} =>
            {
                GradientOp::GradientSelectAddValue{index, first: vf(first), other: vf(other), src: vf(src), added: vf(added)}
            },
            Self::GradientSelectAddTensor{index, first, other, src, added} =>
            {
                GradientOp::GradientSelectAddTensor{index, first: tf(first), other: tf(other), src: tf(src), added: tf(added)}
            },
            Self::IfSetTensor{index, dst, src} => GradientOp::IfSetTensor{index, dst: tf(dst), src: tf(src)},
            Self::IfNotSetTensor{index, dst, src} => GradientOp::IfNotSetTensor{index, dst: tf(dst), src: tf(src)},
            Self::SetOtherSelectorGradient{loop_index, selector_index} => GradientOp::SetOtherSelectorGradient{loop_index, selector_index},
            Self::PushStackValue{loop_index, value} => GradientOp::PushStackValue{loop_index, value: vf(value)},
            Self::PushStackTensor{loop_index, tensor} => GradientOp::PushStackTensor{loop_index, tensor: tf(tensor)},
            Self::PopStackValue{loop_index, output} => GradientOp::PopStackValue{loop_index, output: vf(output)},
            Self::PopStackTensor{loop_index, output} => GradientOp::PopStackTensor{loop_index, output: tf(output)},
            Self::ZeroValue(dst) => GradientOp::ZeroValue(vf(dst)),
            Self::ZeroTensor(dst) => GradientOp::ZeroTensor(tf(dst)),
            Self::Copy{src, dst} => GradientOp::Copy{src: tf(src), dst: tf(dst)},
            Self::CopyScalar{src, dst} => GradientOp::CopyScalar{src: vf(src), dst: vf(dst)},
            Self::AddScalar{lhs, rhs, output} => GradientOp::AddScalar{lhs: tf(lhs), rhs: vf(rhs), output: tf(output)},
            Self::AddScalars{lhs, rhs, output} => GradientOp::AddScalars{lhs: vf(lhs), rhs: vf(rhs), output: vf(output)},
            Self::Add{lhs, rhs, output} => GradientOp::Add{lhs: tf(lhs), rhs: tf(rhs), output: tf(output)},
            Self::Sub{lhs, rhs, output} => GradientOp::Sub{lhs: tf(lhs), rhs: tf(rhs), output: tf(output)},
            Self::SubFromScalar{lhs, rhs, output} => GradientOp::SubFromScalar{lhs: vf(lhs), rhs: tf(rhs), output: tf(output)},
            Self::MulScalar{lhs, rhs, output} => GradientOp::MulScalar{lhs: tf(lhs), rhs: vf(rhs), output: tf(output)},
            Self::MulScalars{lhs, rhs, output} => GradientOp::MulScalars{lhs: vf(lhs), rhs: vf(rhs), output: vf(output)},
            Self::MulComponentwise{lhs, rhs, output} => GradientOp::MulComponentwise{lhs: tf(lhs), rhs: tf(rhs), output: tf(output)},
            Self::MulComponentwiseAdd{lhs, rhs, added, output} =>
            {
                GradientOp::MulComponentwiseAdd{lhs: tf(lhs), rhs: tf(rhs), added: tf(added), output: tf(output)}
            },
            Self::SumTensor{value, output} => GradientOp::SumTensor{value: tf(value), output: vf(output)},
            Self::Fill{value, output} => GradientOp::Fill{value: vf(value), output: tf(output)},
            Self::Pow{lhs, power, output} => GradientOp::Pow{lhs: tf(lhs), power, output: tf(output)},
            Self::LeakyRelu{value, output} => GradientOp::LeakyRelu{value: tf(value), output: tf(output)},
            Self::LeakyReluDiff{value, gradient, output} => GradientOp::LeakyReluDiff{value: tf(value), gradient: tf(gradient), output: tf(output)},
            Self::Sigmoid{value, output} => GradientOp::Sigmoid{value: tf(value), output: tf(output)},
            Self::SigmoidDiff{value, gradient, output} => GradientOp::SigmoidDiff{value: tf(value), gradient: tf(gradient), output: tf(output)},
            Self::Tanh{value, output} => GradientOp::Tanh{value: tf(value), output: tf(output)},
            Self::TanhDiff{value, gradient, output} => GradientOp::TanhDiff{value: tf(value), gradient: tf(gradient), output: tf(output)},
            Self::Dot{lhs, rhs, output} => GradientOp::Dot{lhs: tf(lhs), rhs: tf(rhs), output: vf(output)},
            Self::SoftmaxCrossEntropy{values, targets, softmaxed_output, output} =>
            {
                GradientOp::SoftmaxCrossEntropy{values: tf(values), targets, softmaxed_output: tf(softmaxed_output), output: vf(output)}
            },
            Self::SoftmaxCrossEntropyDiff{softmaxed_values, gradient, targets, output} =>
            {
                GradientOp::SoftmaxCrossEntropyDiff{softmaxed_values: tf(softmaxed_values), gradient: vf(gradient), targets, output: tf(output)}
            },
            Self::Matmulv{lhs, rhs, output} => GradientOp::Matmulv{lhs: tf(lhs), rhs: tf(rhs), output: tf(output)},
            Self::MatmulvAdd{lhs, rhs, added, output} => GradientOp::MatmulvAdd{lhs: tf(lhs), rhs: tf(rhs), added: tf(added), output: tf(output)},
            Self::MatmulOneHotvAdd{lhs, rhs, added, output} =>
            {
                GradientOp::MatmulOneHotvAdd{lhs: tf(lhs), rhs, added: tf(added), output: tf(output)}
            },
            Self::MatmulvTransposed{lhs, rhs, output} => GradientOp::MatmulvTransposed{lhs: tf(lhs), rhs: tf(rhs), output: tf(output)},
            Self::OuterProduct{lhs, rhs, output} => GradientOp::OuterProduct{lhs: tf(lhs), rhs: tf(rhs), output: tf(output)},
            Self::OuterProductAdd{lhs, rhs, added, output} =>
            {
                GradientOp::OuterProductAdd{lhs: tf(lhs), rhs: tf(rhs), added: tf(added), output: tf(output)}
            },
            Self::OuterProductOneHot{lhs, rhs, output} => GradientOp::OuterProductOneHot{lhs: tf(lhs), rhs, output: tf(output)},
            Self::Jump(x) => GradientOp::Jump(jump_f(x)),
            Self::SetInputs(x) => GradientOp::SetInputs(x),
            Self::SoftmaxCrossEntropyNoSoftmaxed{values, targets, output} =>
            {
                GradientOp::SoftmaxCrossEntropyNoSoftmaxed{values: tf(values), targets, output: vf(output)}
            }
        }
    }
}

impl<J, S> GradientOp<TensorPtr, ValueIndex, J, S>
{
    fn map_outputs(self, f: impl FnMut(DiffValue) -> DiffValue) -> Self
    {
        self.map_outputs_with_state(f, |inner_f, t_output|
        {
            inner_f(DiffValue::Tensor(t_output)).into_tensor()
        }, |inner_f, v_output|
        {
            inner_f(DiffValue::Value(v_output)).into_value()
        })
    }

    fn map_args(self, f: impl FnMut(DiffValue) -> DiffValue) -> Self
    {
        self.map_args_with_state(f, |inner_f, t_arg|
        {
            inner_f(DiffValue::Tensor(t_arg)).into_tensor()
        }, |inner_f, v_arg|
        {
            inner_f(DiffValue::Value(v_arg)).into_value()
        }, |inner_f, o_arg|
        {
            inner_f(DiffValue::OneHot(o_arg)).into_one_hot()
        })
    }

    fn for_args(&self, f: impl FnMut(DiffValue))
    where
        J: Clone,
        S: Clone
    {
        self.clone().map_args_with_state(f, |inner_f, x|
        {
            inner_f(DiffValue::Tensor(x));
            x
        }, |inner_f, x|
        {
            inner_f(DiffValue::Value(x));
            x
        }, |inner_f, x|
        {
            inner_f(DiffValue::OneHot(x));
            x
        });
    }

    fn for_outputs(&self, f: impl FnMut(DiffValue))
    where
        J: Clone,
        S: Clone
    {
        self.clone().map_outputs_with_state(f, |inner_f, x|
        {
            inner_f(DiffValue::Tensor(x));
            x
        }, |inner_f, x|
        {
            inner_f(DiffValue::Value(x));
            x
        });
    }
}

impl<T, J, S> GradientOp<T, ValueIndex, J, S>
{
    fn map_outputs_with_state<State>(
        self,
        mut state: State,
        mut tf: impl FnMut(&mut State, T) -> T,
        mut vf: impl FnMut(&mut State, ValueIndex) -> ValueIndex
    ) -> Self
    {
        match self
        {
            Self::ZeroValue(dst) => Self::ZeroValue(vf(&mut state, dst)),
            Self::ZeroTensor(dst) => Self::ZeroTensor(tf(&mut state, dst)),
            Self::PushStackValue{loop_index, value} => Self::PushStackValue{loop_index, value},
            Self::PushStackTensor{loop_index, tensor} => Self::PushStackTensor{loop_index, tensor},
            Self::PopStackValue{loop_index, output} => Self::PopStackValue{loop_index, output: vf(&mut state, output)},
            Self::PopStackTensor{loop_index, output} => Self::PopStackTensor{loop_index, output: tf(&mut state, output)},
            Self::Copy{dst, src} => Self::Copy{dst: tf(&mut state, dst), src},
            Self::AddScalar{output, lhs, rhs} => Self::AddScalar{output: tf(&mut state, output), lhs, rhs},
            Self::Add{output, lhs, rhs} => Self::Add{output: tf(&mut state, output), lhs, rhs},
            Self::Sub{output, lhs, rhs} => Self::Sub{output: tf(&mut state, output), lhs, rhs},
            Self::SubFromScalar{output, lhs, rhs} => Self::SubFromScalar{output: tf(&mut state, output), lhs, rhs},
            Self::MulScalar{output, lhs, rhs} => Self::MulScalar{output: tf(&mut state, output), lhs, rhs},
            Self::MulComponentwise{output, lhs, rhs} => Self::MulComponentwise{output: tf(&mut state, output), lhs, rhs},
            Self::MulComponentwiseAdd{output, lhs, rhs, added} => Self::MulComponentwiseAdd{output: tf(&mut state, output), lhs, rhs, added},
            Self::Fill{output, value} => Self::Fill{output: tf(&mut state, output), value},
            Self::Pow{output, power, lhs} => Self::Pow{output: tf(&mut state, output), power, lhs},
            Self::Sigmoid{output, value} => Self::Sigmoid{output: tf(&mut state, output), value},
            Self::SigmoidDiff{output, gradient, value} => Self::SigmoidDiff{output: tf(&mut state, output), gradient, value},
            Self::Tanh{output, value} => Self::Tanh{output: tf(&mut state, output), value},
            Self::TanhDiff{output, gradient, value} => Self::TanhDiff{output: tf(&mut state, output), gradient, value},
            Self::LeakyRelu{output, value} => Self::LeakyRelu{output: tf(&mut state, output), value},
            Self::LeakyReluDiff{output, gradient, value} => Self::LeakyReluDiff{output: tf(&mut state, output), gradient, value},
            Self::SoftmaxCrossEntropyDiff{output, softmaxed_values, gradient, targets} =>
            {
                Self::SoftmaxCrossEntropyDiff{output: tf(&mut state, output), softmaxed_values, gradient, targets}
            },
            Self::Matmulv{output, lhs, rhs} => Self::Matmulv{output: tf(&mut state, output), lhs, rhs},
            Self::MatmulvAdd{output, lhs, rhs, added} => Self::MatmulvAdd{output: tf(&mut state, output), lhs, rhs, added},
            Self::MatmulOneHotvAdd{output, lhs, rhs, added} => Self::MatmulOneHotvAdd{output: tf(&mut state, output), lhs, rhs, added},
            Self::MatmulvTransposed{output, lhs, rhs} => Self::MatmulvTransposed{output: tf(&mut state, output), lhs, rhs},
            Self::OuterProduct{output, lhs, rhs} => Self::OuterProduct{output: tf(&mut state, output), lhs, rhs},
            Self::OuterProductAdd{output, lhs, rhs, added} => Self::OuterProductAdd{output: tf(&mut state, output), lhs, rhs, added},
            Self::OuterProductOneHot{output, lhs, rhs} => Self::OuterProductOneHot{output: tf(&mut state, output), lhs, rhs},
            Self::CopyScalar{dst, src} => Self::CopyScalar{dst: vf(&mut state, dst), src},
            Self::AddScalars{output, lhs, rhs} => Self::AddScalars{output: vf(&mut state, output), lhs, rhs},
            Self::MulScalars{output, lhs, rhs} => Self::MulScalars{output: vf(&mut state, output), lhs, rhs},
            Self::SumTensor{output, value} => Self::SumTensor{output: vf(&mut state, output), value},
            Self::Dot{output, lhs, rhs} => Self::Dot{output: vf(&mut state, output), lhs, rhs},
            Self::SoftmaxCrossEntropy{softmaxed_output, output, targets, values} =>
            {
                Self::SoftmaxCrossEntropy{softmaxed_output: tf(&mut state, softmaxed_output), output: vf(&mut state, output), targets, values}
            },
            Self::SoftmaxCrossEntropyNoSoftmaxed{output, targets, values} =>
            {
                Self::SoftmaxCrossEntropyNoSoftmaxed{output: vf(&mut state, output), targets, values}
            },
            Self::None => Self::None,
            Self::SetOtherSelector(info) => Self::SetOtherSelector(info),
            Self::GetOtherSelectorValue{info, first, other, output} =>
            {
                Self::GetOtherSelectorValue{info, first, other, output: vf(&mut state, output)}
            },
            Self::GetOtherSelectorTensor{info, first, other, output} =>
            {
                Self::GetOtherSelectorTensor{info, first, other, output: tf(&mut state, output)}
            },
            Self::OtherSelectorValueGradient{index, first, other, src} =>
            {
                Self::OtherSelectorValueGradient{index, first: vf(&mut state, first), other: vf(&mut state, other), src}
            },
            Self::OtherSelectorTensorGradient{index, first, other, src} =>
            {
                Self::OtherSelectorTensorGradient{index, first: tf(&mut state, first), other: tf(&mut state, other), src}
            },
            Self::GradientSelectAddValue{index, first, other, src, added} =>
            {
                Self::GradientSelectAddValue{index, first: vf(&mut state, first), other: vf(&mut state, other), src, added}
            },
            Self::GradientSelectAddTensor{index, first, other, src, added} =>
            {
                Self::GradientSelectAddTensor{index, first: tf(&mut state, first), other: tf(&mut state, other), src, added}
            },
            Self::IfSetTensor{index, dst, src} => Self::IfSetTensor{index, dst: tf(&mut state, dst), src},
            Self::IfNotSetTensor{index, dst, src} => Self::IfNotSetTensor{index, dst: tf(&mut state, dst), src},
            Self::SetOtherSelectorGradient{loop_index, selector_index} => Self::SetOtherSelectorGradient{loop_index, selector_index},
            Self::Jump(x) => Self::Jump(x),
            Self::SetInputs(x) => Self::SetInputs(x),
            Self::FeedforwardEndMarker => Self::FeedforwardEndMarker
        }
    }

    fn map_args_with_state<State>(
        self,
        mut state: State,
        mut tf: impl FnMut(&mut State, T) -> T,
        mut vf: impl FnMut(&mut State, ValueIndex) -> ValueIndex,
        mut of: impl FnMut(&mut State, OneHotIndex) -> OneHotIndex
    ) -> Self
    {
        match self
        {
            Self::ZeroValue(dst) => Self::ZeroValue(dst),
            Self::ZeroTensor(dst) => Self::ZeroTensor(dst),
            Self::PushStackValue{loop_index, value} => Self::PushStackValue{loop_index, value: vf(&mut state, value)},
            Self::PushStackTensor{loop_index, tensor} => Self::PushStackTensor{loop_index, tensor: tf(&mut state, tensor)},
            Self::PopStackValue{loop_index, output} => Self::PopStackValue{loop_index, output},
            Self::PopStackTensor{loop_index, output} => Self::PopStackTensor{loop_index, output},
            Self::Copy{src, dst} => Self::Copy{src: tf(&mut state, src), dst},
            Self::AddScalar{lhs, rhs, output} => Self::AddScalar{lhs: tf(&mut state, lhs), rhs: vf(&mut state, rhs), output},
            Self::Add{lhs, rhs, output} => Self::Add{lhs: tf(&mut state, lhs), rhs: tf(&mut state, rhs), output},
            Self::Sub{lhs, rhs, output} => Self::Sub{lhs: tf(&mut state, lhs), rhs: tf(&mut state, rhs), output},
            Self::SubFromScalar{lhs, rhs, output} => Self::SubFromScalar{rhs: tf(&mut state, rhs), lhs: vf(&mut state, lhs), output},
            Self::MulScalar{lhs, rhs, output} => Self::MulScalar{lhs: tf(&mut state, lhs), rhs: vf(&mut state, rhs), output},
            Self::MulComponentwise{lhs, rhs, output} => Self::MulComponentwise{lhs: tf(&mut state, lhs), rhs: tf(&mut state, rhs), output},
            Self::MulComponentwiseAdd{lhs, rhs, added, output} =>
            {
                Self::MulComponentwiseAdd{lhs: tf(&mut state, lhs), rhs: tf(&mut state, rhs), added: tf(&mut state, added), output}
            },
            Self::SumTensor{value, output} => Self::SumTensor{value: tf(&mut state, value), output},
            Self::Pow{lhs, power, output} => Self::Pow{lhs: tf(&mut state, lhs), power, output},
            Self::LeakyRelu{value, output} => Self::LeakyRelu{value: tf(&mut state, value), output},
            Self::LeakyReluDiff{value, gradient, output} =>
            {
                Self::LeakyReluDiff{value: tf(&mut state, value), gradient: tf(&mut state, gradient), output}
            },
            Self::Sigmoid{value, output} => Self::Sigmoid{value: tf(&mut state, value), output},
            Self::SigmoidDiff{value, gradient, output} => Self::SigmoidDiff{value: tf(&mut state, value), gradient: tf(&mut state, gradient), output},
            Self::Tanh{value, output} => Self::Tanh{value: tf(&mut state, value), output},
            Self::TanhDiff{value, gradient, output} => Self::TanhDiff{value: tf(&mut state, value), gradient: tf(&mut state, gradient), output},
            Self::Dot{lhs, rhs, output} => Self::Dot{lhs: tf(&mut state, lhs), rhs: tf(&mut state, rhs), output},
            Self::SoftmaxCrossEntropy{values, targets, softmaxed_output, output} =>
            {
                Self::SoftmaxCrossEntropy{values: tf(&mut state, values), targets: of(&mut state, targets), softmaxed_output, output}
            },
            Self::SoftmaxCrossEntropyDiff{softmaxed_values, gradient, targets, output} =>
            {
                Self::SoftmaxCrossEntropyDiff{
                    softmaxed_values: tf(&mut state, softmaxed_values),
                    gradient: vf(&mut state, gradient),
                    targets: of(&mut state, targets),
                    output
                }
            },
            Self::Matmulv{lhs, rhs, output} =>
            {
                Self::Matmulv{lhs: tf(&mut state, lhs), rhs: tf(&mut state, rhs), output}
            },
            Self::MatmulvAdd{lhs, rhs, added, output} =>
            {
                Self::MatmulvAdd{lhs: tf(&mut state, lhs), rhs: tf(&mut state, rhs), added: tf(&mut state, added), output}
            },
            Self::MatmulOneHotvAdd{lhs, rhs, added, output} =>
            {
                Self::MatmulOneHotvAdd{lhs: tf(&mut state, lhs), rhs: of(&mut state, rhs), added: tf(&mut state, added), output}
            },
            Self::MatmulvTransposed{lhs, rhs, output} => Self::MatmulvTransposed{lhs: tf(&mut state, lhs), rhs: tf(&mut state, rhs), output},
            Self::OuterProduct{lhs, rhs, output} => Self::OuterProduct{lhs: tf(&mut state, lhs), rhs: tf(&mut state, rhs), output},
            Self::OuterProductAdd{lhs, rhs, added, output} =>
            {
                Self::OuterProductAdd{lhs: tf(&mut state, lhs), rhs: tf(&mut state, rhs), added: tf(&mut state, added), output}
            },
            Self::OuterProductOneHot{lhs, rhs, output} => Self::OuterProductOneHot{lhs: tf(&mut state, lhs), rhs: of(&mut state, rhs), output},
            Self::CopyScalar{src, dst} => Self::CopyScalar{src: vf(&mut state, src), dst},
            Self::AddScalars{lhs, rhs, output} => Self::AddScalars{lhs: vf(&mut state, lhs), rhs: vf(&mut state, rhs), output},
            Self::MulScalars{lhs, rhs, output} => Self::MulScalars{lhs: vf(&mut state, lhs), rhs: vf(&mut state, rhs), output},
            Self::Fill{value, output} => Self::Fill{value: vf(&mut state, value), output},
            Self::SoftmaxCrossEntropyNoSoftmaxed{values, targets, output} =>
            {
                Self::SoftmaxCrossEntropyNoSoftmaxed{values: tf(&mut state, values), targets, output}
            },
            Self::None => Self::None,
            Self::SetOtherSelector(info) => Self::SetOtherSelector(info),
            Self::GetOtherSelectorValue{info, first, other, output} =>
            {
                Self::GetOtherSelectorValue{info, first: vf(&mut state, first), other: vf(&mut state, other), output}
            },
            Self::GetOtherSelectorTensor{info, first, other, output} =>
            {
                Self::GetOtherSelectorTensor{info, first: tf(&mut state, first), other: tf(&mut state, other), output}
            },
            Self::OtherSelectorValueGradient{index, first, other, src} =>
            {
                Self::OtherSelectorValueGradient{index, first, other, src: vf(&mut state, src)}
            },
            Self::OtherSelectorTensorGradient{index, first, other, src} =>
            {
                Self::OtherSelectorTensorGradient{index, first, other, src: tf(&mut state, src)}
            },
            Self::GradientSelectAddValue{index, first, other, src, added} =>
            {
                Self::GradientSelectAddValue{index, first, other, src: vf(&mut state, src), added: vf(&mut state, added)}
            },
            Self::GradientSelectAddTensor{index, first, other, src, added} =>
            {
                Self::GradientSelectAddTensor{index, first, other, src: tf(&mut state, src), added: tf(&mut state, added)}
            },
            Self::IfSetTensor{index, dst, src} => Self::IfSetTensor{index, dst, src: tf(&mut state, src)},
            Self::IfNotSetTensor{index, dst, src} => Self::IfNotSetTensor{index, dst, src: tf(&mut state, src)},
            Self::SetOtherSelectorGradient{loop_index, selector_index} => Self::SetOtherSelectorGradient{loop_index, selector_index},
            Self::Jump(x) => Self::Jump(x),
            Self::SetInputs(x) => Self::SetInputs(x),
            Self::FeedforwardEndMarker => Self::FeedforwardEndMarker
        }
    }
}

#[derive(Debug, Clone)]
pub enum Op
{
    CopyScalar{src: DiffScalar, dst: DiffScalar},
    Copy{src: DiffTensorPtr, dst: DiffTensorPtr},
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
    SetOtherSelector(PhiOtherSelectorRecordingIndex),
    GetOtherSelectorValue{index: PhiOtherSelectorRecordingIndex, output: DiffScalar},
    GetOtherSelectorTensor{index: PhiOtherSelectorRecordingIndex, output: DiffTensorPtr},
    Loop{index: LoopIndex, inputs: Vec<InputTypePtr>, ops: Vec<Op>}
}

impl Op
{
    fn for_outputs(&self, mut f: impl FnMut(DiffWrapper))
    {
        match *self
        {
            Self::CopyScalar{dst, ..} => f(dst.into()),
            Self::Copy{dst, ..} => f(dst.into()),
            Self::AddScalar{output, ..} => f(output.into()),
            Self::AddScalars{output, ..} => f(output.into()),
            Self::Add{output, ..} => f(output.into()),
            Self::Sub{output, ..} => f(output.into()),
            Self::SubFromScalar{output, ..} => f(output.into()),
            Self::MulScalar{output, ..} => f(output.into()),
            Self::MulScalars{output, ..} => f(output.into()),
            Self::MulComponentwise{output, ..} => f(output.into()),
            Self::SumTensor{output, ..} => f(output.into()),
            Self::Pow{output, ..} => f(output.into()),
            Self::LeakyRelu{output, ..} => f(output.into()),
            Self::Sigmoid{output, ..} => f(output.into()),
            Self::Tanh{output, ..} => f(output.into()),
            Self::Dot{output, ..} => f(output.into()),
            Self::SoftmaxCrossEntropy{softmaxed_output, output, ..} => { f(softmaxed_output.into()); f(output.into()) },
            Self::Matmulv{output, ..} => f(output.into()),
            Self::MatmulvAdd{output, ..} => f(output.into()),
            Self::MatmulOneHotvAdd{output, ..} => f(output.into()),
            Self::GetOtherSelectorValue{output, ..} => f(output.into()),
            Self::GetOtherSelectorTensor{output, ..} => f(output.into()),
            Self::SetOtherSelector(_) => (),
            Self::Loop{..} => unreachable!()
        }
    }

    fn for_args(&self, mut f: impl FnMut(DiffWrapper))
    {
        match *self
        {
            Self::CopyScalar{src, ..} => f(src.into()),
            Self::Copy{src, ..} => f(src.into()),
            Self::AddScalar{lhs, rhs, ..} => { f(lhs.into()); f(rhs.into()) },
            Self::AddScalars{lhs, rhs, ..} => { f(lhs.into()); f(rhs.into()) },
            Self::Add{lhs, rhs, ..} => { f(lhs.into()); f(rhs.into()) },
            Self::Sub{lhs, rhs, ..} => { f(lhs.into()); f(rhs.into()) },
            Self::SubFromScalar{lhs, rhs, ..} => { f(lhs.into()); f(rhs.into()) },
            Self::MulScalar{lhs, rhs, ..} => { f(lhs.into()); f(rhs.into()) },
            Self::MulScalars{lhs, rhs, ..} => { f(lhs.into()); f(rhs.into()) },
            Self::MulComponentwise{lhs, rhs, ..} => { f(lhs.into()); f(rhs.into()) },
            Self::SumTensor{value, ..} => { f(value.into()) },
            Self::Pow{lhs, ..} => { f(lhs.into()) },
            Self::LeakyRelu{value, ..} => { f(value.into()) },
            Self::Sigmoid{value, ..} => { f(value.into()) },
            Self::Tanh{value, ..} => { f(value.into()) },
            Self::Dot{lhs, rhs, ..} => { f(lhs.into()); f(rhs.into()) },
            Self::SoftmaxCrossEntropy{values, ..} => f(values.into()),
            Self::Matmulv{lhs, rhs, ..} => { f(lhs.into()); f(rhs.into()) },
            Self::MatmulvAdd{lhs, rhs, added, ..} => { f(lhs.into()); f(rhs.into()); f(added.into()) },
            Self::MatmulOneHotvAdd{lhs, added, ..} => { f(lhs.into()); f(added.into()) },
            Self::GetOtherSelectorValue{..}
            | Self::GetOtherSelectorTensor{..}
            | Self::SetOtherSelector(_) => (),
            Self::Loop{..} => unreachable!()
        }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
impl InputTypePtr
{
    pub fn into_normal(self) -> TensorPtr
    {
        if let Self::Normal(x) = self { x } else { panic!("expected normal") }
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
    pub fn into_normal(self) -> DiffTensorPtr
    {
        if let Self::Normal(x) = self { x } else { panic!("expected normal") }
    }

    pub fn into_one_hot(self) -> OneHotIndex
    {
        if let Self::OneHot(x) = self { x } else { panic!("expected onehot") }
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

    pub fn into_ref_normal(&self) -> &LayerType
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
        correct.as_vec().into_iter().for_each(|x| debug_assert!(!x.is_nan()));

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
        recorder.name_diff_tensor(a, "a");
        recorder.name_diff_tensor(b, "b");

        recorder.allow_discard(a.as_value());
        recorder.allow_discard(b.as_value());

        let out = f(recorder, a, b);

        recorder.name_diff_tensor(out, "out");

        let a_gradient = a.as_gradient().unwrap();
        let b_gradient = b.as_gradient().unwrap();

        recorder.finish();

        recorder.store_tensor_until_end(a_gradient);
        recorder.store_tensor_until_end(b_gradient);

        recorder.gradient(out.into());

        recorder.resolve_memory();

        let a_gradient = recorder.resolve_tensor_ptr(a_gradient);
        let b_gradient = recorder.resolve_tensor_ptr(b_gradient);

        recorder.calculate();

        let a_g = recorder.get_tensor(a_gradient).clone_owned();
        let b_g = recorder.get_tensor(b_gradient).clone_owned();

        let mut new_recorder = OperationsRecorder::new();

        let new_a = new_recorder.new_tensor_no_gradient(a_value.rows(), a_value.columns());
        let new_b = new_recorder.new_tensor_no_gradient(b_value.rows(), b_value.columns());

        new_recorder.allow_discard(new_a.as_value());
        new_recorder.allow_discard(new_b.as_value());

        let output = f(&mut new_recorder, new_a, new_b);
        let output_value = output.as_value();

        new_recorder.store_tensor_until_end(output_value);

        new_recorder.finish();
        new_recorder.no_gradient();

        new_recorder.resolve_memory();

        let output_value = new_recorder.resolve_tensor_ptr(output_value);

        let new_a = new_recorder.resolve_tensor_ptr(new_a.as_value());
        let new_b = new_recorder.resolve_tensor_ptr(new_b.as_value());

        let mut vals = |a: &LayerType, b: &LayerType|
        {
            new_recorder.set_tensor(new_a, a);
            new_recorder.set_tensor(new_b, b);

            new_recorder.calculate();

            new_recorder.get_tensor(output_value).clone_owned()
        };

        let orig = vals(&a_value, &b_value).sum();

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

            let this_fg = fg(vals(&(v.add(epsilon.as_ref())), &b_value));

            a_fg[index] = this_fg;
        }

        let mut b_fg = vec![0.0; b_value.total_len()];
        for index in 0..b_fg.len()
        {
            let v = b_value.clone();
            let epsilon = one_hot(v.clone(), index, epsilon, 0.0);

            let this_fg = fg(vals(&a_value, &(v.add(epsilon.as_ref()))));

            b_fg[index] = this_fg;
        }

        let vec_to_layer = |v, mut layer: LayerType|
        {
            layer.swap_raw_values(v);

            layer
        };

        let a_fg = vec_to_layer(a_fg, a_value);
        let b_fg = vec_to_layer(b_fg, b_value);

        eprintln!("derivative of a ({a_fg:?} vs {a_g:?})");
        eprintln!("derivative of b ({b_fg:?} vs {b_g:?})");

        eprintln!("CHECKING derivative of a");
        compare_tensor(a_fg, a_g);

        eprintln!("CHECKING derivative of b");
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
            recorder.name_diff_tensor(mul_result, "mul_result");

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
    fn copy_scalar()
    {
        check_tensor(|recorder, a, b|
        {
            let a_sum = recorder.sum_tensor(a);

            let a_sum_copy = recorder.copy_scalar(a_sum);

            recorder.add_scalar(b, a_sum_copy)
        })
    }

    #[test]
    fn double_dependency()
    {
        check_tensor(|recorder, a, b|
        {
            let ab = recorder.mul_componentwise(a, b);
            recorder.name_diff_tensor(ab, "ab");

            let abb = recorder.mul_componentwise(ab, b);
            recorder.name_diff_tensor(abb, "abb");

            recorder.add(ab, abb)
        })
    }

    #[test]
    fn simple_sum_state()
    {
        let loops_count = 3;
        let is: Vec<OwnedInputType> = (0..loops_count).map(|_| LayerType::new_with(LAYER_CURR, LAYER_PREV, random_value).into()).collect();

        check_tensor(|recorder, a, b|
        {
            let a_sum = recorder.sum_tensor(a);
            recorder.name_diff_scalar(a_sum, "a_sum");

            let (rows, columns) = recorder.tensor_shape(a.as_value());

            let i = recorder.new_tensor_no_gradient(rows, columns).as_value();
            recorder.name_tensor(i, "i");

            let final_state_selector = recorder.phi_other_selector(a_sum);

            let loop_index = recorder.begin_loop(vec![i.into()]);

            let ai = recorder.mul_componentwise(a, DiffTensorPtr::no_gradient(i));
            recorder.name_diff_tensor(ai, "ai");

            let ai_sum = recorder.sum_tensor(ai);
            recorder.name_diff_scalar(ai_sum, "ai_sum");

            let final_selected = recorder.select_value(final_state_selector);
            recorder.name_diff_scalar(final_selected, "final_selected");

            let final_state = recorder.add_scalars(final_selected, ai_sum);
            recorder.name_diff_scalar(final_state, "final_state");

            recorder.set_phi_other_selector(final_state_selector, final_state);

            recorder.end_loop(loop_index);

            recorder.set_loop_times(loop_index, loops_count);
            recorder.set_loop_inputs(loop_index, is.iter().cloned().collect::<Vec<_>>());

            recorder.add_scalar(b, final_state)
        })
    }

    #[test]
    fn weird_sum_thing()
    {
        let loops_count = 3;

        let zeros = LayerType::new(LAYER_CURR, LAYER_PREV);
        let is: Vec<OwnedInputType> = (0..loops_count).map(|_| LayerType::new_with(LAYER_CURR, LAYER_PREV, random_value).into()).collect();

        check_tensor(|recorder, a, b|
        {
            let (rows, columns) = recorder.tensor_shape(a.as_value());

            let i = recorder.new_tensor_no_gradient(rows, columns).as_value();
            recorder.name_tensor(i, "i");

            let zeros = recorder.set_new_tensor_gradientable(zeros.clone());
            recorder.name_diff_tensor(zeros, "zeros");

            recorder.store_tensor_until_end(zeros.as_value());

            let bc_selector = recorder.phi_other_selector(zeros);

            let loop_index = recorder.begin_loop(vec![i.into()]);

            let bc = {
                let is = recorder.sum_tensor(DiffTensorPtr::no_gradient(i));
                recorder.name_diff_scalar(is, "is");

                let bc_selected = recorder.select_tensor(bc_selector);

                let bc_added = recorder.add(bc_selected, b);
                recorder.name_diff_tensor(bc_added, "bc_added");

                let bis = recorder.mul_scalar(bc_added, is);
                recorder.name_diff_tensor(bis, "bis");

                recorder.add(bis, b)
            };

            recorder.name_diff_tensor(bc, "bc");

            recorder.set_phi_other_selector(bc_selector, bc);

            recorder.end_loop(loop_index);

            recorder.set_loop_times(loop_index, loops_count);
            recorder.set_loop_inputs(loop_index, is.iter().cloned().collect::<Vec<_>>());

            recorder.add(bc, a)
        })
    }

    #[test]
    fn simple_sum_softmax_state()
    {
        let loops_count = 3;

        let output_size = LAYER_CURR;

        let targets: Vec<OneHotLayer> = (0..loops_count).map(|_| OneHotLayer::new([fastrand::usize(0..output_size)], output_size)).collect();

        check_tensor_with_dims((1, LAYER_CURR), (1, LAYER_CURR), |recorder, a, b|
        {
            let a_sum = recorder.sum_tensor(a);
            recorder.name_diff_scalar(a_sum, "a_sum");

            let t = recorder.new_one_hot();
            recorder.name_one_hot(t, "t");

            let final_state_selector = recorder.phi_other_selector(a_sum);

            let loop_index = recorder.begin_loop(vec![t.into()]);

            let (at, a_loss) = recorder.softmax_cross_entropy(a, t);
            recorder.name_diff_tensor(at, "at");
            recorder.name_diff_scalar(a_loss, "a_loss");

            let at_sum = recorder.sum_tensor(at);
            let ata = recorder.add_scalars(at_sum, a_loss);

            let final_selected = recorder.select_value(final_state_selector);
            recorder.name_diff_scalar(final_selected, "final_selected");

            let final_state = recorder.add_scalars(final_selected, ata);
            recorder.name_diff_scalar(final_state, "final_state");

            recorder.set_phi_other_selector(final_state_selector, final_state);

            recorder.end_loop(loop_index);

            recorder.set_loop_times(loop_index, loops_count);
            recorder.set_loop_inputs(loop_index, targets.iter().cloned().map(OwnedInputType::OneHot).collect::<Vec<_>>());

            recorder.add_scalar(b, final_state)
        })
    }

    #[test]
    fn simple_sum_multi_state()
    {
        let loops_count = 3;
        let is: Vec<OwnedInputType> = (0..loops_count).map(|_| LayerType::new_with(LAYER_CURR, LAYER_PREV, random_value).into()).collect();
        let ms: Vec<OwnedInputType> = (0..loops_count).map(|_| LayerType::new_with(LAYER_CURR, LAYER_PREV, random_value).into()).collect();

        check_tensor(|recorder, a, b|
        {
            let a_sum = recorder.sum_tensor(a);
            recorder.name_diff_scalar(a_sum, "a_sum");

            let (rows, columns) = recorder.tensor_shape(a.as_value());

            let i = recorder.new_tensor_no_gradient(rows, columns).as_value();
            recorder.name_tensor(i, "i");

            let m = recorder.new_tensor_no_gradient(rows, columns).as_value();
            recorder.name_tensor(m, "m");

            let final_state_selector = recorder.phi_other_selector(a_sum);

            let loop_index = recorder.begin_loop(vec![m.into(), i.into()]);

            let ai = recorder.mul_componentwise(a, DiffTensorPtr::no_gradient(i));
            recorder.name_diff_tensor(ai, "ai");

            let aim = recorder.mul_componentwise(ai, DiffTensorPtr::no_gradient(m));
            recorder.name_diff_tensor(aim, "aim");

            let aim_sum = recorder.sum_tensor(aim);
            recorder.name_diff_scalar(aim_sum, "aim_sum");

            let final_selected = recorder.select_value(final_state_selector);
            recorder.name_diff_scalar(final_selected, "final_selected");

            let final_state = recorder.add_scalars(final_selected, aim_sum);
            recorder.name_diff_scalar(final_state, "final_state");

            recorder.set_phi_other_selector(final_state_selector, final_state);

            recorder.end_loop(loop_index);

            recorder.set_loop_times(loop_index, loops_count);

            let inputs: Vec<_> = is.iter().cloned().zip(ms.iter().cloned())
                .flat_map(|(i, m)| [i, m])
                .collect();

            recorder.set_loop_inputs(loop_index, inputs);

            recorder.add_scalar(b, final_state)
        })
    }

    #[test]
    fn sum_state_more()
    {
        let loops_count = 3;
        let is: Vec<OwnedInputType> = (0..loops_count).map(|_| LayerType::new_with(LAYER_CURR, LAYER_PREV, random_value).into()).collect();

        check_tensor(|recorder, a, b|
        {
            let a_sum = recorder.sum_tensor(a);
            recorder.name_diff_scalar(a_sum, "a_sum");

            let (rows, columns) = recorder.tensor_shape(a.as_value());

            let i = recorder.new_tensor_no_gradient(rows, columns).as_value();
            recorder.name_tensor(i, "i");

            let final_state_selector = recorder.phi_other_selector(a_sum);

            let loop_index = recorder.begin_loop(vec![i.into()]);

            let ai = recorder.mul_componentwise(a, DiffTensorPtr::no_gradient(i));
            recorder.name_diff_tensor(ai, "ai");

            let ai_sum = recorder.sum_tensor(ai);
            recorder.name_diff_scalar(ai_sum, "ai_sum");

            let final_selected = recorder.select_value(final_state_selector);
            recorder.name_diff_scalar(final_selected, "final_selected");

            let intermediate = recorder.mul_scalar(b, final_selected);
            recorder.name_diff_tensor(intermediate, "intermediate");

            let intermediate_sum = recorder.sum_tensor(intermediate);
            recorder.name_diff_scalar(intermediate_sum, "intermediate_sum");

            let final_state = recorder.add_scalars(intermediate_sum, ai_sum);
            recorder.name_diff_scalar(final_state, "final_state");

            recorder.set_phi_other_selector(final_state_selector, final_state);

            recorder.end_loop(loop_index);

            recorder.set_loop_times(loop_index, loops_count);
            recorder.set_loop_inputs(loop_index, is.iter().cloned().collect::<Vec<_>>());

            recorder.add_scalar(b, final_state)
        })
    }

    #[test]
    fn stateful_loop()
    {
        let loops_count = 3;
        let is: Vec<OwnedInputType> = (0..loops_count + 2).map(|_| LayerType::new_with(LAYER_CURR, LAYER_PREV, random_value).into()).collect();

        check_tensor(|recorder, a, b|
        {
            let s0 = {
                let s0i = recorder.set_new_tensor(is[0].clone().into_normal()).as_value();
                recorder.name_tensor(s0i, "s0i");

                recorder.store_tensor_until_end(s0i);

                recorder.mul_componentwise(a, DiffTensorPtr::no_gradient(s0i))
            };

            recorder.name_diff_tensor(s0, "s0");

            let ss0 = recorder.sum_tensor(s0);
            recorder.name_diff_scalar(ss0, "ss0");

            let s1 = {
                let s1i = recorder.set_new_tensor(is[1].clone().into_normal()).as_value();
                recorder.name_tensor(s1i, "s1i");

                recorder.store_tensor_until_end(s1i);

                let r = recorder.mul_componentwise(a, DiffTensorPtr::no_gradient(s1i));
                recorder.name_diff_tensor(r, "r");

                recorder.mul_componentwise(r, s0)
            };

            recorder.name_diff_tensor(s1, "s1");

            let ss1 = recorder.sum_tensor(s1);
            recorder.name_diff_scalar(ss1, "ss1");

            let combined_state = recorder.add_scalars(ss0, ss1);

            recorder.name_diff_scalar(combined_state, "combined_state");

            let (rows, columns) = recorder.tensor_shape(a.as_value());

            let i = recorder.new_tensor_no_gradient(rows, columns).as_value();
            recorder.name_tensor(i, "i");

            let final_state_selector = recorder.phi_other_selector(combined_state);

            let loop_index = recorder.begin_loop(vec![i.into()]);

            let ai = recorder.mul_componentwise(a, DiffTensorPtr::no_gradient(i));
            recorder.name_diff_tensor(ai, "ai");

            let s2 = recorder.mul_componentwise(ai, s1);
            recorder.name_diff_tensor(s2, "s2");

            let ss2 = recorder.sum_tensor(s2);
            recorder.name_diff_scalar(ss2, "ss2");

            let final_combined_selected = recorder.select_value(final_state_selector);
            recorder.name_diff_scalar(final_combined_selected, "final_combined_selected");

            let final_combined_state = recorder.add_scalars(final_combined_selected, ss2);
            recorder.name_diff_scalar(final_combined_state, "final_combined_state");

            recorder.set_phi_other_selector(final_state_selector, final_combined_state);

            recorder.end_loop(loop_index);

            recorder.set_loop_times(loop_index, loops_count);
            recorder.set_loop_inputs(loop_index, is.iter().cloned().skip(2).collect::<Vec<_>>());

            recorder.add_scalar(b, final_combined_state)
        })
    }

    #[test]
    fn stateful_whatever_loop()
    {
        let loops_count = 3;

        let input_size = LAYER_PREV;
        let hidden_size = LAYER_CURR;
        let output_size = 2;

        let is: Vec<OwnedInputType> = (0..loops_count + 2).map(|_| LayerType::new_with(input_size, 1, fastrand::f32).into()).collect();

        check_tensor_with_dims((input_size, hidden_size), (hidden_size, output_size), |recorder, a, b|
        {
            let mut create_input = |input: OwnedInputType| -> TensorPtr
            {
                let input = recorder.set_new_tensor(input.into_normal());
                recorder.name_diff_tensor(input, "input");

                recorder.store_tensor_until_end(input.as_value());

                input.as_value()
            };

            let i0 = create_input(is[0].clone());
            let i1 = create_input(is[1].clone());

            let do_one = |
                recorder: &mut OperationsRecorder,
                state: Option<DiffTensorPtr>,
                input: TensorPtr
            | -> DiffTensorPtr
            {
                let mut gate = recorder.matmulv(a, DiffTensorPtr::no_gradient(input));
                recorder.name_diff_tensor(gate, "gate");

                if let Some(state) = state
                {
                    gate = recorder.add(gate, state);
                }

                gate
            };

            let s0 = do_one(recorder, None, i0);
            recorder.name_diff_tensor(s0, "s0");

            let s1 = do_one(recorder, Some(s0), i1);
            recorder.name_diff_tensor(s1, "s1");

            let i = recorder.new_tensor_no_gradient(input_size, 1).as_value();
            recorder.name_tensor(i, "loop_i");

            let target = recorder.new_one_hot();
            recorder.name_one_hot(target, "loop_target");

            let previous_state_selector = recorder.phi_other_selector(s1);

            let loop_index = recorder.begin_loop(vec![i.into()]);

            let previous_state_selected = recorder.select_tensor(previous_state_selector);

            let s2 = do_one(recorder, Some(previous_state_selected), i);
            recorder.name_diff_tensor(s2, "s2");

            recorder.set_phi_other_selector(previous_state_selector, s2);

            recorder.end_loop(loop_index);

            recorder.set_loop_times(loop_index, loops_count);

            let inputs_targets: Vec<_> = is.iter().cloned().skip(2)
                .collect();

            recorder.set_loop_inputs(loop_index, inputs_targets);

            let f_sum = recorder.sum_tensor(s2);
            recorder.add_scalar(b, f_sum)
        })
    }

    #[test]
    fn stateful_no_b_loop()
    {
        let loops_count = 3;

        let input_size = LAYER_PREV;
        let hidden_size = LAYER_CURR;
        let output_size = 2;

        let is: Vec<OwnedInputType> = (0..loops_count + 2).map(|_| LayerType::new_with(input_size, 1, fastrand::f32).into()).collect();
        let targets: Vec<OneHotLayer> = (0..loops_count + 2).map(|_| OneHotLayer::new([fastrand::usize(0..output_size)], output_size)).collect();

        check_tensor_with_dims((input_size, hidden_size), (hidden_size, output_size), |recorder, a, b|
        {
            let mut create_input = |input: OwnedInputType| -> TensorPtr
            {
                let input = recorder.set_new_tensor(input.into_normal());
                recorder.name_diff_tensor(input, "input");

                recorder.store_tensor_until_end(input.as_value());

                input.as_value()
            };

            let i0 = create_input(is[0].clone());
            let i1 = create_input(is[1].clone());

            let mut create_target = |targets_values: OneHotLayer| -> OneHotIndex
            {
                let targets = recorder.new_one_hot();
                recorder.name_one_hot(targets, "targets");

                recorder.set_one_hot(targets, targets_values);

                targets
            };

            let t0 = create_target(targets[0].clone());
            let t1 = create_target(targets[1].clone());

            let do_one = |
                recorder: &mut OperationsRecorder,
                state: Option<DiffTensorPtr>,
                input: TensorPtr,
                targets: OneHotIndex
            | -> (DiffTensorPtr, DiffScalar)
            {
                let mut gate = recorder.matmulv(a, DiffTensorPtr::no_gradient(input));
                recorder.name_diff_tensor(gate, "gate");

                if let Some(state) = state
                {
                    gate = recorder.add(gate, state);
                }

                (gate, recorder.softmax_cross_entropy(gate, targets).1)
            };

            let (s0, s0_loss) = do_one(recorder, None, i0, t0);
            recorder.name_diff_tensor(s0, "s0");
            recorder.name_diff_scalar(s0_loss, "s0_loss");

            let (s1, s1_loss) = do_one(recorder, Some(s0), i1, t1);
            recorder.name_diff_tensor(s1, "s1");
            recorder.name_diff_scalar(s1_loss, "s1_loss");

            let compound_loss = recorder.add_scalars(s0_loss, s1_loss);
            recorder.name_diff_scalar(compound_loss, "compound_loss");

            let i = recorder.new_tensor_no_gradient(input_size, 1).as_value();
            recorder.name_tensor(i, "loop_i");

            let target = recorder.new_one_hot();
            recorder.name_one_hot(target, "loop_target");

            let final_state_selector = recorder.phi_other_selector(compound_loss);
            let previous_state_selector = recorder.phi_other_selector(s1);

            let loop_index = recorder.begin_loop(vec![i.into(), target.into()]);

            let previous_state_selected = recorder.select_tensor(previous_state_selector);

            let (s2, s2_loss) = do_one(recorder, Some(previous_state_selected), i, target);
            recorder.name_diff_tensor(s2, "s2");
            recorder.name_diff_scalar(s2_loss, "s2_loss");

            recorder.set_phi_other_selector(previous_state_selector, s2);

            let final_combined_selected = recorder.select_value(final_state_selector);
            recorder.name_diff_scalar(final_combined_selected, "final_combined_selected");

            let final_combined_state = recorder.add_scalars(final_combined_selected, s2_loss);
            recorder.name_diff_scalar(final_combined_state, "final_combined_state");

            recorder.set_phi_other_selector(final_state_selector, final_combined_state);

            recorder.end_loop(loop_index);

            recorder.set_loop_times(loop_index, loops_count);

            let inputs_targets: Vec<_> = is.iter().cloned().zip(targets.iter().cloned()).skip(2)
                .flat_map(|(i, t)| [i, OwnedInputType::OneHot(t)])
                .collect();

            recorder.set_loop_inputs(loop_index, inputs_targets);

            recorder.add_scalar(b, final_combined_state)
        })
    }

    #[test]
    fn stateful_pls_loop()
    {
        let loops_count = 3;

        let input_size = LAYER_PREV;
        let hidden_size = LAYER_CURR;
        let output_size = 2;

        let zeros = LayerType::new(hidden_size, output_size);
        let is: Vec<OwnedInputType> = (0..loops_count + 2).map(|_| LayerType::new_with(input_size, 1, fastrand::f32).into()).collect();
        let targets: Vec<OneHotLayer> = (0..loops_count + 2).map(|_| OneHotLayer::new([fastrand::usize(0..output_size)], output_size)).collect();

        check_tensor_with_dims((input_size, hidden_size), (hidden_size, output_size), |recorder, a, b|
        {
            let mut create_input = |input: OwnedInputType| -> TensorPtr
            {
                let input = recorder.set_new_tensor(input.into_normal());
                recorder.name_diff_tensor(input, "input");

                recorder.store_tensor_until_end(input.as_value());

                input.as_value()
            };

            let i0 = create_input(is[0].clone());
            let i1 = create_input(is[1].clone());

            let mut create_target = |targets_values: OneHotLayer| -> OneHotIndex
            {
                let targets = recorder.new_one_hot();
                recorder.name_one_hot(targets, "targets");

                recorder.set_one_hot(targets, targets_values);

                targets
            };

            let t0 = create_target(targets[0].clone());
            let t1 = create_target(targets[1].clone());

            let do_one = |
                recorder: &mut OperationsRecorder,
                state: Option<DiffTensorPtr>,
                input: TensorPtr,
                targets: OneHotIndex
            | -> (DiffTensorPtr, DiffScalar)
            {
                let mut gate = recorder.matmulv(a, DiffTensorPtr::no_gradient(input));
                recorder.name_diff_tensor(gate, "gate");

                if let Some(state) = state
                {
                    gate = recorder.add(gate, state);
                }

                (gate, recorder.softmax_cross_entropy(gate, targets).1)
            };

            let (s0, s0_loss) = do_one(recorder, None, i0, t0);
            recorder.name_diff_tensor(s0, "s0");
            recorder.name_diff_scalar(s0_loss, "s0_loss");

            let (s1, s1_loss) = do_one(recorder, Some(s0), i1, t1);
            recorder.name_diff_tensor(s1, "s1");
            recorder.name_diff_scalar(s1_loss, "s1_loss");

            let compound_loss = recorder.add_scalars(s0_loss, s1_loss);
            recorder.name_diff_scalar(compound_loss, "compound_loss");

            let i = recorder.new_tensor_no_gradient(input_size, 1).as_value();
            recorder.name_tensor(i, "loop_i");

            let target = recorder.new_one_hot();
            recorder.name_one_hot(target, "loop_target");

            let final_state_selector = recorder.phi_other_selector(compound_loss);
            let previous_state_selector = recorder.phi_other_selector(s1);

            let zeros = recorder.set_new_tensor_gradientable(zeros.clone());
            recorder.name_diff_tensor(zeros, "zeros");

            recorder.store_tensor_until_end(zeros.as_value());

            let bc_selector = recorder.phi_other_selector(zeros);

            let loop_index = recorder.begin_loop(vec![i.into(), target.into()]);

            let bc = {
                let is = recorder.sum_tensor(DiffTensorPtr::no_gradient(i));
                recorder.name_diff_scalar(is, "is");

                let bc_selected = recorder.select_tensor(bc_selector);
                recorder.name_diff_tensor(bc_selected, "bc_selected");

                let bc_added = recorder.add(bc_selected, b);
                recorder.name_diff_tensor(bc_added, "bc_added");

                let bis = recorder.mul_scalar(bc_added, is);
                recorder.name_diff_tensor(bis, "bis");

                recorder.add(bis, b)
            };

            recorder.name_diff_tensor(bc, "bc");

            recorder.set_phi_other_selector(bc_selector, bc);

            let previous_state_selected = recorder.select_tensor(previous_state_selector);

            let (s2, s2_loss) = do_one(recorder, Some(previous_state_selected), i, target);
            recorder.name_diff_tensor(s2, "s2");
            recorder.name_diff_scalar(s2_loss, "s2_loss");

            recorder.set_phi_other_selector(previous_state_selector, s2);

            let final_combined_selected = recorder.select_value(final_state_selector);
            recorder.name_diff_scalar(final_combined_selected, "final_combined_selected");

            let final_combined_state = recorder.add_scalars(final_combined_selected, s2_loss);
            recorder.name_diff_scalar(final_combined_state, "final_combined_state");

            recorder.set_phi_other_selector(final_state_selector, final_combined_state);

            recorder.end_loop(loop_index);

            recorder.set_loop_times(loop_index, loops_count);

            let inputs_targets: Vec<_> = is.iter().cloned().zip(targets.iter().cloned()).skip(2)
                .flat_map(|(i, t)| [i, OwnedInputType::OneHot(t)])
                .collect();

            recorder.set_loop_inputs(loop_index, inputs_targets);

            recorder.add_scalar(bc, final_combined_state)
        })
    }

    #[test]
    fn stateful_more_loop()
    {
        let loops_count = 3;

        let input_size = LAYER_PREV;
        let hidden_size = LAYER_CURR;
        let output_size = 2;

        let zeros = LayerType::new(output_size, 1);
        let is: Vec<OwnedInputType> = (0..loops_count + 2).map(|_| LayerType::new_with(input_size, 1, fastrand::f32).into()).collect();
        let targets: Vec<OneHotLayer> = (0..loops_count + 2).map(|_| OneHotLayer::new([fastrand::usize(0..output_size)], output_size)).collect();

        check_tensor_with_dims((input_size, hidden_size), (hidden_size, output_size), |recorder, a, b|
        {
            let mut create_input = |input: OwnedInputType| -> TensorPtr
            {
                let input = recorder.set_new_tensor(input.into_normal());
                recorder.name_diff_tensor(input, "input");

                recorder.store_tensor_until_end(input.as_value());

                input.as_value()
            };

            let i0 = create_input(is[0].clone());
            let i1 = create_input(is[1].clone());

            let mut create_target = |targets_values: OneHotLayer| -> OneHotIndex
            {
                let targets = recorder.new_one_hot();
                recorder.name_one_hot(targets, "targets");

                recorder.set_one_hot(targets, targets_values);

                targets
            };

            let t0 = create_target(targets[0].clone());
            let t1 = create_target(targets[1].clone());

            let do_one = |
                recorder: &mut OperationsRecorder,
                state: Option<DiffTensorPtr>,
                input: TensorPtr,
                targets: OneHotIndex
            | -> (DiffTensorPtr, DiffScalar)
            {
                let mut gate = recorder.matmulv(a, DiffTensorPtr::no_gradient(input));
                recorder.name_diff_tensor(gate, "gate");

                if let Some(state) = state
                {
                    let mul_value = recorder.set_new_value(0.02);
                    recorder.name_diff_scalar(mul_value, "mul_value");

                    let state_new = recorder.mul_scalar(state, mul_value);

                    let mm = recorder.matmulv(b, state_new);
                    recorder.name_diff_tensor(mm, "mm");

                    gate = recorder.add(gate, mm);
                }

                (gate, recorder.softmax_cross_entropy(gate, targets).1)
            };

            let (s0, s0_loss) = do_one(recorder, None, i0, t0);
            recorder.name_diff_tensor(s0, "s0");
            recorder.name_diff_scalar(s0_loss, "s0_loss");

            let (s1, s1_loss) = do_one(recorder, Some(s0), i1, t1);
            recorder.name_diff_tensor(s1, "s1");
            recorder.name_diff_scalar(s1_loss, "s1_loss");

            let compound_loss = recorder.add_scalars(s0_loss, s1_loss);
            recorder.name_diff_scalar(compound_loss, "compound_loss");

            let i = recorder.new_tensor_no_gradient(input_size, 1).as_value();
            recorder.name_tensor(i, "loop_i");

            let target = recorder.new_one_hot();
            recorder.name_one_hot(target, "loop_target");

            let final_state_selector = recorder.phi_other_selector(compound_loss);
            let previous_state_selector = recorder.phi_other_selector(s1);

            let loop_index = recorder.begin_loop(vec![i.into(), target.into()]);

            let previous_state_selected = recorder.select_tensor(previous_state_selector);

            let (s2, s2_loss) = do_one(recorder, Some(previous_state_selected), i, target);
            recorder.name_diff_tensor(s2, "s2");
            recorder.name_diff_scalar(s2_loss, "s2_loss");

            recorder.set_phi_other_selector(previous_state_selector, s2);

            let final_combined_selected = recorder.select_value(final_state_selector);
            recorder.name_diff_scalar(final_combined_selected, "final_combined_selected");

            let final_combined_state = recorder.add_scalars(final_combined_selected, s2_loss);
            recorder.name_diff_scalar(final_combined_state, "final_combined_state");

            recorder.set_phi_other_selector(final_state_selector, final_combined_state);

            recorder.end_loop(loop_index);

            recorder.set_loop_times(loop_index, loops_count);

            let inputs_targets: Vec<_> = is.iter().cloned().zip(targets.iter().cloned()).skip(2)
                .flat_map(|(i, t)| [i, OwnedInputType::OneHot(t)])
                .collect();

            recorder.set_loop_inputs(loop_index, inputs_targets);

            let zeros = recorder.set_new_tensor(zeros.clone());
            recorder.name_diff_tensor(zeros, "zeros");

            recorder.store_tensor_until_end(zeros.as_value());

            recorder.add_scalar(zeros, final_combined_state)
        })
    }
}
