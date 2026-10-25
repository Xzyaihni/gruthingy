use std::{
    fmt::{self, Debug},
    ops::RangeInclusive
};

use serde::Serialize;

use super::{
    set_inside_loop_context,
    VariableNames,
    OperationsRecorder,
    StandardGradientOp,
    NotationGradientOp,
    GradientOp,
    ValueIndex,
    JumpInfo,
    LoopIndex,
    LoopInfo,
    TensorPtr,
    DiffValue,
    InputTypePtr
};


#[derive(Default, Clone, PartialEq)]
pub struct LiveRange
{
    pub start: Option<i32>,
    pub end: Option<i32>
}

impl Debug for LiveRange
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let start = self.start.map(|x| if x == -1 { "start".to_owned() } else { x.to_string() }).unwrap_or_else(|| "?".to_owned());
        let end = self.end.map(|x| if x == i32::MAX { "end".to_owned() } else { x.to_string() }).unwrap_or_else(|| "?".to_owned());

        write!(f, "{start}..={end}")
    }
}

impl LiveRange
{
    pub fn valid_range(&self) -> bool
    {
        let start = if let Some(x) = self.start { x } else { return false };

        let end = if let Some(x) = self.end { x } else { return false };

        start < end
    }

    pub fn overlaps(&self, other: &Self) -> bool
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

fn live_ranges_non_overlapping(
    live_ranges: impl Iterator<Item=LiveRange>,
    allow_same_start_end: bool
) -> bool
{
    let mut live_ranges: Vec<_> = dbg!(live_ranges.collect());
    live_ranges.sort_by_key(|x| x.start);

    live_ranges.iter().zip(live_ranges.iter().skip(1)).all(|(this, next)|
    {
        if allow_same_start_end { this.end <= next.start } else { this.end < next.start }
    })
}

#[derive(Debug, Default, Clone)]
pub struct LiveRanges
{
    pub loops: Vec<LiveRange>,
    pub values: Vec<LiveRange>,
    pub tensors: Vec<LiveRange>
}

impl LiveRanges
{
    pub fn tensors_non_overlapping(
        &self,
        tensors: &[TensorPtr],
        allow_same_start_end: bool
    ) -> bool
    {
        live_ranges_non_overlapping(tensors.iter().map(|tensor_ptr| self.tensors[tensor_ptr.0].clone()), allow_same_start_end)
    }

    pub fn calculate(
        &mut self,
        gradient_operations: &[StandardGradientOp],
        loop_infos: &[LoopInfo],
        variable_names: &VariableNames
    )
    {
        self.calculate_loop_live_ranges(gradient_operations);

        let mut inside_loop_values = Vec::new();

        {
            let mut inside_loop: Option<LoopIndex> = None;

            gradient_operations.iter().for_each(|op|
            {
                let mut set_loop_lifetime = |inside_loop: Option<LoopIndex>, live_range: &mut LiveRange, value: DiffValue|
                {
                    if let Some(loop_index) = inside_loop
                    {
                        let loop_range = self.loops[loop_index.0].clone();

                        let loop_start = loop_range.start.unwrap();
                        let loop_end = loop_range.end.unwrap();

                        live_range.start = Some(live_range.start.map(|x| x.min(loop_start)).unwrap_or(loop_start));
                        live_range.end = Some(live_range.end.map(|x| x.max(loop_end)).unwrap_or(loop_end));

                        if !inside_loop_values.contains(&value)
                        {
                            inside_loop_values.push(value);
                        }
                    }
                };

                set_inside_loop_context(&mut inside_loop, op);

                match op
                {
                    GradientOp::GetOtherSelectorValue{..}
                    | GradientOp::GetOtherSelectorTensor{..} =>
                    {
                        let (live_range, value): (_, DiffValue) = match op
                        {
                            GradientOp::GetOtherSelectorValue{other, ..} => (&mut self.values[other.0], (*other).into()),
                            GradientOp::GetOtherSelectorTensor{other, ..} => (&mut self.tensors[other.0], (*other).into()),
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
                            GradientOp::OtherSelectorValueGradient{other, ..} => (&mut self.values[other.0], (*other).into()),
                            GradientOp::OtherSelectorTensorGradient{other, ..}
                            | GradientOp::IfNotSetTensor{dst: other, ..} => (&mut self.tensors[other.0], (*other).into()),
                            _ => unreachable!()
                        };

                        set_loop_lifetime(inside_loop, live_range, value);
                    },
                    _ => ()
                }
            });
        }

        gradient_operations.iter().enumerate().for_each(|(op_index, op)|
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
                    if *start == Some(op_index as i32)
                    {
                        return;
                    }

                    debug_assert!(start.is_none(), "{err_name} was defined at operation {} and {op_index}", start.unwrap());

                    *start = Some(new_start);
                }
            };

            let is_allow_reuse = |value: DiffValue| -> bool
            {
                inside_loop_values.contains(&value)
            };

            let mut handle_output = |output: DiffValue|
            {
                let allow_reuse = is_allow_reuse(output);
                let name = variable_names.format_variable(output);

                match output
                {
                    DiffValue::Tensor(tensor_ptr) => handle_output(&mut self.tensors[tensor_ptr.0], allow_reuse, name),
                    DiffValue::Value(value_index) => handle_output(&mut self.values[value_index.0], allow_reuse, name),
                    DiffValue::OneHot(_) => unimplemented!()
                }
            };

            if let GradientOp::OtherSelectorValueGradient{first,  ..} = op
            {
                handle_output((*first).into());
            } else
            {
                op.for_outputs(handle_output);
            }
        });

        gradient_operations.iter().enumerate().rev().for_each(|(op_index, op)|
        {
            let allow_end_before = op.can_use_future();

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
                let name = variable_names.format_variable(arg);

                let live_range = match arg
                {
                    DiffValue::Tensor(tensor_ptr) => &mut self.tensors[tensor_ptr.0],
                    DiffValue::Value(value_index) => &mut self.values[value_index.0],
                    DiffValue::OneHot(_) => return
                };

                handle_arg(live_range, name);
            });
        });

        gradient_operations.iter().for_each(|op|
        {
            match op
            {
                GradientOp::Jump(JumpInfo::JumpTo{index, inputs}) =>
                {
                    inputs.iter().for_each(|input|
                    {
                        if let InputTypePtr::Normal(tensor) = input
                        {
                            let live_range = &mut self.tensors[tensor.0];

                            live_range.start = None;

                            if live_range.end.is_some()
                            {
                                live_range.start = self.loops[index.0].start;
                            }
                        }
                    });
                },
                _ => ()
            }
        });

        self.loops.iter().zip(loop_infos).for_each(|(loop_live_range, loop_info)|
        {
            loop_info.kept_inside.iter().for_each(|kept_inside|
            {
                let live_range = match kept_inside
                {
                    DiffValue::Tensor(tensor) => &mut self.tensors[tensor.0],
                    DiffValue::Value(value) => &mut self.values[value.0],
                    DiffValue::OneHot(_) => return
                };

                let loop_start = loop_live_range.start.unwrap();
                let loop_end = loop_live_range.end.unwrap();

                live_range.start = Some(live_range.start.map(|x| x.min(loop_start)).unwrap_or(loop_start));
                live_range.end = live_range.end.map(|x| x.max(loop_end));
            });
        });
    }

    pub fn remove_unused_ops(&self, gradient_operations: &mut [StandardGradientOp]) -> bool
    {
        let mut any_unused = false;
        gradient_operations.iter_mut().for_each(|op|
        {
            let mut any_unused_single: bool = false;
            let mut all_unused: Option<bool> = None;

            let mut handle_output = |live_range: &LiveRange, err_name: String|
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
                    DiffValue::Tensor(tensor_ptr) => handle_output(&self.tensors[tensor_ptr.0], format!("{tensor_ptr:?}")),
                    DiffValue::Value(value_index) => handle_output(&self.values[value_index.0], format!("{value_index:?}")),
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
                let _is_value_unused = |value_index: ValueIndex| -> bool
                {
                    self.values[value_index.0].end.is_none()
                };

                let is_tensor_unused = |tensor_ptr: TensorPtr| -> bool
                {
                    self.tensors[tensor_ptr.0].end.is_none()
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
                    GradientOp::SoftmaxCrossEntropy{
                        values,
                        softmaxed_output,
                        output,
                        ..
                    } if is_tensor_unused(output) =>
                    {
                        GradientOp::Softmax{values, softmaxed_output}
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

    fn extend_loops_to(&mut self, index: usize)
    {
        while self.loops.len() <= index
        {
            self.loops.push(LiveRange::default());
        }
    }

    fn calculate_loop_live_ranges(&mut self, gradient_operations: &[StandardGradientOp])
    {
        gradient_operations.iter().enumerate().for_each(|(operation_index, op)|
        {
            match op
            {
                GradientOp::Jump(JumpInfo::JumpTo{index, ..}) =>
                {
                    self.extend_loops_to(index.0);

                    self.loops[index.0].start = Some(operation_index as i32);
                },
                GradientOp::Jump(JumpInfo::JumpFrom(index)) =>
                {
                    self.extend_loops_to(index.0);

                    self.loops[index.0].end = Some(operation_index as i32 - 1);
                },
                _ => ()
            }
        });
    }

    pub fn print_info(&self, recorder: &OperationsRecorder)
    {
        #[derive(Serialize)]
        struct LiveRangeInfos
        {
            operations: Vec<String>,
            variables: Vec<(String, RangeInclusive<i32>)>
        }

        let mut variables = Vec::new();

        self.tensors.iter().enumerate().map(|(tensor_ptr_index, live_range)|
        {
            (recorder.memory.format_variable(TensorPtr(tensor_ptr_index)), live_range.clone())
        }).chain(self.values.iter().enumerate().map(|(value_index, live_range)|
        {
            (recorder.memory.format_variable(ValueIndex(value_index)), live_range.clone())
        })).for_each(|(this_name, live_range)|
        {
            if live_range.start.is_none() && live_range.end.is_some()
            {
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

                panic!("{this_name} is malformed, no start but ends at {}", f_live(live_range.end));
            }

            if !live_range.end.is_none()
            {
                let live_range = live_range.start.unwrap()..=live_range.end.unwrap();

                variables.push((this_name, live_range));
            }
        });

        eprintln!("{}", serde_json::to_string_pretty(&LiveRangeInfos{
            operations: recorder.gradient_operations.iter().map(|op|
            {
                format!("{:?}", NotationGradientOp::from_nameable(&recorder.memory, op.clone()))
            }).collect(),
            variables
        }).unwrap());
    }
}
