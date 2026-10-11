@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

var<immediate> one_hot_count: u32;

const WORKGROUP_SIZE = 64;

#include "div_ceil.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    let is_column_major = true;

    let lhs_length = arrayLength(&lhs);
    let output_length = arrayLength(&output);

    let batch_size = arrayLength(&rhs) / one_hot_count;

    let lhs_rows = output_length / batch_size;
    let lhs_single_length = lhs_length / batch_size;
    let lhs_columns = lhs_single_length / lhs_rows;

    let row_stride = select(lhs_columns, 1, is_column_major);
    let column_stride = select(1, lhs_rows, is_column_major);

    let row = workgroup_id.x % lhs_rows;
    let batch_index = workgroup_id.x / lhs_rows;

    let batch_start = batch_index * lhs_single_length;

    for (var o = 0u; o < one_hot_count; o++)
    {
        let one_hot_index = rhs[batch_index * one_hot_count + o];

        output[workgroup_id.x] += lhs[batch_start + row * row_stride + one_hot_index * column_stride];
    }
}
