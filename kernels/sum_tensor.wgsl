@group(0) @binding(0)
var<storage, read_write> sum_values: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

const WORKGROUP_SIZE = 64;

#include "div_ceil.wgsl"

#include "tensor_sum.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    let batch_size = arrayLength(&dst);
    let src_length = arrayLength(&sum_values);
    let src_single_length = src_length / batch_size;

    let batch_index = workgroup_id.x;

    tensor_sum(batch_index, local_invocation_index, src_single_length);

    dst[batch_index] = sum_values[batch_index * src_single_length];
}
