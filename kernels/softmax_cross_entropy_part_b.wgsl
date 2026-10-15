@group(0) @binding(0)
var<storage, read> sum_values: array<f32>;

@group(0) @binding(1)
var<storage, read> z: array<f32>;

@group(0) @binding(2)
var<storage, read_write> softmaxed_values: array<f32>;

@group(0) @binding(3)
var<storage, read_write> entropies: array<f32>;

const WORKGROUP_SIZE = 64;

#include "div_ceil.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    let batch_size = arrayLength(&sum_values);
    let values_length = arrayLength(&softmaxed_values) / batch_size;

    let batch_index = workgroup_id.x;
    let batch_start = batch_index * values_length;

    #include "softmax_common_b.wgsl"

    let entropy = log(sum_values[batch_index]) - z[batch_index];

    entropies[batch_index] = entropy;
}
