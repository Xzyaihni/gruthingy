@group(0) @binding(0)
var<storage, read> targets: array<u32>;

@group(0) @binding(1)
var<storage, read_write> sum_values: array<f32>;

@group(0) @binding(2)
var<storage, read_write> softmaxed_values: array<f32>;

@group(0) @binding(3)
var<storage, read_write> entropies: array<f32>;

const WORKGROUP_SIZE = 64;

#include "div_ceil.wgsl"

#include "tensor_sum.wgsl"

fn target_of_batch(batch_index: u32) -> u32
{
    return targets[batch_index];
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    let batch_size = arrayLength(&entropies);
    let values_length = arrayLength(&softmaxed_values) / batch_size;

    // all threads should quit in here except thread 0 of each workgroup
    #include "softmax_common.wgsl"

    let entropy = log(s) - z;

    entropies[batch_index] = entropy;
}
