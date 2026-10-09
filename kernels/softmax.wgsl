@group(0) @binding(0)
var<storage, read_write> sum_values: array<f32>;

@group(0) @binding(1)
var<storage, read_write> softmaxed_values: array<f32>;

const WORKGROUP_SIZE = 64;

var<immediate> batch_size: u32;

#include "div_ceil.wgsl"

#include "tensor_sum.wgsl"

fn target_of_batch(batch_index: u32) -> u32
{
    // u32::MAX cuz this one has no target
    return 4294967295;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    let values_length = arrayLength(&softmaxed_values) / batch_size;

    #include "softmax_common.wgsl"
}
