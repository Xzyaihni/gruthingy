@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

var<immediate> one_hot_count: u32;

const WORKGROUP_SIZE = 64;

#include "div_ceil.wgsl"

fn map_output(index: u32, value: f32) -> f32
{
    return value + output[index];
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    #include "matmul_onehotv_common.wgsl"
}
