@group(0) @binding(0)
var<uniform> batch_size: u32;

@group(0) @binding(1)
var<storage, read> lhs: array<f32>;

@group(0) @binding(2)
var<storage, read> rhs: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

#include "default_workgroup_size.wgsl"

var<workgroup> workgroup_chunk: array<f32, WORKGROUP_SIZE>;

#include "div_ceil.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    const is_lhs_single_batch = false;
    const is_column_major = false;

    #include "matmulv_common_shared.wgsl"

    if local_invocation_index == 0
    {
        output[workgroup_id.x] = total;
    }
}
