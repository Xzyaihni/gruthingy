@group(0) @binding(0)
var<uniform> batch_size: u32;

@group(0) @binding(1)
var<storage, read> lhs: array<f32>;

@group(0) @binding(2)
var<storage, read> rhs: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

#include "default_workgroup_size.wgsl"

#include "div_ceil.wgsl"

fn map_output(lhs_rows: u32, index: u32, value: f32) -> f32
{
    return value;
}

fn map_output_first(lhs_rows: u32, index: u32, value: f32) -> f32
{
    return map_output(lhs_rows, index, value);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    const is_lhs_single_batch = false;

    #include "matmul_onehotv_common.wgsl"
}
