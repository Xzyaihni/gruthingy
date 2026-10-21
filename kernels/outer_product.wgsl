@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

var<immediate> batch_size: u32;

#include "default_workgroup_size.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let output_length = arrayLength(&output);
    if index >= output_length
    {
        return;
    }

    let lhs_length = arrayLength(&lhs) / batch_size;
    let rhs_length = arrayLength(&rhs) / batch_size;

    let row = index % lhs_length;
    let column = index / lhs_length;

    let batch_index = column / rhs_length;

    output[index] = lhs[batch_index * lhs_length + row] * rhs[column];
}
