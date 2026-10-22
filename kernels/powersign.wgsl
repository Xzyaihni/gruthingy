@group(0) @binding(0)
var<storage, read> gradient: array<f32>;

@group(0) @binding(1)
var<storage, read> m: array<f32>;

@group(0) @binding(2)
var<storage, read_write> weights: array<f32>;

var<immediate> b1: f32;
var<immediate> learning_rate: f32;

#include "default_workgroup_size.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let weights_length = arrayLength(&weights);
    if index >= weights_length
    {
        return;
    }

    weights[index] -= gradient[index] * learning_rate;
}
