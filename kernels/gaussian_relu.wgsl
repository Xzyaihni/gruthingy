@group(0) @binding(0)
var<storage, read> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

const LEAKY_SLOPE: f32 = 0.01;

#include "default_workgroup_size.wgsl"

#include "gaussian_approx.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let arrays_length = arrayLength(&src);
    if index >= arrays_length
    {
        return;
    }

    let x = src[index];

    dst[index] = x * gaussian_relu_approx(x);
}
