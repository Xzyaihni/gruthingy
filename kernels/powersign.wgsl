@group(0) @binding(0)
var<storage, read> gradient: array<f32>;

@group(0) @binding(1)
var<storage, read_write> m: array<f32>;

@group(0) @binding(2)
var<storage, read_write> weights: array<f32>;

struct Info{
    b1: f32,
    learning_rate: f32
};

var<immediate> info: Info;

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

    m[index] = info.b1 * m[index] + (1.0 - info.b1) * gradient[index];

    weights[index] -= gradient[index] * pow(info.learning_rate, sign(m[index]) * sign(gradient[index]));
}
