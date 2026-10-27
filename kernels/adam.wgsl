@group(0) @binding(0)
var<storage, read> gradient: array<f32>;

@group(0) @binding(1)
var<storage, read_write> m: array<f32>;

@group(0) @binding(2)
var<storage, read_write> v: array<f32>;

@group(0) @binding(3)
var<storage, read_write> weights: array<f32>;

struct Info{
    learning_rate: f32,
    b1: f32,
    b2: f32,
    epsilon: f32,
    t: f32
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

    let new_m = info.b1 * m[index] + (1.0 - info.b1) * gradient[index];
    let new_v = info.b2 * v[index] + (1.0 - info.b2) * (gradient[index] * gradient[index]);

    let m_corrected = new_m / (1.0 - pow(info.b1, info.t));
    let v_corrected = new_v / (1.0 - pow(info.b2, info.t));

    weights[index] -= info.learning_rate * m_corrected / (sqrt(v_corrected) + info.epsilon);

    m[index] = new_m;
    v[index] = new_v;
}
