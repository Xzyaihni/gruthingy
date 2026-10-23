@group(0) @binding(0)
var<storage, read> magnitudes: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

var<immediate> cap: f32;

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

    let batch_size = arrayLength(&magnitudes);

    let batch_index = index / batch_size;

    let magnitude = magnitudes[batch_index];
    if magnitude > cap
    {
        output[index] *= cap / magnitude;
    }
}
