@group(0) @binding(0)
var<storage, read> gradient: array<f32>;

@group(0) @binding(1)
var<storage, read> targets: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

const WORKGROUP_SIZE = 64;

#include "div_ceil.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    let batch_size = arrayLength(&gradient);
    let values_length = arrayLength(&output) / batch_size;

    let batch_index = workgroup_id.x;
    let batch_start = batch_index * values_length;

    let inside_spans = div_ceil(values_length, WORKGROUP_SIZE);

    for (var span_id = 0u; span_id < inside_spans; span_id++)
    {
        let i = span_id * WORKGROUP_SIZE + local_invocation_index;

        if i < values_length
        {
            let start = batch_start + i;

            if i == targets[batch_index]
            {
                output[start] -= 1.0;
            }

            output[start] *= gradient[batch_index];
        }
    }
}
