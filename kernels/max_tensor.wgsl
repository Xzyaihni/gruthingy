@group(0) @binding(0)
var<storage, read_write> max_values: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

const WORKGROUP_SIZE = 64;

#include "div_ceil.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    let batch_size = arrayLength(&dst);
    let values_length = arrayLength(&max_values) / batch_size;

    let batch_index = workgroup_id.x;
    let batch_start = batch_index * values_length;

    for (var half_stride = 1u; half_stride < values_length; half_stride *= 2)
    {
        let stride = half_stride * 2;

        let strided_workgroup = WORKGROUP_SIZE * stride;

        let inside_spans = div_ceil(values_length, strided_workgroup);

        for (var span_id = 0u; span_id < inside_spans; span_id++)
        {
            let i = span_id * strided_workgroup + stride * local_invocation_index;

            if (i + half_stride) >= values_length
            {
                return;
            }

            let start = batch_start + i;
            max_values[start] = max(max_values[start], max_values[start + half_stride]);
        }

        storageBarrier();
    }

    dst[batch_index] = max_values[batch_index * values_length];
}
