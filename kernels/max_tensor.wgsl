@group(0) @binding(0)
var<storage, read> max_values: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

const WORKGROUP_SIZE = 64;

var<workgroup> workgroup_chunk: array<f32, WORKGROUP_SIZE>;
var<private> total: f32;

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

    let inside_spans = div_ceil(values_length, WORKGROUP_SIZE);
    for (var span_id = 0u; span_id < inside_spans; span_id++)
    {
        let spanned_column = span_id * WORKGROUP_SIZE + local_invocation_index;

        if spanned_column < values_length
        {
            workgroup_chunk[local_invocation_index] = max_values[batch_start + spanned_column];
        }

        workgroupBarrier();

        for (var stride = (u32(WORKGROUP_SIZE) >> 1); stride > 0; stride >>= 1)
        {
            if (local_invocation_index < stride) && ((spanned_column + stride) < values_length)
            {
                workgroup_chunk[local_invocation_index] = max(
                    workgroup_chunk[local_invocation_index],
                    workgroup_chunk[local_invocation_index + stride]
                );
            }

            workgroupBarrier();
        }

        if local_invocation_index == 0
        {
            if span_id == 0
            {
                total = workgroup_chunk[0];
            } else
            {
                total = max(total, workgroup_chunk[0]);
            }
        }
    }

    if local_invocation_index == 0
    {
        dst[workgroup_id.x] = total;
    }
}
