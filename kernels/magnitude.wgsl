@group(0) @binding(0)
var<storage, read> magnitude_values: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

#include "default_workgroup_size.wgsl"

var<workgroup> workgroup_chunk: array<f32, WORKGROUP_SIZE>;

#include "div_ceil.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    let batch_size = arrayLength(&dst);
    let src_length = arrayLength(&magnitude_values);
    let src_single_length = src_length / batch_size;

    let batch_index = workgroup_id.x;

    let batch_start = batch_index * src_single_length;

    var total = 0.0;

    let inside_spans = div_ceil(src_single_length, WORKGROUP_SIZE);
    for (var span_id = 0u; span_id < inside_spans; span_id++)
    {
        let spanned_column = span_id * WORKGROUP_SIZE + local_invocation_index;

        if spanned_column < src_single_length
        {
            let value = magnitude_values[batch_start + spanned_column];

            workgroup_chunk[local_invocation_index] = value * value;
        } else
        {
            workgroup_chunk[local_invocation_index] = 0.0;
        }

        workgroupBarrier();

        for (var stride = (u32(WORKGROUP_SIZE) >> 1); stride > 0; stride >>= 1)
        {
            if local_invocation_index < stride
            {
                workgroup_chunk[local_invocation_index] += workgroup_chunk[local_invocation_index + stride];
            }

            workgroupBarrier();
        }

        if local_invocation_index == 0
        {
            total += workgroup_chunk[0];
        }
    }

    if local_invocation_index == 0
    {
        dst[workgroup_id.x] = sqrt(total);
    }
}
