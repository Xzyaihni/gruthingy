@group(0) @binding(0)
var<storage, read> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

var<immediate> a: f32;

#include "default_workgroup_size.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let dst_length = arrayLength(&dst);
    let batch_size = arrayLength(&src) / dst_length;

    if index >= dst_length
    {
        return;
    }

    dst[index] = 0.0;

    for (var batch_index = 0u; batch_index < batch_size; batch_index++)
    {
        dst[index] += src[batch_index * dst_length + index] * a;
    }
}
