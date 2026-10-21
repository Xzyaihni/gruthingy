@group(0) @binding(0)
var<storage, read> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

#include "default_workgroup_size.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let src_length = arrayLength(&src);
    let batch_size = arrayLength(&dst) / src_length;

    if index >= src_length
    {
        return;
    }

    for (var batch_index = 0u; batch_index < batch_size; batch_index++)
    {
        dst[batch_index * src_length + index] = src[index];
    }
}
