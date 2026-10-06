@group(0) @binding(0)
var<storage, read> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let batch_size = arrayLength(&src);
    let dst_length = arrayLength(&dst);

    let x = index % dst_length;
    let batch_index = index / dst_length;

    if x >= dst_length || batch_index >= batch_size
    {
        return;
    }

    dst[batch_index * dst_length + x] = src[batch_index];
}
