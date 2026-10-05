@group(0) @binding(0)
var<storage, read> src: array<f32>;

@group(0) @binding(1)
var<storage, read> gradient: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let arrays_length = arrayLength(&src);
    if (index >= arrays_length)
    {
        return;
    }

    let a = src[index];

    dst[index] = (1.0 - a) * a * gradient[index];
}
