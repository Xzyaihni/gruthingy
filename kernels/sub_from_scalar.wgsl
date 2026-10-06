@group(0) @binding(0)
var<storage, read> b: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

var<immediate> a: f32;

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let arrays_length = arrayLength(&b);
    if index >= arrays_length
    {
        return;
    }

    output[index] = a - b[index];
}
