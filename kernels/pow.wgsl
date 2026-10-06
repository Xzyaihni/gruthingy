@group(0) @binding(0)
var<storage, read> a: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

var<immediate> power: u32;

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let arrays_length = arrayLength(&a);
    if index >= arrays_length
    {
        return;
    }

    let a_value = a[index];
    output[index] = select(sign(a_value), 1.0, (power % 2) == 0) * pow(abs(a_value), f32(power));
}
