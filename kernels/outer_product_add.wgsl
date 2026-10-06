@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let output_length = arrayLength(&output);
    if index >= output_length
    {
        return;
    }

    let lhs_length = arrayLength(&lhs);

    let row = index % lhs_length;
    let column = index / lhs_length;

    output[index] += lhs[row] * rhs[column];
}
