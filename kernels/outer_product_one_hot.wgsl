@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

var<immediate> one_hot_count: u32;

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let lhs_length = arrayLength(&lhs);
    let batch_size = arrayLength(&rhs) / one_hot_count;

    if index >= lhs_length
    {
        return;
    }

    let lhs_single_length = lhs_length / batch_size;

    let output_single_length = arrayLength(&output) / batch_size;

    let row = index % lhs_single_length;
    let batch_index = index / lhs_single_length;

    for (var o = 0u; o < one_hot_count; o++)
    {
        let column = rhs[batch_index * one_hot_count + o];

        if column != 4294967295
        {
            output[batch_index * output_single_length + row + column * lhs_single_length] = lhs[index];
        }
    }
}
