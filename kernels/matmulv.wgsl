@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

const WORKGROUP_SIZE = 64;

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
)
{
    let lhs_length = arrayLength(&lhs);
    let rhs_length = arrayLength(&rhs);
    let output_length = arrayLength(&output);

    let lhs_rows = lhs_length / rhs_length;
    let batch_size = output_length / lhs_rows;
    let lhs_columns = rhs_length / batch_size;
}
