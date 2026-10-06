@group(0) @binding(0)
var<storage, read_write> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

const WORKGROUP_SIZE = 64;

fn div_ceil(a: u32, b: u32) -> u32
{
    return (a + b - 1) / b;
}

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
    let lhs_single_length = lhs_rows * lhs_columns;

    let row = workgroup_id.x % lhs_rows;
    let batch_index = workgroup_id.x / lhs_rows;

    let batch_start = batch_index * lhs_single_length;

    {
        let inside_spans = div_ceil(lhs_columns, WORKGROUP_SIZE);
        for (var span_id = 0u; span_id < inside_spans; span_id++)
        {
            let column = span_id * WORKGROUP_SIZE + local_invocation_index;

            if (column >= lhs_columns)
            {
                return;
            }

            lhs[batch_start + row + column * lhs_rows] *= rhs[batch_index * lhs_rows + column];
        }

        storageBarrier();
    }

    for (var half_stride = 1u; half_stride < lhs_columns; half_stride *= 2)
    {
        let stride = half_stride * 2;

        let strided_workgroup = WORKGROUP_SIZE * stride;

        let inside_spans = div_ceil(lhs_columns, strided_workgroup);
        for (var span_id = 0u; span_id < inside_spans; span_id++)
        {
            let lhs_column_index = span_id * strided_workgroup + stride * lhs_rows * local_invocation_index;

            let next_offset = half_stride * lhs_rows;
            let next_column_index = lhs_column_index + next_offset;

            if (next_column_index >= lhs_single_length)
            {
                return;
            }

            let start = batch_start + row + lhs_column_index;
            lhs[start] += lhs[start + next_offset];

            storageBarrier();
        }
    }

    output[workgroup_id.x] = lhs[batch_start + row];
}
