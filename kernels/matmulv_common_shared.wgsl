let lhs_length = arrayLength(&lhs);
let rhs_length = arrayLength(&rhs);
let output_length = arrayLength(&output);

let lhs_rows = lhs_length / rhs_length;
let batch_size = output_length / lhs_rows;
let lhs_columns = rhs_length / batch_size;
let lhs_single_length = lhs_rows * lhs_columns;

let row_stride = select(lhs_columns, 1, is_column_major);
let column_stride = select(1, lhs_rows, is_column_major);

let row = workgroup_id.x % lhs_rows;
let batch_index = workgroup_id.x / lhs_rows;

let batch_start = batch_index * lhs_single_length;

{
    let inside_spans = div_ceil(lhs_columns, WORKGROUP_SIZE);
    for (var span_id = 0u; span_id < inside_spans; span_id++)
    {
        let column = span_id * WORKGROUP_SIZE + local_invocation_index;

        if column < lhs_columns
        {
            lhs[batch_start + row * row_stride + column * column_stride] *= rhs[batch_index * lhs_columns + column];
        }

        storageBarrier();
    }
}

for (var half_stride = 1u; half_stride < lhs_columns; half_stride *= 2)
{
    let stride = half_stride * 2;

    let strided_workgroup = WORKGROUP_SIZE * stride;

    let inside_spans = div_ceil(lhs_columns, strided_workgroup);
    for (var span_id = 0u; span_id < inside_spans; span_id++)
    {
        let column = span_id * strided_workgroup + stride * local_invocation_index;

        if (column + half_stride) < lhs_columns
        {
            let start = batch_start + row * row_stride + column * column_stride;
            lhs[start] += lhs[start + half_stride * column_stride];
        }

        storageBarrier();
    }
}

if local_invocation_index == 0
{
    output[workgroup_id.x] = map_output(workgroup_id.x, lhs[batch_start + row * row_stride]);
}
