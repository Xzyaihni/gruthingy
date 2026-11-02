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

var total = 0.0;

const pre_steps = 8u;
const span_step = WORKGROUP_SIZE * pre_steps;

let inside_spans = div_ceil(lhs_columns, span_step);
for (var span_id = 0u; span_id < inside_spans; span_id++)
{
    workgroup_chunk[local_invocation_index] = 0.0;

    for (var s = 0u; s < pre_steps; s++)
    {
        let spanned_column = span_id * span_step + s * WORKGROUP_SIZE + local_invocation_index;

        if spanned_column < lhs_columns
        {
            let lhs_offset = batch_start + row * row_stride;
            let rhs_offset = batch_index * lhs_columns;

            let value = lhs[lhs_offset + spanned_column * column_stride] * rhs[rhs_offset + spanned_column];

            workgroup_chunk[local_invocation_index] += value;
        }
    }

    workgroupBarrier();

    for (var stride = (u32(WORKGROUP_SIZE) >> 1); stride > 1; stride >>= 1)
    {
        if local_invocation_index < stride
        {
            workgroup_chunk[local_invocation_index] += workgroup_chunk[local_invocation_index + stride];
        }

        workgroupBarrier();
    }

    if local_invocation_index == 0
    {
        // WORKGROUP_SIZE must be at least 2 for this
        total += workgroup_chunk[0] + workgroup_chunk[1];
    }
}

if local_invocation_index == 0
{
    output[workgroup_id.x] = map_output(workgroup_id.x, total);
}
