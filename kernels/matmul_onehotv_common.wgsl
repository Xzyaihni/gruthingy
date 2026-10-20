let lhs_length = arrayLength(&lhs);
let output_length = arrayLength(&output);

let batch_size = arrayLength(&rhs) / one_hot_count;

let lhs_rows = output_length / batch_size;
let lhs_single_length = lhs_length / batch_size;
let lhs_columns = lhs_single_length / lhs_rows;

let column_stride = lhs_rows;

let row = workgroup_id.x % lhs_rows;
let batch_index = workgroup_id.x / lhs_rows;

let batch_start = batch_index * lhs_single_length;

for (var o = 0u; o < one_hot_count; o++)
{
    let one_hot_index = rhs[batch_index * one_hot_count + o];

    if one_hot_index == 4294967295
    {
        if o == 0
        {
            output[workgroup_id.x] = map_output(workgroup_id.x, 0.0);
        }
    } else
    {
        output[workgroup_id.x] = map_output(workgroup_id.x, lhs[batch_start + row + one_hot_index * column_stride]);
    }
}
