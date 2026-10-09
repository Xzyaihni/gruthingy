let batch_index = workgroup_id.x;
let batch_start = batch_index * values_length;

for (var half_stride = 1u; half_stride < values_length; half_stride *= 2)
{
    let stride = half_stride * 2;

    let strided_workgroup = WORKGROUP_SIZE * stride;

    let inside_spans = div_ceil(values_length, strided_workgroup);

    for (var span_id = 0u; span_id < inside_spans; span_id++)
    {
        let i = span_id * strided_workgroup + stride * local_invocation_index;

        if (i + half_stride) >= values_length
        {
            break;
        }

        let start = batch_start + i;
        sum_values[start] = max(sum_values[start], sum_values[start + half_stride]);
    }

    storageBarrier();
}

let biggest_value = sum_values[0];
var z = 0.0;

let inside_spans = div_ceil(values_length, WORKGROUP_SIZE);

for (var span_id = 0u; span_id < inside_spans; span_id++)
{
    let i = span_id * WORKGROUP_SIZE + local_invocation_index;

    if i >= values_length
    {
        break;
    }

    let start = batch_start + i;

    // resets sum_values from the biggest value calculation back to values, softmaxed_values unchanged at this step
    sum_values[start] = softmaxed_values[start];

    softmaxed_values[start] -= biggest_value;

    if i == target_of_batch(batch_index)
    {
        z = softmaxed_values[start];
    }

    softmaxed_values[start] = exp(softmaxed_values[start]);
}

storageBarrier();

tensor_sum(batch_index, local_invocation_index, values_length);

let s = sum_values[batch_start];
let s_r = 1.0 / s;

for (var span_id = 0u; span_id < inside_spans; span_id++)
{
    let i = span_id * WORKGROUP_SIZE + local_invocation_index;

    if i >= values_length
    {
        return;
    }

    let start = batch_start + i;
    softmaxed_values[start] *= s_r;
}
