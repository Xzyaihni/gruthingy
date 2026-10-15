let inside_spans = div_ceil(values_length, WORKGROUP_SIZE);

for (var span_id = 0u; span_id < inside_spans; span_id++)
{
    let i = span_id * WORKGROUP_SIZE + local_invocation_index;

    if i >= values_length
    {
        return;
    }

    let start = batch_start + i;
    softmaxed_values[start] = exp(softmaxed_values[start] - biggest_value);
}
