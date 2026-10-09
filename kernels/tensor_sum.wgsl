fn tensor_sum(batch_index: u32, local_invocation_index: u32, values_length: u32)
{
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
                return;
            }

            let start = batch_start + i;
            sum_values[start] += sum_values[start + half_stride];
        }

        storageBarrier();
    }
}
