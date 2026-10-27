@group(0) @binding(0)
var<storage, read> value: array<f32>;

@group(0) @binding(1)
var<storage, read> states: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

var<immediate> rows: u32;

#include "default_workgroup_size.wgsl"
#include "dropout_keep_this.wgsl"

@compute @workgroup_size(WORKGROUP_SIZE)
fn run(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;

    let values_length = arrayLength(&value);
    if index >= values_length
    {
        return;
    }

    let column = index / rows;

    output[index] = select(0.0, value[index], keep_this(column));
}
