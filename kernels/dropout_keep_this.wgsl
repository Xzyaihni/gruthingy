fn keep_this(column: u32) -> bool
{
    let packed = states[column / 32];
    let bit_index = column % 32;

    return ((packed >> bit_index) & 1) == 1;
}
