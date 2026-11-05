const PI: f32 = 3.14159265358979323846264338327950288;

fn gaussian_relu_approx(x: f32) -> f32
{
    return 0.5 * (1.0 + tanh(sqrt(2.0 / PI) * (x + 0.044715 * (x * x * x))));
}
