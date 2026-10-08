use std::mem;

use super::{
    TensorShape,
    RawArrayMemory
};

use wgpu::util::DeviceExt;


macro_rules! define_op_kernels
{
    ($($variant:ident),+) =>
    {
        pub enum OpKernel
        {
            $(
                $variant,
            )+
        }

        impl OpKernel
        {
            pub const COUNT: usize = 0 $(+ { stringify!($variant); 1 })+;

            pub fn create_kernels<T>(mut f: impl FnMut(OpKernel) -> T) -> [T; Self::COUNT]
            {
                [
                    $(
                        f(Self::$variant),
                    )+
                ]
            }
        }
    }
}

define_op_kernels!{
    Add,
    Sub,
    SubFromScalar,
    MulComponentwise,
    MulScalar,
    SumTensor,
    Fill,
    Pow,
    Tanh,
    TanhDiff,
    Sigmoid,
    SigmoidDiff,
    LeakyRelu,
    LeakyReluDiff,
    Matmulv,
    MatmulvAdd,
    MatmulvTransposed,
    MatmulvTransposedAdd,
    OuterProduct,
    OuterProductAdd,
    SoftmaxCrossEntropyNoEntropy
}

#[derive(Debug, Clone)]
pub struct RawArray
{
    pub output_buffer: Option<wgpu::Buffer>,
    pub buffer: wgpu::Buffer
}

impl RawArray
{
    pub fn from_slice(memory: &RawArrayMemory, values: &[f32], shape: TensorShape) -> Self
    {
        debug_assert_eq!(shape.size(), values.len());

        let buffer = memory.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: None,
            contents: bytemuck::cast_slice(&values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
        });

        Self{
            output_buffer: None,
            buffer
        }
    }

    pub fn uninit(memory: &RawArrayMemory, size: usize) -> Self
    {
        let buffer = memory.device.create_buffer(&wgpu::BufferDescriptor{
            label: None,
            size: size as u64 * mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        Self{
            output_buffer: None,
            buffer
        }
    }
}
