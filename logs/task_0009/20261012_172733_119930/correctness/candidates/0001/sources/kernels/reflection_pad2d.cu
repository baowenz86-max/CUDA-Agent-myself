#include <cuda_runtime.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Kernel for reflection padding 2D
// input: input tensor (4D)
// output: output tensor (4D)
// pad_left, pad_right, pad_top, pad_bottom: padding sizes
// in_height, in_width: input spatial dimensions
// out_height, out_width: output spatial dimensions
// in_stride_batch, in_stride_channel, in_stride_height, in_stride_width: strides of input
// out_stride_batch, out_stride_channel, out_stride_height, out_stride_width: strides of output
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int in_height, int in_width,
    int out_height, int out_width,
    long long in_stride_batch, long long in_stride_channel,
    long long in_stride_height, long long in_stride_width,
    long long out_stride_batch, long long out_stride_channel,
    long long out_stride_height, long long out_stride_width) {

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = out_height * out_width;

    if (global_idx >= total_elements) return;

    // 从输出线性索引推导出输出坐标 (oh, ow)
    int ow = global_idx % out_width;
    int oh = global_idx / out_width;

    // 计算反射坐标 ih, iw
    int ih = oh - pad_top;
    int iw = ow - pad_left;

    // 反射映射
    // 对于高度
    if (ih < 0) ih = -ih;
    if (ih >= in_height) ih = 2 * (in_height - 1) - ih;
    // 对于宽度
    if (iw < 0) iw = -iw;
    if (iw >= in_width) iw = 2 * (in_width - 1) - iw;

    // 注意：当 padding 大于输入尺寸时，反射公式可能需要多次反射，但当前任务 padding=2，输入尺寸足够大，因此单次反射足够。
    // 若需要通用性，可改为循环，但为性能保持单次。

    // 获取 batch 和 channel 索引
    int b = blockIdx.y;
    int c = blockIdx.z;

    // 输入和输出的指针偏移
    const float* in_ptr = input + b * in_stride_batch + c * in_stride_channel
                          + ih * in_stride_height + iw * in_stride_width;
    float* out_ptr = output + b * out_stride_batch + c * out_stride_channel
                     + oh * out_stride_height + ow * out_stride_width;

    *out_ptr = *in_ptr;
}

// Host launcher
void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    long long in_stride_batch,
    long long in_stride_channel,
    long long in_stride_height,
    long long in_stride_width,
    long long out_stride_batch,
    long long out_stride_channel,
    long long out_stride_height,
    long long out_stride_width,
    cudaStream_t stream) {

    int out_height = in_height + pad_top + pad_bottom;
    int out_width = in_width + pad_left + pad_right;
    int total_elements_per_slice = out_height * out_width;

    dim3 block(256);
    dim3 grid((total_elements_per_slice + block.x - 1) / block.x, batch_size, channels);

    reflection_pad2d_kernel<<<grid, block, 0, stream>>>(
        input, output,
        pad_left, pad_right, pad_top, pad_bottom,
        in_height, in_width,
        out_height, out_width,
        in_stride_batch, in_stride_channel, in_stride_height, in_stride_width,
        out_stride_batch, out_stride_channel, out_stride_height, out_stride_width
    );
}

// Binding wrapper
// 注意：这里可以包含 torch 头文件，但为了避免编译慢，我们只包含必要部分。
torch::Tensor reflection_pad2d_forward(torch::Tensor input, std::vector<int64_t> padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(padding.size() == 4, "Padding must have 4 elements");

    // 确保 padding 可以转换为 int
    int pad_left = static_cast<int>(padding[0]);
    int pad_right = static_cast<int>(padding[1]);
    int pad_top = static_cast<int>(padding[2]);
    int pad_bottom = static_cast<int>(padding[3]);

    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    int out_height = in_height + pad_top + pad_bottom;
    int out_width = in_width + pad_left + pad_right;

    // 处理非连续输入：直接使用 strides 即可，Copy kernel 内部使用 strides。
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        stream
    );

    return output;
}

void register_reflection_pad2d(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward",
          py::arg("input"), py::arg("padding"));
}

REGISTER_BINDING(reflection_pad2d, register_reflection_pad2d);
