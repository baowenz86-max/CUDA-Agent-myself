#include <cuda_runtime.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Kernel: applies reflection padding on height and width dimensions for a 4D tensor (N,C,H,W)
// Padding is assumed symmetric on each side: pad_h, pad_w
// input: contiguous, strides: W, H*W, C*H*W, N*C*H*W
template <int PAD_H, int PAD_W>
__global__ void reflection_pad2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int out_H, int out_W) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * out_H * out_W;
    if (out_idx >= total_out) return;

    int out_w = out_idx % out_W;
    int tmp = out_idx / out_W;
    int out_h = tmp % out_H;
    tmp /= out_H;
    int c = tmp % C;
    int n = tmp / C;

    // map output coordinates to input coordinates using reflection
    int in_h = out_h - PAD_H;
    if (in_h < 0) in_h = -in_h;
    else if (in_h >= H) in_h = 2 * (H - 1) - in_h;

    int in_w = out_w - PAD_W;
    if (in_w < 0) in_w = -in_w;
    else if (in_w >= W) in_w = 2 * (W - 1) - in_w;

    // input index
    int in_idx = ((n * C + c) * H + in_h) * W + in_w;
    output[out_idx] = input[in_idx];
}

// Host launcher: allocates output, launches kernel with dynamic padding values (passed as runtime parameters)
extern "C" void reflection_pad2d_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int pad_h, int pad_w,
    cudaStream_t stream) {

    int out_H = H + 2 * pad_h;
    int out_W = W + 2 * pad_w;

    int total_out = N * C * out_H * out_W;
    const int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

    // Dispatch based on pad_h and pad_w at runtime (since they are small, we can use a generic kernel?)
    // For performance, we could have specialized kernels, but for simplicity we use a generic kernel with runtime padding.
    // Here we launch a kernel that takes pad values as parameters instead of templates.
    // We'll declare another kernel or use the templated one with if/else? Better: define a non-template kernel.
    // But for now, we'll use a generic kernel via function pointer? Simpler: use a non-template kernel below.
    // We'll avoid templated version and use a plain kernel with runtime pad values.
    // The above kernel is templated; we can instantiate with common values, but pad is dynamic (fixed for this task: pad_h=pad_w=2).
    // However, the binding will always call with pad=2, so we can hardcode PAD_H=2, PAD_W=2 for performance.
    // But to be safe, we'll implement a non-template kernel that uses runtime pad values.
    // So we need a separate kernel.
}

// Non-template kernel with runtime padding
__global__ void reflection_pad2d_kernel_runtime(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int out_H, int out_W,
    int pad_h, int pad_w) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * out_H * out_W;
    if (out_idx >= total_out) return;

    int out_w = out_idx % out_W;
    int tmp = out_idx / out_W;
    int out_h = tmp % out_H;
    tmp /= out_H;
    int c = tmp % C;
    int n = tmp / C;

    int in_h = out_h - pad_h;
    if (in_h < 0) in_h = -in_h;
    else if (in_h >= H) in_h = 2 * (H - 1) - in_h;

    int in_w = out_w - pad_w;
    if (in_w < 0) in_w = -in_w;
    else if (in_w >= W) in_w = 2 * (W - 1) - in_w;

    int in_idx = ((n * C + c) * H + in_h) * W + in_w;
    output[out_idx] = input[in_idx];
}

// Actual launcher implementation
extern "C" void launch_reflection_pad2d(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int pad_h, int pad_w,
    cudaStream_t stream) {

    int out_H = H + 2 * pad_h;
    int out_W = W + 2 * pad_w;
    int total_out = N * C * out_H * out_W;
    const int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    reflection_pad2d_kernel_runtime<<<blocks, threads, 0, stream>>>(
        output, input, N, C, H, W, out_H, out_W, pad_h, pad_w);
}

// PyTorch binding wrapper
torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(padding >= 0, "Padding must be non-negative");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int pad_h = static_cast<int>(padding);
    int pad_w = static_cast<int>(padding);
    int out_H = H + 2 * pad_h;
    int out_W = W + 2 * pad_w;

    auto output = torch::empty({N, C, out_H, out_W}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_reflection_pad2d(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N, C, H, W,
        pad_h, pad_w,
        stream);

    return output;
}

void register_reflection_pad2d(py::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward, "ReflectionPad2d forward",
          py::arg("input"), py::arg("padding"));
}

REGISTER_BINDING(reflection_pad2d, register_reflection_pad2d);
