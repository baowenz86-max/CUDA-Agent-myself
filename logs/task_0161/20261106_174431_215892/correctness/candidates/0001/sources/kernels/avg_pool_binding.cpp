#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of the CUDA launcher from custom_ops.cu
extern "C" void avg_pool3d_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW,
    long long in_stride_N, long long in_stride_C,
    long long in_stride_D, long long in_stride_H, long long in_stride_W,
    long long out_stride_N, long long out_stride_C,
    long long out_stride_D, long long out_stride_H, long long out_stride_W,
    cudaStream_t stream);

torch::Tensor avg_pool3d_forward(torch::Tensor input, int64_t kernel_size, int64_t stride) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (N,C,D,H,W)");
    // Since kernel_size and stride are both 2 for this task, we enforce them here;
    // the launcher already assumes 2x2x2 pooling with stride 2.
    TORCH_CHECK(kernel_size == 2 && stride == 2, "Only kernel_size=2 and stride=2 are supported");

    // Input shape: [N, C, D, H, W]
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t D = input.size(2);
    int64_t H = input.size(3);
    int64_t W = input.size(4);

    // Compute output spatial sizes
    int64_t outD = D / 2;  // since kernel=2, stride=2, padding=0
    int64_t outH = H / 2;
    int64_t outW = W / 2;

    // Ensure divisibility
    TORCH_CHECK(D % 2 == 0 && H % 2 == 0 && W % 2 == 0, "Input spatial dimensions must be divisible by 2");

    // Allocate output tensor with same options as input
    auto output = torch::empty({N, C, outD, outH, outW}, input.options());

    // Get input strides
    auto strides = input.strides();
    long long in_stride_N = strides[0];
    long long in_stride_C = strides[1];
    long long in_stride_D = strides[2];
    long long in_stride_H = strides[3];
    long long in_stride_W = strides[4];

    // Output is contiguous, compute strides accordingly
    long long out_stride_N = C * outD * outH * outW;
    long long out_stride_C = outD * outH * outW;
    long long out_stride_D = outH * outW;
    long long out_stride_H = outW;
    long long out_stride_W = 1;

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    avg_pool3d_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(D), static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(outD), static_cast<int>(outH), static_cast<int>(outW),
        in_stride_N, in_stride_C, in_stride_D, in_stride_H, in_stride_W,
        out_stride_N, out_stride_C, out_stride_D, out_stride_H, out_stride_W,
        stream);

    return output;
}

void register_avg_pool3d(pybind11::module& m) {
    m.def("avg_pool3d_forward", &avg_pool3d_forward,
          "Average pooling 3D forward (kernel_size=2, stride=2)",
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride"));
}

REGISTER_BINDING(avg_pool3d, register_avg_pool3d);
