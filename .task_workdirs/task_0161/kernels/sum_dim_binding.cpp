#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of the CUDA launcher from custom_ops.cu
extern "C" void sum_dim1_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int D, int H, int W,
    long long in_stride_N, long long in_stride_C,
    long long in_stride_D, long long in_stride_H, long long in_stride_W,
    long long out_stride_N, long long out_stride_D,
    long long out_stride_H, long long out_stride_W,
    cudaStream_t stream);

torch::Tensor sum_dim1_forward(torch::Tensor input) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (N,C,D,H,W)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t D = input.size(2);
    int64_t H = input.size(3);
    int64_t W = input.size(4);

    // Allocate output tensor: [N, D, H, W]
    auto output = torch::empty({N, D, H, W}, input.options());

    // Get input strides
    auto strides = input.strides();
    long long in_stride_N = strides[0];
    long long in_stride_C = strides[1];
    long long in_stride_D = strides[2];
    long long in_stride_H = strides[3];
    long long in_stride_W = strides[4];

    // Output is contiguous, compute strides
    long long out_stride_N = D * H * W;
    long long out_stride_D = H * W;
    long long out_stride_H = W;
    long long out_stride_W = 1;

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    sum_dim1_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(D), static_cast<int>(H), static_cast<int>(W),
        in_stride_N, in_stride_C, in_stride_D, in_stride_H, in_stride_W,
        out_stride_N, out_stride_D, out_stride_H, out_stride_W,
        stream);

    return output;
}

void register_sum_dim1(pybind11::module& m) {
    m.def("sum_dim1_forward", &sum_dim1_forward,
          "Sum over channel dimension (dim=1)",
          py::arg("input"));
}

REGISTER_BINDING(sum_dim1, register_sum_dim1);
