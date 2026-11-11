#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softmax_dim1_launcher(
    const float* input, float* output,
    float* temp_max, float* temp_sum,
    int N, int C, int H, int W,
    long long stride_N, long long stride_C, long long stride_H, long long stride_W,
    double temperature, cudaStream_t stream);

extern "C" void mul_square_launcher(
    const float* input, float* output, long long numel, cudaStream_t stream);

// Wrapper for softmax over dimension 1 with temperature division fused
static torch::Tensor softmax_dim1(torch::Tensor x, double temperature) {
    TORCH_CHECK(x.is_cuda(), "softmax_dim1: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "softmax_dim1: input must be float32");
    TORCH_CHECK(x.dim() == 4, "softmax_dim1: input must be 4D (N,C,H,W)");

    // Ensure contiguous for kernel simplicity (kernel supports strides anyway, but temp buffers indexing assumes contiguous slices? Actually kernel uses strides, so can handle non-contiguous, but we keep contiguous to be safe)
    auto x_contig = x.contiguous();

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);

    auto output = torch::empty_like(x_contig);
    auto temp_max = torch::empty({N * H * W}, x_contig.options());
    auto temp_sum = torch::empty({N * H * W}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_dim1_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        temp_max.data_ptr<float>(),
        temp_sum.data_ptr<float>(),
        N, C, H, W,
        x_contig.stride(0),
        x_contig.stride(1),
        x_contig.stride(2),
        x_contig.stride(3),
        temperature,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("softmax_dim1 launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}

// Wrapper for element-wise squaring
static torch::Tensor mul_square(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "mul_square: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mul_square: input must be float32");

    // The launcher assumes contiguous linear indexing
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mul_square_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        x_contig.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("mul_square launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}

// Registration function
static void register_custom_ops(pybind11::module& m) {
    m.def("softmax_dim1", &softmax_dim1,
          "Softmax over dimension 1 with temperature scaling",
          py::arg("x"), py::arg("temperature"));
    m.def("mul_square", &mul_square,
          "Element-wise square",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
