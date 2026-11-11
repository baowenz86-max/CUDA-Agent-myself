#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declarations of launchers from custom_ops.cu
extern "C" void custom_mean_launcher(const float* x, float* out,
                                      int B, int D, int W,
                                      long long sx0, long long sx1, long long sx2,
                                      cudaStream_t stream);

extern "C" void custom_addmm_launcher(const float* bias, const float* mat1, const float* mat2,
                                        float* out,
                                        int M, int K, int N,
                                        long long bias_s0, long long bias_s1,
                                        long long mat1_s0, long long mat1_s1,
                                        long long mat2_s0, long long mat2_s1,
                                        cudaStream_t stream);

// Custom mean wrapper
torch::Tensor custom_mean(torch::Tensor x, int64_t dim, bool keepdim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, D, W)");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(!keepdim, "keepdim must be false");

    int64_t B = x.size(0);
    int64_t D = x.size(1);
    int64_t W = x.size(2);

    // Output is contiguous, shape (B, W)
    auto out = torch::empty({B, W}, x.options());

    // Get strides in elements
    auto strides = x.strides();
    long long sx0 = strides[0];
    long long sx1 = strides[1];
    long long sx2 = strides[2];

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_mean_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(D),
        static_cast<int>(W),
        sx0, sx1, sx2,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_mean kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Custom addmm wrapper
torch::Tensor custom_addmm(torch::Tensor bias, torch::Tensor mat1, torch::Tensor mat2) {
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(mat1.is_cuda(), "mat1 must be a CUDA tensor");
    TORCH_CHECK(mat2.is_cuda(), "mat2 must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(mat1.dtype() == torch::kFloat32, "mat1 must be float32");
    TORCH_CHECK(mat2.dtype() == torch::kFloat32, "mat2 must be float32");
    TORCH_CHECK(bias.dim() == 2, "bias must be 2D");
    TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2D");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be 2D");

    int64_t M = mat1.size(0);
    int64_t K = mat1.size(1);
    int64_t N = mat2.size(1);

    TORCH_CHECK(bias.size(0) == M && bias.size(1) == N, "bias shape mismatch");
    TORCH_CHECK(mat2.size(0) == K, "mat2 inner dimension mismatch");

    // Output is contiguous, shape (M, N)
    auto out = torch::empty({M, N}, bias.options());

    // Get strides in elements
    auto bias_strides = bias.strides();
    auto mat1_strides = mat1.strides();
    auto mat2_strides = mat2.strides();

    long long bias_s0 = bias_strides[0];
    long long bias_s1 = bias_strides[1];
    long long mat1_s0 = mat1_strides[0];
    long long mat1_s1 = mat1_strides[1];
    long long mat2_s0 = mat2_strides[0];
    long long mat2_s1 = mat2_strides[1];

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_addmm_launcher(
        bias.data_ptr<float>(),
        mat1.data_ptr<float>(),
        mat2.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(M),
        static_cast<int>(K),
        static_cast<int>(N),
        bias_s0, bias_s1,
        mat1_s0, mat1_s1,
        mat2_s0, mat2_s1,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_addmm kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_mean", &custom_mean,
          "Custom mean along dim=1",
          py::arg("x"),
          py::arg("dim") = 1,
          py::arg("keepdim") = false);
    m.def("custom_addmm", &custom_addmm,
          "Custom addmm: bias + mat1 @ mat2",
          py::arg("bias"),
          py::arg("mat1"),
          py::arg("mat2"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
