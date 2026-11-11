// Binding for baddbmm_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void baddbmm_forward_launcher(
    float* output,
    const float* bias,
    const float* mat1,
    const float* mat2,
    int batch, int M, int N, int K,
    float alpha, float beta,
    const long long* bias_strides,
    const long long* mat1_strides,
    const long long* mat2_strides,
    const long long* out_strides,
    cudaStream_t stream);

torch::Tensor baddbmm_forward(
    torch::Tensor bias,
    torch::Tensor mat1,
    torch::Tensor mat2,
    double beta = 1.0,
    double alpha = 1.0)
{
    // Check tensors are CUDA and float32
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(mat1.is_cuda(), "mat1 must be a CUDA tensor");
    TORCH_CHECK(mat2.is_cuda(), "mat2 must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(mat1.dtype() == torch::kFloat32, "mat1 must be float32");
    TORCH_CHECK(mat2.dtype() == torch::kFloat32, "mat2 must be float32");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D [batch, M, N]");
    TORCH_CHECK(mat1.dim() == 3, "mat1 must be 3D [batch, M, K]");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3D [batch, K, N]");

    int batch = mat1.size(0);
    int M = mat1.size(1);
    int K = mat1.size(2);
    TORCH_CHECK(mat2.size(0) == batch && mat2.size(1) == K && mat2.size(2) == mat2.size(2), "mat2 shape mismatch");
    int N = mat2.size(2);
    TORCH_CHECK(bias.size(0) == batch && bias.size(1) == M && bias.size(2) == N, "bias shape mismatch");

    // Make inputs contiguous if needed (launcher assumes contiguous)
    auto bias_contig = bias.contiguous();
    auto mat1_contig = mat1.contiguous();
    auto mat2_contig = mat2.contiguous();

    // Allocate output tensor
    auto output = torch::empty({batch, M, N}, mat1.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Prepare strides (all contiguous: batch stride = M*K or K*N or M*N, row stride = K or N, col stride = 1)
    long long bias_strides[3] = { (long long)M * N, (long long)N, 1 };
    long long mat1_strides[3] = { (long long)M * K, (long long)K, 1 };
    long long mat2_strides[3] = { (long long)K * N, (long long)N, 1 };
    long long out_strides[3] = { (long long)M * N, (long long)N, 1 };

    // Call launcher
    baddbmm_forward_launcher(
        output.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        mat1_contig.data_ptr<float>(),
        mat2_contig.data_ptr<float>(),
        batch, M, N, K,
        (float)alpha, (float)beta,
        bias_strides,
        mat1_strides,
        mat2_strides,
        out_strides,
        stream
    );

    return output;
}

void register_baddbmm(pybind11::module& m) {
    m.def("baddbmm_forward", &baddbmm_forward,
          "baddbmm_forward(bias, mat1, mat2, beta=1.0, alpha=1.0) -> Tensor",
          pybind11::arg("bias"),
          pybind11::arg("mat1"),
          pybind11::arg("mat2"),
          pybind11::arg("beta") = 1.0,
          pybind11::arg("alpha") = 1.0);
}

REGISTER_BINDING(baddbmm, register_baddbmm);
