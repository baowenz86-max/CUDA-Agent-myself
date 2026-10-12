// Bindings for custom CUDA ops: addmm, unsqueeze, interpolate_linear
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void addmm_forward_launcher(
    const float* bias, const float* A, const float* B, float* C,
    int M, int N, int K, int ldA, int ldB, cudaStream_t stream);

extern "C" void unsqueeze_forward_launcher(
    const float* input, float* output, int64_t numel, cudaStream_t stream);

extern "C" void interpolate_linear_forward_launcher(
    const float* input, float* output, int M, int N_in, int N_out,
    float scale_factor, cudaStream_t stream);

// --------------------- addmm_forward ---------------------
torch::Tensor addmm_forward(torch::Tensor bias, torch::Tensor mat1, torch::Tensor mat2) {
    // Input validation
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(mat1.is_cuda(), "mat1 must be a CUDA tensor");
    TORCH_CHECK(mat2.is_cuda(), "mat2 must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(mat1.dtype() == torch::kFloat32, "mat1 must be float32");
    TORCH_CHECK(mat2.dtype() == torch::kFloat32, "mat2 must be float32");

    // bias: [N] (1D)
    // mat1: [M, K]
    // mat2: [K, N]  (the transposed weight)
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2D");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be 2D");

    int64_t K = mat1.size(1);
    int64_t M = mat1.size(0);
    int64_t N = bias.size(0);

    TORCH_CHECK(mat2.size(0) == K, "mat2 rows must equal K");
    TORCH_CHECK(mat2.size(1) == N, "mat2 cols must equal N");

    // Ensure contiguous inputs; mat2 may be non-contiguous (transposed view)
    auto bias_contig = bias.contiguous();
    auto mat1_contig = mat1.contiguous();
    auto mat2_contig = mat2.contiguous();  // For safety, but we can also pass strides

    auto output = torch::empty({M, N}, mat1.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Pass strides of the possibly non-contiguous tensors
    // We use the original strides if we didn't make contiguous, but since we did contig, strides are standard
    int ldA = mat1_contig.stride(0);
    int ldB = mat2_contig.stride(0);

    addmm_forward_launcher(
        bias_contig.data_ptr<float>(),
        mat1_contig.data_ptr<float>(),
        mat2_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K, ldA, ldB, stream
    );

    return output;
}

// --------------------- unsqueeze_forward ---------------------
torch::Tensor unsqueeze_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input, input.options());
    output = output.unsqueeze(dim);  // Change shape, but same storage

    // Since unsqueeze is a view with same number of elements, we can copy
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    unsqueeze_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

// --------------------- interpolate_linear_forward ---------------------
torch::Tensor interpolate_linear_forward(torch::Tensor input, double scale_factor) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [1, M, N_in]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    int64_t C = input.size(0);
    int64_t M = input.size(1);
    int64_t N_in = input.size(2);
    TORCH_CHECK(C == 1, "only batch size 1 supported");

    int64_t N_out = static_cast<int64_t>(std::floor(N_in * scale_factor));
    TORCH_CHECK(N_out > 0, "output size must be positive");

    auto output = torch::empty({C, M, N_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    interpolate_linear_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N_in, N_out, static_cast<float>(scale_factor),
        stream
    );

    return output;
}

// --------------------- Registration ---------------------
void register_custom_ops(pybind11::module& m) {
    m.def("addmm_forward", &addmm_forward, "addmm forward: out = mat1 @ mat2.t() + bias",
          pybind11::arg("bias"), pybind11::arg("mat1"), pybind11::arg("mat2"));
    m.def("unsqueeze_forward", &unsqueeze_forward, "unsqueeze forward",
          pybind11::arg("input"), pybind11::arg("dim"));
    m.def("interpolate_linear_forward", &interpolate_linear_forward, "linear interpolation forward",
          pybind11::arg("input"), pybind11::arg("scale_factor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);