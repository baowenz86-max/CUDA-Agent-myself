#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void hardtanh_launcher(const float* x, float* y, long numel, cudaStream_t stream);
extern "C" void linear_launcher(const float* x, const float* weight, const float* bias, float* y,
                                int M, int N, int K, cudaStream_t stream);

// hardtanh: clamp(x, -1, 1)
torch::Tensor hardtanh_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "hardtanh: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "hardtanh: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "hardtanh: input must be contiguous");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    hardtanh_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "hardtanh kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// linear: y = x @ weight^T + bias
torch::Tensor linear_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "linear: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "linear: weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "linear: bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "linear: x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "linear: weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "linear: bias must be float32");
    TORCH_CHECK(x.dim() == 2, "linear: x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "linear: weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "linear: bias must be 1D");
    TORCH_CHECK(x.size(1) == weight.size(1), "linear: x.size(1) must equal weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "linear: weight.size(0) must equal bias.size(0)");

    // Ensure contiguous for launcher pointer access
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    int64_t M = x_contig.size(0);
    int64_t K = x_contig.size(1);
    int64_t N = weight_contig.size(0);

    auto output = torch::empty({M, N}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    linear_launcher(
        x_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)M,
        (int)N,
        (int)K,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "linear kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("hardtanh", &hardtanh_forward,
          "HardTanh elementwise clamp(x, -1, 1)",
          py::arg("x"));
    m.def("linear", &linear_forward,
          "Linear transformation y = x @ weight.T + bias",
          py::arg("x"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
