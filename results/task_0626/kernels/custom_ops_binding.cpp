#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_softplus_launcher(const float* input,
                                         float* output,
                                         long long total_elements,
                                         cudaStream_t stream);

extern "C" void custom_sum_dim1_launcher(const float* input,
                                         float* output,
                                         int N, int C, int H, int W,
                                         cudaStream_t stream);

// PyTorch wrapper for softplus
torch::Tensor custom_softplus(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    auto output = torch::empty_like(x);
    long long total_elements = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_softplus_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        stream
    );

    return output;
}

// PyTorch wrapper for sum over dim 1 (keepdim=True)
torch::Tensor custom_sum_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty({N, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_sum_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_softplus", &custom_softplus, "Softplus forward",
          pybind11::arg("x"));
    m.def("custom_sum_dim1", &custom_sum_dim1, "Sum over dim 1 with keepdim",
          pybind11::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
