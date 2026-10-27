#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_elu_launcher(float* output, const float* input,
                                    int64_t total_elements, cudaStream_t stream);

extern "C" void custom_triu_launcher(float* output, const float* input,
                                     int64_t total_elements, int64_t H, int64_t W,
                                     cudaStream_t stream);

torch::Tensor custom_elu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    if (output.numel() == 0) {
        return output;
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_elu_launcher(output.data_ptr<float>(), x_contig.data_ptr<float>(),
                        x_contig.numel(), stream);
    return output;
}

torch::Tensor custom_triu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() >= 2, "Input must have at least 2 dimensions");
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    if (output.numel() == 0) {
        return output;
    }
    int64_t H = x_contig.size(-2);
    int64_t W = x_contig.size(-1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_triu_launcher(output.data_ptr<float>(), x_contig.data_ptr<float>(),
                         x_contig.numel(), H, W, stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_elu", &custom_elu, "Custom ELU activation",
          py::arg("x"));
    m.def("custom_triu", &custom_triu, "Custom upper triangular mask",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
