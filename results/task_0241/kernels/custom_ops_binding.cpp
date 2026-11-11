#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void elu_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void mean_all_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);

torch::Tensor elu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input must not be empty");

    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    elu_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        x_contig.numel(),
        stream
    );

    return output;
}

torch::Tensor mean_all_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input must not be empty");

    auto x_contig = x.contiguous();
    auto output = torch::empty({}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mean_all_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        x_contig.numel(),
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("elu_forward", &elu_forward, "ELU activation forward",
          pybind11::arg("x"));
    m.def("mean_all_forward", &mean_all_forward, "Mean of all elements forward",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
