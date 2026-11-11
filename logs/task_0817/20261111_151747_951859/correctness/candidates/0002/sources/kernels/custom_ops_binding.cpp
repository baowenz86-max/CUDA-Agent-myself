#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
void cos_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);

torch::Tensor cos_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cos_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

void register_cos_forward(pybind11::module& m) {
    m.def("cos_forward", &cos_forward, "Elementwise cosine of a float32 CUDA tensor",
          py::arg("input"));
}

REGISTER_BINDING(cos_forward, register_cos_forward);
