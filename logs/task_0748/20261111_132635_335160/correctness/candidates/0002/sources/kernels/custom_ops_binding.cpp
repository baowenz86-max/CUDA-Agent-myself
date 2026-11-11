#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void custom_softmax_launcher(float* out, const float* in, int rows, int cols, cudaStream_t stream);

// Wrapper for custom softmax (column-wise softmax)
torch::Tensor custom_softmax_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    // The kernel only supports dim=0 (column-wise softmax)
    TORCH_CHECK(dim == 0, "Only dim=0 supported");

    int rows = static_cast<int>(input.size(0));
    int cols = static_cast<int>(input.size(1));
    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_softmax_launcher(output.data_ptr<float>(), input.data_ptr<float>(), rows, cols, stream);

    return output;
}

// Wrapper for custom unsqueeze (view)
torch::Tensor custom_unsqueeze_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");

    // Unsqueeze is a view operation; no data copy.
    return torch::unsqueeze(input, dim);
}

// Registration function
void register_custom_ops_binding(pybind11::module& m) {
    m.def("custom_softmax", &custom_softmax_forward, "Column-wise softmax",
          pybind11::arg("input"), pybind11::arg("dim"));
    m.def("custom_unsqueeze", &custom_unsqueeze_forward, "Unsqueeze view",
          pybind11::arg("input"), pybind11::arg("dim"));
}

// Auto-register
REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);
