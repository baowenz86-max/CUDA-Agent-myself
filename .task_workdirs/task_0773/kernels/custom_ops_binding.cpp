#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void stack_dim0_launcher(const float* x, const float* y, float* output,
                                    int64_t outer_size, int64_t inner_size, cudaStream_t stream);
extern "C" void all_dim0_launcher(const float* input, bool* output, int64_t inner_size, cudaStream_t stream);

// stack_dim0: stack two tensors along dim=0 (specialized)
torch::Tensor stack_dim0(torch::Tensor x, torch::Tensor y, int64_t dim) {
    TORCH_CHECK(dim == 0, "stack_dim0 only supports dim=0");
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have same shape");

    auto sizes = x.sizes().vec();
    sizes.insert(sizes.begin(), 2); // output shape [2, ...]
    auto output = torch::empty(sizes, x.options());

    int64_t inner_size = x.numel();
    int64_t outer_size = 1; // dim=0

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    stack_dim0_launcher(x.data_ptr<float>(), y.data_ptr<float>(),
                        output.data_ptr<float>(), outer_size, inner_size, stream);

    return output;
}

// all_dim0: logical and along dim=0 (specialized)
torch::Tensor all_dim0(torch::Tensor stacked_tensors, int64_t dim) {
    TORCH_CHECK(dim == 0, "all_dim0 only supports dim=0");
    TORCH_CHECK(stacked_tensors.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(stacked_tensors.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(stacked_tensors.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(stacked_tensors.dim() >= 1 && stacked_tensors.size(0) == 2, "stacked_tensors must have first dimension 2");

    // output shape: remove first dimension
    auto out_sizes = stacked_tensors.sizes().vec();
    out_sizes.erase(out_sizes.begin());
    auto output = torch::empty(out_sizes, stacked_tensors.options().dtype(torch::kBool));

    int64_t inner_size = output.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    all_dim0_launcher(stacked_tensors.data_ptr<float>(), output.data_ptr<bool>(),
                      inner_size, stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("stack_dim0", &stack_dim0, "Stack two tensors along dim=0",
          py::arg("x"), py::arg("y"), py::arg("dim"));
    m.def("all_dim0", &all_dim0, "Logical and along dim=0",
          py::arg("stacked_tensors"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
