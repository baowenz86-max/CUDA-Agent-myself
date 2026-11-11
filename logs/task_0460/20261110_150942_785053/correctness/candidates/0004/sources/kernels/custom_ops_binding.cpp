#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void unbind_dim1_launcher(const float* x, int64_t B, int64_t C, int64_t H, int64_t W, cudaStream_t stream);
extern "C" void identity_copy_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void stack_dim1_launcher(const float* const* input_ptrs, int64_t B, int64_t C, int64_t H, int64_t W, float* output, cudaStream_t stream);

// Helper to check CUDA tensor requirements
void check_cuda_float32(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

// unbind_dim1: split tensor along dim=1 into tuple of views
py::tuple unbind_dim1(torch::Tensor x) {
    check_cuda_float32(x, "x");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    int64_t C = x.size(1);
    // Calling no-op launcher for side effects (none)
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    unbind_dim1_launcher(x.data_ptr<float>(), x.size(0), C, x.size(2), x.size(3), stream);

    // Create views using select to remove dimension 1, resulting in 3D tensors with stride [524288, 128, 1]
    py::tuple result(C);
    for (int64_t i = 0; i < C; ++i) {
        result[i] = x.select(1, i);
    }
    return result;
}

// identity_new_tensor: copy input to a new contiguous tensor
// Input may be non-contiguous (supports strides). Equivalent to pixel_shuffle factor=1 which returns contiguous tensor.
torch::Tensor identity_new_tensor(torch::Tensor x) {
    check_cuda_float32(x, "x");
    TORCH_CHECK(x.dim() == 3, "x must be 3D ([B,H,W])");
    // Make input contiguous; this creates a new contiguous tensor if needed.
    // This matches the reference behavior where pixel_shuffle returns a new contiguous tensor.
    torch::Tensor x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    identity_copy_launcher(x_contig.data_ptr<float>(), output.data_ptr<float>(), x_contig.numel(), stream);
    return output;
}

// stack_dim1: stack list of 3D tensors along new dim=1
torch::Tensor stack_dim1(std::vector<torch::Tensor> tensors) {
    TORCH_CHECK(!tensors.empty(), "tensors list must not be empty");
    int64_t C = static_cast<int64_t>(tensors.size());
    const auto& first = tensors[0];
    check_cuda_float32(first, "tensors[0]");
    TORCH_CHECK(first.dim() == 3, "each tensor must be 3D ([B,H,W])");
    int64_t B = first.size(0);
    int64_t H = first.size(1);
    int64_t W = first.size(2);

    // Validate all tensors
    for (const auto& t : tensors) {
        check_cuda_float32(t, "tensor in list");
        TORCH_CHECK(t.dim() == 3, "each tensor must be 3D");
        TORCH_CHECK(t.size(0) == B && t.size(1) == H && t.size(2) == W, "all tensors must have same shape");
        // Inputs to stack are expected to be contiguous (after identity_new_tensor).
        TORCH_CHECK(t.is_contiguous(), "each tensor in stack must be contiguous");
    }

    // Allocate output [B, C, H, W]
    auto output = torch::empty({B, C, H, W}, first.options());

    // Build host array of device pointers
    std::vector<const float*> host_ptrs;
    host_ptrs.reserve(C);
    for (const auto& t : tensors) {
        host_ptrs.push_back(t.data_ptr<float>());
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    stack_dim1_launcher(host_ptrs.data(), B, C, H, W, output.data_ptr<float>(), stream);

    return output;
}

// Register all functions
void register_custom_ops(pybind11::module& m) {
    m.def("unbind_dim1", &unbind_dim1, "Split tensor along dim=1 into tuple of views", py::arg("x"));
    m.def("identity_new_tensor", &identity_new_tensor, "Copy tensor to a new contiguous tensor", py::arg("x"));
    m.def("stack_dim1", &stack_dim1, "Stack list of 3D tensors along dim=1", py::arg("tensors"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
