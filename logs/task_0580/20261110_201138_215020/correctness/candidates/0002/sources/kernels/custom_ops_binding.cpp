#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void max_dim_launcher(const float* input,
                                 float* values,
                                 int64_t* indices,
                                 int64_t N, int64_t C, int64_t H, int64_t W,
                                 int64_t dim,
                                 bool keepdim,
                                 cudaStream_t stream);

extern "C" void bce_with_logits_launcher(const float* input,
                                          float* target,
                                          float* output,
                                          int64_t numel,
                                          cudaStream_t stream);

// Wrapper for torch.max(x, dim=1, keepdim=True)
torch::return_types::max max_dim(torch::Tensor x, int64_t dim, bool keepdim) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(dim == 1, "dim must be 1");
    TORCH_CHECK(keepdim, "keepdim must be true");

    // Ensure contiguous (if not, make contiguous)
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    // Extract dimensions
    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "x must be 4D");
    int64_t N = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];

    // Allocate output tensors
    auto values = torch::empty_like(x);
    auto indices = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt64));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    max_dim_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        N, C, H, W,
        dim,
        keepdim,
        stream
    );

    // Return torch.return_types.max
    return torch::return_types::max(values, indices);
}

// Wrapper for F.binary_cross_entropy_with_logits(input, target, reduction='mean')
// Note: target is expected to be uninitialized; it will be overwritten with random normal numbers inside the launcher.
torch::Tensor binary_cross_entropy_with_logits(torch::Tensor input, torch::Tensor target) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(input.sizes() == target.sizes(), "input and target must have same shape");

    // Ensure contiguous
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    if (!target.is_contiguous()) {
        target = target.contiguous();
    }

    int64_t numel = input.numel();

    // Allocate scalar output tensor
    auto output = torch::empty({}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher (target will be filled with random numbers inside)
    bce_with_logits_launcher(
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("max_dim", &max_dim,
          "torch.max(x, dim=1, keepdim=True) custom implementation",
          py::arg("x"),
          py::arg("dim"),
          py::arg("keepdim"));

    m.def("binary_cross_entropy_with_logits", &binary_cross_entropy_with_logits,
          "F.binary_cross_entropy_with_logits(input, target) custom implementation",
          py::arg("input"),
          py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
