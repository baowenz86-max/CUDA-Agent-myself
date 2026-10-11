#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from group_norm_kernel.cu
extern "C" void group_norm_launcher(
    const float* input,
    float* output,
    int N, int G, int Cg, int H, int W, int C,
    float eps,
    cudaStream_t stream);

// PyTorch wrapper for group_norm
torch::Tensor group_norm_forward(torch::Tensor input, int64_t num_groups) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, H, C, W) from transpose");

    // Get dimensions
    int N = input.size(0);
    int H = input.size(1);
    int C = input.size(2);
    int W = input.size(3);

    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    int G = static_cast<int>(num_groups);
    int Cg = C / G;

    // Fixed epsilon as used in PyTorch GroupNorm
    const float eps = 1e-5f;

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    group_norm_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, G, Cg, H, W, C,
        eps,
        stream);

    return output;
}

// Registration function
void register_group_norm(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward, "GroupNorm forward",
          py::arg("input"), py::arg("num_groups"));
}

REGISTER_BINDING(group_norm, register_group_norm);
