#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_softmax_dim1_launcher(const float* input, float* output,
                                              int N, int H, int W, cudaStream_t stream);

extern "C" void custom_kl_div_batchmean_launcher(const float* x, float* output,
                                                  int num_elements, int N, cudaStream_t stream);

// Custom softmax along dim=1 for 3D tensor [N, H, W]
// Contract: input float32 CUDA tensor, dim=1
torch::Tensor custom_softmax_kv_cache(torch::Tensor input, int64_t dim) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "custom_softmax_kv_cache: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_softmax_kv_cache: input must be float32");
    TORCH_CHECK(input.dim() == 3, "custom_softmax_kv_cache: input must be 3D (N, H, W)");
    TORCH_CHECK(dim == 1, "custom_softmax_kv_cache: only dim=1 is supported");
    TORCH_CHECK(input.is_contiguous(), "custom_softmax_kv_cache: input must be contiguous");

    auto N = input.size(0);
    auto H = input.size(1);
    auto W = input.size(2);

    // Allocate output same shape
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_softmax_dim1_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

// Custom KL divergence batchmean for identical input/target
torch::Tensor custom_kl_div_batchmean(torch::Tensor input, torch::Tensor target) {
    // Validate input and target
    TORCH_CHECK(input.is_cuda(), "custom_kl_div_batchmean: input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "custom_kl_div_batchmean: target must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_kl_div_batchmean: input must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "custom_kl_div_batchmean: target must be float32");
    TORCH_CHECK(input.dim() == 3, "custom_kl_div_batchmean: input must be 3D (N, H, W)");
    TORCH_CHECK(target.dim() == 3, "custom_kl_div_batchmean: target must be 3D (N, H, W)");
    TORCH_CHECK(input.sizes() == target.sizes(), "custom_kl_div_batchmean: input and target must have same shape");
    TORCH_CHECK(input.data_ptr() == target.data_ptr(), "custom_kl_div_batchmean: input and target must be the same tensor");
    TORCH_CHECK(input.is_contiguous(), "custom_kl_div_batchmean: input must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "custom_kl_div_batchmean: target must be contiguous");

    auto N = input.size(0);
    auto num_elements = input.numel();

    // Allocate scalar output (shape [])
    auto output = torch::empty({}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_kl_div_batchmean_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(num_elements),
        static_cast<int>(N),
        stream
    );

    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax_kv_cache", &custom_softmax_kv_cache,
          "Custom softmax along dim=1 for 3D tensor",
          py::arg("input"), py::arg("dim"));
    m.def("custom_kl_div_batchmean", &custom_kl_div_batchmean,
          "Custom KL divergence batchmean for identical input/target",
          py::arg("input"), py::arg("target"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
