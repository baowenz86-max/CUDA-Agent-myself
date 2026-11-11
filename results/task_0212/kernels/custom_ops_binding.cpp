#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void custom_l1_loss_mean_launcher(const float* input,
                                             const float* target,
                                             float* output,
                                             int64_t numel,
                                             cudaStream_t stream);

torch::Tensor custom_l1_loss_mean(torch::Tensor input, torch::Tensor target) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(target.scalar_type() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(input.sizes() == target.sizes(), "input and target must have the same shape");
    TORCH_CHECK(input.device() == target.device(), "input and target must be on same device");

    // Make contiguous if necessary (contract supports strided but kernel expects contiguous)
    auto input_contig = input.contiguous();
    auto target_contig = target.contiguous();

    // Get number of elements
    int64_t numel = input_contig.numel();

    // Allocate scalar output tensor
    auto output = torch::empty({}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_l1_loss_mean_launcher(
        input_contig.data_ptr<float>(),
        target_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    // Check for kernel launch errors (does not guarantee kernel completion, but catches launch failures)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_l1_loss_mean kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_l1_loss_mean(pybind11::module& m) {
    m.def("custom_l1_loss_mean", &custom_l1_loss_mean,
          "Compute mean L1 loss (mean absolute difference) between two CUDA tensors",
          pybind11::arg("input"),
          pybind11::arg("target"));
}

REGISTER_BINDING(custom_l1_loss_mean, register_custom_l1_loss_mean);
