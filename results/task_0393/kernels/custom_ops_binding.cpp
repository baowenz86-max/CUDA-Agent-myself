#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void nll_loss_forward_launcher(
    const float* log_prob,
    const int64_t* target,
    float* output,
    int batch_size,
    int num_classes,
    cudaStream_t stream);

// PyTorch wrapper for NLL loss forward
torch::Tensor nll_loss_forward(torch::Tensor log_prob, torch::Tensor target) {
    // Check device
    TORCH_CHECK(log_prob.is_cuda(), "log_prob must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");

    // Check dtype
    TORCH_CHECK(log_prob.dtype() == torch::kFloat32, "log_prob must be float32");
    TORCH_CHECK(target.dtype() == torch::kInt64, "target must be int64");

    // Check contiguity
    TORCH_CHECK(log_prob.is_contiguous(), "log_prob must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "target must be contiguous");

    // Check shape and batch consistency
    TORCH_CHECK(log_prob.dim() == 2, "log_prob must be 2D [batch_size, num_classes]");
    TORCH_CHECK(target.dim() == 1, "target must be 1D [batch_size]");
    int64_t batch_size = log_prob.size(0);
    int64_t num_classes = log_prob.size(1);
    TORCH_CHECK(target.size(0) == batch_size, "batch size mismatch between log_prob and target");

    // Create scalar output tensor on same device
    auto output = torch::empty({}, log_prob.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    nll_loss_forward_launcher(
        log_prob.data_ptr<float>(),
        target.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(num_classes),
        stream);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("nll_loss_forward", &nll_loss_forward,
          "NLL loss forward (mean reduction, no weight)",
          py::arg("log_prob"), py::arg("target"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
