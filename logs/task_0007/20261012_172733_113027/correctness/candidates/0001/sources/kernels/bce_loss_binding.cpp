// Binding for bce_with_logits_loss: mean reduction
// Only float32, contiguous, CUDA tensors supported.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declaration of launcher from bce_loss_kernel.cu
extern "C" float bce_with_logits_loss_launcher(
    const float* logits,
    const float* targets,
    int numel,
    cudaStream_t stream
);

torch::Tensor bce_with_logits_loss(torch::Tensor logits, torch::Tensor target) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
    TORCH_CHECK(logits.sizes() == target.sizes(), "logits and target shapes must match");
    TORCH_CHECK(logits.numel() > 0, "input must have at least one element");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    float loss = bce_with_logits_loss_launcher(
        logits.data_ptr<float>(),
        target.data_ptr<float>(),
        logits.numel(),
        stream
    );

    // Create a scalar tensor on the same device (CPU? Actually loss is computed on CPU, but we need a CUDA scalar tensor)
    // Since the launcher returns a float after synchronous copy, we can create a CUDA tensor and set its value.
    // Simpler: create a CPU scalar tensor and move to CUDA? But model_new.py expects a CUDA scalar tensor (as in reference).
    // We'll create a CUDA tensor with one element and copy the value using cudaMemcpy.
    auto output = torch::empty({}, logits.options());
    cudaMemcpy(output.data_ptr<float>(), &loss, sizeof(float), cudaMemcpyHostToDevice);
    // The above copy is async? Actually cudaMemcpy defaults to synchronous, but we can use stream version if needed.
    // Since stream may be default, cudaMemcpy without stream is synchronous, fine.
    return output;
}

void register_bce_loss(py::module& m) {
    m.def("bce_with_logits_loss", &bce_with_logits_loss, "Binary cross entropy with logits loss",
          py::arg("logits"), py::arg("target"));
}

REGISTER_BINDING(bce_loss, register_bce_loss);
