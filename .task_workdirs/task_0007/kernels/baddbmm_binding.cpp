// Binding for baddbmm_forward: out = beta * input + alpha * (batch1 @ batch2)
// In model_new.py, beta=0, alpha=1, input is zeros_like(target), so effectively out = batch1 @ batch2.
// Only float32, contiguous, CUDA tensors supported.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declaration of launcher from baddbmm_kernel.cu
extern "C" void baddbmm_launcher(
    float* output,
    const float* input,
    const float* batch1,
    const float* batch2,
    float beta,
    float alpha,
    int B,
    int N,
    int M,
    int P,
    cudaStream_t stream
);

torch::Tensor baddbmm_forward(torch::Tensor input, torch::Tensor batch1, torch::Tensor batch2,
                              double beta = 0.0, double alpha = 1.0) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(batch1.is_cuda(), "batch1 must be a CUDA tensor");
    TORCH_CHECK(batch2.is_cuda(), "batch2 must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(batch1.dtype() == torch::kFloat32, "batch1 must be float32");
    TORCH_CHECK(batch2.dtype() == torch::kFloat32, "batch2 must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(batch1.is_contiguous(), "batch1 must be contiguous");
    TORCH_CHECK(batch2.is_contiguous(), "batch2 must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B, N, P)");
    TORCH_CHECK(batch1.dim() == 3, "batch1 must be 3D (B, N, M)");
    TORCH_CHECK(batch2.dim() == 3, "batch2 must be 3D (B, M, P)");
    TORCH_CHECK(batch1.size(0) == batch2.size(0), "batch size mismatch");
    TORCH_CHECK(batch1.size(2) == batch2.size(1), "inner dimension mismatch");
    TORCH_CHECK(input.size(0) == batch1.size(0), "input batch size mismatch");
    TORCH_CHECK(input.size(1) == batch1.size(1), "input N mismatch");
    TORCH_CHECK(input.size(2) == batch2.size(2), "input P mismatch");

    int B = batch1.size(0);
    int N = batch1.size(1);
    int M = batch1.size(2);
    int P = batch2.size(2);

    auto output = torch::empty_like(input); // allocate output; will be overwritten by kernel

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    baddbmm_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(), // may be ignored if beta=0
        batch1.data_ptr<float>(),
        batch2.data_ptr<float>(),
        (float)beta, (float)alpha,
        B, N, M, P,
        stream
    );

    return output;
}

void register_baddbmm(py::module& m) {
    m.def("baddbmm_forward", &baddbmm_forward, "Batched matrix multiply with add",
          py::arg("input"), py::arg("batch1"), py::arg("batch2"),
          py::arg("beta") = 0.0, py::arg("alpha") = 1.0);
}

REGISTER_BINDING(baddbmm, register_baddbmm);
