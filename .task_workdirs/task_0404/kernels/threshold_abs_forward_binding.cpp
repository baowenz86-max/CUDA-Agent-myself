#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void threshold_abs_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    double threshold,
    cudaStream_t stream);

torch::Tensor threshold_abs_forward(torch::Tensor x, double threshold) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");

    // Ensure contiguous layout to simplify kernel indexing
    x = x.contiguous();

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    threshold_abs_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        threshold,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_threshold_abs_forward(pybind11::module& m) {
    m.def("threshold_abs_forward", &threshold_abs_forward,
          "Fused abs and threshold forward (CUDA)",
          py::arg("x"), py::arg("threshold"));
}

REGISTER_BINDING(threshold_abs_forward, register_threshold_abs_forward);
