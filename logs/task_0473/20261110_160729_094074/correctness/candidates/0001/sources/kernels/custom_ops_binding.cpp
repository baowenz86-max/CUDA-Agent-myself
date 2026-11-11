#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void hardtanh_forward_launcher(const float* input,
                                          float* output,
                                          int64_t numel,
                                          double min_val,
                                          double max_val,
                                          cudaStream_t stream);

torch::Tensor hardtanh_forward(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");

    // Ensure input is contiguous; if not, make a contiguous copy.
    auto x_contig = x.is_contiguous() ? x : x.contiguous();

    // Allocate output with same shape and options; use contiguous memory.
    auto output = torch::empty_like(x_contig, x_contig.options());

    int64_t numel = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    hardtanh_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        min_val,
        max_val,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_hardtanh_forward(pybind11::module& m) {
    m.def("hardtanh_forward", &hardtanh_forward,
          "Hardtanh forward (elementwise clamp)",
          py::arg("x"), py::arg("min_val"), py::arg("max_val"));
}

REGISTER_BINDING(hardtanh_forward, register_hardtanh_forward);