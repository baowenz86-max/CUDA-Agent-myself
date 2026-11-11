#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void fused_abs_sign_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream);

// PyTorch wrapper for fused abs-sign operation
torch::Tensor fused_abs_sign_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Ensure contiguous for direct data_ptr usage (preserves values and shape)
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    int64_t numel = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    fused_abs_sign_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        numel,
        stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_fused_abs_sign(pybind11::module& m) {
    m.def("fused_abs_sign_forward", &fused_abs_sign_forward,
          "Fused elementwise abs then sign",
          py::arg("x"));
}

REGISTER_BINDING(fused_abs_sign, register_fused_abs_sign);
