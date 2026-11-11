#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void glu_forward_launcher(const float* input,
                                     float* output,
                                     int B, int N, int F,
                                     cudaStream_t stream);

torch::Tensor glu_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (batch, 2*n, features)");

    int B = x.size(0);
    int twoN = x.size(1);
    TORCH_CHECK(twoN % 2 == 0, "Input dimension 1 must be even (2*n)");
    int N = twoN / 2;
    int F = x.size(2);

    // Allocate output [B, N, F]
    auto output = torch::empty({B, N, F}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher
    glu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, N, F,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_glu(pybind11::module& m) {
    m.def("glu_forward", &glu_forward, "GLU forward along dimension 1",
          py::arg("x"));
}

REGISTER_BINDING(glu, register_glu);
