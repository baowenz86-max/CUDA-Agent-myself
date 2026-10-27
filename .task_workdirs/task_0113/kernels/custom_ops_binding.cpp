#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void argmax_dim1_launcher(
    const float* input,
    int64_t* output,
    int64_t N,
    int C,
    int64_t H,
    int64_t W,
    cudaStream_t stream
);

// PyTorch wrapper for argmax along dim=1
torch::Tensor argmax_dim1_forward(torch::Tensor x) {
    // Checks
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    // Ensure contiguous for simple indexing in kernel
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    // Allocate output tensor of shape [N, H, W] and dtype int64
    auto output = torch::empty({N, H, W}, x.options().dtype(torch::kInt64));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    argmax_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        N,
        static_cast<int>(C),
        H,
        W,
        stream
    );

    return output;
}

// Registration function
void register_argmax_dim1(pybind11::module& m) {
    m.def("argmax_dim1_forward", &argmax_dim1_forward,
          "Compute argmax along dim=1 for 4D input (N, C, H, W)",
          pybind11::arg("x"));
}

// Auto-register
REGISTER_BINDING(argmax_dim1, register_argmax_dim1);
