#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void prelu_forward_launcher(const float* x,
                                      const float* weight,
                                      float* out,
                                      int N, int C, int H, int W,
                                      cudaStream_t stream);

// PyTorch wrapper
torch::Tensor prelu_forward(torch::Tensor x, torch::Tensor weight) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [N, C, H, W]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(weight.size(0) == C, "weight length must equal C (num_channels)");
    TORCH_CHECK(N >= 1 && C >= 1 && H >= 1 && W >= 1, "all dimensions must be positive");
    TORCH_CHECK(N <= INT_MAX && C <= INT_MAX && H <= INT_MAX && W <= INT_MAX,
                "dimensions exceed int range");

    // Handle non-contiguous input by making contiguous if needed (PReLU supports strided?)
    // The launcher assumes contiguous NCHW. Ensure contiguity.
    auto x_contig = x.contiguous();

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    prelu_forward_launcher(
        x_contig.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    // Ensure kernel launch success
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // If input was non-contiguous, output should match original stride? PyTorch contiguous output usually acceptable,
    // but reference may require same stride? The contract does not require stride preservation, only dtype/device/shape.
    // We'll return contiguous output.
    return output;
}

void register_prelu(pybind11::module& m) {
    m.def("prelu_forward", &prelu_forward,
          "PReLU forward (CUDA)",
          py::arg("x"),
          py::arg("weight"));
}

REGISTER_BINDING(prelu, register_prelu);
