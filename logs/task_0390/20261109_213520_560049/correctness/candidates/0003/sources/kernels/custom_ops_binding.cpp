#include <torch/types.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launcher from custom_ops.cu
extern "C" void channel_shuffle_forward_launcher(
    const float* x,
    float* output,
    int N, int C, int H, int W, int groups,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    cudaStream_t stream);

// PyTorch wrapper
torch::Tensor channel_shuffle_forward(torch::Tensor x, int64_t groups) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(groups > 0, "groups must be positive");
    int C = x.size(1);
    TORCH_CHECK(C % groups == 0, "C must be divisible by groups");

    // Retrieve dimensions
    int N = x.size(0);
    int H = x.size(2);
    int W = x.size(3);

    // Get input strides in elements (not bytes)
    auto strides = x.strides(); // in elements
    long long stride_n = strides[0];
    long long stride_c = strides[1];
    long long stride_h = strides[2];
    long long stride_w = strides[3];

    // Output tensor: same shape as input, contiguous
    auto output = torch::empty(x.sizes(), x.options().memory_format(torch::MemoryFormat::Contiguous));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    channel_shuffle_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, (int)groups,
        stride_n, stride_c, stride_h, stride_w,
        stream
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_channel_shuffle(pybind11::module& m) {
    m.def("channel_shuffle_forward", &channel_shuffle_forward,
          "Channel shuffle forward",
          py::arg("x"),
          py::arg("groups"));
}

// Auto-register
REGISTER_BINDING(channel_shuffle, register_channel_shuffle);
