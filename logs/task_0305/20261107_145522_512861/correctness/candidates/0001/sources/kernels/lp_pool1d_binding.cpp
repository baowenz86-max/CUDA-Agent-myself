#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void lp_pool1d_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int length,
    int output_length,
    int kernel_size,
    int stride,
    cudaStream_t stream);

// PyTorch wrapper
static torch::Tensor lp_pool1d(
    torch::Tensor x,
    int64_t norm_type,
    int64_t kernel_size,
    int64_t stride) {

    // Parameter validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3D [batch, channels, length]");
    TORCH_CHECK(norm_type == 2, "Only norm_type=2 is supported");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");

    int batch = x.size(0);
    int channels = x.size(1);
    int length = x.size(2);

    // Output length calculation (matching PyTorch LPPool1d with no padding, ceil_mode=False)
    int output_length = (length - kernel_size) / stride + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    // Make input contiguous if needed (we'll pass a contiguous view to launcher)
    auto x_contig = x.contiguous();

    // Allocate output tensor
    auto output = torch::empty({batch, channels, output_length}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    lp_pool1d_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        length,
        output_length,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Register binding
static void register_lp_pool1d(pybind11::module& m) {
    m.def("lp_pool1d", &lp_pool1d,
          "LPPool1d forward (norm_type=2, Euclidean norm)",
          pybind11::arg("x"),
          pybind11::arg("norm_type"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"));
}

REGISTER_BINDING(lp_pool1d, register_lp_pool1d);
