// Include necessary PyTorch headers without torch/extension.h
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_abs_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream);

extern "C" void custom_pad_forward_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// custom_abs_forward: element-wise absolute value
// -----------------------------------------------------------------------------
torch::Tensor custom_abs_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    // Assume contiguous; if not, make contiguous for simple elementwise op
    auto x_contig = x.contiguous();

    // Allocate output with same shape
    auto output = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_abs_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        stream);

    // Check for CUDA errors after launch
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_abs_forward kernel launch failed");

    return output;
}

// -----------------------------------------------------------------------------
// custom_pad_forward: zero padding (constant mode)
// -----------------------------------------------------------------------------
torch::Tensor custom_pad_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "Padding values must be non-negative");

    // Ensure contiguous input
    auto x_contig = x.contiguous();

    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t H = x_contig.size(2);
    int64_t W = x_contig.size(3);

    int64_t H_out = H + pad_top + pad_bottom;
    int64_t W_out = W + pad_left + pad_right;

    // Allocate output
    auto output = torch::empty({N, C, H_out, W_out}, x_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_pad_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        N, C, H, W,
        pad_left, pad_right, pad_top, pad_bottom,
        stream);

    // Check for CUDA errors after launch
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_pad_forward kernel launch failed");

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs_forward", &custom_abs_forward,
          "Element-wise absolute value",
          py::arg("x"));

    m.def("custom_pad_forward", &custom_pad_forward,
          "Constant zero padding",
          py::arg("x"),
          py::arg("pad_left") = 1,
          py::arg("pad_right") = 2,
          py::arg("pad_top") = 3,
          py::arg("pad_bottom") = 4);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
