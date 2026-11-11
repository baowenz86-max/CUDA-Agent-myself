#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void cat_maximum_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int in_channels,
    int height,
    int width,
    int64_t in_n_stride,
    int64_t in_c_stride,
    int64_t in_h_stride,
    int64_t in_w_stride,
    int64_t out_n_stride,
    int64_t out_c_stride,
    int64_t out_h_stride,
    int64_t out_w_stride,
    cudaStream_t stream);

torch::Tensor cat_maximum_forward(torch::Tensor x) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int batch = x.size(0);
    int in_channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    // Output shape: [batch, 2*in_channels, height, width]
    auto out_sizes = {batch, 2 * in_channels, height, width};
    auto output = torch::empty(out_sizes, x.options());

    // Get strides
    int64_t in_n_stride = x.stride(0);
    int64_t in_c_stride = x.stride(1);
    int64_t in_h_stride = x.stride(2);
    int64_t in_w_stride = x.stride(3);

    int64_t out_n_stride = output.stride(0);
    int64_t out_c_stride = output.stride(1);
    int64_t out_h_stride = output.stride(2);
    int64_t out_w_stride = output.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    cat_maximum_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch,
        in_channels,
        height,
        width,
        in_n_stride,
        in_c_stride,
        in_h_stride,
        in_w_stride,
        out_n_stride,
        out_c_stride,
        out_h_stride,
        out_w_stride,
        stream);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

void register_cat_maximum(pybind11::module& m) {
    m.def("cat_maximum_forward", &cat_maximum_forward,
          "Concatenate input tensor with itself along dim=1 and return elementwise maximum (identity)",
          py::arg("x"));
}

REGISTER_BINDING(cat_maximum, register_cat_maximum);
