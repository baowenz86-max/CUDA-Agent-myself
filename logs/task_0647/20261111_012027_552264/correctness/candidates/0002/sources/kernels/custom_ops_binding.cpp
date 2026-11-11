#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_unfold_launcher(const float* input,
                                      float* output,
                                      int64_t batch,
                                      int64_t in_channels,
                                      int64_t height,
                                      int64_t width,
                                      int64_t kernel_h,
                                      int64_t kernel_w,
                                      cudaStream_t stream);

extern "C" void custom_abs_launcher(const float* input,
                                    float* output,
                                    int64_t total,
                                    cudaStream_t stream);

extern "C" void custom_unfold_abs_launcher(const float* input,
                                           float* output,
                                           int64_t batch,
                                           int64_t in_channels,
                                           int64_t height,
                                           int64_t width,
                                           int64_t kernel_h,
                                           int64_t kernel_w,
                                           cudaStream_t stream);

// Helper to validate tensor requirements
static void check_cuda_float_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// custom_unfold: x -> [batch, in_channels*kernel_h*kernel_w, 1]
torch::Tensor custom_unfold(torch::Tensor x, int64_t kernel_h, int64_t kernel_w) {
    check_cuda_float_contiguous(x, "x");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (batch, channels, height, width)");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "kernel sizes must be positive");

    int64_t batch = x.size(0);
    int64_t in_channels = x.size(1);
    int64_t height = x.size(2);
    int64_t width = x.size(3);
    TORCH_CHECK(kernel_h <= height && kernel_w <= width, "kernel size must not exceed input spatial dims");

    int64_t out_channels = in_channels * kernel_h * kernel_w;
    auto output = torch::empty({batch, out_channels, 1}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_unfold_launcher(x.data_ptr<float>(),
                           output.data_ptr<float>(),
                           batch,
                           in_channels,
                           height,
                           width,
                           kernel_h,
                           kernel_w,
                           stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_unfold kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// custom_abs: elementwise absolute value
torch::Tensor custom_abs(torch::Tensor x) {
    check_cuda_float_contiguous(x, "x");
    auto output = torch::empty_like(x);
    int64_t total = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_abs_launcher(x.data_ptr<float>(),
                        output.data_ptr<float>(),
                        total,
                        stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_abs kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// custom_unfold_abs: fused unfold + abs, same output shape as custom_unfold
torch::Tensor custom_unfold_abs(torch::Tensor x, int64_t kernel_h, int64_t kernel_w) {
    check_cuda_float_contiguous(x, "x");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (batch, channels, height, width)");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "kernel sizes must be positive");

    int64_t batch = x.size(0);
    int64_t in_channels = x.size(1);
    int64_t height = x.size(2);
    int64_t width = x.size(3);
    TORCH_CHECK(kernel_h <= height && kernel_w <= width, "kernel size must not exceed input spatial dims");

    int64_t out_channels = in_channels * kernel_h * kernel_w;
    auto output = torch::empty({batch, out_channels, 1}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_unfold_abs_launcher(x.data_ptr<float>(),
                               output.data_ptr<float>(),
                               batch,
                               in_channels,
                               height,
                               width,
                               kernel_h,
                               kernel_w,
                               stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_unfold_abs kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_unfold", &custom_unfold, "custom unfold operation",
          py::arg("x"), py::arg("kernel_h"), py::arg("kernel_w"));
    m.def("custom_abs", &custom_abs, "custom abs operation",
          py::arg("x"));
    m.def("custom_unfold_abs", &custom_unfold_abs, "fused unfold + abs operation",
          py::arg("x"), py::arg("kernel_h"), py::arg("kernel_w"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
