#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void hardshrink_launcher(float* x, float lambd, int total_elements, cudaStream_t stream);
extern "C" void max_pool2d_launcher(const float* x, float* out,
                                    int N, int C, int H_in, int W_in,
                                    int kernel_h, int kernel_w, int stride_h, int stride_w,
                                    int pad_h, int pad_w, int H_out, int W_out,
                                    cudaStream_t stream);

// Hardshrink forward wrapper
torch::Tensor hardshrink_forward(torch::Tensor x, double lambd) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");
    
    // Ensure contiguous; if not, make a copy
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call launcher (in-place operation on x)
    hardshrink_launcher(x.data_ptr<float>(), static_cast<float>(lambd), x.numel(), stream);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in hardshrink_launcher: ", cudaGetErrorString(err));
    
    return x;
}

// MaxPool2d forward wrapper
torch::Tensor max_pool2d_forward(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N,C,H,W)");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 && x.size(3) > 0, "Input dimensions must be positive");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    
    // Contiguous check (launcher assumes contiguous input)
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;
    
    TORCH_CHECK(H_out > 0 && W_out > 0, "Output dimensions must be positive");
    
    // Allocate output tensor
    auto output = torch::empty({N, C, H_out, W_out}, x.options());
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call launcher
    max_pool2d_launcher(x.data_ptr<float>(), output.data_ptr<float>(),
                        N, C, H_in, W_in,
                        static_cast<int>(kernel_size), static_cast<int>(kernel_size),
                        static_cast<int>(stride), static_cast<int>(stride),
                        static_cast<int>(padding), static_cast<int>(padding),
                        H_out, W_out,
                        stream);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in max_pool2d_launcher: ", cudaGetErrorString(err));
    
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("hardshrink_forward", &hardshrink_forward,
          "HardShrink forward (CUDA)",
          py::arg("x"), py::arg("lambd"));
    m.def("max_pool2d_forward", &max_pool2d_forward,
          "MaxPool2d forward (CUDA)",
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
