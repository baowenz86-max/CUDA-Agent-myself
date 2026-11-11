#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_abs_forward_launcher(const float* input, float* output,
                                             long long numel, cudaStream_t stream);

// Launcher for adaptive avg pool
extern "C" void custom_adaptive_avg_pool1d_forward_launcher(
        const float* input, float* output,
        int batch, int channels, int input_width, int output_width,
        int window_size, cudaStream_t stream);

// Element-wise absolute value
torch::Tensor custom_abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    
    // Ensure contiguous
    x = x.contiguous();
    
    auto output = torch::empty_like(x);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    custom_abs_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream
    );
    
    return output;
}

// Adaptive average pooling 1d
torch::Tensor custom_adaptive_avg_pool1d_forward(torch::Tensor x, int64_t output_size) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (batch, channels, width)");
    TORCH_CHECK(output_size > 0, "output_size must be positive");
    
    // Ensure contiguous
    x = x.contiguous();
    
    int batch = x.size(0);
    int channels = x.size(1);
    int input_width = x.size(2);
    
    TORCH_CHECK(input_width % output_size == 0, "input_width must be divisible by output_size");
    int window_size = input_width / output_size;
    
    auto output = torch::empty({batch, channels, output_size}, x.options());
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    custom_adaptive_avg_pool1d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, input_width,
        static_cast<int>(output_size), window_size,
        stream
    );
    
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs_forward", &custom_abs_forward,
          "Element-wise absolute value",
          py::arg("x"));
    m.def("custom_adaptive_avg_pool1d_forward", &custom_adaptive_avg_pool1d_forward,
          "Adaptive average pooling 1d",
          py::arg("x"), py::arg("output_size") = 16);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
