#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void adaptive_avg_pool3d_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w,
    int O,
    int out_stride_n, int out_stride_c, int out_stride_d, int out_stride_h, int out_stride_w,
    cudaStream_t stream);

// PyTorch binding function
torch::Tensor adaptive_avg_pool3d_2(torch::Tensor x, int64_t output_size) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (N,C,D,H,W)");
    
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int O = static_cast<int>(output_size);
    TORCH_CHECK(O > 0, "output_size must be positive");
    
    // Allocate contiguous output tensor
    auto output = torch::empty({N, C, O, O, O}, x.options());
    
    // Obtain current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Input strides (x may be non-contiguous)
    int in_stride_n = x.stride(0);
    int in_stride_c = x.stride(1);
    int in_stride_d = x.stride(2);
    int in_stride_h = x.stride(3);
    int in_stride_w = x.stride(4);
    
    // Output strides (contiguous)
    int out_stride_n = output.stride(0);
    int out_stride_c = output.stride(1);
    int out_stride_d = output.stride(2);
    int out_stride_h = output.stride(3);
    int out_stride_w = output.stride(4);
    
    // Call launcher
    adaptive_avg_pool3d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
        O,
        out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        stream);
    
    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// Registration function
void register_custom_ops_binding(pybind11::module& m) {
    m.def("adaptive_avg_pool3d_2", &adaptive_avg_pool3d_2,
          "Adaptive average pooling 3D to output_size (O,O,O)",
          pybind11::arg("x"), pybind11::arg("output_size"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops_binding);
