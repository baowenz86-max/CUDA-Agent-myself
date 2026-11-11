#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void pixel_shuffle_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int scale_factor,
    cudaStream_t stream
);

torch::Tensor pixel_shuffle(torch::Tensor x, int64_t scale_factor) {
    // Input validation
    TORCH_CHECK(x.is_cpu(), "pixel_shuffle expects CPU input tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "pixel_shuffle expects float32 input");
    TORCH_CHECK(x.dim() == 4, "pixel_shuffle expects 4D input (N,C,H,W)");
    TORCH_CHECK(x.is_contiguous(), "pixel_shuffle expects contiguous input");
    TORCH_CHECK(scale_factor >= 1, "scale_factor must be >= 1");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int r = static_cast<int>(scale_factor);
    TORCH_CHECK(C % (r * r) == 0, "C must be divisible by scale_factor^2");
    int64_t C_out = C / (r * r);

    // Allocate output on CUDA
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto output = torch::empty({N, C_out, H * r, W * r}, options);

    // Copy input from CPU to CUDA
    auto x_cuda = x.to(torch::kCUDA);
    TORCH_CHECK(x_cuda.is_contiguous(), "CUDA tensor must be contiguous");

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Call launcher with CUDA pointers
    pixel_shuffle_launcher(
        x_cuda.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        r,
        stream
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "pixel_shuffle kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_pixel_shuffle(pybind11::module& m) {
    m.def("pixel_shuffle", &pixel_shuffle, "Pixel shuffle operator (CPU input, CUDA output)",
          pybind11::arg("x"), pybind11::arg("scale_factor"));
}

REGISTER_BINDING(pixel_shuffle, register_pixel_shuffle);
