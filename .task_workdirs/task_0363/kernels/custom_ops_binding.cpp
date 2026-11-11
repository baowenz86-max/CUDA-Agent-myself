#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void pixel_unshuffle_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W, int ds,
    const int* in_strides,
    const int* out_strides,
    cudaStream_t stream);

torch::Tensor pixel_unshuffle_forward(torch::Tensor input, int64_t downscale_factor) {
    // Check input tensor properties
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(downscale_factor > 0, "downscale_factor must be positive");

    int ds = static_cast<int>(downscale_factor);
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    TORCH_CHECK(H % ds == 0 && W % ds == 0, "H and W must be divisible by downscale_factor");

    // Make input contiguous if needed
    torch::Tensor input_contig = input.contiguous();

    // Allocate output with correct shape
    int64_t out_H = H / ds;
    int64_t out_W = W / ds;
    int64_t out_C = C * ds * ds;
    auto output = torch::empty({N, out_C, out_H, out_W}, input_contig.options());

    // Get strides (in int for launcher, but ensure no overflow for typical sizes)
    int in_strides[4];
    int out_strides[4];
    for (int i = 0; i < 4; ++i) {
        in_strides[i] = static_cast<int>(input_contig.stride(i));
        out_strides[i] = static_cast<int>(output.stride(i));
    }

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    pixel_unshuffle_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        ds,
        in_strides,
        out_strides,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_pixel_unshuffle(pybind11::module& m) {
    m.def("pixel_unshuffle_forward", &pixel_unshuffle_forward,
          "Pixel unshuffle forward (CUDA)",
          py::arg("input"),
          py::arg("downscale_factor"));
}

REGISTER_BINDING(pixel_unshuffle, register_pixel_unshuffle);
