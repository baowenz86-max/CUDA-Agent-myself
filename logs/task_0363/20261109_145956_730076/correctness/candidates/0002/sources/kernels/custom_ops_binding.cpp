#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void pixel_unshuffle_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int upscale_factor,
    int64_t in_n_stride,
    int64_t in_c_stride,
    int64_t in_h_stride,
    int64_t in_w_stride,
    cudaStream_t stream
);

torch::Tensor pixel_unshuffle_forward(torch::Tensor x, int64_t upscale_factor) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(upscale_factor > 0, "upscale_factor must be positive");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(H % upscale_factor == 0 && W % upscale_factor == 0,
                "H and W must be divisible by upscale_factor");

    int H_out = H / upscale_factor;
    int W_out = W / upscale_factor;
    int C_out = C * upscale_factor * upscale_factor;

    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());

    // Get strides (int64_t)
    int64_t in_n_stride = x.stride(0);
    int64_t in_c_stride = x.stride(1);
    int64_t in_h_stride = x.stride(2);
    int64_t in_w_stride = x.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    pixel_unshuffle_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        H,
        W,
        static_cast<int>(upscale_factor),
        in_n_stride,
        in_c_stride,
        in_h_stride,
        in_w_stride,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_pixel_unshuffle(pybind11::module& m) {
    m.def("pixel_unshuffle_forward", &pixel_unshuffle_forward,
          "Pixel unshuffle forward",
          pybind11::arg("x"),
          pybind11::arg("upscale_factor"));
}

REGISTER_BINDING(pixel_unshuffle, register_pixel_unshuffle);
