#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void pixel_unshuffle_custom_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W, int r,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

torch::Tensor pixel_unshuffle_custom(torch::Tensor x, int64_t downscale_factor) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    int r = static_cast<int>(downscale_factor);
    TORCH_CHECK(r > 0, "downscale_factor must be positive");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(H % r == 0 && W % r == 0, "H and W must be divisible by downscale_factor");

    int64_t out_H = H / r;
    int64_t out_W = W / r;
    int64_t out_C = C * r * r;

    // Allocate output tensor (contiguous)
    auto output = torch::empty({N, out_C, out_H, out_W}, x.options());

    // Get strides
    auto in_strides = x.strides();
    auto out_strides = output.strides();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher; note launcher expects int sizes but input may exceed int range? For safety use int64_t passing as int is okay as long as dimensions fit in int (which they do here)
    pixel_unshuffle_custom_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        r,
        static_cast<long long>(in_strides[0]),
        static_cast<long long>(in_strides[1]),
        static_cast<long long>(in_strides[2]),
        static_cast<long long>(in_strides[3]),
        static_cast<long long>(out_strides[0]),
        static_cast<long long>(out_strides[1]),
        static_cast<long long>(out_strides[2]),
        static_cast<long long>(out_strides[3]),
        stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_pixel_unshuffle(pybind11::module& m) {
    m.def("pixel_unshuffle_custom", &pixel_unshuffle_custom,
          "Pixel unshuffle custom op",
          pybind11::arg("x"), pybind11::arg("downscale_factor"));
}

REGISTER_BINDING(pixel_unshuffle, register_pixel_unshuffle);
