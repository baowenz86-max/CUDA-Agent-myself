#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void pixel_unshuffle_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W, int r,
    long long total_elements,
    cudaStream_t stream
);

torch::Tensor pixel_unshuffle_forward(torch::Tensor x, int64_t downscale_factor) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t r = downscale_factor;

    TORCH_CHECK(r > 0, "downscale_factor must be positive");
    TORCH_CHECK(H % r == 0 && W % r == 0, "H and W must be divisible by downscale_factor");

    // Make input contiguous if necessary (we rely on contiguous memory in the kernel)
    auto x_contig = x.contiguous();

    // Allocate output tensor with required shape
    int64_t H_out = H / r;
    int64_t W_out = W / r;
    int64_t C_out = C * r * r;
    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());

    long long total_elements = output.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    pixel_unshuffle_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(r),
        total_elements,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_pixel_unshuffle(pybind11::module& m) {
    m.def("pixel_unshuffle_forward", &pixel_unshuffle_forward,
          "PixelUnshuffle forward (CUDA)",
          py::arg("x"), py::arg("downscale_factor"));
}

REGISTER_BINDING(pixel_unshuffle, register_pixel_unshuffle);
