#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void pixel_shuffle_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int height,
    int width,
    int upscale_factor,
    const long long* input_strides,
    cudaStream_t stream);

torch::Tensor pixel_shuffle_forward(torch::Tensor x, int64_t upscale_factor) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B, C*r^2, H, W)");
    TORCH_CHECK(upscale_factor > 0, "upscale_factor must be positive");

    int64_t r = upscale_factor;
    int64_t B = x.size(0);
    int64_t C_in = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(C_in % (r * r) == 0, "Input channel dimension must be divisible by r^2");

    int64_t C = C_in / (r * r);
    int64_t H_out = H * r;
    int64_t W_out = W * r;

    auto output = torch::empty({B, C, H_out, W_out}, x.options());

    // Extract input strides in elements (not bytes) as long long array
    long long input_strides[4];
    for (int i = 0; i < 4; ++i) {
        input_strides[i] = static_cast<long long>(x.stride(i));
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    pixel_shuffle_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(r),
        input_strides,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("pixel_shuffle_forward", &pixel_shuffle_forward,
          "PixelShuffle forward",
          pybind11::arg("x"),
          pybind11::arg("upscale_factor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
