#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void gelu_forward_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream);
extern "C" void pixel_shuffle_forward_launcher(float* output, const float* input,
                                                int N, int C_out, int H, int W,
                                                cudaStream_t stream);
extern "C" void upsample_bilinear2x_forward_launcher(float* output, const float* input,
                                                      int N, int C, int H_in, int W_in,
                                                      cudaStream_t stream);

// GELU wrapper
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gelu_forward_launcher(output.data_ptr<float>(), x.data_ptr<float>(), numel, stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "GELU kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// PixelShuffle wrapper
torch::Tensor pixel_shuffle_forward(torch::Tensor x, int64_t upscale_factor) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(upscale_factor == 2, "Only upscale_factor=2 is supported in this kernel");

    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(C % (upscale_factor * upscale_factor) == 0, "C must be divisible by upscale_factor^2");
    int64_t C_out = C / (upscale_factor * upscale_factor);
    int64_t H_out = H * upscale_factor;
    int64_t W_out = W * upscale_factor;

    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    pixel_shuffle_forward_launcher(output.data_ptr<float>(), x.data_ptr<float>(),
                                   (int)N, (int)C_out, (int)H, (int)W, stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "PixelShuffle kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Upsample bilinear 2x wrapper
torch::Tensor upsample_bilinear2x_forward(torch::Tensor x, c10::optional<bool> align_corners = c10::nullopt) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // align_corners=None must be treated as False for bilinear mode
    bool align_corners_val = align_corners.has_value() ? align_corners.value() : false;
    TORCH_CHECK(align_corners_val == false, "Only align_corners=False is supported");

    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);

    int64_t H_out = H_in * 2;
    int64_t W_out = W_in * 2;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    upsample_bilinear2x_forward_launcher(output.data_ptr<float>(), x.data_ptr<float>(),
                                         (int)N, (int)C, (int)H_in, (int)W_in, stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Upsample kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward, "GELU activation forward",
          py::arg("x"));
    m.def("pixel_shuffle_forward", &pixel_shuffle_forward, "PixelShuffle forward (r=2)",
          py::arg("x"), py::arg("upscale_factor"));
    m.def("upsample_bilinear2x_forward", &upsample_bilinear2x_forward, "Bilinear 2x upsample forward",
          py::arg("x"), py::arg("align_corners") = py::none());
}

REGISTER_BINDING(custom_ops, register_custom_ops);
