#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_softmax_launcher(
    float* input,
    float* output,
    int dim,
    const int* shape,
    const int* strides,
    int rank,
    cudaStream_t stream);

extern "C" void custom_pixel_shuffle_launcher(
    float* input,
    float* output,
    int N,
    int C_in,
    int H,
    int W,
    int r,
    const int* in_strides,
    const int* out_strides,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// custom_softmax binding
// -----------------------------------------------------------------------------
torch::Tensor custom_softmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "custom_softmax: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_softmax: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_softmax: only 4D tensors are supported");
    TORCH_CHECK(dim >= 0 && dim <= 3, "custom_softmax: dim must be in [0, 3]");

    // Ensure contiguous input to match launcher's assumptions about strides
    auto x_contig = x.contiguous();

    // Create output with same shape as input, contiguous
    auto output = torch::empty_like(x_contig);

    // Prepare shape and strides arrays
    int shape[4];
    int strides[4];
    for (int i = 0; i < 4; ++i) {
        shape[i] = static_cast<int>(x_contig.size(i));
        strides[i] = static_cast<int>(x_contig.stride(i));
    }

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    custom_softmax_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(dim),
        shape,
        strides,
        4,
        stream);

    // Check for errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_softmax: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// custom_pixel_shuffle binding
// -----------------------------------------------------------------------------
torch::Tensor custom_pixel_shuffle_forward(torch::Tensor x, int64_t upscale_factor) {
    TORCH_CHECK(x.is_cuda(), "custom_pixel_shuffle: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_pixel_shuffle: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_pixel_shuffle: only 4D tensors are supported");

    int r = static_cast<int>(upscale_factor);
    TORCH_CHECK(r > 0, "custom_pixel_shuffle: upscale_factor must be positive");

    // Ensure contiguous input
    auto x_contig = x.contiguous();

    int N = static_cast<int>(x_contig.size(0));
    int C_in_times_r_sq = static_cast<int>(x_contig.size(1));
    int H = static_cast<int>(x_contig.size(2));
    int W = static_cast<int>(x_contig.size(3));

    TORCH_CHECK(C_in_times_r_sq % (r * r) == 0, "custom_pixel_shuffle: channel dimension must be divisible by r^2");
    int C_in = C_in_times_r_sq / (r * r);

    // Allocate output contiguous: [N, C_in, H*r, W*r]
    auto output = torch::empty({N, C_in, H * r, W * r}, x_contig.options());

    // Prepare strides arrays for input and output (contiguous)
    int in_strides[4];
    int out_strides[4];
    for (int i = 0; i < 4; ++i) {
        in_strides[i] = static_cast<int>(x_contig.stride(i));
        out_strides[i] = static_cast<int>(output.stride(i));
    }

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    custom_pixel_shuffle_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C_in,
        H,
        W,
        r,
        in_strides,
        out_strides,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_pixel_shuffle: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax", &custom_softmax_forward,
          "Compute softmax along a dimension",
          py::arg("x"), py::arg("dim"));
    m.def("custom_pixel_shuffle", &custom_pixel_shuffle_forward,
          "PixelShuffle operation",
          py::arg("x"), py::arg("upscale_factor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
