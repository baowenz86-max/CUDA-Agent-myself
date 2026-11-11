#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void abs_forward_launcher(const float* input,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream);

extern "C" void pixel_shuffle_forward_launcher(const float* input,
                                               float* output,
                                               int64_t N, int64_t C, int64_t H, int64_t W,
                                               int64_t r, int64_t out_channels,
                                               const int64_t* input_strides,
                                               const int64_t* output_strides,
                                               cudaStream_t stream);

// Helper function to get strides as int64_t array on device
static inline std::vector<int64_t> get_strides(const torch::Tensor& t) {
    std::vector<int64_t> s(t.dim());
    for (int64_t i = 0; i < t.dim(); ++i) {
        s[i] = t.stride(i);
    }
    return s;
}

// abs_forward: element-wise absolute value
torch::Tensor abs_forward(torch::Tensor x) {
    // Check device
    TORCH_CHECK(x.is_cuda(), "abs_forward: input must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "abs_forward: input must be float32");
    // The kernel supports non-contiguous strides via grid-stride loop; no need to enforce contiguity
    // However, data_ptr must be valid. We will call with original strides not needed for abs.
    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(x.data_ptr<float>(),
                         output.data_ptr<float>(),
                         numel,
                         stream);
    return output;
}

// pixel_shuffle_forward
torch::Tensor pixel_shuffle_forward(torch::Tensor x, int64_t upscale_factor) {
    TORCH_CHECK(x.is_cuda(), "pixel_shuffle_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "pixel_shuffle_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "pixel_shuffle_forward: input must be 4D (N, C, H, W)");
    TORCH_CHECK(upscale_factor > 0, "upscale_factor must be positive");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t r = upscale_factor;
    TORCH_CHECK(C % (r * r) == 0, "C must be divisible by r^2");
    int64_t out_channels = C / (r * r);

    auto output = torch::empty({N, out_channels, H * r, W * r}, x.options());

    // Get strides for input and output (support non-contiguous via kernel strides)
    std::vector<int64_t> input_strides = get_strides(x);
    std::vector<int64_t> output_strides = get_strides(output);

    // Allocate device memory for strides and copy
    int64_t* d_input_strides;
    int64_t* d_output_strides;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaMalloc(&d_input_strides, 4 * sizeof(int64_t));
    cudaMalloc(&d_output_strides, 4 * sizeof(int64_t));
    cudaMemcpyAsync(d_input_strides, input_strides.data(), 4 * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_output_strides, output_strides.data(), 4 * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    pixel_shuffle_forward_launcher(x.data_ptr<float>(),
                                   output.data_ptr<float>(),
                                   N, C, H, W,
                                   r, out_channels,
                                   d_input_strides,
                                   d_output_strides,
                                   stream);

    // Free device memory after launch (safe because kernel is async and copy is async on same stream)
    cudaFree(d_input_strides);
    cudaFree(d_output_strides);
    // Note: cudaFree is synchronous but after kernel completes on default ordering; acceptable for correctness.
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward,
          "Element-wise absolute value (float32 CUDA tensor)",
          py::arg("x"));
    m.def("pixel_shuffle_forward", &pixel_shuffle_forward,
          "PixelShuffle forward (float32 CUDA tensor, upscale_factor)",
          py::arg("x"),
          py::arg("upscale_factor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
