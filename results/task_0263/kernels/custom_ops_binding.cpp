#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <tuple>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void pool2d_avgmax_launcher(
    const float* x,
    float* avg,
    float* max,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t sN,
    int64_t sC,
    int64_t sH,
    int64_t sW,
    int64_t kh,
    int64_t kw,
    cudaStream_t stream
);

// PyTorch wrapper for combined avg/max pooling
std::tuple<torch::Tensor, torch::Tensor> pool2d_avgmax_forward(
    torch::Tensor x,
    int64_t kernel_h,
    int64_t kernel_w
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "kernel sizes must be positive");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(H % kernel_h == 0 && W % kernel_w == 0,
                "Spatial dimensions must be divisible by kernel sizes");

    int64_t H_out = H / kernel_h;
    int64_t W_out = W / kernel_w;

    // Allocate contiguous outputs
    auto avg = torch::empty({N, C, H_out, W_out}, x.options());
    auto max = torch::empty({N, C, H_out, W_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get strides (note: x may be non-contiguous)
    int64_t sN = x.stride(0);
    int64_t sC = x.stride(1);
    int64_t sH = x.stride(2);
    int64_t sW = x.stride(3);

    // Call launcher
    pool2d_avgmax_launcher(
        x.data_ptr<float>(),
        avg.data_ptr<float>(),
        max.data_ptr<float>(),
        N, C, H, W,
        sN, sC, sH, sW,
        kernel_h, kernel_w,
        stream
    );

    // Return tuple (avg, max)
    return std::make_tuple(avg, max);
}

// Registration function
void register_pool2d_avgmax(pybind11::module& m) {
    m.def("pool2d_avgmax", &pool2d_avgmax_forward,
          "Compute average and max pooling over non-overlapping windows",
          py::arg("x"),
          py::arg("kernel_h"),
          py::arg("kernel_w"));
}

// Auto-register
REGISTER_BINDING(pool2d_avgmax, register_pool2d_avgmax);
