#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void pad_sum_channels_launcher(
    const float* x,
    float* out,
    int B, int C, int H, int W,
    int left, int right, int top, int bottom,
    cudaStream_t stream);

extern "C" void pad2d_launcher(
    const float* x,
    float* out,
    int B, int C, int H, int W,
    int left, int right, int top, int bottom,
    cudaStream_t stream);

extern "C" void sum_channels_launcher(
    const float* x,
    float* out,
    int B, int C, int H, int W,
    cudaStream_t stream);

// Helper to check CUDA errors after kernel launch
static void check_cuda_error() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// ===================== pad_sum_channels =====================
torch::Tensor pad_sum_channels(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom)
{
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D [B, C, H, W]");
    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "Padding sizes must be non-negative");

    // Ensure contiguous (launcher assumes contiguous)
    x = x.contiguous();

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t outH = H + pad_top + pad_bottom;
    int64_t outW = W + pad_left + pad_right;

    // Allocate output
    auto out = torch::empty({B, 1, outH, outW}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    pad_sum_channels_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        stream
    );
    check_cuda_error();

    return out;
}

// ===================== pad2d_custom =====================
torch::Tensor pad2d_custom(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom)
{
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D [B, C, H, W]");
    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "Padding sizes must be non-negative");

    x = x.contiguous();

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t outH = H + pad_top + pad_bottom;
    int64_t outW = W + pad_left + pad_right;

    auto out = torch::empty({B, C, outH, outW}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    pad2d_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        stream
    );
    check_cuda_error();

    return out;
}

// ===================== sum_channels_custom =====================
torch::Tensor sum_channels_custom(
    torch::Tensor x,
    bool keepdim)
{
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D [B, C, H, W]");
    TORCH_CHECK(keepdim, "sum_channels_custom currently only supports keepdim=True as required by the task");

    x = x.contiguous();

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto out = torch::empty({B, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_channels_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );
    check_cuda_error();

    return out;
}

// ===================== Registration =====================
void register_custom_ops(pybind11::module& m) {
    m.def("pad_sum_channels", &pad_sum_channels,
          "Fused ZeroPad2d + sum over channels",
          py::arg("x"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"));
    m.def("pad2d_custom", &pad2d_custom,
          "ZeroPad2d only",
          py::arg("x"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"));
    m.def("sum_channels_custom", &sum_channels_custom,
          "Sum over channel dimension",
          py::arg("x"),
          py::arg("keepdim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
