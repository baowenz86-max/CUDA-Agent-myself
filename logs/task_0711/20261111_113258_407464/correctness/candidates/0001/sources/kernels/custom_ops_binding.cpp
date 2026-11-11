#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void sum_channels_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    cudaStream_t stream);

extern "C" void expand_channels_launcher(
    const float* x,
    float* y,
    int N, int channels_factor, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    cudaStream_t stream);

extern "C" void pixel_shuffle_launcher(
    const float* x,
    float* y,
    int N, int C_in, int H, int W, int r,
    long long sN, long long sC, long long sH, long long sW,
    cudaStream_t stream);

// Helper to validate CUDA float32 tensor and get strides
static void check_cuda_float_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.dim() == 4, name, " must be 4-dimensional");
}

static std::tuple<long long, long long, long long, long long> get_strides(const torch::Tensor& t) {
    // Strides are in elements
    return std::make_tuple(
        t.stride(0),
        t.stride(1),
        t.stride(2),
        t.stride(3)
    );
}

torch::Tensor sum_channels_forward(torch::Tensor x) {
    check_cuda_float_tensor(x, "sum_channels_forward input");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty({N, 1, H, W}, x.options());

    auto [sN, sC, sH, sW] = get_strides(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_channels_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        sN, sC, sH, sW,
        stream
    );

    return output;
}

torch::Tensor expand_channels_forward(torch::Tensor x, int64_t channels_factor) {
    check_cuda_float_tensor(x, "expand_channels_forward input");
    TORCH_CHECK(channels_factor > 0, "channels_factor must be positive");
    TORCH_CHECK(x.size(1) == 1, "expand_channels_forward expects x.size(1) == 1");

    int N = x.size(0);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty({N, channels_factor, H, W}, x.options());

    auto [sN, sC, sH, sW] = get_strides(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    expand_channels_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, (int)channels_factor, H, W,
        sN, sC, sH, sW,
        stream
    );

    return output;
}

torch::Tensor pixel_shuffle_forward(torch::Tensor x, int64_t upscale_factor) {
    check_cuda_float_tensor(x, "pixel_shuffle_forward input");
    TORCH_CHECK(upscale_factor > 0, "upscale_factor must be positive");
    int r = (int)upscale_factor;
    TORCH_CHECK(x.size(1) % (r * r) == 0, "channels must be divisible by r^2");

    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_out = C_in / (r * r);

    auto output = torch::empty({N, C_out, H * r, W * r}, x.options());

    auto [sN, sC, sH, sW] = get_strides(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    pixel_shuffle_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W, r,
        sN, sC, sH, sW,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("sum_channels_forward", &sum_channels_forward, "Sum over channels with keepdim=True",
          py::arg("x"));
    m.def("expand_channels_forward", &expand_channels_forward, "Expand channels to given factor",
          py::arg("x"), py::arg("channels_factor"));
    m.def("pixel_shuffle_forward", &pixel_shuffle_forward, "Pixel shuffle operation",
          py::arg("x"), py::arg("upscale_factor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
