#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
void upsample_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    bool align_corners,
    const long long* in_strides,
    cudaStream_t stream);

void constant_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int left, int right, int top, int bottom,
    float value,
    const long long* in_strides,
    cudaStream_t stream);
}

// Helper to extract strides from a tensor; returns true if contiguous (and sets strides to nullptr)
static bool get_strides(const torch::Tensor& t, long long* strides) {
    if (t.is_contiguous()) {
        return true;
    }
    auto sizes = t.sizes();
    auto strides_vec = t.strides();
    for (int i = 0; i < 4; ++i) {
        strides[i] = static_cast<long long>(strides_vec[i]);
    }
    return false;
}

// Upsampling bilinear 2D forward
torch::Tensor upsampling_bilinear2d_forward(
    torch::Tensor x,
    double scale_factor,
    bool align_corners) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    int H_out = static_cast<int>(std::floor(H_in * scale_factor));
    int W_out = static_cast<int>(std::floor(W_in * scale_factor));

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    long long strides[4];
    bool is_contiguous = get_strides(x, strides);
    const long long* strides_ptr = is_contiguous ? nullptr : strides;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    upsample_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        static_cast<float>(scale_factor),
        align_corners,
        strides_ptr,
        stream);

    return output;
}

// Constant pad 2D forward
torch::Tensor constant_pad2d_forward(
    torch::Tensor x,
    std::vector<int64_t> padding,
    double value) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(padding.size() == 4, "Padding must contain 4 integers (left, right, top, bottom)");

    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    int left = static_cast<int>(padding[0]);
    int right = static_cast<int>(padding[1]);
    int top = static_cast<int>(padding[2]);
    int bottom = static_cast<int>(padding[3]);

    int H_out = H_in + top + bottom;
    int W_out = W_in + left + right;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    long long strides[4];
    bool is_contiguous = get_strides(x, strides);
    const long long* strides_ptr = is_contiguous ? nullptr : strides;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    constant_pad2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        left, right, top, bottom,
        static_cast<float>(value),
        strides_ptr,
        stream);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("upsampling_bilinear2d_forward", &upsampling_bilinear2d_forward,
          "Upsampling bilinear 2D forward",
          py::arg("x"), py::arg("scale_factor"), py::arg("align_corners"));
    m.def("constant_pad2d_forward", &constant_pad2d_forward,
          "Constant pad 2D forward",
          py::arg("x"), py::arg("padding"), py::arg("value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);