#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void adaptive_avg_pool2d_5d_launcher(
    const float* x, float* y, int batch, int channels, int depth,
    int height, int width, cudaStream_t stream);

extern "C" void leaky_relu_launcher(
    const float* x, float* y, long long n, float negative_slope,
    cudaStream_t stream);

extern "C" void add_bias_launcher(
    const float* x, const float* bias, float* y, long long total,
    int channels, int inner_size, cudaStream_t stream);

extern "C" void clamp_launcher(
    const float* x, float* y, long long n, float min_val, float max_val,
    cudaStream_t stream);

extern "C" void conv3d_launcher(
    const float* x, const float* weight, const float* bias, float* y,
    int batch, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int kernel_d, int kernel_h, int kernel_w,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w,
    int out_depth, int out_height, int out_width,
    cudaStream_t stream);

extern "C" void conv_transpose3d_launcher(
    const float* x, const float* weight, const float* bias, float* y,
    int batch, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int kernel_d, int kernel_h, int kernel_w,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w,
    int out_depth, int out_height, int out_width,
    int output_pad_d, int output_pad_h, int output_pad_w,
    cudaStream_t stream);

// Helper to check CUDA tensor float32
static void check_cuda_float32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

// -----------------------------------------------------------------------------
// adaptive_avg_pool2d_5d
// -----------------------------------------------------------------------------
torch::Tensor adaptive_avg_pool2d_5d(torch::Tensor x, int64_t output_size) {
    check_cuda_float32(x, "x");
    TORCH_CHECK(output_size == 1, "Only output_size=1 is supported for last two dims");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (batch, channels, depth, height, width)");

    // Input may be non-contiguous; contract says supports strides.
    // We'll make it contiguous to simplify indexing and ensure correct strides.
    auto x_contig = x.contiguous();

    int batch = x_contig.size(0);
    int channels = x_contig.size(1);
    int depth = x_contig.size(2);
    int height = x_contig.size(3);
    int width = x_contig.size(4);

    auto y = torch::empty({batch, channels, depth, 1, 1}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    adaptive_avg_pool2d_5d_launcher(
        x_contig.data_ptr<float>(), y.data_ptr<float>(),
        batch, channels, depth, height, width, stream);

    return y;
}

// -----------------------------------------------------------------------------
// conv3d_forward using custom CUDA kernel
// -----------------------------------------------------------------------------
torch::Tensor conv3d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding) {

    check_cuda_float32(x, "x");
    check_cuda_float32(weight, "weight");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D");
    TORCH_CHECK(stride.size() == 3 && padding.size() == 3, "stride/padding must be length 3");

    auto x_contig = x.contiguous();
    auto w_contig = weight.contiguous();

    int batch = x_contig.size(0);
    int in_channels = x_contig.size(1);
    int in_depth = x_contig.size(2);
    int in_height = x_contig.size(3);
    int in_width = x_contig.size(4);

    int out_channels = w_contig.size(0);
    int kernel_d = w_contig.size(2);
    int kernel_h = w_contig.size(3);
    int kernel_w = w_contig.size(4);

    TORCH_CHECK(in_channels == w_contig.size(1), "Input channels must match weight");

    int pad_d = padding[0], pad_h = padding[1], pad_w = padding[2];
    int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];

    int out_depth = (in_depth + 2 * pad_d - kernel_d) / stride_d + 1;
    int out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    auto y = torch::empty({batch, out_channels, out_depth, out_height, out_width}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // bias pointer: may be null
    const float* bias_ptr = nullptr;
    torch::Tensor b_contig;
    if (bias.has_value() && bias.value().defined()) {
        b_contig = bias.value().contiguous();
        check_cuda_float32(b_contig, "bias");
        TORCH_CHECK(b_contig.numel() == out_channels, "bias size mismatch");
        bias_ptr = b_contig.data_ptr<float>();
    }

    conv3d_launcher(
        x_contig.data_ptr<float>(),
        w_contig.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        batch, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        pad_d, pad_h, pad_w,
        stride_d, stride_h, stride_w,
        out_depth, out_height, out_width,
        stream);

    return y;
}

// -----------------------------------------------------------------------------
// conv_transpose3d_forward using custom CUDA kernel
// -----------------------------------------------------------------------------
torch::Tensor conv_transpose3d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding) {

    check_cuda_float32(x, "x");
    check_cuda_float32(weight, "weight");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D");
    TORCH_CHECK(stride.size() == 3 && padding.size() == 3 && output_padding.size() == 3,
                "stride/padding/output_padding must be length 3");

    auto x_contig = x.contiguous();
    auto w_contig = weight.contiguous();

    int batch = x_contig.size(0);
    int in_channels = x_contig.size(1);
    int in_depth = x_contig.size(2);
    int in_height = x_contig.size(3);
    int in_width = x_contig.size(4);

    // For ConvTranspose, weight shape is [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
    int out_channels = w_contig.size(1);
    int kernel_d = w_contig.size(2);
    int kernel_h = w_contig.size(3);
    int kernel_w = w_contig.size(4);

    int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
    int pad_d = padding[0], pad_h = padding[1], pad_w = padding[2];
    int opad_d = output_padding[0], opad_h = output_padding[1], opad_w = output_padding[2];

    int out_depth = (in_depth - 1) * stride_d - 2 * pad_d + kernel_d + opad_d;
    int out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h + opad_h;
    int out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w + opad_w;

    auto y = torch::empty({batch, out_channels, out_depth, out_height, out_width}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = nullptr;
    torch::Tensor b_contig;
    if (bias.has_value() && bias.value().defined()) {
        b_contig = bias.value().contiguous();
        check_cuda_float32(b_contig, "bias");
        TORCH_CHECK(b_contig.numel() == out_channels, "bias size mismatch");
        bias_ptr = b_contig.data_ptr<float>();
    }

    conv_transpose3d_launcher(
        x_contig.data_ptr<float>(),
        w_contig.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        batch, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        pad_d, pad_h, pad_w,
        stride_d, stride_h, stride_w,
        out_depth, out_height, out_width,
        opad_d, opad_h, opad_w,
        stream);

    return y;
}

// -----------------------------------------------------------------------------
// Elementwise ops
// -----------------------------------------------------------------------------
torch::Tensor leaky_relu_forward(torch::Tensor x, double negative_slope) {
    check_cuda_float32(x, "x");
    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);
    long long n = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    leaky_relu_launcher(x_contig.data_ptr<float>(), y.data_ptr<float>(), n,
                        static_cast<float>(negative_slope), stream);
    return y;
}

torch::Tensor add_bias(torch::Tensor x, torch::Tensor bias) {
    check_cuda_float32(x, "x");
    check_cuda_float32(bias, "bias");
    TORCH_CHECK(x.dim() == 5, "x must be 5D");
    TORCH_CHECK(bias.dim() == 4, "bias must be 4D [channels,1,1,1]");
    TORCH_CHECK(bias.size(0) == x.size(1), "bias channel mismatch");

    auto x_contig = x.contiguous();
    auto bias_contig = bias.contiguous();
    auto y = torch::empty_like(x_contig);
    long long total = x_contig.numel();
    int channels = x_contig.size(1);
    int inner_size = x_contig.size(2) * x_contig.size(3) * x_contig.size(4);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(x_contig.data_ptr<float>(), bias_contig.data_ptr<float>(),
                      y.data_ptr<float>(), total, channels, inner_size, stream);
    return y;
}

torch::Tensor clamp_forward(torch::Tensor x, double min_val, double max_val) {
    check_cuda_float32(x, "x");
    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);
    long long n = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    clamp_launcher(x_contig.data_ptr<float>(), y.data_ptr<float>(), n,
                   static_cast<float>(min_val), static_cast<float>(max_val), stream);
    return y;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("adaptive_avg_pool2d_5d", &adaptive_avg_pool2d_5d,
          "5D adaptive average pooling over last two dims",
          py::arg("x"), py::arg("output_size"));
    m.def("conv3d_forward", &conv3d_forward,
          "3D convolution forward (custom CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"));
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "LeakyReLU forward",
          py::arg("x"), py::arg("negative_slope"));
    m.def("add_bias", &add_bias,
          "Add bias with broadcasting",
          py::arg("x"), py::arg("bias"));
    m.def("clamp_forward", &clamp_forward,
          "Clamp tensor",
          py::arg("x"), py::arg("min_val"), py::arg("max_val"));
    m.def("conv_transpose3d_forward", &conv_transpose3d_forward,
          "3D transposed convolution forward (custom CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
