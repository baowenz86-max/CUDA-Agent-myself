#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void zero_pad2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int left, int top,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_n,
    long long output_stride_c,
    long long output_stride_h,
    long long output_stride_w,
    cudaStream_t stream);

extern "C" void log_softmax_channel_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_n,
    long long output_stride_c,
    long long output_stride_h,
    long long output_stride_w,
    cudaStream_t stream);

extern "C" void conv_transpose2d_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int output_pad_h, int output_pad_w,
    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// zero_pad2d_forward
// Contract: padding is int (scalar) indicating same padding on all sides. But model_new passes a list.
// We accept py::object to support both list and scalar; our function will parse accordingly.
torch::Tensor zero_pad2d_forward(torch::Tensor input, py::object padding) {
    TORCH_CHECK(input.is_cuda(), "zero_pad2d_forward: input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "zero_pad2d_forward: input must be float32");
    TORCH_CHECK(input.dim() == 4, "zero_pad2d_forward: input must be 4D");

    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    // Parse padding object
    std::vector<int64_t> pad_vals;
    if (py::isinstance<py::list>(padding)) {
        py::list plist = py::reinterpret_borrow<py::list>(padding);
        for (auto item : plist) {
            pad_vals.push_back(item.cast<int64_t>());
        }
    } else if (py::isinstance<py::int_>(padding)) {
        int64_t p = padding.cast<int64_t>();
        pad_vals = {p, p, p, p};
    } else {
        TORCH_CHECK(false, "zero_pad2d_forward: padding must be list of 4 ints or int");
    }
    TORCH_CHECK(pad_vals.size() == 4, "zero_pad2d_forward: padding must have 4 elements");

    int left = static_cast<int>(pad_vals[0]);
    int right = static_cast<int>(pad_vals[1]);
    int top = static_cast<int>(pad_vals[2]);
    int bottom = static_cast<int>(pad_vals[3]);

    int H_out = H_in + top + bottom;
    int W_out = W_in + left + right;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    cudaStream_t stream = getCurrentCUDAStream();

    long long input_stride_n = static_cast<long long>(input.stride(0));
    long long input_stride_c = static_cast<long long>(input.stride(1));
    long long input_stride_h = static_cast<long long>(input.stride(2));
    long long input_stride_w = static_cast<long long>(input.stride(3));
    long long output_stride_n = static_cast<long long>(output.stride(0));
    long long output_stride_c = static_cast<long long>(output.stride(1));
    long long output_stride_h = static_cast<long long>(output.stride(2));
    long long output_stride_w = static_cast<long long>(output.stride(3));

    zero_pad2d_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        left, top,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_c, output_stride_h, output_stride_w,
        stream
    );

    return output;
}

// log_softmax_channel_forward
torch::Tensor log_softmax_channel_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "log_softmax_channel_forward: input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "log_softmax_channel_forward: input must be float32");
    TORCH_CHECK(input.dim() == 4, "log_softmax_channel_forward: input must be 4D");
    TORCH_CHECK(dim == 1, "log_softmax_channel_forward: dim must be 1");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input);

    cudaStream_t stream = getCurrentCUDAStream();

    long long input_stride_n = static_cast<long long>(input.stride(0));
    long long input_stride_c = static_cast<long long>(input.stride(1));
    long long input_stride_h = static_cast<long long>(input.stride(2));
    long long input_stride_w = static_cast<long long>(input.stride(3));
    long long output_stride_n = static_cast<long long>(output.stride(0));
    long long output_stride_c = static_cast<long long>(output.stride(1));
    long long output_stride_h = static_cast<long long>(output.stride(2));
    long long output_stride_w = static_cast<long long>(output.stride(3));

    log_softmax_channel_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N, C, H, W,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_c, output_stride_h, output_stride_w,
        stream
    );

    return output;
}

// conv_transpose2d_forward
// Contract: stride, padding, output_padding are each a single int (assuming square).
// We accept int64_t scalars.
torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(input.is_cuda(), "conv_transpose2d_forward: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose2d_forward: weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "conv_transpose2d_forward: bias must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int C_out = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    // Assume symmetric stride/padding/output_padding (same for H and W)
    int stride_h = static_cast<int>(stride);
    int stride_w = static_cast<int>(stride);
    int pad_h = static_cast<int>(padding);
    int pad_w = static_cast<int>(padding);
    int output_pad_h = static_cast<int>(output_padding);
    int output_pad_w = static_cast<int>(output_padding);

    int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + output_pad_h;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + output_pad_w;

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    cudaStream_t stream = getCurrentCUDAStream();

    // Ensure contiguous for simplicity
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    conv_transpose2d_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        kH, kW, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w,
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("zero_pad2d_forward", &zero_pad2d_forward,
          "ZeroPad2d forward",
          py::arg("input"), py::arg("padding"));
    m.def("log_softmax_channel_forward", &log_softmax_channel_forward,
          "LogSoftmax along channel",
          py::arg("input"), py::arg("dim"));
    m.def("conv_transpose2d_forward", &conv_transpose2d_forward,
          "ConvTranspose2d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
