#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_lr, int pad_tb,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    cudaStream_t stream);

extern "C" void glu_forward_launcher(
    const float* input,
    float* output,
    int total_outer, int C, int inner,
    int input_stride_outer, int input_stride_dim, int input_stride_inner,
    int output_stride_outer, int output_stride_dim, int output_stride_inner,
    cudaStream_t stream);

extern "C" void min_dim_forward_launcher(
    const float* input,
    float* output,
    int total_outer, int D, int inner,
    int input_stride_outer, int input_stride_dim, int input_stride_inner,
    cudaStream_t stream);

extern "C" void conv2d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out,
    cudaStream_t stream);

// Helper to check CUDA error after kernel launch
static void check_cuda_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }
}

// ---------------- reflection_pad2d_forward ----------------
torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t pad_left_right, int64_t pad_top_bottom) {
    TORCH_CHECK(input.is_cuda(), "reflection_pad2d input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "reflection_pad2d input must be float32");
    TORCH_CHECK(input.dim() == 4, "reflection_pad2d input must be 4D (N, C, H, W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int out_H = H + 2 * (int)pad_top_bottom;
    int out_W = W + 2 * (int)pad_left_right;

    auto output = torch::empty({N, C, out_H, out_W}, input.options());

    int64_t in_stride_n = input.stride(0);
    int64_t in_stride_c = input.stride(1);
    int64_t in_stride_h = input.stride(2);
    int64_t in_stride_w = input.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        (int)pad_left_right, (int)pad_top_bottom,
        (int)in_stride_n, (int)in_stride_c, (int)in_stride_h, (int)in_stride_w,
        stream
    );

    check_cuda_error();
    return output;
}

// ---------------- conv2d_forward using custom CUDA kernel ----------------
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

    TORCH_CHECK(input.is_cuda(), "conv2d input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv2d weight must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "conv2d input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "conv2d weight must be float32");
    TORCH_CHECK(input.dim() == 4, "conv2d input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "conv2d weight must be 4D (C_out, C_in/groups, kH, kW)");

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias->numel() > 0) {
        TORCH_CHECK(bias->is_cuda(), "conv2d bias must be a CUDA tensor");
        TORCH_CHECK(bias->scalar_type() == torch::kFloat32, "conv2d bias must be float32");
        TORCH_CHECK(bias->dim() == 1, "conv2d bias must be 1D");
        TORCH_CHECK(bias->size(0) == weight.size(0), "conv2d bias size must match output channels");
        bias_ptr = bias->data_ptr<float>();
    }

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(0);
    int kH = weight.size(2);
    int kW = weight.size(3);

    int stride_h = (int)stride;
    int stride_w = (int)stride;
    int pad_h = (int)padding;
    int pad_w = (int)padding;
    int dilation_h = (int)dilation;
    int dilation_w = (int)dilation;
    int groups_i = (int)groups;

    int H_out = (H_in + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv2d_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups_i,
        H_out, W_out,
        stream
    );

    check_cuda_error();
    return output;
}

// ---------------- glu_forward ----------------
torch::Tensor glu_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "glu input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "glu input must be float32");

    int64_t dim_size = input.size(dim);
    TORCH_CHECK(dim_size % 2 == 0, "glu split dimension must be even");

    if (dim < 0) dim += input.dim();
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "glu dim out of range");

    int C = (int)dim_size;
    int half_C = C / 2;

    int total_outer = 1;
    for (int i = 0; i < (int)dim; ++i) {
        total_outer *= input.size(i);
    }
    int inner = 1;
    for (int i = (int)dim + 1; i < input.dim(); ++i) {
        inner *= input.size(i);
    }

    std::vector<int64_t> output_sizes;
    output_sizes.reserve(input.dim());
    for (int i = 0; i < input.dim(); ++i) {
        if (i == (int)dim) {
            output_sizes.push_back(half_C);
        } else {
            output_sizes.push_back(input.size(i));
        }
    }
    auto output = torch::empty(output_sizes, input.options());

    int64_t input_stride_outer = input.stride(0);
    int64_t input_stride_dim = input.stride((int)dim);
    int64_t input_stride_inner = (dim + 1 < input.dim()) ? input.stride((int)dim + 1) : 0;

    int64_t output_stride_outer = (dim > 0) ? output.stride(0) : 0;
    int64_t output_stride_dim = output.stride((int)dim);
    int64_t output_stride_inner = (dim + 1 < output.dim()) ? output.stride((int)dim + 1) : 0;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    glu_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_outer, C, inner,
        (int)input_stride_outer, (int)input_stride_dim, (int)input_stride_inner,
        (int)output_stride_outer, (int)output_stride_dim, (int)output_stride_inner,
        stream
    );

    check_cuda_error();
    return output;
}

// ---------------- min_dim_forward ----------------
torch::Tensor min_dim_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "min_dim input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "min_dim input must be float32");

    if (dim < 0) dim += input.dim();
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "min_dim dim out of range");

    int D = (int)input.size(dim);
    TORCH_CHECK(D > 0, "min_dim dimension size must be positive");

    int total_outer = 1;
    for (int i = 0; i < (int)dim; ++i) {
        total_outer *= input.size(i);
    }
    int inner = 1;
    for (int i = (int)dim + 1; i < input.dim(); ++i) {
        inner *= input.size(i);
    }

    std::vector<int64_t> output_sizes;
    output_sizes.reserve(input.dim() - 1);
    for (int i = 0; i < input.dim(); ++i) {
        if (i != (int)dim) {
            output_sizes.push_back(input.size(i));
        }
    }
    auto output = torch::empty(output_sizes, input.options());

    int64_t input_stride_outer = (dim > 0) ? input.stride(0) : 0;
    int64_t input_stride_dim = input.stride((int)dim);
    int64_t input_stride_inner = (dim + 1 < input.dim()) ? input.stride((int)dim + 1) : 0;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    min_dim_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_outer, D, inner,
        (int)input_stride_outer, (int)input_stride_dim, (int)input_stride_inner,
        stream
    );

    check_cuda_error();
    return output;
}

// ---------------- Registration ----------------
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward, "ReflectionPad2d forward",
          pybind11::arg("input"), pybind11::arg("pad_left_right"), pybind11::arg("pad_top_bottom"));
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward using custom CUDA kernel",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("dilation"), pybind11::arg("groups"));
    m.def("glu_forward", &glu_forward, "GLU forward",
          pybind11::arg("input"), pybind11::arg("dim"));
    m.def("min_dim_forward", &min_dim_forward, "Min along dimension forward",
          pybind11::arg("input"), pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);