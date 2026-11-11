#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare CUDA launchers from custom_ops.cu
extern "C" {
void custom_hardtanh_launcher(const float* input, float* output, int64_t numel,
                              float min_val, float max_val, cudaStream_t stream);

void custom_max_pool2d_launcher(const float* input, float* output,
                                int batch_size, int channels,
                                int input_height, int input_width,
                                int output_height, int output_width,
                                int kernel_h, int kernel_w,
                                int stride_h, int stride_w,
                                int padding_h, int padding_w,
                                int dilation_h, int dilation_w,
                                cudaStream_t stream);

void custom_flatten_launcher(const float* input, float* output, int64_t numel,
                             cudaStream_t stream);

void custom_linear_launcher(const float* input, const float* weight,
                            const float* bias, float* output,
                            int N, int in_features, int out_features,
                            cudaStream_t stream);
}

// -----------------------------------------------------------------------------
// custom_hardtanh wrapper
// -----------------------------------------------------------------------------
torch::Tensor custom_hardtanh(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "custom_hardtanh: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_hardtanh: input must be float32");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_hardtanh_launcher(x.data_ptr<float>(), output.data_ptr<float>(),
                             x.numel(), static_cast<float>(min_val),
                             static_cast<float>(max_val), stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_hardtanh kernel failed: ", cudaGetErrorString(err));
    return output;
}

// -----------------------------------------------------------------------------
// custom_max_pool2d wrapper
// -----------------------------------------------------------------------------
torch::Tensor custom_max_pool2d(torch::Tensor x, int64_t kernel_size,
                               int64_t stride, int64_t padding,
                               int64_t dilation, bool ceil_mode) {
    TORCH_CHECK(x.is_cuda(), "custom_max_pool2d: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_max_pool2d: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_max_pool2d: input must be 4D (N,C,H,W)");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0 && dilation > 0,
                "custom_max_pool2d: invalid pooling parameters");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t out_h, out_w;
    if (ceil_mode) {
        out_h = (H + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
        out_w = (W + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
    } else {
        out_h = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        out_w = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    }
    TORCH_CHECK(out_h > 0 && out_w > 0, "custom_max_pool2d: output size too small");

    auto output = torch::empty({N, C, out_h, out_w}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_max_pool2d_launcher(x.data_ptr<float>(), output.data_ptr<float>(),
                               static_cast<int>(N), static_cast<int>(C),
                               static_cast<int>(H), static_cast<int>(W),
                               static_cast<int>(out_h), static_cast<int>(out_w),
                               static_cast<int>(kernel_size), static_cast<int>(kernel_size),
                               static_cast<int>(stride), static_cast<int>(stride),
                               static_cast<int>(padding), static_cast<int>(padding),
                               static_cast<int>(dilation), static_cast<int>(dilation),
                               stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_max_pool2d kernel failed: ", cudaGetErrorString(err));
    return output;
}

// -----------------------------------------------------------------------------
// custom_flatten wrapper
// -----------------------------------------------------------------------------
torch::Tensor custom_flatten(torch::Tensor x, int64_t start_dim) {
    TORCH_CHECK(x.is_cuda(), "custom_flatten: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_flatten: input must be float32");
    TORCH_CHECK(start_dim >= 0 && start_dim < x.dim(), "custom_flatten: invalid start_dim");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < start_dim; ++i) {
        out_shape.push_back(x.size(i));
    }
    int64_t flattened_size = 1;
    for (int64_t i = start_dim; i < x.dim(); ++i) {
        flattened_size *= x.size(i);
    }
    out_shape.push_back(flattened_size);

    auto output = torch::empty(out_shape, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_flatten_launcher(x.data_ptr<float>(), output.data_ptr<float>(),
                            x.numel(), stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_flatten kernel failed: ", cudaGetErrorString(err));
    return output;
}

// -----------------------------------------------------------------------------
// custom_linear wrapper
// -----------------------------------------------------------------------------
torch::Tensor custom_linear(torch::Tensor x, torch::Tensor weight,
                            torch::Tensor bias = torch::Tensor()) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "custom_linear: inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32,
                "custom_linear: inputs must be float32");
    TORCH_CHECK(x.dim() == 2, "custom_linear: x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "custom_linear: weight must be 2D");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32,
                    "custom_linear: bias must be CUDA float32 if provided");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == weight.size(0),
                    "custom_linear: bias dimension mismatch");
    }

    if (!x.is_contiguous()) x = x.contiguous();
    if (!weight.is_contiguous()) weight = weight.contiguous();
    if (bias.defined() && !bias.is_contiguous()) bias = bias.contiguous();

    int64_t N = x.size(0);
    int64_t in_features = x.size(1);
    int64_t out_features = weight.size(0);
    TORCH_CHECK(in_features == weight.size(1), "custom_linear: in_features mismatch");

    auto output = torch::empty({N, out_features}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    custom_linear_launcher(x.data_ptr<float>(), weight.data_ptr<float>(),
                           bias_ptr, output.data_ptr<float>(),
                           static_cast<int>(N),
                           static_cast<int>(in_features),
                           static_cast<int>(out_features),
                           stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_linear kernel failed: ", cudaGetErrorString(err));
    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_hardtanh", &custom_hardtanh,
          "Custom hardtanh",
          pybind11::arg("x"), pybind11::arg("min_val"), pybind11::arg("max_val"));
    m.def("custom_max_pool2d", &custom_max_pool2d,
          "Custom max_pool2d",
          pybind11::arg("x"), pybind11::arg("kernel_size"),
          pybind11::arg("stride"), pybind11::arg("padding"),
          pybind11::arg("dilation"), pybind11::arg("ceil_mode"));
    m.def("custom_flatten", &custom_flatten,
          "Custom flatten",
          pybind11::arg("x"), pybind11::arg("start_dim"));
    m.def("custom_linear", &custom_linear,
          "Custom linear",
          pybind11::arg("x"), pybind11::arg("weight"),
          pybind11::arg("bias") = pybind11::none());
}

REGISTER_BINDING(custom_ops, register_custom_ops);