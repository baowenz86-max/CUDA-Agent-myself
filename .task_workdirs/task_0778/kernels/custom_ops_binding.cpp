#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void isin_custom_launcher(
    const float* x,
    const float* values,
    bool* out,
    long long shape0, long long shape1, long long shape2,
    long long x_stride0, long long x_stride1, long long x_stride2,
    long long out_stride0, long long out_stride1, long long out_stride2,
    int values_size,
    long long total_elements,
    cudaStream_t stream);

extern "C" void maxpool1d_custom_launcher(
    const float* x,
    float* y,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride,
    cudaStream_t stream);

torch::Tensor isin_custom(torch::Tensor x, torch::Tensor values) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
    TORCH_CHECK(values.dim() == 1, "values must be 1D tensor");
    TORCH_CHECK(values.is_contiguous(), "values must be contiguous");

    auto out = torch::empty(x.sizes(), x.options().dtype(torch::kBool));
    out.as_strided_(x.sizes(), x.strides());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, channels, length]");
    long long shape0 = x.size(0);
    long long shape1 = x.size(1);
    long long shape2 = x.size(2);
    long long x_stride0 = x.stride(0);
    long long x_stride1 = x.stride(1);
    long long x_stride2 = x.stride(2);
    long long out_stride0 = out.stride(0);
    long long out_stride1 = out.stride(1);
    long long out_stride2 = out.stride(2);

    long long total_elements = x.numel();
    int values_size = values.size(0);

    isin_custom_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        out.data_ptr<bool>(),
        shape0, shape1, shape2,
        x_stride0, x_stride1, x_stride2,
        out_stride0, out_stride1, out_stride2,
        values_size,
        total_elements,
        stream
    );

    return out;
}

torch::Tensor maxpool1d_custom(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool ceil_mode) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation > 0, "dilation must be positive");
    TORCH_CHECK(!ceil_mode, "ceil_mode must be false for this contract");

    TORCH_CHECK(x.dim() >= 2, "x must have at least batch and channel dimensions");
    int64_t batch_size = x.size(0);
    int64_t channels = x.size(1);
    int64_t input_length = x.size(2);
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, channels, length]");

    int64_t output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(output_length > 0, "invalid output length");

    auto y = torch::empty({batch_size, channels, output_length}, x.options());

    long long x_batch_stride = x.stride(0);
    long long x_channel_stride = x.stride(1);
    long long x_length_stride = x.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    maxpool1d_custom_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(channels),
        static_cast<int>(input_length),
        static_cast<int>(output_length),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(dilation),
        x_batch_stride,
        x_channel_stride,
        x_length_stride,
        stream
    );

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("isin_custom", &isin_custom, "isin custom op",
          py::arg("x"), py::arg("values"));
    m.def("maxpool1d_custom", &maxpool1d_custom, "maxpool1d custom op",
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"), py::arg("ceil_mode"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
