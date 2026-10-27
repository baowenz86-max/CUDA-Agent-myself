#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_conv1d_launcher(const float* input,
                                       const float* weight,
                                       const float* bias,
                                       float* output,
                                       int batch,
                                       int in_channels,
                                       int out_channels,
                                       int in_len,
                                       int kernel_size,
                                       int out_len,
                                       int has_bias,
                                       cudaStream_t stream);

extern "C" void custom_adaptive_avg_pool1d_launcher(const float* input,
                                                    float* output,
                                                    int batch,
                                                    int channels,
                                                    int in_len,
                                                    int out_len,
                                                    cudaStream_t stream);

extern "C" void custom_slice_launcher(const float* input,
                                      float* output,
                                      int batch,
                                      int channels,
                                      int in_len,
                                      int dim,
                                      int start,
                                      int end,
                                      cudaStream_t stream);

extern "C" void custom_std_launcher(const float* input,
                                    float* output,
                                    int batch,
                                    int channels,
                                    int length,
                                    cudaStream_t stream);

static cudaStream_t getCurrentCudaStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

torch::Tensor custom_conv1d(torch::Tensor input,
                            torch::Tensor weight,
                            torch::Tensor bias,
                            int64_t stride,
                            int64_t padding,
                            int64_t dilation,
                            int64_t groups) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(stride == 1 && padding == 0 && dilation == 1 && groups == 1,
               "Only stride=1, padding=0, dilation=1, groups=1 supported");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    }
    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_len = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_len = in_len - kernel_size + 1;
    auto output = torch::empty({batch, out_channels, out_len}, input.options());
    cudaStream_t stream = getCurrentCudaStream();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    int has_bias = bias.defined() ? 1 : 0;
    custom_conv1d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        in_len,
        kernel_size,
        out_len,
        has_bias,
        stream
    );
    return output;
}

torch::Tensor custom_adaptive_avg_pool1d(torch::Tensor input, int64_t output_size) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    auto output = torch::empty({input.size(0), input.size(1), output_size}, input.options());
    cudaStream_t stream = getCurrentCudaStream();
    custom_adaptive_avg_pool1d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        input.size(1),
        input.size(2),
        static_cast<int>(output_size),
        stream
    );
    return output;
}

torch::Tensor custom_slice(torch::Tensor input, int64_t dim, int64_t start, int64_t end) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(dim == 2, "only dim=2 slice is supported");
    TORCH_CHECK(start >= 0 && end <= input.size(2) && start < end, "invalid slice range");
    auto output = torch::empty({input.size(0), input.size(1), end - start}, input.options());
    cudaStream_t stream = getCurrentCudaStream();
    custom_slice_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        input.size(1),
        input.size(2),
        static_cast<int>(dim),
        static_cast<int>(start),
        static_cast<int>(end),
        stream
    );
    return output;
}

torch::Tensor custom_std(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(dim == 2, "only dim=2 reduction is supported");
    auto output = torch::empty({input.size(0), input.size(1)}, input.options());
    cudaStream_t stream = getCurrentCudaStream();
    custom_std_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        input.size(1),
        input.size(2),
        stream
    );
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_conv1d", &custom_conv1d,
          "Custom 1D convolution (simple kernel)",
          py::arg("input"), py::arg("weight"), py::arg("bias").none(true),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"), py::arg("groups"));
    m.def("custom_adaptive_avg_pool1d", &custom_adaptive_avg_pool1d,
          "Custom adaptive average pooling 1D",
          py::arg("input"), py::arg("output_size"));
    m.def("custom_slice", &custom_slice,
          "Custom slice along dimension",
          py::arg("input"), py::arg("dim"), py::arg("start"), py::arg("end"));
    m.def("custom_std", &custom_std,
          "Custom standard deviation reduction",
          py::arg("input"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
