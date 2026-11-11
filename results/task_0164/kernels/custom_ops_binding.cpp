#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void maxpool3d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    cudaStream_t stream);

extern "C" void prod_dim_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    cudaStream_t stream);

// Helper to convert pybind11 iterable to std::vector<int64_t>
static std::vector<int64_t> to_int_vector(const pybind11::object& obj) {
    std::vector<int64_t> vec;
    for (auto item : obj) {
        vec.push_back(item.cast<int64_t>());
    }
    return vec;
}

// -----------------------------------------------------------------------------
// maxpool3d_forward
// -----------------------------------------------------------------------------
torch::Tensor maxpool3d_forward(
    torch::Tensor input,
    pybind11::object kernel_size_obj,
    pybind11::object stride_obj,
    pybind11::object padding_obj,
    int64_t dilation,
    bool ceil_mode)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (N, C, D, H, W)");

    // Convert kernel_size, stride, padding to vectors (support scalar or tuple)
    std::vector<int64_t> kernel_size = to_int_vector(kernel_size_obj);
    std::vector<int64_t> stride = to_int_vector(stride_obj);
    std::vector<int64_t> padding = to_int_vector(padding_obj);

    TORCH_CHECK(kernel_size.size() == 3, "kernel_size must have 3 elements");
    TORCH_CHECK(stride.size() == 3, "stride must have 3 elements");
    TORCH_CHECK(padding.size() == 3, "padding must have 3 elements");

    // dilation is scalar, replicate to 3 elements
    int64_t dilation_vals[3] = {dilation, dilation, dilation};

    input = input.contiguous();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int output_depth = (input_depth + 2 * padding[0] - dilation_vals[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    int output_height = (input_height + 2 * padding[1] - dilation_vals[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
    int output_width = (input_width + 2 * padding[2] - dilation_vals[2] * (kernel_size[2] - 1) - 1) / stride[2] + 1;

    TORCH_CHECK(output_depth > 0 && output_height > 0 && output_width > 0, "Invalid pooling output dimensions");

    auto output = torch::empty({batch_size, in_channels, output_depth, output_height, output_width}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    maxpool3d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_depth,
        input_height,
        input_width,
        static_cast<int>(kernel_size[0]), static_cast<int>(kernel_size[1]), static_cast<int>(kernel_size[2]),
        static_cast<int>(stride[0]), static_cast<int>(stride[1]), static_cast<int>(stride[2]),
        static_cast<int>(padding[0]), static_cast<int>(padding[1]), static_cast<int>(padding[2]),
        static_cast<int>(dilation_vals[0]), static_cast<int>(dilation_vals[1]), static_cast<int>(dilation_vals[2]),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "maxpool3d_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// prod_dim_forward
// -----------------------------------------------------------------------------
torch::Tensor prod_dim_forward(
    torch::Tensor input,
    int64_t dim,
    bool keepdim)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (N, C, D, H, W)");

    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(keepdim == true, "Only keepdim=True is supported");

    input = input.contiguous();

    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    auto output = torch::empty({batch_size, 1, depth, height, width}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    prod_dim_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "prod_dim_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("maxpool3d_forward", &maxpool3d_forward,
          "3D max pooling forward",
          pybind11::arg("input"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"),
          pybind11::arg("ceil_mode"));

    m.def("prod_dim_forward", &prod_dim_forward,
          "Product along dim=1 with keepdim=True",
          pybind11::arg("input"),
          pybind11::arg("dim"),
          pybind11::arg("keepdim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
