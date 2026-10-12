// Binding for custom reflection_pad2d, ge_scalar, remainder_scalar kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_forward(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    int out_height, int out_width,
    cudaStream_t stream);

extern "C" void ge_scalar_forward(
    const float* input,
    unsigned char* output,
    float scalar,
    int total_elements,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    int H, int W, int C,
    cudaStream_t stream);

extern "C" void remainder_scalar_forward(
    const unsigned char* input,
    long long* output,
    double scalar,
    int total_elements,
    cudaStream_t stream);

// Wrapper for reflection_pad2d
static torch::Tensor reflection_pad2d_forward_wrapper(torch::Tensor input, std::vector<int64_t> padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(padding.size() == 4, "padding must have 4 elements");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t pad_left = padding[0];
    int64_t pad_right = padding[1];
    int64_t pad_top = padding[2];
    int64_t pad_bottom = padding[3];

    int64_t out_H = H + pad_top + pad_bottom;
    int64_t out_W = W + pad_left + pad_right;
    TORCH_CHECK(out_H > 0 && out_W > 0, "Output dimensions must be positive");

    auto output = torch::empty({N, C, out_H, out_W}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)N, (int)C, (int)H, (int)W,
        (int)pad_left, (int)pad_right, (int)pad_top, (int)pad_bottom,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        (int)out_H, (int)out_W,
        stream);

    return output;
}

// Wrapper for ge_scalar (input >= scalar)
static torch::Tensor ge_scalar_forward_wrapper(torch::Tensor input, double scalar) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t total_elements = N * C * H * W;

    auto output = torch::empty({N, C, H, W}, input.options().dtype(torch::kBool));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    ge_scalar_forward(
        input.data_ptr<float>(),
        reinterpret_cast<unsigned char*>(output.data_ptr<bool>()),
        (float)scalar,
        (int)total_elements,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        (int)H, (int)W, (int)C,
        stream);

    return output;
}

// Wrapper for remainder_scalar (input % scalar)
static torch::Tensor remainder_scalar_forward_wrapper(torch::Tensor input, double scalar) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kBool, "Input must be bool");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t total_elements = N * C * H * W;

    auto output = torch::empty({N, C, H, W}, input.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    remainder_scalar_forward(
        reinterpret_cast<const unsigned char*>(input.data_ptr<bool>()),
        output.data_ptr<int64_t>(),
        scalar,
        (int)total_elements,
        stream);

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward_wrapper, "ReflectionPad2d forward",
          pybind11::arg("input"), pybind11::arg("padding"));
    m.def("ge_scalar_forward", &ge_scalar_forward_wrapper, "GreaterEqual scalar forward",
          pybind11::arg("input"), pybind11::arg("scalar"));
    m.def("remainder_scalar_forward", &remainder_scalar_forward_wrapper, "Remainder scalar forward",
          pybind11::arg("input"), pybind11::arg("scalar"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
