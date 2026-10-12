#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void prelu_forward_launcher(
    const float* input,
    const float* weight,
    float* output,
    int64_t total_elements,
    int64_t channel_stride,
    int64_t channels,
    cudaStream_t stream);

extern "C" void bilinear_interpolate_forward_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    float scale_h,
    float scale_w,
    cudaStream_t stream);

extern "C" void argmax_dim_forward_launcher(
    const float* input,
    int64_t* output,
    int64_t N,
    int64_t C,
    int64_t spatial_size,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// PReLU forward binding
// -----------------------------------------------------------------------------
torch::Tensor prelu_forward(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D (C,)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    TORCH_CHECK(weight.size(0) == C, "weight size must match input channel dimension");

    auto output = torch::empty_like(input);
    int64_t total_elements = input.numel();
    int64_t channel_stride = H * W;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    prelu_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        channel_stride,
        C,
        stream);

    return output;
}

// -----------------------------------------------------------------------------
// Bilinear interpolate forward binding (align_corners=false)
// -----------------------------------------------------------------------------
torch::Tensor bilinear_interpolate_forward(torch::Tensor input, double scale_factor) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(scale_factor > 0, "scale_factor must be positive");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    // PyTorch bilinear with align_corners=False uses output size = floor(input_size * scale_factor)
    int64_t H_out = static_cast<int64_t>(std::floor(H_in * scale_factor));
    int64_t W_out = static_cast<int64_t>(std::floor(W_in * scale_factor));

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    float scale_h = static_cast<float>(scale_factor);
    float scale_w = static_cast<float>(scale_factor);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bilinear_interpolate_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        scale_h,
        scale_w,
        stream);

    return output;
}

// -----------------------------------------------------------------------------
// Argmax over dimension 1 (keepdim=True) binding
// -----------------------------------------------------------------------------
torch::Tensor argmax_dim_forward(torch::Tensor input, int64_t dim, bool keepdim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");
    TORCH_CHECK(keepdim == true, "only keepdim=true is supported");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t spatial_size = H * W;

    auto output = torch::empty({N, 1, H, W}, input.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    argmax_dim_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        N,
        C,
        spatial_size,
        stream);

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("prelu_forward", &prelu_forward, "PReLU forward",
          py::arg("input"), py::arg("weight"));
    m.def("bilinear_interpolate_forward", &bilinear_interpolate_forward,
          "Bilinear interpolation forward",
          py::arg("input"), py::arg("scale_factor"));
    m.def("argmax_dim_forward", &argmax_dim_forward,
          "Argmax over dimension 1 with keepdim",
          py::arg("input"), py::arg("dim"), py::arg("keepdim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
