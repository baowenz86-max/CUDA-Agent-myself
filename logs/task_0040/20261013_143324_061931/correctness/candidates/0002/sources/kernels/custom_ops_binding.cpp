#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void channel_add_launcher(const float* x, const float* bias, float* y,
                                     int64_t N, int64_t C, int64_t L, cudaStream_t stream);
extern "C" void roll_last_dim_launcher(const float* x, float* y,
                                       int64_t N, int64_t C, int64_t L, int64_t shift, cudaStream_t stream);
extern "C" void linear_forward_launcher(const float* x, const float* weight, const float* bias,
                                        float* y, int64_t N, int64_t in_features, int64_t out_features,
                                        cudaStream_t stream);
extern "C" void flatten_to_2d_launcher(cudaStream_t stream);

// Declare conv_transpose1d_forward from conv_transpose1d.cpp
torch::Tensor conv_transpose1d_forward(torch::Tensor x,
                                       torch::Tensor weight,
                                       torch::Tensor bias,
                                       int64_t stride,
                                       int64_t padding,
                                       int64_t output_padding,
                                       int64_t groups = 1,
                                       int64_t dilation = 1);

// Helper to get current CUDA stream
static cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Check CUDA, dtype float32, and make contiguous if needed
torch::Tensor ensure_cuda_float_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    if (!t.is_contiguous()) {
        return t.contiguous();
    }
    return t;
}

// ===================== channel_add =====================
torch::Tensor channel_add(torch::Tensor x, torch::Tensor bias) {
    auto x_contig = ensure_cuda_float_contiguous(x, "x");
    auto bias_contig = ensure_cuda_float_contiguous(bias, "bias");

    TORCH_CHECK(x_contig.dim() == 3, "x must be 3D [N,C,L]");
    TORCH_CHECK(bias_contig.dim() == 2 && bias_contig.size(1) == 1, "bias must be [C,1]");
    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t L = x_contig.size(2);
    TORCH_CHECK(bias_contig.size(0) == C, "bias first dim must match channels");

    auto output = torch::empty_like(x_contig);
    cudaStream_t stream = getCurrentCUDAStream();
    channel_add_launcher(x_contig.data_ptr<float>(),
                         bias_contig.data_ptr<float>(),
                         output.data_ptr<float>(),
                         N, C, L, stream);
    return output;
}

// ===================== roll_last_dim =====================
torch::Tensor roll_last_dim(torch::Tensor x, int64_t shift) {
    auto x_contig = ensure_cuda_float_contiguous(x, "x");
    TORCH_CHECK(x_contig.dim() == 3, "x must be 3D [N,C,L]");
    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t L = x_contig.size(2);

    auto output = torch::empty_like(x_contig);
    cudaStream_t stream = getCurrentCUDAStream();
    roll_last_dim_launcher(x_contig.data_ptr<float>(),
                           output.data_ptr<float>(),
                           N, C, L, shift, stream);
    return output;
}

// ===================== flatten_to_2d =====================
torch::Tensor flatten_to_2d(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [N,C,L]");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    return x.view({x.size(0), -1});
}

// ===================== linear_forward =====================
torch::Tensor linear_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    auto x_contig = ensure_cuda_float_contiguous(x, "x");
    auto weight_contig = ensure_cuda_float_contiguous(weight, "weight");
    TORCH_CHECK(x_contig.dim() == 2, "x must be 2D [N, in_features]");
    TORCH_CHECK(weight_contig.dim() == 2, "weight must be 2D [out_features, in_features]");
    int64_t N = x_contig.size(0);
    int64_t in_features = x_contig.size(1);
    int64_t out_features = weight_contig.size(0);
    TORCH_CHECK(weight_contig.size(1) == in_features, "weight shape mismatch");

    torch::Tensor bias_contig;
    if (bias.defined()) {
        bias_contig = ensure_cuda_float_contiguous(bias, "bias");
        TORCH_CHECK(bias_contig.dim() == 1 && bias_contig.size(0) == out_features, "bias shape mismatch");
    }

    auto output = torch::empty({N, out_features}, x_contig.options());
    cudaStream_t stream = getCurrentCUDAStream();
    linear_forward_launcher(x_contig.data_ptr<float>(),
                            weight_contig.data_ptr<float>(),
                            bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr,
                            output.data_ptr<float>(),
                            N, in_features, out_features, stream);
    return output;
}

// Register all ops
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups") = 1, py::arg("dilation") = 1);
    m.def("channel_add", &channel_add, "Add channel-wise bias",
          py::arg("x"), py::arg("bias"));
    m.def("roll_last_dim", &roll_last_dim, "Roll last dim by shift",
          py::arg("x"), py::arg("shift"));
    m.def("flatten_to_2d", &flatten_to_2d, "Flatten to 2D view",
          py::arg("x"));
    m.def("linear_forward", &linear_forward, "Linear forward",
          py::arg("x"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
