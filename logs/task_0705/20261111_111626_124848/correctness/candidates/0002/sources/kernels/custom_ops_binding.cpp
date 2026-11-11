// Binding for custom CUDA kernels: argmax_dim, float_cast, batch_norm_eval
// This file contains no CUDA kernels, only launcher wrappers and pybind registration.

#include <torch/types.h>
#include <torch/nn/functional/activation.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

#include "../binding_registry.h"

// ---- Declare launchers from custom_ops.cu ----
extern "C" void argmax_dim_launcher(
    const float* input,
    int64_t* indices,
    float* values,
    int batch,
    int length,
    int channels,
    int dim,
    cudaStream_t stream);

extern "C" void float_cast_launcher(
    const int64_t* input,
    float* output,
    int64_t n,
    cudaStream_t stream);

extern "C" void batch_norm_eval_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    float* output,
    int batch,
    int channels,
    cudaStream_t stream);

// ---- Helper: ensure CUDA tensor, dtype, contiguous ----
static torch::Tensor ensure_cuda_float32_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    if (!t.is_contiguous()) {
        return t.contiguous();
    }
    return t;
}

static torch::Tensor ensure_cuda_int64_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kInt64, name, " must be int64");
    if (!t.is_contiguous()) {
        return t.contiguous();
    }
    return t;
}

// ---- argmax_dim wrapper: returns torch.return_types.max ----
torch::return_types::max argmax_dim_forward(torch::Tensor x, int64_t dim) {
    // Validate input shape: expected [batch, length, channels]
    TORCH_CHECK(x.dim() == 3, "argmax_dim: input must be 3D [batch, length, channels]");
    TORCH_CHECK(dim == 1, "argmax_dim: only dim=1 is supported");

    // Ensure contiguous and correct dtype
    x = ensure_cuda_float32_contiguous(x, "argmax_dim input");

    int batch = static_cast<int>(x.size(0));
    int length = static_cast<int>(x.size(1));
    int channels = static_cast<int>(x.size(2));

    // Allocate outputs: [batch, channels] float32 and int64
    auto values = torch::empty({batch, channels}, x.options().dtype(torch::kFloat32));
    auto indices = torch::empty({batch, channels}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    argmax_dim_launcher(
        x.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        values.data_ptr<float>(),
        batch,
        length,
        channels,
        static_cast<int>(dim),
        stream
    );

    // Return torch.return_types.max container
    return {std::move(values), std::move(indices)};
}

// ---- float_cast wrapper ----
torch::Tensor float_cast_forward(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "float_cast: input must be 2D [batch, channels]");
    x = ensure_cuda_int64_contiguous(x, "float_cast input");

    auto output = torch::empty_like(x, x.options().dtype(torch::kFloat32));

    int64_t n = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    float_cast_launcher(
        x.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        n,
        stream
    );

    return output;
}

// ---- batch_norm_eval wrapper ----
torch::Tensor batch_norm_eval_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    TORCH_CHECK(input.dim() == 2, "batch_norm_eval: input must be 2D [batch, channels]");
    input = ensure_cuda_float32_contiguous(input, "batch_norm_eval input");
    weight = ensure_cuda_float32_contiguous(weight, "batch_norm_eval weight");
    bias = ensure_cuda_float32_contiguous(bias, "batch_norm_eval bias");
    running_mean = ensure_cuda_float32_contiguous(running_mean, "batch_norm_eval running_mean");
    running_var = ensure_cuda_float32_contiguous(running_var, "batch_norm_eval running_var");

    int batch = static_cast<int>(input.size(0));
    int channels = static_cast<int>(input.size(1));

    // Ensure parameter tensors have size [channels]
    TORCH_CHECK(weight.numel() == channels, "batch_norm_eval: weight size mismatch");
    TORCH_CHECK(bias.numel() == channels, "batch_norm_eval: bias size mismatch");
    TORCH_CHECK(running_mean.numel() == channels, "batch_norm_eval: running_mean size mismatch");
    TORCH_CHECK(running_var.numel() == channels, "batch_norm_eval: running_var size mismatch");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm_eval_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        output.data_ptr<float>(),
        batch,
        channels,
        stream
    );

    return output;
}

// ---- Registration ----
void register_custom_ops(pybind11::module& m) {
    m.def("argmax_dim", &argmax_dim_forward,
          "argmax_dim: torch.max(x, dim=1) returning torch.return_types.max",
          py::arg("x"), py::arg("dim"));
    m.def("float_cast", &float_cast_forward,
          "float_cast: cast int64 tensor to float32",
          py::arg("x"));
    m.def("batch_norm_eval", &batch_norm_eval_forward,
          "batch_norm_eval: apply eval-mode BatchNorm1d",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
