#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void log_softmax_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int dim,
    cudaStream_t stream);

extern "C" void softmax_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int dim,
    cudaStream_t stream);

extern "C" void reflection_pad2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int pad,
    cudaStream_t stream);

// Helper to check tensor properties and return contiguous version if needed
static torch::Tensor ensure_contiguous_float_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, name, " must be float32");
    if (!x.is_contiguous()) {
        return x.contiguous();
    }
    return x;
}

// log_softmax forward (dim=1)
torch::Tensor log_softmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(dim == 1, "log_softmax only supports dim=1");
    x = ensure_contiguous_float_cuda(x, "log_softmax input");
    TORCH_CHECK(x.dim() == 4, "log_softmax input must be 4D (N,C,H,W)");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    log_softmax_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(dim),
        stream);
    return output;
}

// softmax forward (dim=1)
torch::Tensor softmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(dim == 1, "softmax only supports dim=1");
    x = ensure_contiguous_float_cuda(x, "softmax input");
    TORCH_CHECK(x.dim() == 4, "softmax input must be 4D (N,C,H,W)");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(dim),
        stream);
    return output;
}

// reflection_pad2d forward
torch::Tensor reflection_pad2d_forward(torch::Tensor x, int64_t padding) {
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    x = ensure_contiguous_float_cuda(x, "reflection_pad2d input");
    TORCH_CHECK(x.dim() == 4, "reflection_pad2d input must be 4D (N,C,H,W)");
    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int H_out = H_in + 2 * static_cast<int>(padding);
    int W_out = W_in + 2 * static_cast<int>(padding);
    auto output = torch::empty({N, C, H_out, W_out}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H_in, W_in,
        static_cast<int>(padding),
        stream);
    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax", &log_softmax_forward,
          "Compute log_softmax along dim=1",
          py::arg("x"), py::arg("dim"));
    m.def("softmax", &softmax_forward,
          "Compute softmax along dim=1",
          py::arg("x"), py::arg("dim"));
    m.def("reflection_pad2d", &reflection_pad2d_forward,
          "Apply reflection padding of given size",
          py::arg("x"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
