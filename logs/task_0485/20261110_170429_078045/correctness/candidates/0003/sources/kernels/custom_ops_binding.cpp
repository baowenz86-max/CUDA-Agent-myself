#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_launcher(
    const float* input,
    float* output,
    int inner_dim,
    int total_rows,
    cudaStream_t stream);

extern "C" void tanh_launcher(
    const float* input,
    float* output,
    long long n,
    cudaStream_t stream);

// log_softmax_cuda: input may be CPU or CUDA float32; output is CUDA float32 tensor.
torch::Tensor log_softmax_cuda(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "log_softmax_cuda: input must be float32");
    TORCH_CHECK(x.dim() >= 1, "log_softmax_cuda: input must have at least 1 dimension");
    TORCH_CHECK(dim == x.dim() - 1, "log_softmax_cuda: only supports dim equal to last dimension");

    // Ensure input is on CUDA
    auto x_cuda = x.is_cuda() ? x : x.cuda();

    auto inner_dim = x_cuda.size(-1);
    auto total_rows = x_cuda.numel() / inner_dim;

    // Allocate output on CUDA with same shape
    auto output = torch::empty_like(x_cuda);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_launcher(
        x_cuda.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(inner_dim),
        static_cast<int>(total_rows),
        stream);

    return output;
}

// tanh_cuda: input is CUDA float32, output is same shape CUDA float32.
torch::Tensor tanh_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "tanh_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "tanh_cuda: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "tanh_cuda: input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    tanh_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_cuda", &log_softmax_cuda, "log_softmax along last dimension",
          py::arg("x"), py::arg("dim"));
    m.def("tanh_cuda", &tanh_cuda, "elementwise tanh",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
