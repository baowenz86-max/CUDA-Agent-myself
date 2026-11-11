#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_abs_launcher(const float* input, float* output, long long total, cudaStream_t stream);
extern "C" void custom_softmax_dim1_launcher(const float* input, float* output, int batch, int height, int width, cudaStream_t stream);

// Check tensor properties: CUDA, float32, 3D for softmax
static void check_common(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, name, " must be float32");
}

// Elementwise absolute value
torch::Tensor custom_abs_forward(torch::Tensor x) {
    check_common(x, "custom_abs_forward: x");
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_abs_launcher(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(), stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_abs_forward kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// Softmax along dimension 1
torch::Tensor custom_softmax_dim1_forward(torch::Tensor x) {
    check_common(x, "custom_softmax_dim1_forward: x");
    TORCH_CHECK(x.dim() == 3, "custom_softmax_dim1_forward: input must be 3D (batch, height, width)");
    int64_t batch = x.size(0);
    int64_t height = x.size(1);
    int64_t width = x.size(2);
    TORCH_CHECK(batch > 0 && height > 0 && width > 0, "custom_softmax_dim1_forward: non-empty dimensions required");
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_softmax_dim1_launcher(x.data_ptr<float>(), output.data_ptr<float>(),
                                 static_cast<int>(batch), static_cast<int>(height), static_cast<int>(width), stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_softmax_dim1_forward kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs_forward", &custom_abs_forward, "Element-wise absolute value",
          py::arg("x"));
    m.def("custom_softmax_dim1_forward", &custom_softmax_dim1_forward, "Softmax along dimension 1",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
