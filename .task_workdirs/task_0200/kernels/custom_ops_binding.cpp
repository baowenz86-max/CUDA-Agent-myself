#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void hardshrink_forward_launcher(const float* input,
                                             float* output,
                                             int64_t total,
                                             cudaStream_t stream);

extern "C" void elu_forward_launcher(const float* input,
                                      float* output,
                                      int64_t total,
                                      cudaStream_t stream);

extern "C" void hardshrink_elu_forward_launcher(const float* input,
                                                 float* output,
                                                 int64_t total,
                                                 cudaStream_t stream);

namespace {

void validate_cuda_float32(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input must have at least one element");
    // Ensure no overlapping strides that would break elementwise kernel correctness.
    // A simple contiguity check is not sufficient for broadcasted views; but our
    // elementwise kernel reads with a single linear index assuming no overlap.
    // Overlapping strides typically indicate a broadcasted tensor, which we cannot handle.
    // We require the input to be non-overlapping: the stride of each dimension must either
    // be the product of sizes of subsequent dimensions (contiguous) or the tensor has
    // fewer than 2 dims where overlap is impossible, and no stride is zero.
    // This simple check ensures no zero strides and for contiguous case works.
    // The model will pass contiguous tensors, so this is acceptable.
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous for elementwise kernel");
}

} // anonymous namespace

torch::Tensor hardshrink_forward(torch::Tensor x) {
    validate_cuda_float32(x);
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    hardshrink_forward_launcher(x.data_ptr<float>(),
                                output.data_ptr<float>(),
                                x.numel(),
                                stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "hardshrink_forward kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

torch::Tensor elu_forward(torch::Tensor x) {
    validate_cuda_float32(x);
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    elu_forward_launcher(x.data_ptr<float>(),
                         output.data_ptr<float>(),
                         x.numel(),
                         stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "elu_forward kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

torch::Tensor hardshrink_elu_forward(torch::Tensor x) {
    validate_cuda_float32(x);
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    hardshrink_elu_forward_launcher(x.data_ptr<float>(),
                                    output.data_ptr<float>(),
                                    x.numel(),
                                    stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "hardshrink_elu_forward kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops_binding(pybind11::module& m) {
    m.def("hardshrink_forward", &hardshrink_forward,
          "hardshrink_forward(x: Tensor) -> Tensor",
          py::arg("x"));
    m.def("elu_forward", &elu_forward,
          "elu_forward(x: Tensor) -> Tensor",
          py::arg("x"));
    m.def("hardshrink_elu_forward", &hardshrink_elu_forward,
          "hardshrink_elu_forward(x: Tensor) -> Tensor",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops_binding);
