#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void abs_forward_launcher(
    float* out,
    const float* in,
    const long long* shape,
    const long long* strides,
    long long numel,
    cudaStream_t stream);

extern "C" void prelu_forward_launcher(
    float* out,
    const float* in,
    const float* weight,
    const long long* shape,
    const long long* strides,
    long long numel,
    cudaStream_t stream);

namespace {

// Convert torch::Tensor to contiguous float32 CUDA tensor and provide raw pointers
void prepare_tensor(torch::Tensor& tensor, float** data_ptr, long long* numel, std::vector<long long>& shape, std::vector<long long>& strides) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, "Tensor must be float32");
    tensor = tensor.contiguous();
    *data_ptr = tensor.data_ptr<float>();
    *numel = tensor.numel();
    shape.clear();
    strides.clear();
    for (int64_t dim : tensor.sizes()) {
        shape.push_back(static_cast<long long>(dim));
    }
    for (int64_t stride : tensor.strides()) {
        strides.push_back(static_cast<long long>(stride));
    }
    // Pad to 4 dimensions for the launcher (launcher assumes 4D)
    while (shape.size() < 4) {
        shape.insert(shape.begin(), 1);
        strides.insert(strides.begin(), 0);
    }
    TORCH_CHECK(shape.size() == 4, "Expected tensor with up to 4 dimensions");
}

} // anonymous namespace

// Binding for abs_forward
torch::Tensor abs_forward(torch::Tensor x) {
    float* in_ptr = nullptr;
    long long numel = 0;
    std::vector<long long> shape, strides;
    prepare_tensor(x, &in_ptr, &numel, shape, strides);

    auto out = torch::empty_like(x);
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kFloat32, "Output must be CUDA float32");
    float* out_ptr = out.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(out_ptr, in_ptr, shape.data(), strides.data(), numel, stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "abs_forward kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Binding for prelu_forward
torch::Tensor prelu_forward(torch::Tensor x, torch::Tensor weight) {
    float* in_ptr = nullptr;
    long long numel = 0;
    std::vector<long long> shape, strides;
    prepare_tensor(x, &in_ptr, &numel, shape, strides);

    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    weight = weight.contiguous();
    float* weight_ptr = weight.data_ptr<float>();

    // Validate weight size: either 1 or equal to channel dimension (shape[1])
    TORCH_CHECK(weight.numel() == 1 || weight.numel() == shape[1],
                "Weight must have 1 element or equal to number of channels");

    auto out = torch::empty_like(x);
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kFloat32, "Output must be CUDA float32");
    float* out_ptr = out.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    prelu_forward_launcher(out_ptr, in_ptr, weight_ptr, shape.data(), strides.data(), numel, stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "prelu_forward kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Elementwise absolute value",
          pybind11::arg("x"));
    m.def("prelu_forward", &prelu_forward, "PReLU activation",
          pybind11::arg("x"), pybind11::arg("weight"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
