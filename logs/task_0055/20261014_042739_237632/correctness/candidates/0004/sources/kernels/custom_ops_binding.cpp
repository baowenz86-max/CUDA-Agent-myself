#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(float* output, const float* input,
                                      int64_t total_elements, int num_dims,
                                      const int64_t* dims, const int64_t* strides,
                                      cudaStream_t stream);

extern "C" void hardtanh_forward_launcher(float* output, const float* input,
                                           int64_t total_elements, int num_dims,
                                           const int64_t* dims, const int64_t* strides,
                                           float min_val, float max_val,
                                           cudaStream_t stream);

// Helper to copy sizes to int64_t vector
static std::vector<int64_t> get_sizes(const torch::Tensor& t) {
    return std::vector<int64_t>(t.sizes().begin(), t.sizes().end());
}

// Helper to copy strides to int64_t vector
static std::vector<int64_t> get_strides(const torch::Tensor& t) {
    return std::vector<int64_t>(t.strides().begin(), t.strides().end());
}

torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    std::vector<int64_t> dims = get_sizes(x);
    std::vector<int64_t> strides = get_strides(x);
    int64_t total_elements = x.numel();
    int num_dims = static_cast<int>(dims.size());

    relu_forward_launcher(output.data_ptr<float>(), x.data_ptr<float>(),
                          total_elements, num_dims, dims.data(), strides.data(),
                          stream);

    return output;
}

torch::Tensor hardtanh_forward(torch::Tensor x, double min_val = -1.0, double max_val = 1.0) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    std::vector<int64_t> dims = get_sizes(x);
    std::vector<int64_t> strides = get_strides(x);
    int64_t total_elements = x.numel();
    int num_dims = static_cast<int>(dims.size());

    hardtanh_forward_launcher(output.data_ptr<float>(), x.data_ptr<float>(),
                              total_elements, num_dims, dims.data(), strides.data(),
                              static_cast<float>(min_val), static_cast<float>(max_val),
                              stream);

    return output;
}

void register_custom_ops(py::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward", py::arg("x"));
    m.def("hardtanh_forward", &hardtanh_forward, "Hardtanh forward",
          py::arg("x"), py::arg("min_val") = -1.0, py::arg("max_val") = 1.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);