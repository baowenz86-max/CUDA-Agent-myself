#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void one_hot_forward_launcher(const long long* x,
                                          long long* out,
                                          long long num_elements,
                                          int num_classes,
                                          int L,
                                          cudaStream_t stream);

extern "C" void maximum_scalar_forward_launcher(const float* x,
                                                  float* out,
                                                  long long num_elements,
                                                  cudaStream_t stream);

extern "C" void bucketize_forward_launcher(const float* values,
                                             const float* boundaries,
                                             long long* out,
                                             long long num_elements,
                                             int nb,
                                             cudaStream_t stream);

// one_hot_forward: x (int64) shape [B, L], returns int64 shape [B, L, num_classes]
torch::Tensor one_hot_forward(torch::Tensor x, int64_t num_classes) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kInt64, "Input x must be int64");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "Input x must have at least 1 dimension");

    auto sizes = x.sizes();
    auto output_sizes = sizes.vec();
    output_sizes.push_back(num_classes);
    auto out = torch::empty(output_sizes, x.options());

    long long num_elements = x.numel();
    int L = (x.dim() == 1) ? x.size(0) : x.size(-1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    one_hot_forward_launcher(
        reinterpret_cast<long long*>(x.data_ptr()),
        reinterpret_cast<long long*>(out.data_ptr()),
        num_elements,
        static_cast<int>(num_classes),
        L,
        stream
    );

    return out;
}

// maximum_scalar_forward: x (float32) shape [B, L, C], returns float32 same shape
torch::Tensor maximum_scalar_forward(torch::Tensor x, double scalar) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(scalar == 0.0, "Only scalar=0.0 is supported");

    auto out = torch::empty_like(x);
    long long num_elements = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    maximum_scalar_forward_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements,
        stream
    );

    return out;
}

// bucketize_forward: values (float32) shape [B, L], boundaries (float32) 1D, returns int64 shape [B, L]
torch::Tensor bucketize_forward(torch::Tensor values, torch::Tensor boundaries) {
    TORCH_CHECK(values.is_cuda(), "Input values must be a CUDA tensor");
    TORCH_CHECK(values.dtype() == torch::kFloat32, "Input values must be float32");
    TORCH_CHECK(values.is_contiguous(), "Input values must be contiguous");
    TORCH_CHECK(boundaries.is_cuda(), "Boundaries must be a CUDA tensor");
    TORCH_CHECK(boundaries.dtype() == torch::kFloat32, "Boundaries must be float32");
    TORCH_CHECK(boundaries.is_contiguous(), "Boundaries must be contiguous");
    TORCH_CHECK(boundaries.dim() == 1, "Boundaries must be 1-dimensional");

    auto out = torch::empty_like(values, values.options().dtype(torch::kInt64));
    long long num_elements = values.numel();
    int nb = static_cast<int>(boundaries.numel());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bucketize_forward_launcher(
        values.data_ptr<float>(),
        boundaries.data_ptr<float>(),
        reinterpret_cast<long long*>(out.data_ptr()),
        num_elements,
        nb,
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("one_hot_forward", &one_hot_forward,
          "One-hot encode int64 tensor",
          pybind11::arg("x"), pybind11::arg("num_classes"));
    m.def("maximum_scalar_forward", &maximum_scalar_forward,
          "Element-wise maximum with scalar",
          pybind11::arg("x"), pybind11::arg("scalar"));
    m.def("bucketize_forward", &bucketize_forward,
          "Bucketize values using boundaries",
          pybind11::arg("values"), pybind11::arg("boundaries"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
