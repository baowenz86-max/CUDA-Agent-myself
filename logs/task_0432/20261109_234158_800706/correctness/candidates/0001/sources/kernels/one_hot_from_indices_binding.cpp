#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void one_hot_from_indices_launcher(
    const int64_t* indices,
    int64_t* output,
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t num_classes,
    int64_t indices_stride_n,
    int64_t indices_stride_h,
    int64_t indices_stride_w,
    cudaStream_t stream);

torch::Tensor one_hot_from_indices(torch::Tensor indices, int64_t num_classes) {
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(indices.dim() == 4, "indices must be 4D (N, 1, H, W)");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(num_classes > 0, "num_classes must be positive");

    int64_t N = indices.size(0);
    TORCH_CHECK(indices.size(1) == 1, "indices must have size 1 in dim 1");
    int64_t H = indices.size(2);
    int64_t W = indices.size(3);

    // Output shape [N, 1, H, W, num_classes]
    auto output = torch::empty({N, 1, H, W, num_classes},
                               indices.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Strides are element counts, not bytes
    int64_t indices_stride_n = indices.stride(0);
    int64_t indices_stride_h = indices.stride(2);
    int64_t indices_stride_w = indices.stride(3);

    one_hot_from_indices_launcher(
        indices.data_ptr<int64_t>(),
        output.data_ptr<int64_t>(),
        N, H, W, num_classes,
        indices_stride_n, indices_stride_h, indices_stride_w,
        stream);

    return output;
}

void register_one_hot_from_indices_binding(pybind11::module& m) {
    m.def("one_hot_from_indices", &one_hot_from_indices,
          "Convert indices to one-hot encoding along last dimension",
          py::arg("indices"), py::arg("num_classes"));
}

REGISTER_BINDING(one_hot_from_indices, register_one_hot_from_indices_binding);
