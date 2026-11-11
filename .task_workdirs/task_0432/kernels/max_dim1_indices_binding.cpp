#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void max_dim1_indices_launcher(
    const float* input,
    float* values,
    int64_t* indices,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t input_stride_n,
    int64_t input_stride_c,
    int64_t input_stride_h,
    int64_t input_stride_w,
    int64_t output_stride_n,
    int64_t output_stride_h,
    int64_t output_stride_w,
    cudaStream_t stream);

std::tuple<torch::Tensor, torch::Tensor> max_dim1_indices(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto values = torch::empty({N, 1, H, W}, x.options());
    auto indices = torch::empty({N, 1, H, W},
                                x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Strides are element counts, not bytes
    int64_t input_stride_n = x.stride(0);
    int64_t input_stride_c = x.stride(1);
    int64_t input_stride_h = x.stride(2);
    int64_t input_stride_w = x.stride(3);

    // Output tensor is contiguous by construction
    int64_t output_stride_n = values.stride(0);
    int64_t output_stride_h = values.stride(2);
    int64_t output_stride_w = values.stride(3);

    max_dim1_indices_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        N, C, H, W,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_h, output_stride_w,
        stream);

    return std::make_tuple(values, indices);
}

void register_max_dim1_indices_binding(pybind11::module& m) {
    m.def("max_dim1_indices", &max_dim1_indices,
          "Compute max and indices along dim 1 with keepdim=True",
          py::arg("x"));
}

REGISTER_BINDING(max_dim1_indices, register_max_dim1_indices_binding);
