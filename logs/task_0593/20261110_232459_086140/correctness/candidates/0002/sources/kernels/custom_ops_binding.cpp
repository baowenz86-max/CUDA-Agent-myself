#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void abs_forward_launcher(
    float* output,
    const float* input,
    const int* dims,
    const long long* strides_in,
    const long long* strides_out,
    cudaStream_t stream);

extern "C" void layer_norm_forward_launcher(
    float* output,
    const float* input,
    const int* dims,
    const long long* strides_in,
    const long long* strides_out,
    float eps,
    cudaStream_t stream);

// Helper to create a dims array and strides arrays from tensor
static void tensor_dims_and_strides(const torch::Tensor& t, std::vector<int>& dims,
                                    std::vector<long long>& strides_in, std::vector<long long>& strides_out) {
    TORCH_CHECK(t.dim() == 4, "Expected 4D tensor [B,C,H,W], got ", t.dim(), "D");
    dims.resize(4);
    for (int i = 0; i < 4; ++i) {
        dims[i] = (int)t.size(i);
    }
    strides_in.resize(4);
    strides_out.resize(4);
    for (int i = 0; i < 4; ++i) {
        strides_in[i] = (long long)t.stride(i);
        strides_out[i] = (long long)t.stride(i); // output will have same strides
    }
}

// abs_forward binding
torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Expected 4D tensor, got ", x.dim(), "D");

    // Output with same shape, strides, dtype, device
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    std::vector<int> dims;
    std::vector<long long> strides_in, strides_out;
    tensor_dims_and_strides(x, dims, strides_in, strides_out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        dims.data(),
        strides_in.data(),
        strides_out.data(),
        stream);

    return output;
}

// layer_norm_forward binding
torch::Tensor layer_norm_forward(torch::Tensor x, int64_t normalized_shape, double eps) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Expected 4D tensor, got ", x.dim(), "D");
    TORCH_CHECK(normalized_shape == 3, "Only normalized_shape=3 (C,H,W) is supported");
    // The launcher normalizes over the last 3 dimensions, regardless of their actual sizes.
    // No need to check trailing dims equality; contract specifies [B,C,H,W] and normalized_shape=(C,H,W).

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    std::vector<int> dims;
    std::vector<long long> strides_in, strides_out;
    tensor_dims_and_strides(x, dims, strides_in, strides_out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    layer_norm_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        dims.data(),
        strides_in.data(),
        strides_out.data(),
        (float)eps,
        stream);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Element-wise absolute value",
          py::arg("x"));
    m.def("layer_norm_forward", &layer_norm_forward, "Layer normalization over trailing dims",
          py::arg("x"), py::arg("normalized_shape"), py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
