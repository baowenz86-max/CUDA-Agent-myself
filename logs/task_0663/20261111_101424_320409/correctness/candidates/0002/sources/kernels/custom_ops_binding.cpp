#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    int ndim,
    const int64_t* dims,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream);

extern "C" void sum_dim1_forward_launcher(
    const float* input,
    float* output,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream);

extern "C" void expand_forward_launcher(
    const float* input,
    float* output,
    int64_t B,
    int64_t out_features,
    int64_t H,
    int64_t W,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream);

static std::vector<int64_t> get_contiguous_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t acc = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = acc;
        acc *= shape[i];
    }
    return strides;
}

torch::Tensor relu6_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu6_forward: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "relu6_forward: input must be float32");
    TORCH_CHECK(x.dim() >= 1, "relu6_forward: input must have at least 1 dimension");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t total_elements = x.numel();
    int ndim = x.dim();
    std::vector<int64_t> dims(x.sizes().begin(), x.sizes().end());
    std::vector<int64_t> input_strides(x.strides().begin(), x.strides().end());
    std::vector<int64_t> output_strides(output.strides().begin(), output.strides().end());

    relu6_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        ndim,
        dims.data(),
        input_strides.data(),
        output_strides.data(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "relu6_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor sum_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "sum_dim1_forward: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "sum_dim1_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "sum_dim1_forward: input must be 4D [B, C, H, W]");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Output shape [B, 1, H, W], contiguous
    auto output = torch::empty({B, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    std::vector<int64_t> input_strides(x.strides().begin(), x.strides().end());
    std::vector<int64_t> output_strides(output.strides().begin(), output.strides().end());

    sum_dim1_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        C,
        H,
        W,
        input_strides.data(),
        output_strides.data(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sum_dim1_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor expand_forward(torch::Tensor x, int64_t out_features) {
    TORCH_CHECK(x.is_cuda(), "expand_forward: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "expand_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "expand_forward: input must be 4D [B, 1, H, W]");
    TORCH_CHECK(x.size(1) == 1, "expand_forward: dimension 1 must be size 1");

    int64_t B = x.size(0);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Create output shape [B, out_features, H, W] with stride[1] = 0 (broadcast view)
    std::vector<int64_t> output_shape = {B, out_features, H, W};
    std::vector<int64_t> output_strides = {x.stride(0), 0, x.stride(2), x.stride(3)};
    auto output = torch::empty_strided(output_shape, output_strides, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    std::vector<int64_t> input_strides(x.strides().begin(), x.strides().end());

    expand_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        out_features,
        H,
        W,
        input_strides.data(),
        output_strides.data(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "expand_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("relu6_forward", &relu6_forward, "ReLU6 activation", py::arg("x"));
    m.def("sum_dim1_forward", &sum_dim1_forward, "Sum over dimension 1", py::arg("x"));
    m.def("expand_forward", &expand_forward, "Expand dimension 1", py::arg("x"), py::arg("out_features"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
