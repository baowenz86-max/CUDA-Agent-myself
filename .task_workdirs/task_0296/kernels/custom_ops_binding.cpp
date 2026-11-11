#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

extern "C" void sign_launcher(
    float* output,
    const float* input,
    int64_t numel,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* shape,
    int ndim,
    cudaStream_t stream
);

torch::Tensor sign_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input dtype must be float32");
    TORCH_CHECK(x.dim() > 0, "Input must have at least one dimension");

    const int ndim = x.dim();
    TORCH_CHECK(ndim <= 8, "Too many dimensions, max 8 supported");

    std::vector<int64_t> input_strides(ndim);
    std::vector<int64_t> output_strides(ndim);
    std::vector<int64_t> shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = x.size(i);
        input_strides[i] = x.stride(i);
        output_strides[i] = x.stride(i);
    }

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sign_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        input_strides.data(),
        output_strides.data(),
        shape.data(),
        ndim,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("sign_forward", &sign_forward, "Sign function forward",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
