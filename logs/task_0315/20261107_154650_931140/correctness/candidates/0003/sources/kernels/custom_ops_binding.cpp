#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void fused_tanh_leaky_relu_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* shape,
    int ndim,
    cudaStream_t stream);

torch::Tensor fused_tanh_leaky_relu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    auto output = torch::empty_like(x);

    int ndim = x.dim();
    std::vector<int64_t> input_strides(ndim);
    std::vector<int64_t> shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        input_strides[i] = x.stride(i);
        shape[i] = x.size(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // The launcher will use the host-side arrays to pass values to the kernel.
    // This is safe because the launcher reads them on the host and passes by value.
    fused_tanh_leaky_relu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        input_strides.data(),
        shape.data(),
        ndim,
        stream
    );

    // Optionally check errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_fused_tanh_leaky_relu(pybind11::module& m) {
    m.def("fused_tanh_leaky_relu", &fused_tanh_leaky_relu,
          "Fused Tanh + LeakyReLU (negative_slope=0.01)",
          pybind11::arg("x"));
}

REGISTER_BINDING(fused_tanh_leaky_relu, register_fused_tanh_leaky_relu);
