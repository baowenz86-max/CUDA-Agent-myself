#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void hardsigmoid_forward_launcher(
    const float* input,
    float* output,
    int total_elements,
    int ndims,
    const int* shape_device,
    const int* strides_device,
    cudaStream_t stream);

torch::Tensor hardsigmoid_forward(torch::Tensor x) {
    // Check device and dtype
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Get sizes and strides
    auto sizes = x.sizes();
    auto strides = x.strides();
    int ndims = x.dim();
    int total_elements = x.numel();

    // Allocate output with same sizes and strides
    auto output = torch::empty_strided(sizes, strides, x.options());

    // Copy shape and strides to device
    int* shape_host = new int[ndims];
    int* strides_host = new int[ndims];
    for (int i = 0; i < ndims; ++i) {
        shape_host[i] = sizes[i];
        strides_host[i] = strides[i];
    }

    int* shape_device = nullptr;
    int* strides_device = nullptr;
    cudaMalloc(&shape_device, ndims * sizeof(int));
    cudaMalloc(&strides_device, ndims * sizeof(int));
    cudaMemcpy(shape_device, shape_host, ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(strides_device, strides_host, ndims * sizeof(int), cudaMemcpyHostToDevice);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    hardsigmoid_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        ndims,
        shape_device,
        strides_device,
        stream);

    // Free device memory and host memory
    cudaFree(shape_device);
    cudaFree(strides_device);
    delete[] shape_host;
    delete[] strides_host;

    return output;
}

void register_hardsigmoid(pybind11::module& m) {
    m.def("hardsigmoid_forward", &hardsigmoid_forward,
          "Hardsigmoid forward (CUDA)",
          pybind11::arg("x"));
}

REGISTER_BINDING(hardsigmoid, register_hardsigmoid);
