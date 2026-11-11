#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void stack_like_launcher(
    float* output,
    const float* input,
    int total_elems,
    int ndims,
    const int64_t* dims,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream);

torch::Tensor stack_like(torch::Tensor x) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "stack_like: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "stack_like: input dtype must be float32");
    TORCH_CHECK(x.dim() >= 1, "stack_like: input must have at least one dimension");

    // Ensure output is contiguous with same shape
    auto output = torch::empty_like(x);

    // Collect dims and strides
    int ndims = x.dim();
    std::vector<int64_t> dims_vec(ndims);
    std::vector<int64_t> input_strides_vec(ndims);
    std::vector<int64_t> output_strides_vec(ndims);

    for (int i = 0; i < ndims; ++i) {
        dims_vec[i] = x.size(i);
        input_strides_vec[i] = x.stride(i);
        output_strides_vec[i] = output.stride(i);
    }

    int total_elems = x.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    stack_like_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        total_elems,
        ndims,
        dims_vec.data(),
        input_strides_vec.data(),
        output_strides_vec.data(),
        stream
    );

    // Check for kernel errors (optional but recommended)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "stack_like: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_stack_like(pybind11::module& m) {
    m.def("stack_like", &stack_like, "Copy input tensor to contiguous output preserving values",
          py::arg("x"));
}

REGISTER_BINDING(stack_like, register_stack_like);
