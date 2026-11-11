#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void custom_stack_relu_launcher(
    const float* input,
    float* output,
    int64_t outer,
    int64_t inner,
    int64_t num_repeats,
    cudaStream_t stream);

// Wrapper function: torch.stack along dim followed by relu
torch::Tensor custom_stack_relu(torch::Tensor x, int64_t dim, int64_t num_repeats) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension");

    int64_t ndim = x.dim();
    // Normalize dim to [0, ndim]
    if (dim < 0) dim += ndim + 1;
    TORCH_CHECK(dim >= 0 && dim <= ndim, "dim out of range for stacking");

    // Calculate outer and inner sizes
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= x.size(i);
    int64_t inner = 1;
    for (int64_t i = dim; i < ndim; ++i) inner *= x.size(i);

    // Build output shape: insert num_repeats at dim
    std::vector<int64_t> output_shape;
    output_shape.reserve(ndim + 1);
    for (int64_t i = 0; i < dim; ++i) output_shape.push_back(x.size(i));
    output_shape.push_back(num_repeats);
    for (int64_t i = dim; i < ndim; ++i) output_shape.push_back(x.size(i));

    // Allocate output tensor (contiguous, float32, on same device)
    auto output = torch::empty(output_shape, x.options());

    // Ensure input is contiguous; if not, make it contiguous
    auto x_contig = x.contiguous();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_stack_relu_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        outer,
        inner,
        num_repeats,
        stream
    );

    // Check for CUDA errors after launch (optional but good practice)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_stack_relu(pybind11::module& m) {
    m.def("custom_stack_relu", &custom_stack_relu,
          "Stack tensor along dim and apply ReLU",
          py::arg("x"), py::arg("dim"), py::arg("num_repeats"));
}

REGISTER_BINDING(custom_stack_relu, register_custom_stack_relu);
