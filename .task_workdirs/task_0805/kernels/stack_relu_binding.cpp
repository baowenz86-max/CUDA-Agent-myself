// Binding for stack_relu_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <vector>

// Forward declaration of the launcher from .cu
extern "C" void stack_relu_forward_launcher(
    const float* input,
    float* output,
    int64_t num_channels,
    int64_t stack_dim,
    const int64_t* input_shape,
    int64_t ndim,
    int64_t total_output_elements,
    cudaStream_t stream);

// PyTorch wrapper with validation and allocation
torch::Tensor stack_relu_forward(torch::Tensor x, int64_t num_channels, int64_t stack_dim) {
    // Validate dtype and device
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() > 0, "Input must have at least one dimension");
    TORCH_CHECK(num_channels > 0, "num_channels must be positive");
    TORCH_CHECK(stack_dim >= 0 && stack_dim <= x.dim(), "stack_dim must be in [0, input.dim()]");

    // Make input contiguous, since our kernel assumes contiguous memory
    x = x.contiguous();

    // Build output shape: insert num_channels at stack_dim
    std::vector<int64_t> output_sizes;
    output_sizes.reserve(x.dim() + 1);
    for (int64_t d = 0; d < x.dim(); ++d) {
        if (d == stack_dim) {
            output_sizes.push_back(num_channels);
        }
        output_sizes.push_back(x.size(d));
    }
    if (stack_dim == x.dim()) {
        output_sizes.push_back(num_channels);
    }

    // Allocate contiguous output tensor of the required shape
    auto output = torch::empty(output_sizes, x.options());

    // Prepare input_shape array for launcher (host memory)
    std::vector<int64_t> input_shape(x.dim());
    for (int64_t d = 0; d < x.dim(); ++d) {
        input_shape[d] = x.size(d);
    }

    int64_t ndim = x.dim();
    int64_t total_output_elements = output.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    stack_relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        num_channels,
        stack_dim,
        input_shape.data(),
        ndim,
        total_output_elements,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Register the binding
void register_stack_relu(pybind11::module& m) {
    m.def("stack_relu_forward", &stack_relu_forward,
          "Apply ReLU and stack along a new dimension",
          py::arg("x"),
          py::arg("num_channels"),
          py::arg("stack_dim"));
}

REGISTER_BINDING(stack_relu, register_stack_relu);
