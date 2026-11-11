#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void custom_leaky_relu_launcher(float* out,
                                           const float* in,
                                           int64_t n,
                                           double negative_slope,
                                           cudaStream_t stream);

// Wrapper function
static torch::Tensor custom_leaky_relu(torch::Tensor x, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    
    // Make contiguous if necessary (handles non-contiguous strides)
    auto x_contig = x.contiguous();
    
    // Allocate output tensor with same sizes and options as contiguous input
    auto output = torch::empty_like(x_contig);
    
    int64_t n = x_contig.numel();
    if (n == 0) {
        return output;
    }
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call launcher with raw pointers
    custom_leaky_relu_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        n,
        negative_slope,
        stream
    );
    
    return output;
}

// Registration function
static void register_custom_leaky_relu(pybind11::module& m) {
    m.def("custom_leaky_relu", &custom_leaky_relu,
          "LeakyReLU forward",
          py::arg("x"),
          py::arg("negative_slope"));
}

// Auto-register
REGISTER_BINDING(custom_leaky_relu, register_custom_leaky_relu);
