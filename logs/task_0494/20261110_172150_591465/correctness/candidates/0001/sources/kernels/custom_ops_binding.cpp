#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void softplus_leaky_relu_launcher(const float* input,
                                              float* output,
                                              int64_t numel,
                                              cudaStream_t stream);

// PyTorch binding: fused softplus + leaky_relu
torch::Tensor softplus_leaky_relu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    // Ensure contiguous for optimized kernel; launcher assumes contiguity
    torch::Tensor x_contig = x.contiguous();
    
    // Allocate output tensor with same shape and options
    torch::Tensor output = torch::empty_like(x_contig);
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call launcher
    softplus_leaky_relu_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        x_contig.numel(),
        stream
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// Registration
void register_softplus_leaky_relu(pybind11::module& m) {
    m.def("softplus_leaky_relu", &softplus_leaky_relu,
          "Fused softplus + leaky_relu forward",
          py::arg("x"));
}

REGISTER_BINDING(softplus_leaky_relu, register_softplus_leaky_relu);
