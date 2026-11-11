#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void linear_celu_forward(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    cudaStream_t stream);

// PyTorch binding function
torch::Tensor linear_celu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    // Check CUDA
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    
    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    
    // Check dimensions
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch_size, in_features]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out_features, in_features]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [out_features]");
    
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    
    TORCH_CHECK(weight.size(1) == in_features, "weight.size(1) must equal in_features");
    TORCH_CHECK(bias.size(0) == out_features, "bias.size(0) must equal out_features");
    
    // Make tensors contiguous (no-op if already contiguous)
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    
    // Allocate output tensor [batch_size, out_features]
    auto output = torch::empty({batch_size, out_features}, x.options());
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call launcher
    linear_celu_forward(
        x_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        stream
    );
    
    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// Registration function
void register_linear_celu(pybind11::module& m) {
    m.def("linear_celu", &linear_celu,
          "Fused linear + CELU activation",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"));
}

// Auto-register
REGISTER_BINDING(linear_celu, register_linear_celu);
