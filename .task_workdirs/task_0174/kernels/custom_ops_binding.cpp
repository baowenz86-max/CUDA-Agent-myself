#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declaration of the CUDA launcher from custom_ops.cu
extern "C" void hardtanh_forward_launcher(float* output,
                                          const float* input,
                                          int64_t numel,
                                          cudaStream_t stream);

// PyTorch binding for hardtanh forward
torch::Tensor hardtanh_forward(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // Ensure input is contiguous; if not, make it contiguous
    auto x_contig = x.contiguous();
    
    // Allocate output with same shape as input, preserving strides via empty_like
    auto output = torch::empty_like(x_contig);
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call launcher with raw pointers
    hardtanh_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        stream
    );
    
    // Return output (contiguous). The model expects output with same shape/dtype/device,
    // and since input from model is contiguous, output will match.
    return output;
}

// Registration function
void register_hardtanh(pybind11::module& m) {
    m.def("hardtanh_forward", &hardtanh_forward, 
          "HardTanh forward (clamp to [-1,1])",
          pybind11::arg("x"));
}

// Auto-register
REGISTER_BINDING(hardtanh, register_hardtanh);
