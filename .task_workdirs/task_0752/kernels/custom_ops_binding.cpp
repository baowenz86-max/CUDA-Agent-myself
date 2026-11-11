#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declaration of CUDA launcher from custom_ops.cu
extern "C" void custom_cat_erf_launcher(
    float* out,
    const float* in0,
    const float* in1,
    int N, int C0, int C1, int H, int W,
    cudaStream_t stream);

// Binding implementation: concatenates two tensors along dim=1 and applies erf.
// The contract expects a single argument 'inputs' which is a list of tensors.
// Since the launcher handles exactly two 4D contiguous tensors, we enforce that.
static torch::Tensor custom_cat_erf(std::vector<torch::Tensor> inputs) {
    TORCH_CHECK(inputs.size() == 2, "custom_cat_erf expects exactly 2 input tensors");
    
    torch::Tensor a = inputs[0];
    torch::Tensor b = inputs[1];
    
    // Validate device, dtype, and contiguity (make contiguous if needed)
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Input tensors must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32,
                "Input tensors must be float32");
    TORCH_CHECK(a.dim() == 4 && b.dim() == 4, "Input tensors must be 4D");
    
    // Make contiguous (the reference inputs are contiguous, but we should handle non-contiguous safely)
    if (!a.is_contiguous()) a = a.contiguous();
    if (!b.is_contiguous()) b = b.contiguous();
    
    // Extract dimensions: assume [N, C, H, W]
    int N = a.size(0);
    int C0 = a.size(1);
    int H = a.size(2);
    int W = a.size(3);
    
    int N1 = b.size(0);
    int C1 = b.size(1);
    int H1 = b.size(2);
    int W1 = b.size(3);
    
    TORCH_CHECK(N == N1 && H == H1 && W == W1,
                "Input tensors must have same N, H, W dimensions for concatenation along dim=1");
    
    // Allocate output tensor with concatenated channels
    auto output = torch::empty({N, C0 + C1, H, W}, a.options());
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call CUDA launcher
    custom_cat_erf_launcher(
        output.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        N, C0, C1, H, W,
        stream
    );
    
    // Check for kernel launch errors (optional but good practice)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// Registration function
static void register_custom_cat_erf(pybind11::module& m) {
    m.def("custom_cat_erf", &custom_cat_erf, py::arg("inputs"));
}

// Auto-register
REGISTER_BINDING(custom_cat_erf, register_custom_cat_erf);
