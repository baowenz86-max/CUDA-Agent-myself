// Binding for cumprod CUDA kernel
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void cumprod_launcher(
    const float* input,
    float* output,
    int dim0,
    int dim1,
    int dim2,
    long long input_stride0,
    long long input_stride1,
    long long input_stride2,
    long long output_stride0,
    long long output_stride1,
    long long output_stride2,
    cudaStream_t stream);

// Wrapper function called from Python
torch::Tensor cumprod(torch::Tensor input, int64_t dim) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be a 3D tensor");
    TORCH_CHECK(dim == 0, "Only dim=0 is supported for cumprod");

    // Contract shape is [8192, 3, 4], but we keep generic for 3D tensors.
    int64_t dim0 = input.size(0);
    int64_t dim1 = input.size(1);
    int64_t dim2 = input.size(2);

    // Get strides in elements (they are in bytes in PyTorch, convert to element count)
    auto strides = input.strides();
    long long input_stride0 = static_cast<long long>(strides[0]);
    long long input_stride1 = static_cast<long long>(strides[1]);
    long long input_stride2 = static_cast<long long>(strides[2]);

    // Allocate output with same sizes and strides
    auto output = torch::empty_strided(input.sizes(), input.strides(), input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    cumprod_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(dim0),
        static_cast<int>(dim1),
        static_cast<int>(dim2),
        input_stride0,
        input_stride1,
        input_stride2,
        static_cast<long long>(output.strides()[0]),
        static_cast<long long>(output.strides()[1]),
        static_cast<long long>(output.strides()[2]),
        stream);

    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

// Registration function
void register_cumprod(pybind11::module& m) {
    m.def("cumprod", &cumprod, "Cumulative product along dimension 0",
          pybind11::arg("input"), pybind11::arg("dim"));
}

// Auto-register
REGISTER_BINDING(cumprod, register_cumprod);
