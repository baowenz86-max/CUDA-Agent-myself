#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration from custom_ops.cu
extern "C" void custom_isin_launcher(const float* elements,
                                     const float* test_elements,
                                     bool* output,
                                     int64_t numel,
                                     int64_t test_size,
                                     cudaStream_t stream);

// PyTorch binding wrapper
torch::Tensor custom_isin(torch::Tensor elements, torch::Tensor test_elements) {
    // Ensure inputs are CUDA and float32
    TORCH_CHECK(elements.is_cuda(), "elements must be a CUDA tensor");
    TORCH_CHECK(test_elements.is_cuda(), "test_elements must be a CUDA tensor");
    TORCH_CHECK(elements.dtype() == torch::kFloat32, "elements must be float32");
    TORCH_CHECK(test_elements.dtype() == torch::kFloat32, "test_elements must be float32");

    // Ensure test_elements is contiguous (launcher assumes contiguous)
    auto test_contig = test_elements.contiguous();

    // Handle non-contiguous elements: make a contiguous copy
    auto elements_contig = elements.contiguous();

    // Output boolean tensor with same shape as elements
    auto output = torch::empty(elements.sizes(), elements.options().dtype(torch::kBool));

    int64_t numel = elements_contig.numel();
    int64_t test_size = test_contig.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with raw pointers
    custom_isin_launcher(
        elements_contig.data_ptr<float>(),
        test_contig.data_ptr<float>(),
        output.data_ptr<bool>(),
        numel,
        test_size,
        stream
    );

    // Check for CUDA errors (optional but useful)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_isin(pybind11::module& m) {
    m.def("custom_isin", &custom_isin,
          "CUDA implementation of torch.isin for float32",
          py::arg("elements"),
          py::arg("test_elements"));
}

// Auto-register
REGISTER_BINDING(custom_isin, register_custom_isin);
