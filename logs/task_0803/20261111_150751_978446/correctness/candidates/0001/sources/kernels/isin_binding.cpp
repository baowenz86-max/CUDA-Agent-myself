// Binding for isin and is_complex custom operators.
// Contains only launcher declaration and PyTorch tensor wrappers.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher from custom_ops.cu
extern "C" void isin_launcher(const float* x,
                             const float* test_elements,
                             bool* output,
                             int64_t numel,
                             int64_t test_size,
                             cudaStream_t stream);

// isin_forward: checks membership of x elements in test_elements.
// Inputs must be CUDA float32 tensors; x and test_elements are 1D or 2D,
// no broadcasting is applied (reference uses same shapes).
torch::Tensor isin_forward(torch::Tensor x, torch::Tensor test_elements) {
    // Validate device and dtype
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(test_elements.is_cuda(), "test_elements must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(test_elements.scalar_type() == torch::kFloat32, "test_elements must be float32");

    // Ensure contiguity for simple pointer passing (kernel expects contiguous)
    auto x_contig = x.contiguous();
    auto test_contig = test_elements.contiguous();

    // Allocate output bool tensor with same shape as x
    auto output = torch::empty_like(x_contig, x_contig.options().dtype(torch::kBool));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t numel = x_contig.numel();
    int64_t test_size = test_contig.numel();

    // Call launcher
    isin_launcher(
        x_contig.data_ptr<float>(),
        test_contig.data_ptr<float>(),
        output.data_ptr<bool>(),
        numel,
        test_size,
        stream
    );

    return output;
}

// is_complex_forward: returns True if input is a complex tensor.
// Since reference always passes complex64, we simply check dtype.
// This does not require a CUDA kernel.
bool is_complex_forward(torch::Tensor input) {
    // None of the checks require CUDA; input is passed as complex64.
    // Type check suffices; device is irrelevant for is_complex.
    TORCH_CHECK(input.is_complex(), "Input must be a complex tensor");
    return true;
}

// Register functions
void register_isin_binding(pybind11::module& m) {
    m.def("isin_forward", &isin_forward,
          "Check membership of x in test_elements (CUDA float32 -> bool tensor)",
          pybind11::arg("x"), pybind11::arg("test_elements"));
    m.def("is_complex_forward", &is_complex_forward,
          "Return True if input tensor is complex",
          pybind11::arg("input"));
}

REGISTER_BINDING(isin_binding, register_isin_binding);
