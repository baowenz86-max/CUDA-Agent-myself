#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void abs_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* dims,
    int ndims,
    cudaStream_t stream);

extern "C" void relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* dims,
    int ndims,
    cudaStream_t stream);

extern "C" void abs_relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* dims,
    int ndims,
    cudaStream_t stream);

namespace {

// Helper to validate tensor and extract metadata for stride-aware kernel launch
void validate_and_get_metadata(
    const torch::Tensor& x,
    torch::Tensor& output,
    std::vector<int64_t>& input_strides,
    std::vector<int64_t>& output_strides,
    std::vector<int64_t>& dims,
    int& ndims)
{
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");

    // Output same shape, dtype, device; stride matching input by default (empty_like preserves stride)
    output = torch::empty_like(x);

    ndims = static_cast<int>(x.dim());
    TORCH_CHECK(ndims > 0, "Input tensor must have at least one dimension");

    dims.resize(ndims);
    input_strides.resize(ndims);
    output_strides.resize(ndims);

    for (int i = 0; i < ndims; ++i) {
        dims[i] = x.size(i);
        input_strides[i] = x.stride(i);
        output_strides[i] = output.stride(i);
    }
}

// Generic launch wrapper to reduce duplication
template <typename Launcher>
void launch_elementwise(
    const torch::Tensor& x,
    torch::Tensor& output,
    Launcher launcher)
{
    if (x.numel() == 0) {
        return;
    }

    std::vector<int64_t> input_strides, output_strides, dims;
    int ndims = 0;
    validate_and_get_metadata(x, output, input_strides, output_strides, dims, ndims);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        input_strides.data(),
        output_strides.data(),
        dims.data(),
        ndims,
        stream);

    // Check for kernel launch errors (cudaGetLastError) and synchronize? Better not to sync for perf, but we can check async error immediately after launch.
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

} // namespace

// PyTorch wrapper functions

torch::Tensor abs_forward(torch::Tensor x) {
    torch::Tensor output;
    launch_elementwise(x, output, [](const float* in, float* out, int64_t n, const int64_t* is, const int64_t* os, const int64_t* d, int nd, cudaStream_t s) {
        abs_forward_launcher(in, out, n, is, os, d, nd, s);
    });
    return output;
}

torch::Tensor relu6_forward(torch::Tensor x) {
    torch::Tensor output;
    launch_elementwise(x, output, [](const float* in, float* out, int64_t n, const int64_t* is, const int64_t* os, const int64_t* d, int nd, cudaStream_t s) {
        relu6_forward_launcher(in, out, n, is, os, d, nd, s);
    });
    return output;
}

torch::Tensor abs_relu6_forward(torch::Tensor x) {
    torch::Tensor output;
    launch_elementwise(x, output, [](const float* in, float* out, int64_t n, const int64_t* is, const int64_t* os, const int64_t* d, int nd, cudaStream_t s) {
        abs_relu6_forward_launcher(in, out, n, is, os, d, nd, s);
    });
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Element-wise absolute value", py::arg("x"));
    m.def("relu6_forward", &relu6_forward, "Element-wise ReLU6", py::arg("x"));
    m.def("abs_relu6_forward", &abs_relu6_forward, "Fused abs then ReLU6", py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
