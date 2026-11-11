// Binding file for custom CUDA operators: view_as_real_complex64, log_softmax_dim_last, group_norm_fp32
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu (extern "C")
extern "C" void view_as_real_complex64_launcher(const void* input, float* output, long long n_complex, cudaStream_t stream);
extern "C" void log_softmax_dim_last_launcher(const float* input, float* output, long long total_rows, cudaStream_t stream);
extern "C" void group_norm_fp32_launcher(const float* input, float* output, int N, int C, int spatial_size, int num_groups, float eps, cudaStream_t stream);

// Wrapper for view_as_real_complex64
torch::Tensor view_as_real_complex64(torch::Tensor x) {
    // Check input
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kComplexFloat, "Input tensor must have dtype torch.complex64");
    TORCH_CHECK(x.dim() >= 1, "Input tensor must have at least one dimension");

    // Ensure input is contiguous to safely view as float2 array.
    // For complex64, contiguous means the memory layout is (real0, imag0, real1, imag1, ...)
    // If not contiguous, we can make it contiguous; torch.view_as_real expects last dimension added, but
    // the underlying memory order of complex is interleaved, so making it contiguous is safe for the launcher
    // which treats it as a flat array of complex elements in row-major order.
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    // Number of complex elements
    long long n_complex = x.numel();

    // Output shape: input shape + [2]
    std::vector<int64_t> output_shape(x.sizes().begin(), x.sizes().end());
    output_shape.push_back(2);
    auto output = torch::empty(output_shape, x.options().dtype(at::kFloat));

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    // Pass x.data_ptr() as const void*, output.data_ptr<float>() as float*
    view_as_real_complex64_launcher(
        x.data_ptr(),
        output.data_ptr<float>(),
        n_complex,
        stream
    );

    return output;
}

// Wrapper for log_softmax_dim_last
torch::Tensor log_softmax_dim_last(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Input tensor must have dtype torch.float32");
    TORCH_CHECK(x.dim() >= 1, "Input tensor must have at least one dimension");
    TORCH_CHECK(x.size(-1) == 2, "Last dimension must have size 2");

    // Ensure contiguous for flat processing. The kernel assumes a flat array of pairs (a,b).
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    long long total_rows = x.numel() / 2; // number of (real, imag) pairs
    auto output = torch::empty_like(x);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    log_softmax_dim_last_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_rows,
        stream
    );

    return output;
}

// Wrapper for group_norm_fp32
torch::Tensor group_norm_fp32(torch::Tensor x, int64_t num_groups, double eps) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Input tensor must have dtype torch.float32");
    TORCH_CHECK(x.dim() == 5, "Expected 5D input tensor [N, C, D1, D2, D3]");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(x.size(1) % num_groups == 0, "Number of channels must be divisible by num_groups");

    // Ensure contiguous for the launcher which linearizes all dimensions after N,C.
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int N = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int spatial_size = 1;
    for (int i = 2; i < x.dim(); ++i) {
        spatial_size *= static_cast<int>(x.size(i));
    }

    auto output = torch::empty_like(x);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    group_norm_fp32_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, spatial_size,
        static_cast<int>(num_groups),
        static_cast<float>(eps),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("view_as_real_complex64", &view_as_real_complex64,
          "Convert complex64 tensor to real representation with last dimension 2",
          py::arg("x"));
    m.def("log_softmax_dim_last", &log_softmax_dim_last,
          "Apply log_softmax over the last dimension (size 2)",
          py::arg("x"));
    m.def("group_norm_fp32", &group_norm_fp32,
          "Apply group normalization without affine parameters",
          py::arg("x"), py::arg("num_groups"), py::arg("eps"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
