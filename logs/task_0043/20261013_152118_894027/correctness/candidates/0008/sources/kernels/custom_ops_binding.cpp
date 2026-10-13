// Binding for custom non-convolution kernels defined in custom_ops.cu
#include <torch/types.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launchers from custom_ops.cu
extern "C" void view_as_real_launcher(const float2* input, float* output,
                                      int64_t total_complex_elements,
                                      const int64_t* input_strides,
                                      const int64_t* input_shape,
                                      int ndim,
                                      cudaStream_t stream);

extern "C" void einsum_permute_launcher(const float* input, float* output,
                                        int64_t total_output_elements,
                                        const int64_t* input_strides,
                                        const int64_t* output_strides,
                                        const int64_t* output_shape,
                                        int ndim,
                                        cudaStream_t stream);

extern "C" void clip_launcher(float* data, int64_t total_elements,
                              float min_val, float max_val,
                              cudaStream_t stream);

// Helper to check CUDA errors after kernel launches
static void check_cuda_error(const char* name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, name, " failed with error: ", cudaGetErrorString(err));
}

// view_as_real_forward: CPU complex64 -> CUDA float32 with extra trailing dim of size 2
torch::Tensor view_as_real_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cpu(), "view_as_real_forward: input must be CPU tensor");
    TORCH_CHECK(x.scalar_type() == c10::kComplex64, "view_as_real_forward: input must be complex64");
    TORCH_CHECK(!x.is_complex(), "view_as_real_forward: input should be complex64 but got ", x.scalar_type());

    // Original shape
    auto sizes = x.sizes();
    int64_t ndim = x.dim();
    TORCH_CHECK(ndim > 0, "view_as_real_forward: input must have at least one dimension");

    // Output shape: input shape + [2]
    std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
    out_sizes.push_back(2);

    // Allocate contiguous CUDA output
    auto out = torch::empty(out_sizes, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Ensure input is contiguous for easier copying to GPU; if not, make contiguous
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    // Copy complex tensor to GPU (as float2). Total number of complex elements = numel of input.
    int64_t total_complex_elements = x.numel();
    auto x_gpu = torch::empty_like(x, torch::dtype(c10::kComplex64).device(torch::kCUDA));
    x_gpu.copy_(x);

    // For input strides, we pass the strides of the complex tensor (in units of float2 elements)
    // Convert to int64_t array on host; kernel expects strides in terms of complex elements.
    auto stride_opt = x.strides();
    std::vector<int64_t> input_strides(stride_opt.begin(), stride_opt.end());
    auto shape_opt = x.sizes();
    std::vector<int64_t> input_shape(shape_opt.begin(), shape_opt.end());

    // Move strides and shape to GPU? The launcher expects device pointers to int64 arrays.
    // In our .cu, the kernel takes const int64_t* __restrict__ input_strides and input_shape,
    // which are declared as device pointers (kernel memory space). So we need to copy them to GPU.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t* d_input_strides = nullptr;
    int64_t* d_input_shape = nullptr;
    cudaMalloc(&d_input_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_input_shape, ndim * sizeof(int64_t));
    cudaMemcpyAsync(d_input_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_input_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    // Call launcher with GPU complex pointer
    view_as_real_launcher(
        reinterpret_cast<const float2*>(x_gpu.data_ptr()),
        out.data_ptr<float>(),
        total_complex_elements,
        d_input_strides,
        d_input_shape,
        static_cast<int>(ndim),
        stream
    );
    check_cuda_error("view_as_real_launcher");

    // Free device arrays
    cudaFree(d_input_strides);
    cudaFree(d_input_shape);

    return out;
}

// einsum_bchw_to_bhwc_forward: CUDA float32 [B,C,H,W,2] -> [B,H,W,C,2]
torch::Tensor einsum_bchw_to_bhwc_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "einsum_bchw_to_bhwc_forward: input must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "einsum_bchw_to_bhwc_forward: input must be float32");

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 5, "einsum_bchw_to_bhwc_forward: input must be 5D [B,C,H,W,2]");
    int64_t B = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];
    int64_t R = sizes[4]; // == 2
    TORCH_CHECK(R == 2, "einsum_bchw_to_bhwc_forward: last dim must be 2");

    // Output shape [B,H,W,C,2]
    std::vector<int64_t> out_sizes = {B, H, W, C, R};
    auto out = torch::empty(out_sizes, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Input and output strides
    auto in_strides = x.strides(); // 5 strides, may be non-contiguous
    auto out_strides = out.strides();

    // Total output elements = B*H*W*C*2
    int64_t total_out = out.numel();

    // Prepare strides and shape for launcher (device arrays)
    int ndim = 5;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t* d_input_strides = nullptr;
    int64_t* d_output_strides = nullptr;
    int64_t* d_output_shape = nullptr;
    cudaMalloc(&d_input_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_output_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_output_shape, ndim * sizeof(int64_t));

    std::vector<int64_t> input_strides(in_strides.begin(), in_strides.end());
    std::vector<int64_t> output_strides(out_strides.begin(), out_strides.end());
    std::vector<int64_t> output_shape(out_sizes.begin(), out_sizes.end());

    cudaMemcpyAsync(d_input_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_output_strides, output_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_output_shape, output_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    einsum_permute_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        total_out,
        d_input_strides,
        d_output_strides,
        d_output_shape,
        ndim,
        stream
    );
    check_cuda_error("einsum_permute_launcher");

    cudaFree(d_input_strides);
    cudaFree(d_output_strides);
    cudaFree(d_output_shape);

    return out;
}

// clip_forward: in-place clamp on CUDA float32 tensor
torch::Tensor clip_forward(torch::Tensor x, float min = 0.0f, float max = 1.0f) {
    TORCH_CHECK(x.is_cuda(), "clip_forward: input must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "clip_forward: input must be float32");

    // We will modify in-place to preserve any strides and avoid extra memory.
    // The launcher modifies data in-place.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int64_t total_elements = x.numel();

    clip_launcher(x.data_ptr<float>(), total_elements, min, max, stream);
    check_cuda_error("clip_launcher");

    return x;
}

void register_custom_ops(pybind11::module& m) {
    m.def("view_as_real_forward", &view_as_real_forward,
          "Convert CPU complex64 tensor to CUDA float32 tensor with extra trailing dim of 2",
          py::arg("x"));
    m.def("einsum_bchw_to_bhwc_forward", &einsum_bchw_to_bhwc_forward,
          "Permute bchw->bhwc for a 5D tensor where last dim is 2",
          py::arg("x"));
    m.def("clip_forward", &clip_forward,
          "Clamp tensor element-wise to [min, max]",
          py::arg("x"), py::arg("min") = 0.0f, py::arg("max") = 1.0f);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
