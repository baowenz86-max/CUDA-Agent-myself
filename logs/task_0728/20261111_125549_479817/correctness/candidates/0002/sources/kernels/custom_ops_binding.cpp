// Binding for custom ops: squeeze_dim and concat_dim
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void squeeze_dim_launcher(
    const float* input,
    float* output,
    int64_t dim,
    int64_t input_ndim,
    const int64_t* input_sizes,
    const int64_t* input_strides,
    cudaStream_t stream
);

extern "C" void concat_two_launcher(
    const float* a,
    const float* b,
    float* out,
    int64_t a_sizes[8],
    int64_t b_sizes[8],
    int64_t ndim,
    int64_t dim,
    cudaStream_t stream
);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// squeeze_dim: If size(dim)==1, return x.as_strided(new_sizes) (view), otherwise return x unchanged.
// This matches torch.squeeze(input, dim) behavior for the given dimension.
static torch::Tensor squeeze_dim(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "squeeze_dim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "squeeze_dim: input must be float32");

    auto ndim = x.dim();
    // Normalize negative dim
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "squeeze_dim: dim out of range");

    if (x.size(dim) == 1) {
        // Remove dimension dim: new sizes and strides
        std::vector<int64_t> new_sizes;
        std::vector<int64_t> new_strides;
        for (int64_t d = 0; d < ndim; ++d) {
            if (d != dim) {
                new_sizes.push_back(x.size(d));
                new_strides.push_back(x.stride(d));
            }
        }
        return x.as_strided(new_sizes, new_strides);
    } else {
        // No change, return original tensor (view)
        return x;
    }
}

// concat_dim: Concatenate a list of tensors along dimension dim.
// All tensors must be CUDA float32, have same number of dimensions and identical sizes except along dim.
// Output is a new contiguous tensor.
static torch::Tensor concat_dim(std::vector<torch::Tensor> tensors, int64_t dim) {
    TORCH_CHECK(tensors.size() > 0, "concat_dim: tensor list must not be empty");
    
    const auto& first = tensors[0];
    TORCH_CHECK(first.is_cuda(), "concat_dim: all tensors must be CUDA");
    TORCH_CHECK(first.dtype() == torch::kFloat32, "concat_dim: all tensors must be float32");
    
    int64_t ndim = first.dim();
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "concat_dim: dim out of range");
    
    // Validate all tensors
    std::vector<torch::Tensor> processed_tensors;
    processed_tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
        TORCH_CHECK(t.is_cuda(), "concat_dim: all tensors must be CUDA");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "concat_dim: all tensors must be float32");
        TORCH_CHECK(t.dim() == ndim, "concat_dim: all tensors must have same number of dimensions");
        for (int64_t d = 0; d < ndim; ++d) {
            if (d != dim) {
                TORCH_CHECK(t.size(d) == first.size(d), "concat_dim: all tensors must have same sizes except along dim");
            }
        }
        // Ensure contiguous for kernel (kernel assumes contiguous)
        processed_tensors.push_back(t.contiguous());
    }
    
    // Compute total size along dim
    int64_t total_dim_size = 0;
    for (const auto& t : processed_tensors) {
        total_dim_size += t.size(dim);
    }
    
    // Output sizes
    std::vector<int64_t> out_sizes(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
        out_sizes[d] = (d == dim) ? total_dim_size : first.size(d);
    }
    
    auto out = torch::empty(out_sizes, first.options());
    TORCH_CHECK(out.is_contiguous(), "concat_dim: output must be contiguous");
    
    // For simplicity and correctness, handle exactly two tensors using concat_two_launcher.
    // The launcher expects arrays of sizes and computes concatenation along dim.
    // Since contract only requires two tensors in current usage, we can call it directly.
    // If more than two, we could call multiple times, but current model uses two.
    TORCH_CHECK(processed_tensors.size() == 2, "concat_dim: currently only two tensors supported");
    
    const auto& a = processed_tensors[0];
    const auto& b = processed_tensors[1];
    
    // Prepare size arrays for launcher (max 8 dims)
    TORCH_CHECK(ndim <= 8, "concat_dim: too many dimensions");
    int64_t a_sizes[8] = {0};
    int64_t b_sizes[8] = {0};
    for (int64_t d = 0; d < ndim; ++d) {
        a_sizes[d] = a.size(d);
        b_sizes[d] = b.size(d);
    }
    
    cudaStream_t stream = getCurrentStream();
    
    concat_two_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        a_sizes,
        b_sizes,
        ndim,
        dim,
        stream
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "concat_dim: CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return out;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("squeeze_dim", &squeeze_dim, "Squeeze dimension if size is 1, otherwise return input",
          py::arg("x"), py::arg("dim"));
    m.def("concat_dim", &concat_dim, "Concatenate list of tensors along dimension",
          py::arg("tensors"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
