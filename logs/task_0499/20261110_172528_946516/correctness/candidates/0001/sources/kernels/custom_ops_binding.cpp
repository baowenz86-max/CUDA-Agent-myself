#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void clamp_forward_launcher(
    float* output,
    const float* input,
    const int64_t* sizes,
    const int64_t* strides,
    int ndim,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream
);

extern "C" void argmin_forward_launcher(
    const float* input,
    int64_t* output,
    const int64_t* input_sizes,
    const int64_t* input_strides,
    const int64_t* output_sizes,
    const int64_t* output_strides,
    int ndim,
    int dim,
    int64_t output_numel,
    cudaStream_t stream
);

// Helper to convert tensor sizes/strides to int64_t vectors
static std::vector<int64_t> get_int64_vector(torch::IntArrayRef arr) {
    return std::vector<int64_t>(arr.begin(), arr.end());
}

// clamp_forward implementation
torch::Tensor clamp_forward(torch::Tensor x, double min, double max) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // x can be non-contiguous; launcher handles strides
    auto output = torch::empty_like(x);

    // Get sizes and strides as int64_t pointers
    auto sizes_vec = get_int64_vector(x.sizes());
    auto strides_vec = get_int64_vector(x.strides());
    int ndim = static_cast<int>(x.dim());
    int64_t numel = x.numel();

    if (numel == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clamp_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        sizes_vec.data(),
        strides_vec.data(),
        ndim,
        numel,
        static_cast<float>(min),
        static_cast<float>(max),
        stream
    );

    return output;
}

// argmin_forward implementation
torch::Tensor argmin_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    int ndim = static_cast<int>(x.dim());
    TORCH_CHECK(ndim > 0, "argmin requires at least 1 dimension");

    // Normalize negative dim
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "Dimension out of range (expected to be in range of [-", ndim, ", ", ndim - 1, "], but got ", dim, ")");

    // Compute output shape (remove dim)
    std::vector<int64_t> output_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) {
            output_shape.push_back(x.size(i));
        }
    }

    auto output = torch::empty(output_shape, x.options().dtype(torch::kInt64));

    int64_t output_numel = output.numel();
    if (output_numel == 0) {
        return output;
    }

    // Prepare input sizes/strides as int64_t arrays
    auto input_sizes = get_int64_vector(x.sizes());
    auto input_strides = get_int64_vector(x.strides());

    // Output sizes/strides: output is contiguous, strides computed from shape
    auto output_sizes = get_int64_vector(output.sizes());
    std::vector<int64_t> output_strides(output_sizes.size());
    if (!output_sizes.empty()) {
        output_strides.back() = 1;
        for (int i = static_cast<int>(output_sizes.size()) - 2; i >= 0; --i) {
            output_strides[i] = output_strides[i + 1] * output_sizes[i + 1];
        }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    argmin_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        input_sizes.data(),
        input_strides.data(),
        output_sizes.data(),
        output_strides.data(),
        ndim,
        static_cast<int>(dim),
        output_numel,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("clamp_forward", &clamp_forward,
          py::arg("x"), py::arg("min"), py::arg("max"));
    m.def("argmin_forward", &argmin_forward,
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
