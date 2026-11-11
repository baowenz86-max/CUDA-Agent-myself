#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_dim_launcher(const float* x, float* out, int64_t dim, const int64_t* shape, int64_t ndim, cudaStream_t stream);
extern "C" void equal_expand_as_launcher(const float* x, const float* mean, int64_t batch_size, int64_t spatial_size, int* equal_out, cudaStream_t stream);

// PyTorch wrapper for mean_dim
torch::Tensor mean_dim(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(dim == 0, "Only dim=0 is supported for this task");
    TORCH_CHECK(x.dim() == 4, "Expected 4D tensor");

    // Output shape: remove dim dimension
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) out_shape.push_back(x.size(i));
    }
    auto output = torch::empty(out_shape, x.options());

    // Prepare shape array for launcher
    std::vector<int64_t> shape(x.dim());
    for (int64_t i = 0; i < x.dim(); ++i) {
        shape[i] = x.size(i);
    }

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    mean_dim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        shape.data(),
        x.dim(),
        stream
    );

    return output;
}

// PyTorch wrapper for equal_expand_as
bool equal_expand_as(torch::Tensor x, torch::Tensor mean) {
    TORCH_CHECK(x.is_cuda() && mean.is_cuda(), "Both inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && mean.dtype() == torch::kFloat32, "Both inputs must be float32");
    TORCH_CHECK(x.is_contiguous() && mean.is_contiguous(), "Both inputs must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(mean.dim() == 3, "mean must be 3D");
    TORCH_CHECK(x.size(0) > 0, "batch size must be positive");
    TORCH_CHECK(x.size(1) == mean.size(0) && x.size(2) == mean.size(1) && x.size(3) == mean.size(2), "Shape mismatch for expand_as");

    int64_t batch_size = x.size(0);
    int64_t spatial_size = mean.numel();

    // Allocate host int for equal flag (will be filled by launcher)
    int equal_flag = 0;

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    equal_expand_as_launcher(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        batch_size,
        spatial_size,
        &equal_flag,
        stream
    );

    // Ensure kernel completion before reading flag (launcher uses cudaMemcpy sync internally?)
    // Actually the launcher performs cudaMemcpy (synchronous) as noted, so equal_flag is ready after call.
    // But we must still synchronize stream to be safe.
    cudaStreamSynchronize(stream);

    return equal_flag != 0;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim", &mean_dim, "Compute mean over dimension", py::arg("x"), py::arg("dim"));
    m.def("equal_expand_as", &equal_expand_as, "Check equality after expand_as", py::arg("x"), py::arg("mean"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
