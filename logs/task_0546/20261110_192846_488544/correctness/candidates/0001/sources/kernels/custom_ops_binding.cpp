#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void max_dim1_kernel_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream
);

extern "C" void expm1_kernel_launcher(
    const float* x,
    float* y,
    long long total,
    cudaStream_t stream
);

// Wrapper for max_dim1_kernel
// Returns torch.return_types.max with values and indices
torch::return_types::max max_dim1_kernel(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [batch, channels, height, width]");

    // Ensure input is contiguous for the launcher
    torch::Tensor x_contig = x.contiguous();

    int batch = x_contig.size(0);
    int channels = x_contig.size(1);
    int height = x_contig.size(2);
    int width = x_contig.size(3);

    // Output shapes: [batch, 1, height, width]
    auto options_values = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(x.device());
    auto options_indices = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(x.device());

    torch::Tensor values = torch::empty({batch, 1, height, width}, options_values);
    torch::Tensor indices = torch::empty({batch, 1, height, width}, options_indices);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_dim1_kernel_launcher(
        x_contig.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        batch,
        channels,
        height,
        width,
        stream
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in max_dim1_kernel: ", cudaGetErrorString(err));

    return {values, indices};
}

// Wrapper for expm1_kernel
torch::Tensor expm1_kernel(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Launcher expects contiguous input
    torch::Tensor x_contig = x.contiguous();

    auto output = torch::empty_like(x_contig);

    long long total = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    expm1_kernel_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in expm1_kernel: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("max_dim1_kernel", &max_dim1_kernel, "Compute max over dim=1 with keepdim=True. Returns (values, indices)",
          py::arg("x"));
    m.def("expm1_kernel", &expm1_kernel, "Compute exp(x)-1 elementwise",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
