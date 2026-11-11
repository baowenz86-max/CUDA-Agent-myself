#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_mean_dim1_launcher(float* output,
                                           const float* input,
                                           int batch,
                                           int channels,
                                           int height,
                                           int width,
                                           cudaStream_t stream);

extern "C" void custom_gelu_launcher(float* output,
                                      const float* input,
                                      int numel,
                                      cudaStream_t stream);

// Wrapper for custom_mean_dim1
torch::Tensor custom_mean_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must have 4 dimensions");

    int batch = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    auto output = torch::empty({batch, 1, height, width}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_mean_dim1_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch,
        channels,
        height,
        width,
        stream
    );

    return output;
}

// Wrapper for custom_gelu
torch::Tensor custom_gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);
    int numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_gelu_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_mean_dim1", &custom_mean_dim1_forward,
          "Compute mean over dim=1 with keepdim=True",
          py::arg("x"));
    m.def("custom_gelu", &custom_gelu_forward,
          "Apply GELU activation",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
