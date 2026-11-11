#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void mean_dim1_keepdim_launcher(const float* input,
                                            float* output,
                                            int64_t numel,
                                            cudaStream_t stream);

extern "C" void floor_launcher(const float* input,
                                float* output,
                                int64_t numel,
                                cudaStream_t stream);

// PyTorch wrapper for mean_dim1_keepdim
torch::Tensor mean_dim1_keepdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "mean_dim1_keepdim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_dim1_keepdim: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "mean_dim1_keepdim: input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mean_dim1_keepdim_launcher(x.data_ptr<float>(),
                                output.data_ptr<float>(),
                                x.numel(),
                                stream);

    return output;
}

// PyTorch wrapper for floor (renamed to avoid conflict with ::floor)
torch::Tensor floor_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "floor: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "floor: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "floor: input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    floor_launcher(x.data_ptr<float>(),
                   output.data_ptr<float>(),
                   x.numel(),
                   stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim1_keepdim", &mean_dim1_keepdim,
          "Mean over dim 1 with keepdim=True (dim size is 1)",
          py::arg("x"));
    m.def("floor", &floor_forward,
          "Elementwise floor",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);