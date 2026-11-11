#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_mean_reduce_launcher(const float* input,
                                           float* output,
                                           int batch_size,
                                           long long elements_per_batch,
                                           cudaStream_t stream);

extern "C" void custom_tan_launcher(const float* input,
                                    float* output,
                                    int numel,
                                    cudaStream_t stream);

// custom_mean_reduce: torch.mean(x, dim=(1,2,3), keepdim=True)
torch::Tensor custom_mean_reduce(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_mean_reduce: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "custom_mean_reduce: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_mean_reduce: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_mean_reduce: input must be 4D");
    TORCH_CHECK(x.size(1) == 3 && x.size(2) == 2048 && x.size(3) == 2048,
                "custom_mean_reduce: input shape must be [N, 3, 2048, 2048]");

    int batch_size = static_cast<int>(x.size(0));
    long long elements_per_batch = static_cast<long long>(x.size(1)) * x.size(2) * x.size(3);

    auto output = torch::empty({batch_size, 1, 1, 1}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_mean_reduce_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        elements_per_batch,
        stream
    );

    return output;
}

// custom_tan: torch.tan(x)
torch::Tensor custom_tan(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_tan: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "custom_tan: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_tan: input must be float32");

    auto output = torch::empty_like(x);

    int numel = static_cast<int>(x.numel());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_tan_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_mean_reduce", &custom_mean_reduce, "Mean reduction over dims (1,2,3) with keepdim=True",
          py::arg("x"));
    m.def("custom_tan", &custom_tan, "Elementwise tangent",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
