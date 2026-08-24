#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

extern "C" void ceil_transpose_group_norm_launcher(
    const float*, float*, int, int, int, int, int, cudaStream_t);

torch::Tensor ceil_transpose_group_norm(torch::Tensor input, int64_t groups) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "input must have dtype float32");
    TORCH_CHECK(input.dim() == 4, "input must be a 4D tensor");
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    TORCH_CHECK(groups > 0 && height % groups == 0,
                "transposed channels must be divisible by groups");
    TORCH_CHECK(width % 4 == 0, "width must be divisible by four");
    auto output = torch::empty({batch, height, channels, width}, input.options());
    ceil_transpose_group_norm_launcher(
        input.data_ptr<float>(), output.data_ptr<float>(), batch, channels,
        height, width, int(groups), c10::cuda::getCurrentCUDAStream().stream());
    return output;
}

void register_ceil_transpose_group_norm(pybind11::module& m) {
    m.def("ceil_transpose_group_norm", &ceil_transpose_group_norm);
}
REGISTER_BINDING(ceil_transpose_group_norm, register_ceil_transpose_group_norm);
