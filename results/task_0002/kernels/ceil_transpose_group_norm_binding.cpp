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
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW");

    const int64_t batch = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t height = input.size(2);
    const int64_t width = input.size(3);
    TORCH_CHECK(groups > 0 && height % groups == 0,
                "transposed channel count must be divisible by groups");
    TORCH_CHECK(batch <= 65535 && channels <= INT_MAX && height <= INT_MAX &&
                width <= INT_MAX && groups <= INT_MAX,
                "tensor dimensions exceed kernel limits");

    auto output = torch::empty_like(input);
    output.unsafeGetTensorImpl()->set_sizes_contiguous(
        {batch, height, channels, width});
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    ceil_transpose_group_norm_launcher(
        input.data_ptr<float>(), output.data_ptr<float>(),
        static_cast<int>(batch), static_cast<int>(channels),
        static_cast<int>(height), static_cast<int>(width),
        static_cast<int>(groups), stream);
    return output;
}

void register_ceil_transpose_group_norm(pybind11::module& m) {
    m.def("ceil_transpose_group_norm", &ceil_transpose_group_norm);
}

REGISTER_BINDING(ceil_transpose_group_norm, register_ceil_transpose_group_norm);
