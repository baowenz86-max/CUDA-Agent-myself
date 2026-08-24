#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

extern "C" void ceil_transpose_group_norm_launcher(
    const float*, float*, int, int, int, int, int, cudaStream_t);

torch::Tensor ceil_transpose_group_norm(torch::Tensor input, int64_t groups) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    const auto n = input.size(0), c = input.size(1);
    const auto h = input.size(2), w = input.size(3);
    TORCH_CHECK(groups > 0 && h % groups == 0,
                "transposed channels must be divisible by groups");
    auto output = torch::empty_like(input);
    output.unsafeGetTensorImpl()->set_sizes_contiguous({n, h, c, w});
    ceil_transpose_group_norm_launcher(
        input.data_ptr<float>(), output.data_ptr<float>(), int(n), int(c),
        int(h), int(w), int(groups), c10::cuda::getCurrentCUDAStream().stream());
    return output;
}

void register_ceil_transpose_group_norm(pybind11::module& m) {
    m.def("ceil_transpose_group_norm", &ceil_transpose_group_norm);
}
REGISTER_BINDING(ceil_transpose_group_norm, register_ceil_transpose_group_norm);
