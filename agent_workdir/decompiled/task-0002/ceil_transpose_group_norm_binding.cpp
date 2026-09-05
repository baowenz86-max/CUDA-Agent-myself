#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <climits>

#include "../binding_registry.h"

extern "C" void launch_ceil_transpose_group_norm(
    const float* arg0,
    float* arg1,
    int arg2,
    int arg3,
    int arg4,
    int arg5,
    int arg6,
    cudaStream_t stream);

torch::Tensor ceil_transpose_group_norm(torch::Tensor arg0, int64_t arg6) {
    TORCH_CHECK(arg0.is_cuda(), "arg0 must be a CUDA tensor");
    TORCH_CHECK(arg0.is_contiguous(), "arg0 must be contiguous");
    TORCH_CHECK(arg0.scalar_type() == torch::kFloat32,
                "arg0 must have dtype float32");
    TORCH_CHECK(arg0.dim() == 4, "arg0 must be a 4D tensor");
    TORCH_CHECK(arg6 > 0 && arg6 <= INT_MAX, "arg6 must be a valid int");

    const int arg2 = static_cast<int>(arg0.size(0));
    const int arg3 = static_cast<int>(arg0.size(1));
    const int arg4 = static_cast<int>(arg0.size(2));
    const int arg5 = static_cast<int>(arg0.size(3));
    TORCH_CHECK(arg0.size(0) == arg2 && arg0.size(1) == arg3 &&
                    arg0.size(2) == arg4 && arg0.size(3) == arg5,
                "arg0 dimensions exceed launcher integer range");
    TORCH_CHECK(arg4 % arg6 == 0, "arg4 must be divisible by arg6");
    TORCH_CHECK(arg5 % 4 == 0, "arg5 must be divisible by 4");

    auto arg1 = torch::empty({arg2, arg4, arg3, arg5}, arg0.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_ceil_transpose_group_norm(
        arg0.data_ptr<float>(), arg1.data_ptr<float>(), arg2, arg3, arg4,
        arg5, static_cast<int>(arg6), stream);
    const cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess,
                "launch_ceil_transpose_group_norm failed: ",
                cudaGetErrorString(status));
    return arg1;
}

void register_ceil_transpose_group_norm(pybind11::module& arg0) {
    arg0.def("ceil_transpose_group_norm", &ceil_transpose_group_norm,
             pybind11::arg("arg0"), pybind11::arg("arg6"));
}

REGISTER_BINDING(ceil_transpose_group_norm, register_ceil_transpose_group_norm);
