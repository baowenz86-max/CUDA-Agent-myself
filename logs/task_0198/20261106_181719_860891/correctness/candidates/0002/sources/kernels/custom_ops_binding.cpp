#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu (C linkage)
extern "C" void custom_sigmoid_launcher(
    float* output,
    const float* input,
    int total_elements,
    int dim0, int dim1, int dim2, int dim3,
    int in_stride0, int in_stride1, int in_stride2, int in_stride3,
    int out_stride0, int out_stride1, int out_stride2, int out_stride3,
    cudaStream_t stream);

extern "C" void custom_instance_norm2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int in_stride0, int in_stride1, int in_stride2, int in_stride3,
    int out_stride0, int out_stride1, int out_stride2, int out_stride3,
    float eps,
    cudaStream_t stream);

static torch::Tensor custom_sigmoid(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    int64_t dim0 = x.size(0);
    int64_t dim1 = x.size(1);
    int64_t dim2 = x.size(2);
    int64_t dim3 = x.size(3);
    int64_t in_stride0 = x.stride(0);
    int64_t in_stride1 = x.stride(1);
    int64_t in_stride2 = x.stride(2);
    int64_t in_stride3 = x.stride(3);
    int64_t out_stride0 = output.stride(0);
    int64_t out_stride1 = output.stride(1);
    int64_t out_stride2 = output.stride(2);
    int64_t out_stride3 = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_sigmoid_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(x.numel()),
        static_cast<int>(dim0), static_cast<int>(dim1), static_cast<int>(dim2), static_cast<int>(dim3),
        static_cast<int>(in_stride0), static_cast<int>(in_stride1), static_cast<int>(in_stride2), static_cast<int>(in_stride3),
        static_cast<int>(out_stride0), static_cast<int>(out_stride1), static_cast<int>(out_stride2), static_cast<int>(out_stride3),
        stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

static torch::Tensor custom_instance_norm2d(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    int64_t in_stride0 = x.stride(0);
    int64_t in_stride1 = x.stride(1);
    int64_t in_stride2 = x.stride(2);
    int64_t in_stride3 = x.stride(3);
    int64_t out_stride0 = output.stride(0);
    int64_t out_stride1 = output.stride(1);
    int64_t out_stride2 = output.stride(2);
    int64_t out_stride3 = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_instance_norm2d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(in_stride0), static_cast<int>(in_stride1), static_cast<int>(in_stride2), static_cast<int>(in_stride3),
        static_cast<int>(out_stride0), static_cast<int>(out_stride1), static_cast<int>(out_stride2), static_cast<int>(out_stride3),
        static_cast<float>(eps),
        stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_sigmoid", &custom_sigmoid, "Element-wise sigmoid");
    m.def("custom_instance_norm2d", &custom_instance_norm2d, "InstanceNorm2d without affine parameters",
          py::arg("x"), py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
