#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu with scalar arguments
extern "C" void custom_sigmoid_launcher(
    float* output,
    const float* input,
    int total_elements,
    int ndims,
    int in_strides0, int in_strides1, int in_strides2, int in_strides3,
    int out_strides0, int out_strides1, int out_strides2, int out_strides3,
    int shape0, int shape1, int shape2, int shape3,
    cudaStream_t stream);

extern "C" void custom_instance_norm2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int in_strides0, int in_strides1, int in_strides2, int in_strides3,
    int out_strides0, int out_strides1, int out_strides2, int out_strides3,
    float eps,
    cudaStream_t stream);

static torch::Tensor custom_sigmoid(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    // Strides and shape as int64_t, but values fit in int (they are within int range for this task)
    int64_t in_strides[4];
    int64_t out_strides[4];
    int64_t shape[4];
    for (int i = 0; i < 4; ++i) {
        in_strides[i] = x.stride(i);
        out_strides[i] = output.stride(i);
        shape[i] = x.size(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Pass scalar arguments directly, avoiding device pointer issues
    custom_sigmoid_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(x.numel()),
        4,
        static_cast<int>(in_strides[0]),
        static_cast<int>(in_strides[1]),
        static_cast<int>(in_strides[2]),
        static_cast<int>(in_strides[3]),
        static_cast<int>(out_strides[0]),
        static_cast<int>(out_strides[1]),
        static_cast<int>(out_strides[2]),
        static_cast<int>(out_strides[3]),
        static_cast<int>(shape[0]),
        static_cast<int>(shape[1]),
        static_cast<int>(shape[2]),
        static_cast<int>(shape[3]),
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

    int64_t in_strides[4];
    int64_t out_strides[4];
    for (int i = 0; i < 4; ++i) {
        in_strides[i] = x.stride(i);
        out_strides[i] = output.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_instance_norm2d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(in_strides[0]),
        static_cast<int>(in_strides[1]),
        static_cast<int>(in_strides[2]),
        static_cast<int>(in_strides[3]),
        static_cast<int>(out_strides[0]),
        static_cast<int>(out_strides[1]),
        static_cast<int>(out_strides[2]),
        static_cast<int>(out_strides[3]),
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
