#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
    int pad,
    cudaStream_t stream);

extern "C" void elementwise_max_launcher(
    const float* a,
    const float* b,
    float* out,
    int64_t numel,
    cudaStream_t stream);

extern "C" void elementwise_min_launcher(
    const float* a,
    const float* b,
    float* out,
    int64_t numel,
    cudaStream_t stream);

// Helper to check CUDA errors after kernel launch
static void check_cuda_error(const char* op_name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, op_name, " CUDA error: ", cudaGetErrorString(err));
}

// ReflectionPad2d binding
at::Tensor reflection_pad2d_forward(at::Tensor input, int64_t pad) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be a 4D tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(pad >= 0, "pad must be non-negative");

    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto H_out = H + 2 * pad;
    auto W_out = W + 2 * pad;

    // 获取各维度 stride（对于非连续输入同样适用）
    auto strides = input.strides();
    int64_t stride_n = strides[0];
    int64_t stride_c = strides[1];
    int64_t stride_h = strides[2];
    int64_t stride_w = strides[3];

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 分配输出张量，连续，形状 [N, C, H_out, W_out]
    auto output = at::empty({N, C, H_out, W_out}, input.options());

    reflection_pad2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        static_cast<int>(pad),
        stream
    );
    check_cuda_error("reflection_pad2d_forward");

    return output;
}

// Element-wise max binding
at::Tensor elementwise_max_forward(at::Tensor a, at::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "a must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "b must be float32");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    auto output = at::empty_like(a);
    int64_t numel = a.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    elementwise_max_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );
    check_cuda_error("elementwise_max_forward");

    return output;
}

// Element-wise min binding
at::Tensor elementwise_min_forward(at::Tensor a, at::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "a must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "b must be float32");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    auto output = at::empty_like(a);
    int64_t numel = a.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    elementwise_min_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );
    check_cuda_error("elementwise_min_forward");

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward (CUDA)",
          pybind11::arg("input"), pybind11::arg("pad"));
    m.def("elementwise_max_forward", &elementwise_max_forward,
          "Element-wise max forward (CUDA)",
          pybind11::arg("a"), pybind11::arg("b"));
    m.def("elementwise_min_forward", &elementwise_min_forward,
          "Element-wise min forward (CUDA)",
          pybind11::arg("a"), pybind11::arg("b"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
