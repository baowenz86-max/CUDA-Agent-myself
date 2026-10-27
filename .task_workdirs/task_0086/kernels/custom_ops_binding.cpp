// PyTorch bindings for custom reflection padding and scaled dot product attention.
// This file uses torch types and pybind11 but contains NO CUDA kernels.
// CUDA kernels are implemented in custom_ops.cu.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long in_stride_N, long long in_stride_C, long long in_stride_H, long long in_stride_W,
    cudaStream_t stream);

extern "C" void scaled_dot_product_attention_launcher(
    float* output,
    const float* q,
    const float* k,
    const float* v,
    int N, int H, int L, int S, int D, int V,
    bool is_causal,
    long long q_stride_N, long long q_stride_H, long long q_stride_L, long long q_stride_D,
    long long k_stride_N, long long k_stride_H, long long k_stride_S, long long k_stride_D,
    long long v_stride_N, long long v_stride_H, long long v_stride_S, long long v_stride_V,
    cudaStream_t stream);

// Helper to extract integer strides from a tensor dimension; negative not expected
static inline long long get_stride(const torch::Tensor& t, int64_t dim) {
    TORCH_CHECK(dim >= 0 && dim < t.dim(), "dimension index out of range");
    return (long long)t.stride(dim);
}

// Forward function for reflection padding
// Parameters: input tensor, pad_left, pad_right, pad_top, pad_bottom
// Returns: padded output tensor (contiguous, same dtype/device)
torch::Tensor reflection_pad2d_forward(
    torch::Tensor input,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4-dimensional (N, C, H, W)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "Padding must be non-negative");

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);

    const int64_t out_H = H + pad_top + pad_bottom;
    const int64_t out_W = W + pad_left + pad_right;

    // Check that reflections are valid (padding <= H-1 and W-1)
    TORCH_CHECK(pad_top <= H - 1 && pad_bottom <= H - 1,
                "Reflection padding top/bottom must not exceed H-1");
    TORCH_CHECK(pad_left <= W - 1 && pad_right <= W - 1,
                "Reflection padding left/right must not exceed W-1");

    // Allocate contiguous output tensor with same dtype/device
    auto output = torch::empty({N, C, out_H, out_W}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Extract input strides (can be non-contiguous but must be positive)
    long long in_stride_N = get_stride(input, 0);
    long long in_stride_C = get_stride(input, 1);
    long long in_stride_H = get_stride(input, 2);
    long long in_stride_W = get_stride(input, 3);

    // Call launcher
    reflection_pad2d_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        (int)N, (int)C, (int)H, (int)W,
        (int)pad_left, (int)pad_right, (int)pad_top, (int)pad_bottom,
        in_stride_N, in_stride_C, in_stride_H, in_stride_W,
        stream);

    // Check for kernel launch errors (optional but recommended)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "reflection_pad2d_forward CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Forward function for scaled dot product attention
// Parameters: q, k, v tensors, is_causal bool
// Returns: attention output tensor with shape of q
torch::Tensor scaled_dot_product_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal) {
    // Check devices and dtypes
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "All inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 &&
                k.scalar_type() == torch::kFloat32 &&
                v.scalar_type() == torch::kFloat32,
                "All inputs must be float32");

    // Check shapes: q: [N, H, L, D], k: [N, H, S, D], v: [N, H, S, V]
    TORCH_CHECK(q.dim() == 4, "q must be 4-dimensional");
    TORCH_CHECK(k.dim() == 4, "k must be 4-dimensional");
    TORCH_CHECK(v.dim() == 4, "v must be 4-dimensional");
    const int64_t N = q.size(0);
    const int64_t H = q.size(1);
    const int64_t L = q.size(2);
    const int64_t D = q.size(3);
    TORCH_CHECK(k.size(0) == N && k.size(1) == H, "k batch/head dims must match q");
    const int64_t S = k.size(2);
    TORCH_CHECK(k.size(3) == D, "k last dim must match q last dim");
    TORCH_CHECK(v.size(0) == N && v.size(1) == H, "v batch/head dims must match q");
    TORCH_CHECK(v.size(2) == S, "v second-to-last dim must match k");
    const int64_t V = v.size(3);

    // Allocate output: same shape as q
    auto output = torch::empty_like(q, q.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Extract strides (support non-contiguous tensors)
    long long q_stride_N = get_stride(q, 0);
    long long q_stride_H = get_stride(q, 1);
    long long q_stride_L = get_stride(q, 2);
    long long q_stride_D = get_stride(q, 3);

    long long k_stride_N = get_stride(k, 0);
    long long k_stride_H = get_stride(k, 1);
    long long k_stride_S = get_stride(k, 2);
    long long k_stride_D = get_stride(k, 3);

    long long v_stride_N = get_stride(v, 0);
    long long v_stride_H = get_stride(v, 1);
    long long v_stride_S = get_stride(v, 2);
    long long v_stride_V = get_stride(v, 3);

    // Call launcher
    scaled_dot_product_attention_launcher(
        output.data_ptr<float>(),
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        (int)N, (int)H, (int)L, (int)S, (int)D, (int)V,
        is_causal,
        q_stride_N, q_stride_H, q_stride_L, q_stride_D,
        k_stride_N, k_stride_H, k_stride_S, k_stride_D,
        v_stride_N, v_stride_H, v_stride_S, v_stride_V,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "scaled_dot_product_attention_forward CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("pad_left"),
          pybind11::arg("pad_right"),
          pybind11::arg("pad_top"),
          pybind11::arg("pad_bottom"));
    m.def("scaled_dot_product_attention_forward", &scaled_dot_product_attention_forward,
          "Scaled Dot Product Attention forward (CUDA)",
          pybind11::arg("q"),
          pybind11::arg("k"),
          pybind11::arg("v"),
          pybind11::arg("is_causal"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
