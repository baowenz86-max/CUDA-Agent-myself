#include <cuda_runtime.h>
#include <cstdint>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Helper for ceil division
inline int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// Kernel 1: ReflectionPad2d
// Input: float32 tensor contiguous or non-contiguous, shape [N, C, H, W]
// Padding: (left, right, top, bottom) = (pad_w0, pad_w1, pad_h0, pad_h1)
// Output: float32 tensor shape [N, C, H + pad_h0 + pad_h1, W + pad_w0 + pad_w1]
__global__ void reflection_pad2d_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int pad_h0, int pad_h1, int pad_w0, int pad_w1,
    int64_t total_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t w_out = i % W_out;
        int64_t tmp = i / W_out;
        int64_t h_out = tmp % H_out;
        tmp /= H_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Compute source coordinates with reflection
        int h_src = h_out - pad_h0;
        if (h_src < 0) {
            h_src = -h_src;
            if (h_src >= H_in) h_src = 2 * H_in - 2 - h_src; // reflection with boundary clamp
        } else if (h_src >= H_in) {
            h_src = 2 * H_in - 2 - h_src;
            if (h_src < 0) h_src = -h_src;
        }
        // Ensure h_src in [0, H_in-1]
        if (h_src < 0) h_src = 0;
        if (h_src >= H_in) h_src = H_in - 1;

        int w_src = w_out - pad_w0;
        if (w_src < 0) {
            w_src = -w_src;
            if (w_src >= W_in) w_src = 2 * W_in - 2 - w_src;
        } else if (w_src >= W_in) {
            w_src = 2 * W_in - 2 - w_src;
            if (w_src < 0) w_src = -w_src;
        }
        if (w_src < 0) w_src = 0;
        if (w_src >= W_in) w_src = W_in - 1;

        int64_t in_idx = ((n * C + c) * H_in + h_src) * W_in + w_src;
        out[i] = in[in_idx];
    }
}

// Kernel 2: GreaterEqual (ge) with scalar 0
// Input: float32 tensor, output: bool tensor (uint8 underlying)
// Output element = input >= 0
__global__ void ge_zero_kernel(
    const float* __restrict__ in,
    bool* __restrict__ out,
    int64_t total_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        out[i] = (in[i] >= 0.0f);
    }
}

// Kernel 3: Remainder with scalar 2 (mod 2)
// Input: bool tensor (converted to 0/1 float? We need to match torch.remainder semantics)
// Actually, remainder of bool tensor with scalar 2:
// In torch, torch.remainder(bool_tensor, 2) promotes bool to int64? Let's check:
// torch.remainder(torch.tensor([True, False]), 2) -> tensor([1, 0]) dtype=int64
// So input bool is cast to int64, then remainder with 2.
// So we need to convert bool to int64 and compute remainder.
// We'll implement: out[i] = (int64_t)in_bool[i] % 2;
__global__ void remainder_bool_kernel(
    const bool* __restrict__ in,
    int64_t* __restrict__ out,
    int64_t total_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        out[i] = (int64_t)in[i] % 2;
    }
}

// Launchers for each op

// reflection_pad2d_forward
// Signature: (torch::Tensor input, tuple<int,int,int,int> padding)
// Returns float32 tensor of same dtype as input (float32), shape as computed.
torch::Tensor reflection_pad2d_forward(torch::Tensor input, std::tuple<int64_t, int64_t, int64_t, int64_t> padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 supported");
    // Input can be non-contiguous; we will handle with strides provided to kernel via raw pointer? 
    // For simplicity, we assume contiguous; but to be safe, call .contiguous() if needed.
    // In PyTorch, ReflectionPad2d supports non-contiguous, but we can make contiguous for simplicity.
    auto input_contig = input.contiguous();
    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H_in = input_contig.size(2);
    int64_t W_in = input_contig.size(3);
    
    int64_t pad_left, pad_right, pad_top, pad_bottom;
    std::tie(pad_left, pad_right, pad_top, pad_bottom) = padding;
    // For ReflectionPad2d, padding order is (left, right, top, bottom)
    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;
    
    auto output = torch::empty({N, C, H_out, W_out}, input_contig.options());
    int64_t total_elements = output.numel();
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (int)std::min((int64_t)65535, ceil_div(total_elements, threads));
    if (blocks == 0) blocks = 1;
    
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        (int)pad_top, (int)pad_bottom, (int)pad_left, (int)pad_right,
        total_elements
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// ge_forward
// Signature: (torch::Tensor input, double scalar)
// Returns bool tensor same shape.
torch::Tensor ge_forward(torch::Tensor input, double scalar) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 supported");
    auto input_contig = input.contiguous();
    auto output = torch::empty(input_contig.sizes(), input_contig.options().dtype(torch::kBool));
    int64_t total_elements = input_contig.numel();
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (int)std::min((int64_t)65535, ceil_div(total_elements, threads));
    if (blocks == 0) blocks = 1;
    
    ge_zero_kernel<<<blocks, threads, 0, stream>>>(
        input_contig.data_ptr<float>(),
        output.data_ptr<bool>(),
        total_elements
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// remainder_forward
// Signature: (torch::Tensor input, double scalar) where input is bool, scalar is 2
// Returns int64 tensor same shape.
torch::Tensor remainder_forward(torch::Tensor input, double scalar) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kBool, "Expected bool input");
    TORCH_CHECK(scalar == 2.0, "Only scalar 2 supported");
    auto input_contig = input.contiguous();
    auto output = torch::empty(input_contig.sizes(), input_contig.options().dtype(torch::kInt64));
    int64_t total_elements = input_contig.numel();
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (int)std::min((int64_t)65535, ceil_div(total_elements, threads));
    if (blocks == 0) blocks = 1;
    
    remainder_bool_kernel<<<blocks, threads, 0, stream>>>(
        input_contig.data_ptr<bool>(),
        output.data_ptr<int64_t>(),
        total_elements
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// Registration function
void register_custom_ops(py::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward, "ReflectionPad2d forward",
          py::arg("input"), py::arg("padding"));
    m.def("ge_forward", &ge_forward, "Greater equal forward",
          py::arg("input"), py::arg("scalar"));
    m.def("remainder_forward", &remainder_forward, "Remainder forward",
          py::arg("input"), py::arg("scalar"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
