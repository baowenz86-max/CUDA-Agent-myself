#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../binding_registry.h"

namespace py = pybind11;

extern "C" void unfold_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kh, int64_t kw,
    int64_t dh, int64_t dw,
    int64_t ph, int64_t pw,
    int64_t sh, int64_t sw,
    cudaStream_t stream);

extern "C" void cat_forward_launcher(
    const float* tensor_a,
    const float* tensor_b,
    float* output,
    int64_t total_elements_per_tensor,
    cudaStream_t stream);

static inline void unpack_pair(int64_t packed, int64_t& first, int64_t& second) {
    int64_t low = packed & 0xffffffffLL;
    int64_t high = packed >> 32;
    if (low == 0xffffffffLL) {
        first = high;
        second = high;
    } else {
        first = high;
        second = low;
    }
}

torch::Tensor unfold_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t dilation,
    int64_t padding,
    int64_t stride) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");

    auto input_contig = input.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);

    int64_t kh, kw;
    unpack_pair(kernel_size, kh, kw);
    int64_t dh, dw;
    unpack_pair(dilation, dh, dw);
    int64_t ph, pw;
    unpack_pair(padding, ph, pw);
    int64_t sh, sw;
    unpack_pair(stride, sh, sw);

    int64_t out_h = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    int64_t out_w = (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    int64_t L = out_h * out_w;
    int64_t C_out = C * kh * kw;

    auto output = torch::empty({N, C_out, L}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    unfold_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kh, kw,
        dh, dw,
        ph, pw,
        sh, sw,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "unfold_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor cat_forward(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    int64_t dim) {
    TORCH_CHECK(tensor_a.is_cuda(), "tensor_a must be a CUDA tensor");
    TORCH_CHECK(tensor_b.is_cuda(), "tensor_b must be a CUDA tensor");
    TORCH_CHECK(tensor_a.dtype() == torch::kFloat32, "tensor_a must be float32");
    TORCH_CHECK(tensor_b.dtype() == torch::kFloat32, "tensor_b must be float32");
    TORCH_CHECK(dim == 1, "only dim=1 supported");

    auto a_contig = tensor_a.contiguous();
    auto b_contig = tensor_b.contiguous();

    TORCH_CHECK(a_contig.sizes() == b_contig.sizes(), "tensors must have same shape");

    int64_t total_elements_per_tensor = a_contig.numel();
    int64_t C = a_contig.size(1);
    int64_t N = a_contig.size(0);
    int64_t L = total_elements_per_tensor / (N * C);

    auto output = torch::empty({N, 2 * C, L}, a_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cat_forward_launcher(
        a_contig.data_ptr<float>(),
        b_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements_per_tensor,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "cat_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(py::module& m) {
    m.def("unfold_forward", &unfold_forward,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("dilation"),
          py::arg("padding"),
          py::arg("stride"));
    m.def("cat_forward", &cat_forward,
          py::arg("tensor_a"),
          py::arg("tensor_b"),
          py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
