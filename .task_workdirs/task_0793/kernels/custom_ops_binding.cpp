#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void isin_forward_launcher(
    const float* x,
    const float* elements,
    bool* output,
    long long total_elements,
    int elements_count,
    cudaStream_t stream);

extern "C" void zero_pad2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long total_output_elements,
    cudaStream_t stream);

static torch::Tensor isin_forward(torch::Tensor x, torch::Tensor elements) {
    TORCH_CHECK(x.is_cuda(), "isin_forward: x must be a CUDA tensor");
    TORCH_CHECK(elements.is_cuda(), "isin_forward: elements must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "isin_forward: x must be float32");
    TORCH_CHECK(elements.scalar_type() == torch::kFloat32, "isin_forward: elements must be float32");
    TORCH_CHECK(elements.dim() == 1, "isin_forward: elements must be 1D");
    TORCH_CHECK(elements.numel() > 0, "isin_forward: elements must not be empty");

    auto x_contig = x.contiguous();
    auto elements_contig = elements.contiguous();

    auto output = torch::empty(x_contig.sizes(), x_contig.options().dtype(torch::kBool));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    isin_forward_launcher(
        x_contig.data_ptr<float>(),
        elements_contig.data_ptr<float>(),
        output.data_ptr<bool>(),
        static_cast<long long>(x_contig.numel()),
        static_cast<int>(elements_contig.numel()),
        stream);

    return output;
}

static torch::Tensor zero_pad2d_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom) {
    TORCH_CHECK(x.is_cuda(), "zero_pad2d_forward: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "zero_pad2d_forward: x must be float32");
    TORCH_CHECK(x.dim() == 4, "zero_pad2d_forward: x must be 4D (N,C,H,W)");
    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "zero_pad2d_forward: padding sizes must be non-negative");

    auto x_contig = x.contiguous();
    const int64_t N = x_contig.size(0);
    const int64_t C = x_contig.size(1);
    const int64_t H = x_contig.size(2);
    const int64_t W = x_contig.size(3);
    const int64_t out_H = H + pad_top + pad_bottom;
    const int64_t out_W = W + pad_left + pad_right;

    auto output = torch::empty({N, C, out_H, out_W}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    zero_pad2d_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        static_cast<long long>(output.numel()),
        stream);

    return output;
}

static void register_custom_ops(pybind11::module& m) {
    m.def("isin_forward", &isin_forward,
          "Check element membership (torch.isin equivalent)",
          py::arg("x"), py::arg("elements"));
    m.def("zero_pad2d_forward", &zero_pad2d_forward,
          "Zero padding on last two dimensions",
          py::arg("x"), py::arg("pad_left"), py::arg("pad_right"),
          py::arg("pad_top"), py::arg("pad_bottom"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
