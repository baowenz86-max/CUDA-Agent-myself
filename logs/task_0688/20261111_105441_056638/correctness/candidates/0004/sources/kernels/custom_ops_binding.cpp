#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Launcher declarations
extern "C" void custom_cat_launcher(
    float* output,
    const float* x1,
    const float* x2,
    int64_t total_elements,
    int64_t C1,
    int64_t C2,
    int64_t HW,
    int64_t C_out,
    cudaStream_t stream);

extern "C" void custom_l1_loss_launcher(
    const float* input,
    const float* target,
    int64_t total_elements,
    float* output,
    cudaStream_t stream);

static void check_cuda_float32_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor custom_cat(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    check_cuda_float32_contiguous(x1, "x1");
    check_cuda_float32_contiguous(x2, "x2");
    TORCH_CHECK(dim == 1, "custom_cat only supports dim=1");
    TORCH_CHECK(x1.dim() == 4 && x2.dim() == 4, "inputs must be 4D");
    TORCH_CHECK(x1.size(0) == x2.size(0) && x1.size(2) == x2.size(2) && x1.size(3) == x2.size(3),
                "non-channel dimensions must match");

    int64_t N = x1.size(0);
    int64_t C1 = x1.size(1);
    int64_t C2 = x2.size(1);
    int64_t H = x1.size(2);
    int64_t W = x1.size(3);
    int64_t HW = H * W;
    int64_t C_out = C1 + C2;
    int64_t total_elements = N * C_out * HW;

    auto output = torch::empty({N, C_out, H, W}, x1.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_cat_launcher(
        output.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        total_elements,
        C1,
        C2,
        HW,
        C_out,
        stream);

    return output;
}

torch::Tensor custom_l1_loss(torch::Tensor input, torch::Tensor target) {
    check_cuda_float32_contiguous(input, "input");
    check_cuda_float32_contiguous(target, "target");
    TORCH_CHECK(input.sizes() == target.sizes(), "input and target must have same shape");
    TORCH_CHECK(input.numel() > 0, "empty tensor not supported");

    auto output = torch::empty({}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_l1_loss_launcher(
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        input.numel(),
        output.data_ptr<float>(),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_cat", &custom_cat, "Concatenate two tensors along dim=1",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
    m.def("custom_l1_loss", &custom_l1_loss, "Mean L1 loss",
          py::arg("input"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
