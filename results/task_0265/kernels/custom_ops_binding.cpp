#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void fma_addcdiv_launcher(
    float* out,
    const float* x,
    const float* tensor1,
    const float* tensor2,
    float value,
    int total_elements,
    int dim0, int dim1, int dim2, int dim3,
    int stride_x0, int stride_x1, int stride_x2, int stride_x3,
    int stride_t1_0, int stride_t1_1, int stride_t1_2, int stride_t1_3,
    int stride_t2_0, int stride_t2_1, int stride_t2_2, int stride_t2_3,
    cudaStream_t stream);

extern "C" void add_bias_launcher(
    float* out,
    const float* x,
    const float* bias,
    int total_elements,
    int dim0, int dim1, int dim2, int dim3,
    int stride_x0, int stride_x1, int stride_x2, int stride_x3,
    cudaStream_t stream);

// Helper to validate 4D contiguous or non-contiguous float32 CUDA tensor
static void check_fma_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.dim() == 4, name, " must be 4-dimensional");
    TORCH_CHECK(t.numel() > 0, name, " must have at least one element");
}

torch::Tensor fma_addcdiv_forward(
    torch::Tensor x,
    torch::Tensor tensor1,
    torch::Tensor tensor2,
    double value = 1.0) {

    check_fma_tensor(x, "x");
    check_fma_tensor(tensor1, "tensor1");
    check_fma_tensor(tensor2, "tensor2");

    // Check same shape
    TORCH_CHECK(x.sizes() == tensor1.sizes() && x.sizes() == tensor2.sizes(),
                "All input tensors must have the same shape");

    auto out = torch::empty_like(x);

    int total_elements = static_cast<int>(x.numel());
    TORCH_CHECK(total_elements > 0, "Input tensors must not be empty");

    int dim0 = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);
    int dim3 = x.size(3);

    int stride_x0 = x.stride(0);
    int stride_x1 = x.stride(1);
    int stride_x2 = x.stride(2);
    int stride_x3 = x.stride(3);

    int stride_t1_0 = tensor1.stride(0);
    int stride_t1_1 = tensor1.stride(1);
    int stride_t1_2 = tensor1.stride(2);
    int stride_t1_3 = tensor1.stride(3);

    int stride_t2_0 = tensor2.stride(0);
    int stride_t2_1 = tensor2.stride(1);
    int stride_t2_2 = tensor2.stride(2);
    int stride_t2_3 = tensor2.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    fma_addcdiv_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        tensor1.data_ptr<float>(),
        tensor2.data_ptr<float>(),
        static_cast<float>(value),
        total_elements,
        dim0, dim1, dim2, dim3,
        stride_x0, stride_x1, stride_x2, stride_x3,
        stride_t1_0, stride_t1_1, stride_t1_2, stride_t1_3,
        stride_t2_0, stride_t2_1, stride_t2_2, stride_t2_3,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fma_addcdiv kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

torch::Tensor add_bias_forward(
    torch::Tensor x,
    torch::Tensor bias,
    int64_t bias_shape_tuple = 3) {

    check_fma_tensor(x, "x");

    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3-dimensional");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias must have shape [C,1,1]");
    TORCH_CHECK(bias.stride(0) == 1 && bias.stride(1) == 1 && bias.stride(2) == 1,
                "bias must be contiguous");

    TORCH_CHECK(x.size(1) == bias.size(0), "bias channel dimension must match x dimension 1");
    TORCH_CHECK(bias_shape_tuple == 3, "bias_shape_tuple must be 3 for this task");

    auto out = torch::empty_like(x);

    int total_elements = static_cast<int>(x.numel());
    int dim0 = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);
    int dim3 = x.size(3);

    int stride_x0 = x.stride(0);
    int stride_x1 = x.stride(1);
    int stride_x2 = x.stride(2);
    int stride_x3 = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    add_bias_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        total_elements,
        dim0, dim1, dim2, dim3,
        stride_x0, stride_x1, stride_x2, stride_x3,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "add_bias kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("fma_addcdiv", &fma_addcdiv_forward,
          "Fused addcdiv: out = x + value * (tensor1 / tensor2)",
          py::arg("x"),
          py::arg("tensor1"),
          py::arg("tensor2"),
          py::arg("value") = 1.0);

    m.def("add_bias", &add_bias_forward,
          "Add bias with shape [C,1,1] broadcasting",
          py::arg("x"),
          py::arg("bias"),
          py::arg("bias_shape_tuple") = 3);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
