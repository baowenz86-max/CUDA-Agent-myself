#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_cat_launcher(float* output,
                                   const float* input1,
                                   const float* input2,
                                   int N, int C_in, int H, int W,
                                   cudaStream_t stream);

extern "C" void custom_mul_launcher(float* output,
                                    const float* input,
                                    const float* multiplier,
                                    int total,
                                    cudaStream_t stream);

// Utility to check that a tensor is a CUDA float32 contiguous tensor
torch::Tensor custom_cat_forward(const std::vector<torch::Tensor>& inputs, int64_t dim) {
    TORCH_CHECK(dim == 1, "custom_cat_forward only supports dim=1");
    TORCH_CHECK(inputs.size() == 2, "custom_cat_forward expects exactly 2 input tensors");
    
    const auto& input1 = inputs[0];
    const auto& input2 = inputs[1];
    
    TORCH_CHECK(input1.is_cuda() && input2.is_cuda(), "Input tensors must be CUDA");
    TORCH_CHECK(input1.dtype() == torch::kFloat32 && input2.dtype() == torch::kFloat32,
                "Input tensors must be float32");
    TORCH_CHECK(input1.is_contiguous() && input2.is_contiguous(),
                "Input tensors must be contiguous");
    
    // Expected shape: [N, C_in, H, W]
    const auto sizes1 = input1.sizes();
    const auto sizes2 = input2.sizes();
    TORCH_CHECK(sizes1.size() == 4 && sizes2.size() == 4, "Input tensors must be 4D");
    TORCH_CHECK(sizes1 == sizes2, "Input tensors must have identical shapes");
    
    int N = sizes1[0];
    int C_in = sizes1[1];
    int H = sizes1[2];
    int W = sizes1[3];
    int C_out = 2 * C_in;
    
    // Allocate output tensor of shape [N, C_out, H, W]
    auto output = torch::empty({N, C_out, H, W}, input1.options());
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    custom_cat_launcher(
        output.data_ptr<float>(),
        input1.data_ptr<float>(),
        input2.data_ptr<float>(),
        N, C_in, H, W,
        stream
    );
    
    return output;
}

torch::Tensor custom_mul_forward(const torch::Tensor& x, const torch::Tensor& multiplier) {
    TORCH_CHECK(x.is_cuda() && multiplier.is_cuda(), "Input tensors must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && multiplier.dtype() == torch::kFloat32,
                "Input tensors must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(multiplier.is_contiguous(), "multiplier must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(multiplier.numel() == 1, "multiplier must have exactly 1 element");
    
    int total = x.numel();
    
    auto output = torch::empty_like(x);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    custom_mul_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        total,
        stream
    );
    
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_cat_forward", &custom_cat_forward,
          "Concatenate two float32 CUDA tensors along dim 1",
          py::arg("inputs"), py::arg("dim") = 1);
    m.def("custom_mul_forward", &custom_mul_forward,
          "Multiply tensor x by scalar multiplier",
          py::arg("x"), py::arg("multiplier"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
