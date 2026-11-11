#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void gelu_forward_launcher(float* output,
                                      const float* input,
                                      long long total_elements,
                                      cudaStream_t stream);

extern "C" void prod_dim1_keepdim_launcher(float* output,
                                            const float* input,
                                            float* temp_product,
                                            int B, int C, int H, int W,
                                            cudaStream_t stream);

// PyTorch wrapper for gelu_forward
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "gelu_forward: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "gelu_forward: input must be float32");

    auto output = torch::empty_like(x);
    long long total_elements = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gelu_forward_launcher(output.data_ptr<float>(),
                          x.data_ptr<float>(),
                          total_elements,
                          stream);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gelu_forward: CUDA launch failed: ", cudaGetErrorString(err));

    return output;
}

// PyTorch wrapper for prod_dim1_keepdim_forward
torch::Tensor prod_dim1_keepdim_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "prod_dim1_keepdim_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "prod_dim1_keepdim_forward: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "prod_dim1_keepdim_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "prod_dim1_keepdim_forward: input must be 4D");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate output with keepdim shape [B, 1, H, W]
    auto output = torch::empty({B, 1, H, W}, x.options());
    // Allocate temporary contiguous buffer for products: B*H*W floats
    auto temp_product = torch::empty({B * H * W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    prod_dim1_keepdim_launcher(output.data_ptr<float>(),
                               x.data_ptr<float>(),
                               temp_product.data_ptr<float>(),
                               B, C, H, W,
                               stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "prod_dim1_keepdim_forward: CUDA launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward,
          "GELU activation (approximate='none')",
          py::arg("x"));
    m.def("prod_dim1_keepdim_forward", &prod_dim1_keepdim_forward,
          "Product along dim=1 with keepdim=True",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
