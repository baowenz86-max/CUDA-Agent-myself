#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void cat_dim1_forward_launcher(const float* x1,
                                          const float* x2,
                                          float* out,
                                          int N, int C1, int C2, int H, int W,
                                          cudaStream_t stream);

extern "C" void smooth_l1_loss_mean_forward_launcher(const float* x,
                                                       float* out,
                                                       int numel,
                                                       cudaStream_t stream);

// Helper to check CUDA error after kernel launch
static inline void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

torch::Tensor cat_dim1_forward(torch::Tensor x1, torch::Tensor x2) {
    // Parameter validation
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.is_contiguous(), "x1 must be contiguous");
    TORCH_CHECK(x2.is_contiguous(), "x2 must be contiguous");
    TORCH_CHECK(x1.dim() == 4, "x1 must be 4D");
    TORCH_CHECK(x2.dim() == 4, "x2 must be 4D");
    TORCH_CHECK(x1.size(0) == x2.size(0), "Batch size mismatch");
    TORCH_CHECK(x1.size(2) == x2.size(2) && x1.size(3) == x2.size(3), "H/W mismatch");

    // Extract dimensions
    int N = static_cast<int>(x1.size(0));
    int C1 = static_cast<int>(x1.size(1));
    int C2 = static_cast<int>(x2.size(1));
    int H = static_cast<int>(x1.size(2));
    int W = static_cast<int>(x1.size(3));

    // Allocate output tensor of shape [N, C1+C2, H, W]
    auto out = torch::empty({N, C1 + C2, H, W}, x1.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    cat_dim1_forward_launcher(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C1, C2, H, W,
        stream
    );

    // Check for errors
    check_cuda_error("cat_dim1_forward_launcher failed");

    return out;
}

torch::Tensor smooth_l1_loss_mean_forward(torch::Tensor x) {
    // Parameter validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int numel = static_cast<int>(x.numel());
    TORCH_CHECK(numel > 0, "x must not be empty");

    // Allocate scalar output
    auto out = torch::empty({}, x.options());

    // Get current stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    smooth_l1_loss_mean_forward_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        stream
    );

    // Check errors
    check_cuda_error("smooth_l1_loss_mean_forward_launcher failed");

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("cat_dim1_forward", &cat_dim1_forward,
          "Concatenate two tensors along dimension 1",
          py::arg("x1"), py::arg("x2"));
    m.def("smooth_l1_loss_mean_forward", &smooth_l1_loss_mean_forward,
          "Compute smooth_l1_loss mean against zeros",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
