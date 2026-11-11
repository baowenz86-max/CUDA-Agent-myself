#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void cat_forward_launcher(const float* x, float* out,
                                     int64_t B, int64_t S, int64_t H,
                                     int64_t cat_dim, cudaStream_t stream);

extern "C" void argmax_forward_launcher(const float* x, int64_t* out,
                                        int64_t B, int64_t S, int64_t D,
                                        int64_t dim, cudaStream_t stream);

// cat_forward wrapper
torch::Tensor cat_forward(torch::Tensor x, int64_t half_size, int64_t cat_dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, S, 2*H)");

    // For this task, cat_dim is expected to be 1, and input's last dim is 2*half_size.
    TORCH_CHECK(cat_dim == 1, "Only cat_dim=1 is supported");
    int64_t B = x.size(0);
    int64_t S = x.size(1);
    int64_t H = half_size;
    TORCH_CHECK(x.size(2) == 2 * H, "Input last dimension must be 2*half_size");

    auto out = torch::empty({B, 2 * S, H}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cat_forward_launcher(x.data_ptr<float>(), out.data_ptr<float>(),
                         B, S, H, cat_dim, stream);
    return out;
}

// argmax_forward wrapper
torch::Tensor argmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, S, D)");
    TORCH_CHECK(dim == 2, "Only dim=2 is supported");

    int64_t B = x.size(0);
    int64_t S = x.size(1);
    int64_t D = x.size(2);

    auto out = torch::empty({B, S}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    argmax_forward_launcher(x.data_ptr<float>(), out.data_ptr<int64_t>(),
                            B, S, D, dim, stream);
    return out;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("cat_forward", &cat_forward,
          "Concatenate two halves along dim=1",
          py::arg("x"), py::arg("half_size"), py::arg("cat_dim"));
    m.def("argmax_forward", &argmax_forward,
          "Argmax along last dim",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
