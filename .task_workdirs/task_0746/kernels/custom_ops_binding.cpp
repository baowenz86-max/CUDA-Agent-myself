#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launchers from custom_ops.cu
extern "C" void custom_softmax_launcher(const float* x, float* y,
                                        int64_t N, int64_t C,
                                        int64_t H, int64_t W,
                                        cudaStream_t stream);

extern "C" void custom_rot90_launcher(const float* x, float* y,
                                      int64_t N, int64_t C,
                                      int64_t H, int64_t W,
                                      cudaStream_t stream);

namespace {

torch::Tensor custom_softmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "custom_softmax: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_softmax: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_softmax: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "custom_softmax: input must be 4D (N,C,H,W)");
    TORCH_CHECK(dim == 1, "custom_softmax: only dim=1 is supported");

    auto y = torch::empty_like(x);
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_softmax_launcher(x.data_ptr<float>(), y.data_ptr<float>(),
                            N, C, H, W, stream);
    return y;
}

torch::Tensor custom_rot90_forward(torch::Tensor x, int64_t k, int64_t axis1, int64_t axis2) {
    TORCH_CHECK(x.is_cuda(), "custom_rot90: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_rot90: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_rot90: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "custom_rot90: input must be 4D (N,C,H,W)");
    TORCH_CHECK(k == 1, "custom_rot90: only k=1 is supported");
    TORCH_CHECK(axis1 == 2 && axis2 == 3, "custom_rot90: only axes=(2,3) is supported");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Output shape [N, C, W, H]
    auto y = torch::empty({N, C, W, H}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_rot90_launcher(x.data_ptr<float>(), y.data_ptr<float>(),
                          N, C, H, W, stream);
    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax", &custom_softmax_forward,
          "Softmax over channel dimension (dim=1)",
          py::arg("x"), py::arg("dim"));
    m.def("custom_rot90", &custom_rot90_forward,
          "Rot90 for 4D tensor with k=1, axes=(2,3)",
          py::arg("x"), py::arg("k"), py::arg("axis1"), py::arg("axis2"));
}

} // anonymous namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
