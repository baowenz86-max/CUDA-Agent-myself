#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launchers exported from custom_ops.cu
extern "C" void instance_norm_launcher(const float* x,
                                        float* y,
                                        int N,
                                        int C,
                                        int H,
                                        int W,
                                        float eps,
                                        cudaStream_t stream);

extern "C" void add_scalar_launcher(const float* x,
                                     float* y,
                                     int n,
                                     float scalar,
                                     cudaStream_t stream);

torch::Tensor instance_norm_forward(torch::Tensor x, int64_t num_features, double eps) {
    TORCH_CHECK(x.is_cuda(), "instance_norm_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "instance_norm_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "instance_norm_forward: input must be 4D (N,C,H,W)");
    TORCH_CHECK(x.size(1) == num_features, "instance_norm_forward: num_features mismatch");

    // Ensure contiguous; if not, make contiguous to match launcher expectation
    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);

    int N = static_cast<int>(x_contig.size(0));
    int C = static_cast<int>(x_contig.size(1));
    int H = static_cast<int>(x_contig.size(2));
    int W = static_cast<int>(x_contig.size(3));
    float eps_f = static_cast<float>(eps);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    instance_norm_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        eps_f,
        stream
    );

    return y;
}

torch::Tensor add_scalar(torch::Tensor x, double scalar) {
    TORCH_CHECK(x.is_cuda(), "add_scalar: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "add_scalar: input must be float32");

    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);
    int64_t n64 = x_contig.numel();
    TORCH_CHECK(n64 <= static_cast<int64_t>(INT_MAX), "add_scalar: tensor too large for int");
    int n = static_cast<int>(n64);
    float scalar_f = static_cast<float>(scalar);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    add_scalar_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        n,
        scalar_f,
        stream
    );

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("instance_norm_forward", &instance_norm_forward,
          "Instance normalization forward (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("num_features"),
          pybind11::arg("eps"));
    m.def("add_scalar", &add_scalar,
          "Add scalar to tensor (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("scalar"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
