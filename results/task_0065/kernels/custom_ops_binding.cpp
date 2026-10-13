#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void channel_mean_launcher(const float* x, float* mean, int N, int C, int H, int W, cudaStream_t stream);
extern "C" void subtract_mean_launcher(const float* x, const float* mean, float subtract_value, float* out, int N, int C, int H, int W, cudaStream_t stream);

// Compute mean over dimensions 1,2,3 for each batch element, output shape [N,1,1,1]
torch::Tensor channel_mean_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto mean = torch::empty({N, 1, 1, 1}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    channel_mean_launcher(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    return mean;
}

// output = x - mean - subtract_value, broadcasting mean over C,H,W
torch::Tensor subtract_mean_forward(torch::Tensor x, torch::Tensor mean, double subtract_value) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mean.is_cuda(), "mean must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mean.is_contiguous(), "mean must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(mean.dtype() == torch::kFloat32, "mean must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(mean.dim() == 4, "mean must be 4D");
    TORCH_CHECK(mean.size(0) == x.size(0), "mean batch size must match x");
    TORCH_CHECK(mean.size(1) == 1 && mean.size(2) == 1 && mean.size(3) == 1, "mean shape must be [N,1,1,1]");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    subtract_mean_launcher(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        static_cast<float>(subtract_value),
        out.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("channel_mean_forward", &channel_mean_forward,
          "Compute mean over dims (1,2,3) keeping dims",
          py::arg("x"));
    m.def("subtract_mean_forward", &subtract_mean_forward,
          "Subtract mean and scalar from input",
          py::arg("x"), py::arg("mean"), py::arg("subtract_value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
