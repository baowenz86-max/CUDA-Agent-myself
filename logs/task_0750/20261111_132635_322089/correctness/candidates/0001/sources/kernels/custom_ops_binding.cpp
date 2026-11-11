#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_softmax_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    long long oN, long long oC, long long oH, long long oW,
    cudaStream_t stream);

extern "C" void custom_cross_entropy_launcher(
    const float* input,
    const long long* target,
    float* output,
    int N, int C, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    long long tN, long long tH, long long tW,
    cudaStream_t stream);

// Wrapper for softmax along dim=1
torch::Tensor custom_softmax(torch::Tensor x, int64_t dim) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "custom_softmax: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_softmax: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_softmax: input must be 4D (N,C,H,W)");
    // Contract restricts dim=1, but launcher hardcodes dim=1. Verify passed dim.
    TORCH_CHECK(dim == 1, "custom_softmax: only dim=1 supported");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Output tensor with same shape
    auto y = torch::empty_like(x);

    // Strides (in elements)
    auto stride = x.strides();
    long long sN = (long long)stride[0];
    long long sC = (long long)stride[1];
    long long sH = (long long)stride[2];
    long long sW = (long long)stride[3];
    auto ostride = y.strides();
    long long oN = (long long)ostride[0];
    long long oC = (long long)ostride[1];
    long long oH = (long long)ostride[2];
    long long oW = (long long)ostride[3];

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        sN, sC, sH, sW,
        oN, oC, oH, oW,
        stream
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_softmax kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

// Wrapper for cross entropy (mean)
torch::Tensor custom_cross_entropy(torch::Tensor input, torch::Tensor target) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "custom_cross_entropy: input must be CUDA");
    TORCH_CHECK(target.is_cuda(), "custom_cross_entropy: target must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_cross_entropy: input must be float32");
    TORCH_CHECK(target.dtype() == torch::kInt64, "custom_cross_entropy: target must be int64");
    TORCH_CHECK(input.dim() == 4, "custom_cross_entropy: input must be 4D (N,C,H,W)");
    TORCH_CHECK(target.dim() == 3, "custom_cross_entropy: target must be 3D (N,H,W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    TORCH_CHECK(target.size(0) == N && target.size(1) == H && target.size(2) == W,
                "custom_cross_entropy: target shape mismatch");

    // Output scalar tensor (shape [])
    auto output = torch::zeros({}, torch::dtype(torch::kFloat32).device(input.device()));

    // Strides
    auto istride = input.strides();
    long long sN = (long long)istride[0];
    long long sC = (long long)istride[1];
    long long sH = (long long)istride[2];
    long long sW = (long long)istride[3];
    auto tstride = target.strides();
    long long tN = (long long)tstride[0];
    long long tH = (long long)tstride[1];
    long long tW = (long long)tstride[2];

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_cross_entropy_launcher(
        input.data_ptr<float>(),
        target.data_ptr<int64_t>(),  // int64_t is long long
        output.data_ptr<float>(),
        N, C, H, W,
        sN, sC, sH, sW,
        tN, tH, tW,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_cross_entropy kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax", &custom_softmax, "Softmax along dim=1",
          py::arg("x"), py::arg("dim") = 1);
    m.def("custom_cross_entropy", &custom_cross_entropy, "Cross entropy (mean)",
          py::arg("input"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
