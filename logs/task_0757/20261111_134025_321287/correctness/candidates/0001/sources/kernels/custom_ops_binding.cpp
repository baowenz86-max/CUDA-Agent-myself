#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_cat_launcher(
    float* output,
    const float* x1,
    const float* x2,
    int64_t out_numel,
    int N,
    int C1,
    int C2,
    int H,
    int W,
    int64_t s1_N, int64_t s1_C, int64_t s1_H, int64_t s1_W,
    int64_t s2_N, int64_t s2_C, int64_t s2_H, int64_t s2_W,
    cudaStream_t stream
);

extern "C" void custom_ne_launcher(
    const float* input,
    bool* output,
    int64_t numel,
    float other,
    cudaStream_t stream
);

// Wrapper for torch.cat equivalent
torch::Tensor custom_cat(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    // Input validation
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == 4 && x2.dim() == 4, "Expected 4D tensors");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    // Assume x1 and x2 have same sizes except on dim=1
    TORCH_CHECK(x1.size(0) == x2.size(0) && x1.size(2) == x2.size(2) && x1.size(3) == x2.size(3),
                "Non-cat dimensions must match");

    int N = x1.size(0);
    int C1 = x1.size(1);
    int C2 = x2.size(1);
    int H = x1.size(2);
    int W = x1.size(3);
    int C_out = C1 + C2;
    int64_t out_numel = static_cast<int64_t>(N) * C_out * H * W;

    // Allocate contiguous output
    auto output = torch::empty({N, C_out, H, W}, x1.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with strides from input tensors
    custom_cat_launcher(
        output.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out_numel,
        N, C1, C2, H, W,
        x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        stream
    );

    // Check for kernel launch errors (cudaGetLastError is already called in launcher,
    // but we can sync here for debugging; not required for production)
    // cudaDeviceSynchronize(); // Optional

    return output;
}

// Wrapper for torch.ne equivalent
torch::Tensor custom_ne(torch::Tensor x, double other) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous for custom_ne");

    auto output = torch::empty_like(x, x.options().dtype(torch::kBool));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_ne_launcher(
        x.data_ptr<float>(),
        output.data_ptr<bool>(),
        x.numel(),
        static_cast<float>(other),
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_cat", &custom_cat,
          "Concatenate two float32 tensors along dim=1",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
    m.def("custom_ne", &custom_ne,
          "Element-wise not equal to scalar (float32 -> bool)",
          py::arg("x"), py::arg("other"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
