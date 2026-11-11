// Binding for custom operators used by model_new.py
// All kernels are implemented in custom_ops.cu
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_hardtanh_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream
);

extern "C" void custom_slice_last_dim_launcher(
    float* output,
    const float* input,
    int64_t output_numel,
    cudaStream_t stream
);

extern "C" void custom_instance_norm_3d_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t channels,
    int64_t depth,
    int64_t height,
    int64_t width,
    double eps,
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// custom_hardtanh
// -----------------------------------------------------------------------------
torch::Tensor custom_hardtanh(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "custom_hardtanh: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "custom_hardtanh: input must be float32");
    x = x.contiguous();
    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_hardtanh_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream
    );
    return output;
}

// -----------------------------------------------------------------------------
// custom_slice_last_dim
// -----------------------------------------------------------------------------
torch::Tensor custom_slice_last_dim(torch::Tensor x, int64_t index) {
    TORCH_CHECK(x.is_cuda(), "custom_slice_last_dim: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "custom_slice_last_dim: input must be float32");
    TORCH_CHECK(index == 0, "custom_slice_last_dim: only index 0 is supported");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == 1, "custom_slice_last_dim: last dimension must be 1");

    x = x.contiguous();
    auto output_sizes = x.sizes().vec();
    output_sizes.pop_back();
    auto output = torch::empty(output_sizes, x.options());

    int64_t output_numel = output.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_slice_last_dim_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        output_numel,
        stream
    );
    return output;
}

// -----------------------------------------------------------------------------
// custom_instance_norm_3d
// -----------------------------------------------------------------------------
torch::Tensor custom_instance_norm_3d(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "custom_instance_norm_3d: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "custom_instance_norm_3d: input must be float32");
    TORCH_CHECK(x.dim() == 5, "custom_instance_norm_3d: expected [B,C,D,H,W]");

    x = x.contiguous();
    auto output = torch::empty_like(x);

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t D = x.size(2);
    int64_t H = x.size(3);
    int64_t W = x.size(4);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_instance_norm_3d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        B,
        C,
        D,
        H,
        W,
        eps,
        stream
    );
    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_hardtanh", &custom_hardtanh,
          "Apply hardtanh activation (clip to [min_val, max_val])",
          py::arg("x"),
          py::arg("min_val") = -1.0,
          py::arg("max_val") = 1.0);

    m.def("custom_slice_last_dim", &custom_slice_last_dim,
          "Remove the last dimension (size 1) of a tensor",
          py::arg("x"),
          py::arg("index") = 0);

    m.def("custom_instance_norm_3d", &custom_instance_norm_3d,
          "Instance normalization over spatial dimensions (D,H,W)",
          py::arg("x"),
          py::arg("eps") = 1e-05);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
