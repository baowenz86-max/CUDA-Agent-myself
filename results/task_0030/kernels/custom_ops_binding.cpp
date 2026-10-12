#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
void polar_launcher(const float* mag, const float* angle, float2* out, int64_t total, cudaStream_t stream);
void sdpa_launcher(const float* q, const float* k, const float* v, float* out, int B, int H, int L, int D, cudaStream_t stream);
}

torch::Tensor polar_forward(torch::Tensor magnitudes, torch::Tensor angles) {
    TORCH_CHECK(magnitudes.is_cuda(), "magnitudes must be a CUDA tensor");
    TORCH_CHECK(angles.is_cuda(), "angles must be a CUDA tensor");
    TORCH_CHECK(magnitudes.dtype() == torch::kFloat32, "magnitudes must be float32");
    TORCH_CHECK(angles.dtype() == torch::kFloat32, "angles must be float32");
    TORCH_CHECK(magnitudes.sizes() == angles.sizes(), "magnitudes and angles must have the same shape");
    TORCH_CHECK(magnitudes.is_contiguous(), "magnitudes must be contiguous");
    TORCH_CHECK(angles.is_contiguous(), "angles must be contiguous");

    auto output = torch::empty_like(magnitudes, torch::TensorOptions().dtype(torch::kComplexFloat));
    TORCH_CHECK(output.is_contiguous(), "output allocation failed");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    polar_launcher(
        magnitudes.data_ptr<float>(),
        angles.data_ptr<float>(),
        reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>()),
        magnitudes.numel(),
        stream
    );

    return output;
}

torch::Tensor one_hot_forward(int64_t num_classes) {
    TORCH_CHECK(num_classes > 0, "num_classes must be positive");
    auto output = torch::empty({num_classes, num_classes}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* data = output.data_ptr<float>();
    // Fill identity matrix on CPU (no CUDA kernel)
    for (int64_t i = 0; i < num_classes; ++i) {
        for (int64_t j = 0; j < num_classes; ++j) {
            data[i * num_classes + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    return output;
}

torch::Tensor sdpa_forward(torch::Tensor query, torch::Tensor key, torch::Tensor value) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(query.dtype() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(key.dtype() == torch::kFloat32, "key must be float32");
    TORCH_CHECK(value.dtype() == torch::kFloat32, "value must be float32");
    TORCH_CHECK(query.dim() == 4, "query must be 4D [B,H,L,D]");
    TORCH_CHECK(key.dim() == 4, "key must be 4D [B,H,L,D]");
    TORCH_CHECK(value.dim() == 4, "value must be 4D [B,H,L,D]");
    TORCH_CHECK(query.size(0) == key.size(0) && key.size(0) == value.size(0), "batch size mismatch");
    TORCH_CHECK(query.size(1) == key.size(1) && key.size(1) == value.size(1), "num_heads mismatch");
    TORCH_CHECK(key.size(2) == value.size(2), "key/value sequence length mismatch");
    TORCH_CHECK(query.size(3) == key.size(3), "head dim mismatch");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");

    int B = static_cast<int>(query.size(0));
    int H = static_cast<int>(query.size(1));
    int L = static_cast<int>(query.size(2));
    int D = static_cast<int>(query.size(3));

    auto output = torch::empty_like(query);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sdpa_launcher(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        output.data_ptr<float>(),
        B, H, L, D,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("polar_forward", &polar_forward, "Compute complex numbers from magnitudes and angles (CUDA)",
          pybind11::arg("magnitudes"), pybind11::arg("angles"));
    m.def("one_hot_forward", &one_hot_forward, "Generate one-hot identity matrix on CPU",
          pybind11::arg("num_classes"));
    m.def("sdpa_forward", &sdpa_forward, "Scaled dot-product attention (CUDA)",
          pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
