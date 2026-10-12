#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void custom_unbind_max_launcher(
    const float* input,
    float* values,
    int64_t* indices,
    int B, int C, int H, int W,
    cudaStream_t stream);

std::vector<std::pair<torch::Tensor, torch::Tensor>> custom_unbind_max(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_unbind_max: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_unbind_max: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_unbind_max: input must be 4D (B, C, H, W)");
    TORCH_CHECK(x.is_contiguous(), "custom_unbind_max: input must be contiguous");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto options_long  = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
    auto values = torch::empty({B, H, W}, options_float);
    auto indices = torch::empty({B, H, W}, options_long);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_unbind_max_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    std::vector<std::pair<torch::Tensor, torch::Tensor>> result;
    result.reserve(static_cast<size_t>(W));
    for (int64_t w = 0; w < W; ++w) {
        torch::Tensor val_w = values.index({torch::indexing::Slice(), torch::indexing::Slice(), w}).contiguous();
        torch::Tensor idx_w = indices.index({torch::indexing::Slice(), torch::indexing::Slice(), w}).contiguous();
        result.emplace_back(val_w, idx_w);
    }

    return result;
}

void register_custom_unbind_max(pybind11::module& m) {
    m.def("custom_unbind_max", &custom_unbind_max,
          "Perform unbind along last dimension and max over channel dimension",
          py::arg("x"));
}

REGISTER_BINDING(custom_unbind_max, register_custom_unbind_max);
