#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void custom_unbind_max_launcher(
    const float* input,
    float* values,
    long long* indices,
    int B, int C, int H, int W,
    cudaStream_t stream);

// Custom op: performs torch.unbind(x, dim=-1) then torch.max(xi, dim=1)
// Input: x shape (B, C, H, W) float32 CUDA contiguous
// Output: std::vector of W pairs (values, indices), each values/indices shape (B,H)
// values dtype float32, indices dtype int64
std::vector<std::pair<torch::Tensor, torch::Tensor>> custom_unbind_max(torch::Tensor x) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "custom_unbind_max: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_unbind_max: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_unbind_max: input must be 4D (B, C, H, W)");
    TORCH_CHECK(x.is_contiguous(), "custom_unbind_max: input must be contiguous");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Allocate output tensors
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto options_long  = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
    auto values = torch::empty({B, H, W}, options_float);
    auto indices = torch::empty({B, H, W}, options_long);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
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

    // Build list of W max-tuples. Each max-tuple is (values[..., w], indices[..., w])
    // Use slicing to get contiguous views? In fact, values[:, :, w] is not contiguous in memory
    // because stride for W is 1, so each slice has stride (H*W, W, 1) -> not contiguous.
    // However, torch.max returns contiguous tensors. We should call .contiguous() on each slice.
    // But contract shows stride [128,1] for each output, meaning contiguous. So we make them contiguous.
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
