// PyTorch binding for custom ops implemented in custom_ops.cu
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_gt_scalar_launcher(const float* x, bool* out, double threshold, int64_t numel, cudaStream_t stream);
extern "C" void custom_logical_and_launcher(const bool* a, const bool* b, bool* out, int64_t numel, cudaStream_t stream);
extern "C" void custom_where_mask_launcher(const float* x, const bool* mask, float* out, int B, int C, int H, int W, const int64_t* x_strides, const int64_t* mask_strides, const int64_t* out_strides, cudaStream_t stream);
extern "C" void custom_max_dim_launcher(const float* x, float* max_values, int64_t* max_indices, int B, int C, int H, int W, const int64_t* x_strides, const int64_t* out_val_strides, const int64_t* out_idx_strides, cudaStream_t stream);
extern "C" void custom_any_gt_launcher(const float* x, bool* any_mask, int B, int C, int H, int W, const int64_t* x_strides, const int64_t* out_any_strides, double any_threshold, cudaStream_t stream);
extern "C" void custom_threshold_mask_launcher(const float* x, float* out, int B, int C, int H, int W, const int64_t* x_strides, const int64_t* out_strides, double max_threshold, double any_threshold, cudaStream_t stream);

namespace {

// Helper to get current CUDA stream
cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Helper to check tensor on CUDA and dtype float32
void check_cuda_float(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
}

void check_cuda_bool(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kBool, name, " must be bool");
}

// Helper to get int64 strides array from tensor (must be 4D)
std::array<int64_t, 4> get_strides4(const torch::Tensor& t) {
    TORCH_CHECK(t.dim() == 4, "Expected 4D tensor");
    return {t.stride(0), t.stride(1), t.stride(2), t.stride(3)};
}

// custom_gt_scalar
// Input: float tensor any shape; output bool same shape
// Launcher signature: (const float* x, bool* out, double threshold, int64_t numel, stream)
torch::Tensor custom_gt_scalar(torch::Tensor x, double threshold) {
    check_cuda_float(x, "x");
    auto out = torch::empty_like(x, x.options().dtype(torch::kBool));
    int64_t numel = x.numel();
    cudaStream_t stream = getCurrentCUDAStream();
    custom_gt_scalar_launcher(
        x.data_ptr<float>(),
        out.data_ptr<bool>(),
        threshold,
        numel,
        stream
    );
    return out;
}

// custom_logical_and
// Inputs: bool tensors of same shape; output bool same shape
// Launcher: (const bool* a, const bool* b, bool* out, int64_t numel, stream)
torch::Tensor custom_logical_and(torch::Tensor a, torch::Tensor b) {
    check_cuda_bool(a, "a");
    check_cuda_bool(b, "b");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have same shape");
    auto out = torch::empty_like(a);
    int64_t numel = a.numel();
    cudaStream_t stream = getCurrentCUDAStream();
    custom_logical_and_launcher(
        a.data_ptr<bool>(),
        b.data_ptr<bool>(),
        out.data_ptr<bool>(),
        numel,
        stream
    );
    return out;
}

// custom_where_mask
// mask: [B,1,H,W] bool; x: [B,C,H,W] float; output same as x
// Launcher: (const float* x, const bool* mask, float* out, B,C,H,W,
//            x_strides, mask_strides, out_strides, stream)
torch::Tensor custom_where_mask(torch::Tensor mask, torch::Tensor x) {
    check_cuda_bool(mask, "mask");
    check_cuda_float(x, "x");
    TORCH_CHECK(mask.dim() == 4, "mask must be 4D");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(mask.size(0) == x.size(0), "batch mismatch");
    TORCH_CHECK(mask.size(1) == 1, "mask channel dim must be 1");
    TORCH_CHECK(mask.size(2) == x.size(2) && mask.size(3) == x.size(3), "spatial mismatch");
    auto out = torch::empty_like(x);
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    auto x_strides = get_strides4(x);
    auto mask_strides = get_strides4(mask);
    auto out_strides = get_strides4(out);
    cudaStream_t stream = getCurrentCUDAStream();
    custom_where_mask_launcher(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        out.data_ptr<float>(),
        B, C, H, W,
        x_strides.data(),
        mask_strides.data(),
        out_strides.data(),
        stream
    );
    return out;
}

// custom_max_dim
// Input: x float [B,C,H,W], dim=1, keepdim=true
// Returns tuple (values float, indices int64) both [B,1,H,W]
// Launcher: (const float* x, float* max_values, int64_t* max_indices,
//            B,C,H,W, x_strides, out_val_strides, out_idx_strides, stream)
std::tuple<torch::Tensor, torch::Tensor> custom_max_dim(torch::Tensor x, int64_t dim, bool keepdim) {
    check_cuda_float(x, "x");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(keepdim == true, "Only keepdim=True is supported");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    // Output shapes [B,1,H,W]
    auto values = torch::empty({B, 1, H, W}, x.options());
    auto indices = torch::empty({B, 1, H, W}, x.options().dtype(torch::kInt64));
    auto x_strides = get_strides4(x);
    auto val_strides = get_strides4(values);
    auto idx_strides = get_strides4(indices);
    cudaStream_t stream = getCurrentCUDAStream();
    custom_max_dim_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        B, C, H, W,
        x_strides.data(),
        val_strides.data(),
        idx_strides.data(),
        stream
    );
    return std::make_tuple(values, indices);
}

// custom_any_gt
// Input: x float [B,C,H,W], threshold, dim=1, keepdim=true
// Output bool [B,1,H,W]
// Launcher: (const float* x, bool* any_mask, B,C,H,W,
//            x_strides, out_any_strides, threshold, stream)
torch::Tensor custom_any_gt(torch::Tensor x, double threshold, int64_t dim, bool keepdim) {
    check_cuda_float(x, "x");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(keepdim == true, "Only keepdim=True is supported");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    auto out = torch::empty({B, 1, H, W}, x.options().dtype(torch::kBool));
    auto x_strides = get_strides4(x);
    auto out_strides = get_strides4(out);
    cudaStream_t stream = getCurrentCUDAStream();
    custom_any_gt_launcher(
        x.data_ptr<float>(),
        out.data_ptr<bool>(),
        B, C, H, W,
        x_strides.data(),
        out_strides.data(),
        threshold,
        stream
    );
    return out;
}

// custom_threshold_mask (fused)
// Input: x float [B,C,H,W], max_threshold, any_threshold
// Output: float same shape as x
// Launcher: (const float* x, float* out, B,C,H,W,
//            x_strides, out_strides, max_threshold, any_threshold, stream)
torch::Tensor custom_threshold_mask(torch::Tensor x, double max_threshold, double any_threshold) {
    check_cuda_float(x, "x");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    auto out = torch::empty_like(x);
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    auto x_strides = get_strides4(x);
    auto out_strides = get_strides4(out);
    cudaStream_t stream = getCurrentCUDAStream();
    custom_threshold_mask_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        x_strides.data(),
        out_strides.data(),
        max_threshold,
        any_threshold,
        stream
    );
    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_threshold_mask", &custom_threshold_mask,
          "Fused threshold mask",
          py::arg("x"), py::arg("max_threshold"), py::arg("any_threshold"));
    m.def("custom_max_dim", &custom_max_dim,
          "Max over channel dim",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));
    m.def("custom_gt_scalar", &custom_gt_scalar,
          "Elementwise greater-than scalar",
          py::arg("x"), py::arg("threshold"));
    m.def("custom_any_gt", &custom_any_gt,
          "Any greater-than threshold over channel dim",
          py::arg("x"), py::arg("threshold"), py::arg("dim"), py::arg("keepdim"));
    m.def("custom_logical_and", &custom_logical_and,
          "Elementwise logical AND",
          py::arg("a"), py::arg("b"));
    m.def("custom_where_mask", &custom_where_mask,
          "Where with mask broadcast",
          py::arg("mask"), py::arg("x"));
}

} // namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
