// PyTorch binding for custom ops implemented in custom_ops.cu
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_gt_scalar_launcher(const float* x, bool* out, double threshold, int64_t numel, cudaStream_t stream);
extern "C" void custom_logical_and_launcher(const bool* a, const bool* b, bool* out, int64_t numel, cudaStream_t stream);
extern "C" void custom_where_mask_launcher(const float* x, const bool* mask, float* out, int B, int C, int H, int W, int64_t x_s0, int64_t x_s1, int64_t x_s2, int64_t x_s3, int64_t mask_s0, int64_t mask_s1, int64_t mask_s2, int64_t mask_s3, int64_t out_s0, int64_t out_s1, int64_t out_s2, int64_t out_s3, cudaStream_t stream);
extern "C" void custom_max_dim_launcher(const float* x, float* max_values, int64_t* max_indices, int B, int C, int H, int W, int64_t x_s0, int64_t x_s1, int64_t x_s2, int64_t x_s3, int64_t out_val_s0, int64_t out_val_s1, int64_t out_val_s2, int64_t out_val_s3, int64_t out_idx_s0, int64_t out_idx_s1, int64_t out_idx_s2, int64_t out_idx_s3, cudaStream_t stream);
extern "C" void custom_any_gt_launcher(const float* x, bool* any_mask, int B, int C, int H, int W, int64_t x_s0, int64_t x_s1, int64_t x_s2, int64_t x_s3, int64_t out_any_s0, int64_t out_any_s1, int64_t out_any_s2, int64_t out_any_s3, double any_threshold, cudaStream_t stream);
extern "C" void custom_threshold_mask_launcher(const float* x, float* out, int B, int C, int H, int W, int64_t x_s0, int64_t x_s1, int64_t x_s2, int64_t x_s3, int64_t out_s0, int64_t out_s1, int64_t out_s2, int64_t out_s3, double max_threshold, double any_threshold, cudaStream_t stream);

namespace {

cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

void check_cuda_float(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
}

void check_cuda_bool(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kBool, name, " must be bool");
}

// Get 4D strides as individual values (avoid passing host pointer to kernel)
void get_strides4(const torch::Tensor& t, int64_t& s0, int64_t& s1, int64_t& s2, int64_t& s3) {
    TORCH_CHECK(t.dim() == 4, "Expected 4D tensor");
    s0 = t.stride(0);
    s1 = t.stride(1);
    s2 = t.stride(2);
    s3 = t.stride(3);
}

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
    int64_t x_s0, x_s1, x_s2, x_s3;
    int64_t mask_s0, mask_s1, mask_s2, mask_s3;
    int64_t out_s0, out_s1, out_s2, out_s3;
    get_strides4(x, x_s0, x_s1, x_s2, x_s3);
    get_strides4(mask, mask_s0, mask_s1, mask_s2, mask_s3);
    get_strides4(out, out_s0, out_s1, out_s2, out_s3);
    cudaStream_t stream = getCurrentCUDAStream();
    custom_where_mask_launcher(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        out.data_ptr<float>(),
        B, C, H, W,
        x_s0, x_s1, x_s2, x_s3,
        mask_s0, mask_s1, mask_s2, mask_s3,
        out_s0, out_s1, out_s2, out_s3,
        stream
    );
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> custom_max_dim(torch::Tensor x, int64_t dim, bool keepdim) {
    check_cuda_float(x, "x");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(keepdim == true, "Only keepdim=True is supported");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    auto values = torch::empty({B, 1, H, W}, x.options());
    auto indices = torch::empty({B, 1, H, W}, x.options().dtype(torch::kInt64));
    int64_t x_s0, x_s1, x_s2, x_s3;
    int64_t val_s0, val_s1, val_s2, val_s3;
    int64_t idx_s0, idx_s1, idx_s2, idx_s3;
    get_strides4(x, x_s0, x_s1, x_s2, x_s3);
    get_strides4(values, val_s0, val_s1, val_s2, val_s3);
    get_strides4(indices, idx_s0, idx_s1, idx_s2, idx_s3);
    cudaStream_t stream = getCurrentCUDAStream();
    custom_max_dim_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        B, C, H, W,
        x_s0, x_s1, x_s2, x_s3,
        val_s0, val_s1, val_s2, val_s3,
        idx_s0, idx_s1, idx_s2, idx_s3,
        stream
    );
    return std::make_tuple(values, indices);
}

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
    int64_t x_s0, x_s1, x_s2, x_s3;
    int64_t out_s0, out_s1, out_s2, out_s3;
    get_strides4(x, x_s0, x_s1, x_s2, x_s3);
    get_strides4(out, out_s0, out_s1, out_s2, out_s3);
    cudaStream_t stream = getCurrentCUDAStream();
    custom_any_gt_launcher(
        x.data_ptr<float>(),
        out.data_ptr<bool>(),
        B, C, H, W,
        x_s0, x_s1, x_s2, x_s3,
        out_s0, out_s1, out_s2, out_s3,
        threshold,
        stream
    );
    return out;
}

torch::Tensor custom_threshold_mask(torch::Tensor x, double max_threshold, double any_threshold) {
    check_cuda_float(x, "x");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    auto out = torch::empty_like(x);
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int64_t x_s0, x_s1, x_s2, x_s3;
    int64_t out_s0, out_s1, out_s2, out_s3;
    get_strides4(x, x_s0, x_s1, x_s2, x_s3);
    get_strides4(out, out_s0, out_s1, out_s2, out_s3);
    cudaStream_t stream = getCurrentCUDAStream();
    custom_threshold_mask_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        x_s0, x_s1, x_s2, x_s3,
        out_s0, out_s1, out_s2, out_s3,
        max_threshold,
        any_threshold,
        stream
    );
    return out;
}

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
