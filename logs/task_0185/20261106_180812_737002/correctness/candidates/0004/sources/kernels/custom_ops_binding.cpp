// Binding for custom mean and tile operators
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "../binding_registry.h"

// Launchers from custom_ops.cu
extern "C" void mean_dim1_keepdim_launcher(
    const float* x,
    float* out,
    int B,
    int C,
    int H,
    int W,
    const long long* x_strides,
    const long long* out_strides,
    cudaStream_t stream);

extern "C" void tile_dim1_launcher(
    const float* x,
    float* out,
    int B,
    int repeats,
    int H,
    int W,
    const long long* x_strides,
    const long long* out_strides,
    cudaStream_t stream);

// Helper to copy strides (in elements, as long long) from tensor
static void fill_strides(const torch::Tensor& t, long long* strides) {
    TORCH_CHECK(t.dim() <= 4, "Unexpected tensor rank");
    for (int i = 0; i < t.dim(); ++i) {
        strides[i] = t.stride(i);
    }
}

torch::Tensor mean_dim1_keepdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B,C,H,W)");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(C > 0, "Channel dimension must be positive");

    // Output shape [B,1,H,W] with reference strides [65536, 65536, 256, 1]
    std::vector<int64_t> out_shape = {B, 1, H, W};
    std::vector<int64_t> out_strides = {static_cast<int64_t>(H) * W, static_cast<int64_t>(H) * W, static_cast<int64_t>(W), 1};

    auto options = x.options();
    auto out_contig = torch::empty(out_shape, options);
    auto out = out_contig.as_strided(out_shape, out_strides);

    // Input strides (in elements)
    long long x_strides[4];
    fill_strides(x, x_strides);

    long long out_strides_arr[4] = {out_strides[0], out_strides[1], out_strides[2], out_strides[3]};

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mean_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        x_strides,
        out_strides_arr,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor tile_dim1(torch::Tensor x, int64_t repeats = 8) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B,1,H,W)");
    TORCH_CHECK(x.size(1) == 1, "Input dim1 must be 1 for tiling along that dimension");
    TORCH_CHECK(repeats > 0, "repeats must be positive");

    int B = x.size(0);
    int H = x.size(2);
    int W = x.size(3);

    // Output shape [B, repeats, H, W], contiguous allocation matches reference strides
    std::vector<int64_t> out_shape = {B, repeats, H, W};
    auto options = x.options();
    auto out = torch::empty(out_shape, options);

    // Input strides
    long long x_strides[4];
    fill_strides(x, x_strides);
    long long out_strides_arr[4] = {static_cast<long long>(repeats) * H * W, static_cast<long long>(H) * W, static_cast<long long>(W), 1};

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    tile_dim1_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, static_cast<int>(repeats), H, W,
        x_strides,
        out_strides_arr,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void register_custom_ops_ops(pybind11::module& m) {
    m.def("mean_dim1_keepdim", &mean_dim1_keepdim,
          "Mean over dim=1 with keepdim=True",
          py::arg("x"));
    m.def("tile_dim1", &tile_dim1,
          "Tile along dim=1 by repeats",
          py::arg("x"),
          py::arg("repeats") = 8);
}

REGISTER_BINDING(custom_ops, register_custom_ops_ops);
