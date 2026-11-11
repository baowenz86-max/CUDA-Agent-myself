#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_gelu_launcher(
    const float* x,
    float* y,
    long long numel,
    int B, int C, int H, int W,
    long long stride_batch,
    long long stride_channel,
    long long stride_height,
    long long stride_width,
    cudaStream_t stream);

extern "C" void custom_mean_dim1_launcher(
    const float* x,
    float* y,
    int B, int C, int H, int W,
    long long stride_batch_in, long long stride_channel_in, long long stride_height_in, long long stride_width_in,
    long long stride_batch_out, long long stride_height_out, long long stride_width_out,
    cudaStream_t stream);

static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

torch::Tensor custom_gelu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_gelu: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_gelu: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_gelu: input must be 4D [B,C,H,W]");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Ensure input is contiguous for simple stride calculation; if not, make contiguous copy.
    auto x_contig = x.contiguous();

    auto y = torch::empty_like(x_contig);

    long long numel = x_contig.numel();
    long long stride_batch = x_contig.stride(0);
    long long stride_channel = x_contig.stride(1);
    long long stride_height = x_contig.stride(2);
    long long stride_width = x_contig.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_gelu_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        numel,
        (int)B, (int)C, (int)H, (int)W,
        stride_batch, stride_channel, stride_height, stride_width,
        stream
    );
    check_cuda_error("custom_gelu kernel launch failed");

    return y;
}

torch::Tensor custom_mean_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_mean_dim1: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_mean_dim1: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_mean_dim1: input must be 4D [B,C,H,W]");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto x_contig = x.contiguous();

    // Output shape: [B, H, W]
    auto y = torch::empty({B, H, W}, x_contig.options());

    long long stride_batch_in = x_contig.stride(0);
    long long stride_channel_in = x_contig.stride(1);
    long long stride_height_in = x_contig.stride(2);
    long long stride_width_in = x_contig.stride(3);

    long long stride_batch_out = y.stride(0);
    long long stride_height_out = y.stride(1);
    long long stride_width_out = y.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_mean_dim1_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)B, (int)C, (int)H, (int)W,
        stride_batch_in, stride_channel_in, stride_height_in, stride_width_in,
        stride_batch_out, stride_height_out, stride_width_out,
        stream
    );
    check_cuda_error("custom_mean_dim1 kernel launch failed");

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_gelu", &custom_gelu, "Apply GELU activation (exact erf formula)",
          py::arg("x"));
    m.def("custom_mean_dim1", &custom_mean_dim1, "Mean over dim=1",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
