#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// 声明 .cu 导出的 launcher
extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int padding,
    long long in_stride_n, long long in_stride_c,
    long long in_stride_h, long long in_stride_w,
    cudaStream_t stream);

extern "C" void gather_custom_forward_launcher(
    const float* input,
    const int64_t* indices,
    float* output,
    int N, int C_x, int C_out, int H, int W,
    long long in_stride_n, long long in_stride_c,
    long long in_stride_h, long long in_stride_w,
    long long idx_stride_n, long long idx_stride_c,
    long long idx_stride_h, long long idx_stride_w,
    cudaStream_t stream);

extern "C" void conv2d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int H_out, int W_out,
    cudaStream_t stream);

// 辅助函数：检查 CUDA 错误
static void checkCudaError(cudaError_t err, const char* msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// reflection_pad2d_forward 包装
static torch::Tensor reflection_pad2d_forward_cpp(torch::Tensor x, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    auto x_contig = x.contiguous();

    int H_out = H_in + 2 * (int)padding;
    int W_out = W_in + 2 * (int)padding;
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    long long in_stride_n = x_contig.stride(0);
    long long in_stride_c = x_contig.stride(1);
    long long in_stride_h = x_contig.stride(2);
    long long in_stride_w = x_contig.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        (int)padding,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        stream
    );

    checkCudaError(cudaGetLastError(), "reflection_pad2d_forward_launcher failed");
    return output;
}

// conv2d_cudnn_forward 包装（实际使用自定义 kernel 替代 cuDNN，因为环境中缺少 cuDNN 头文件）
static torch::Tensor conv2d_cudnn_forward_cpp(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(stride == 1, "only stride=1 supported");
    TORCH_CHECK(padding == 0, "only padding=0 supported");

    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    int C_out = weight.size(0);
    int K_h = weight.size(2);
    int K_w = weight.size(3);

    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    int H_out = (H_in + 2 * (int)padding - K_h) / (int)stride + 1;
    int W_out = (W_in + 2 * (int)padding - K_w) / (int)stride + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv2d_forward_launcher(
        x.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        H_out, W_out,
        stream
    );

    checkCudaError(cudaGetLastError(), "conv2d_forward_launcher failed");
    return output;
}

// gather_custom_forward 包装
static torch::Tensor gather_custom_forward_cpp(torch::Tensor x, int64_t dim, torch::Tensor indices) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(indices.dim() == 4, "indices must be 4D");
    TORCH_CHECK(dim == 1, "only dim=1 supported");

    int N = x.size(0);
    int C_x = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = indices.size(1);
    TORCH_CHECK(indices.size(0) == N, "batch size mismatch");
    TORCH_CHECK(indices.size(2) == H, "height mismatch");
    TORCH_CHECK(indices.size(3) == W, "width mismatch");

    auto x_contig = x.contiguous();
    auto indices_contig = indices.contiguous();

    auto output = torch::empty({N, C_out, H, W}, x.options());

    long long in_stride_n = x_contig.stride(0);
    long long in_stride_c = x_contig.stride(1);
    long long in_stride_h = x_contig.stride(2);
    long long in_stride_w = x_contig.stride(3);

    long long idx_stride_n = indices_contig.stride(0);
    long long idx_stride_c = indices_contig.stride(1);
    long long idx_stride_h = indices_contig.stride(2);
    long long idx_stride_w = indices_contig.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    gather_custom_forward_launcher(
        x_contig.data_ptr<float>(),
        indices_contig.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        N, C_x, C_out, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        idx_stride_n, idx_stride_c, idx_stride_h, idx_stride_w,
        stream
    );

    checkCudaError(cudaGetLastError(), "gather_custom_forward_launcher failed");
    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward_cpp,
          "reflection pad 2d forward",
          py::arg("x"), py::arg("padding"));
    m.def("conv2d_cudnn_forward", &conv2d_cudnn_forward_cpp,
          "conv2d cudnn forward (using custom kernel fallback)",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride") = 1, py::arg("padding") = 0);
    m.def("gather_custom_forward", &gather_custom_forward_cpp,
          "gather custom forward",
          py::arg("x"), py::arg("dim"), py::arg("indices"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
