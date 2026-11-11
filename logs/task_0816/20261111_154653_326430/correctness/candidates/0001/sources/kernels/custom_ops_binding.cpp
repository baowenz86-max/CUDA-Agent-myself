#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void upsample_bilinear2d_launcher(
    const float* input, float* output,
    int N, int C, int in_h, int in_w, int out_h, int out_w,
    float scale_factor,
    const int64_t* in_strides, const int64_t* out_strides,
    cudaStream_t stream);

extern "C" void mean_dim1_keepdim_launcher(
    const float* input, float* output,
    int N, int C, int H, int W,
    const int64_t* in_strides, const int64_t* out_strides,
    cudaStream_t stream);

namespace {

// 检查 CUDA 错误，失败时抛出异常
void check_cuda_error(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

// 确保输入 tensor 满足 requirements，必要时连续化
// 返回是否已连续化（用于释放临时 tensor）
bool ensure_cuda_float_4d(const torch::Tensor& input, const char* op_name) {
    TORCH_CHECK(input.is_cuda(), op_name, ": input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, op_name, ": input must be float32");
    TORCH_CHECK(input.dim() == 4, op_name, ": input must be 4D (N, C, H, W)");
    return input.is_contiguous();
}

} // anonymous namespace

// Upsample bilinear wrapper
torch::Tensor upsample_bilinear2d(torch::Tensor input, double scale_factor = 2.0) {
    // 检查输入
    bool contiguous = ensure_cuda_float_4d(input, "upsample_bilinear2d");
    TORCH_CHECK(scale_factor > 0.0, "upsample_bilinear2d: scale_factor must be positive");

    // 获取原始尺寸（连续化后尺寸不变）
    auto input_contig = contiguous ? input : input.contiguous();
    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int in_h = input_contig.size(2);
    int in_w = input_contig.size(3);

    // 计算输出尺寸
    int out_h = static_cast<int>(std::floor(in_h * scale_factor));
    int out_w = static_cast<int>(std::floor(in_w * scale_factor));

    // 分配输出 tensor，连续布局
    auto output = torch::empty({N, C, out_h, out_w}, input_contig.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 准备 strides（对于 contiguous tensor，可以使用默认 strides，但 launcher 期望数组）
    // 我们直接使用 input_contig 和 output 的 strides（它们都是 contiguous，因此可直接使用）
    int64_t in_strides[4] = {input_contig.stride(0), input_contig.stride(1), input_contig.stride(2), input_contig.stride(3)};
    int64_t out_strides[4] = {output.stride(0), output.stride(1), output.stride(2), output.stride(3)};

    // 调用 launcher
    upsample_bilinear2d_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, in_h, in_w, out_h, out_w,
        static_cast<float>(scale_factor),
        in_strides,
        out_strides,
        stream
    );

    // 检查 CUDA 错误（同步点仅用于捕获错误，性能影响可接受）
    check_cuda_error(cudaGetLastError(), "upsample_bilinear2d_launcher");

    return output;
}

// Mean over channel dim (keepdim=True) wrapper
torch::Tensor mean_dim1_keepdim(torch::Tensor input) {
    // 检查输入
    bool contiguous = ensure_cuda_float_4d(input, "mean_dim1_keepdim");

    // 连续化
    auto input_contig = contiguous ? input : input.contiguous();
    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);

    // 输出形状 [N, 1, H, W]
    auto output = torch::empty({N, 1, H, W}, input_contig.options());

    // 获取 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // strides（contiguous）
    int64_t in_strides[4] = {input_contig.stride(0), input_contig.stride(1), input_contig.stride(2), input_contig.stride(3)};
    int64_t out_strides[4] = {output.stride(0), output.stride(1), output.stride(2), output.stride(3)};

    // 调用 launcher
    mean_dim1_keepdim_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        in_strides,
        out_strides,
        stream
    );

    check_cuda_error(cudaGetLastError(), "mean_dim1_keepdim_launcher");

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("upsample_bilinear2d", &upsample_bilinear2d,
          "Bilinear upsampling (align_corners=False)",
          py::arg("input"), py::arg("scale_factor") = 2.0);
    m.def("mean_dim1_keepdim", &mean_dim1_keepdim,
          "Mean over channel dim with keepdim=True",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
