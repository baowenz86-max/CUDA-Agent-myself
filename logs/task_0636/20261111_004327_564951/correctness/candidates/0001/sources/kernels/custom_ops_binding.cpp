#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher 函数
extern "C" void maxpool1d_forward_launcher(
    const float* input, float* output,
    int N, int C, int L, int L_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream);

extern "C" void tanh_forward_launcher(
    const float* input, float* output,
    size_t total,
    cudaStream_t stream);

// MaxPool1d 前向包装
static torch::Tensor maxpool1d_forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must have dtype torch.float32");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3D (N, C, L)");

    // 确保输入是连续的，如果需要则显式拷贝；
    // 转置输入按实际 stride 处理或显式连续化，此处连续化保证 launcher 使用简单指针索引。
    torch::Tensor x_contig = x.contiguous();

    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t L = x_contig.size(2);

    // 计算输出长度，与 PyTorch MaxPool1d 公式一致
    int64_t L_out = (L + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(L_out > 0, "Invalid output length: kernel_size, stride, padding combination produces non-positive L_out");

    auto output = torch::empty({N, C, L_out}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    maxpool1d_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(L),
        static_cast<int>(L_out),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        stream);

    return output;
}

// Tanh 前向包装
static torch::Tensor tanh_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must have dtype torch.float32");

    torch::Tensor x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    size_t total = static_cast<size_t>(x_contig.numel());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    tanh_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream);

    return output;
}

// 注册函数
static void register_custom_ops(pybind11::module& m) {
    m.def("maxpool1d_forward", &maxpool1d_forward,
          "MaxPool1d forward (CUDA)",
          py::arg("x"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"));

    m.def("tanh_forward", &tanh_forward,
          "Tanh forward (CUDA)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
