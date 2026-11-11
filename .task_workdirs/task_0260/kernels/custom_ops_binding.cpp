#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void avg_pool2d_launcher(const float* input,
                                    float* output,
                                    int N, int C, int H, int W,
                                    int kernel_size, int stride, int padding,
                                    cudaStream_t stream);

extern "C" void log_softmax_dim1_launcher(const float* input,
                                           float* output,
                                           int N, int C, int H, int W,
                                           cudaStream_t stream);

// avg_pool2d 绑定函数
torch::Tensor avg_pool2d(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    // 转换为 contiguous（如果不是）
    auto x_contig = x.contiguous();

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avg_pool2d_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        stream
    );

    return output;
}

// log_softmax_dim1 绑定函数
torch::Tensor log_softmax_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    auto x_contig = x.contiguous();

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_dim1_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("avg_pool2d", &avg_pool2d,
          "Average pooling 2D",
          py::arg("x"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"));

    m.def("log_softmax_dim1", &log_softmax_dim1,
          "Log softmax along dimension 1",
          py::arg("x"));
}

// 自动注册
REGISTER_BINDING(custom_ops, register_custom_ops);
