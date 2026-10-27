#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void adaptive_avg_pool2d_squeeze_launcher(
    const float* input_gpu,
    float* output_gpu,
    int B, int C, int H, int W,
    cudaStream_t stream);

extern "C" void triplet_margin_loss_launcher(
    const float* anchor,
    const float* positive,
    const float* negative,
    float* output,
    int B, int C,
    cudaStream_t stream);

// 包装: adaptive_avg_pool2d_squeeze
// 输入: tensor [B, C, H, W] float32，可以是 CPU 或 CUDA
// 输出: CUDA tensor [B, C] float32
torch::Tensor adaptive_avg_pool2d_squeeze(torch::Tensor input) {
    TORCH_CHECK(input.dtype() == torch::kFloat32, "adaptive_avg_pool2d_squeeze: input must be float32");
    TORCH_CHECK(input.dim() == 4, "adaptive_avg_pool2d_squeeze: input must be 4D (B,C,H,W)");

    // 连续化，避免处理 strides
    auto input_contig = input.contiguous();
    int B = input_contig.size(0);
    int C = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);

    // 确保输入在 CUDA 上
    auto input_cuda = input_contig.is_cuda() ? input_contig : input_contig.to(torch::kCUDA);

    // 分配输出 [B, C] CUDA float32
    auto output = torch::empty({B, C}, input_cuda.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    adaptive_avg_pool2d_squeeze_launcher(
        input_cuda.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        stream
    );

    return output;
}

// 包装: triplet_margin_loss
// 输入: 三个 CUDA tensor [B, C] float32
// 输出: CUDA scalar tensor [] float32
torch::Tensor triplet_margin_loss(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative) {
    TORCH_CHECK(anchor.is_cuda(), "triplet_margin_loss: anchor must be a CUDA tensor");
    TORCH_CHECK(positive.is_cuda(), "triplet_margin_loss: positive must be a CUDA tensor");
    TORCH_CHECK(negative.is_cuda(), "triplet_margin_loss: negative must be a CUDA tensor");
    TORCH_CHECK(anchor.dtype() == torch::kFloat32, "triplet_margin_loss: anchor must be float32");
    TORCH_CHECK(positive.dtype() == torch::kFloat32, "triplet_margin_loss: positive must be float32");
    TORCH_CHECK(negative.dtype() == torch::kFloat32, "triplet_margin_loss: negative must be float32");
    TORCH_CHECK(anchor.dim() == 2, "triplet_margin_loss: anchor must be 2D (B,C)");
    TORCH_CHECK(positive.dim() == 2, "triplet_margin_loss: positive must be 2D (B,C)");
    TORCH_CHECK(negative.dim() == 2, "triplet_margin_loss: negative must be 2D (B,C)");
    TORCH_CHECK(anchor.size(0) == positive.size(0) && anchor.size(0) == negative.size(0),
                "triplet_margin_loss: batch dimensions must match");
    TORCH_CHECK(anchor.size(1) == positive.size(1) && anchor.size(1) == negative.size(1),
                "triplet_margin_loss: channel dimensions must match");

    // 连续化以确保 data_ptr 适用于 launcher
    auto anchor_contig = anchor.contiguous();
    auto positive_contig = positive.contiguous();
    auto negative_contig = negative.contiguous();

    int B = anchor_contig.size(0);
    int C = anchor_contig.size(1);

    // 分配标量输出
    auto output = torch::empty({}, anchor_contig.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    triplet_margin_loss_launcher(
        anchor_contig.data_ptr<float>(),
        positive_contig.data_ptr<float>(),
        negative_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C,
        stream
    );

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("adaptive_avg_pool2d_squeeze", &adaptive_avg_pool2d_squeeze,
          "adaptive_avg_pool2d_squeeze(Tensor input) -> Tensor",
          py::arg("input"));
    m.def("triplet_margin_loss", &triplet_margin_loss,
          "triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative) -> Tensor",
          py::arg("anchor"), py::arg("positive"), py::arg("negative"));
}

// 自动注册
REGISTER_BINDING(custom_ops, register_custom_ops);
