// custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// 前向声明 .cu 中的 launcher
extern "C" void custom_pad_mean_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t top, int64_t bottom, int64_t left, int64_t right,
    cudaStream_t stream);

extern "C" void custom_pad_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t top, int64_t bottom, int64_t left, int64_t right,
    cudaStream_t stream);

extern "C" void custom_mean_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream);

// 辅助函数：从 py::tuple 提取 pad 四个分量
static inline void parse_pad_tuple(const py::tuple& pad_tuple, int64_t& left, int64_t& right, int64_t& top, int64_t& bottom) {
    if (pad_tuple.size() != 4) {
        throw std::invalid_argument("pad must be a tuple of 4 ints (left, right, top, bottom)");
    }
    left = py::cast<int64_t>(pad_tuple[0]);
    right = py::cast<int64_t>(pad_tuple[1]);
    top = py::cast<int64_t>(pad_tuple[2]);
    bottom = py::cast<int64_t>(pad_tuple[3]);
}

// custom_pad_mean_forward: 执行 pad + mean
static torch::Tensor custom_pad_mean_forward(torch::Tensor x, py::tuple pad, int64_t dim) {
    // 输入检查
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    // 确保 dim 是有效维度，本任务固定为 1，但保留泛化检查
    TORCH_CHECK(dim >= 0 && dim < 4, "dim must be in [0, 3]");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t left, right, top, bottom;
    parse_pad_tuple(pad, left, right, top, bottom);

    int64_t H_out = H + top + bottom;
    int64_t W_out = W + left + right;

    // 分配输出: 形状取决于 dim 是否为 1（本任务 dim=1，输出 3D）
    torch::Tensor output;
    if (dim == 1) {
        output = torch::empty({N, H_out, W_out}, x.options());
    } else {
        // 通用情况：去掉 dim 维，其他维度保留
        std::vector<int64_t> out_shape;
        for (int64_t d = 0; d < 4; ++d) {
            if (d == dim) continue;
            if (d == 2) {
                out_shape.push_back(H_out);
            } else if (d == 3) {
                out_shape.push_back(W_out);
            } else {
                out_shape.push_back(x.size(d));
            }
        }
        output = torch::empty(out_shape, x.options());
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 对于 dim == 1，调用 fused pad+mean launcher
    if (dim == 1) {
        custom_pad_mean_launcher(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W,
            top, bottom, left, right,
            stream);
    } else {
        // 对于其他 dim（虽然当前任务不会用到），可先 pad 再 mean，但为了完整性，此处可以简化：
        // 仅支持 dim == 1 的快速路径，其他 dim 抛错或退化为两步。
        // 但根据契约仅需要 dim=1，这里给出通用实现：先 pad 后 mean
        auto padded = torch::empty({N, C, H_out, W_out}, x.options());
        custom_pad_launcher(
            x.data_ptr<float>(),
            padded.data_ptr<float>(),
            N, C, H, W,
            top, bottom, left, right,
            stream);
        // 然后调用 mean，注意 dim 可能不是 1，但 launcher 仅支持 dim=1，因此若 dim != 1 需另行处理
        // 但本任务 dim 固定为 1，所以不会进入此分支；保留基本实现
        TORCH_CHECK(dim == 1, "custom_pad_mean currently only supports dim=1");
        custom_mean_launcher(
            padded.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H_out, W_out,
            stream);
    }

    return output;
}

// custom_pad_forward: 仅 padding
static torch::Tensor custom_pad_forward(torch::Tensor x, py::tuple pad) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t left, right, top, bottom;
    parse_pad_tuple(pad, left, right, top, bottom);

    int64_t H_out = H + top + bottom;
    int64_t W_out = W + left + right;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_pad_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        top, bottom, left, right,
        stream);
    return output;
}

// custom_mean_forward: 沿 dim=1 求均值
static torch::Tensor custom_mean_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    TORCH_CHECK(dim == 1, "custom_mean currently only supports dim=1");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty({N, H, W}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_mean_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream);
    return output;
}

// 注册绑定
void register_custom_ops(py::module& m) {
    m.def("custom_pad_mean", &custom_pad_mean_forward,
          "Fused pad+mean",
          py::arg("x"), py::arg("pad"), py::arg("dim"));
    m.def("custom_pad", &custom_pad_forward,
          "Pad only",
          py::arg("x"), py::arg("pad"));
    m.def("custom_mean", &custom_mean_forward,
          "Mean over dim=1",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
