#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <ATen/cuda/CUDAContext.h>

// Declare launchers from custom_ops.cu
extern "C" void maxpool2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int Hout, int Wout,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    bool ceil_mode,
    long long in_stride_N, long long in_stride_C, long long in_stride_H, long long in_stride_W,
    cudaStream_t stream);

extern "C" void elu_launcher(
    float* output,
    const float* input,
    int numel,
    long long in_stride_N, long long in_stride_C, long long in_stride_H, long long in_stride_W,
    float alpha,
    cudaStream_t stream);

static std::pair<int, int> get_maxpool_output_size(
    int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    bool ceil_mode) {
    int Hout, Wout;
    if (ceil_mode) {
        Hout = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1 + stride_h - 1) / stride_h + 1;
        Wout = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1 + stride_w - 1) / stride_w + 1;
    } else {
        Hout = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        Wout = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    }
    return {Hout, Wout};
}

torch::Tensor maxpool2d_forward_impl(
    torch::Tensor x,
    int kernel_size_h, int kernel_size_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    bool ceil_mode) {

    TORCH_CHECK(x.is_cuda(), "maxpool2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "maxpool2d_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "maxpool2d_forward: input must be 4D (N,C,H,W)");
    TORCH_CHECK(kernel_size_h > 0 && kernel_size_w > 0, "maxpool2d_forward: kernel size must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "maxpool2d_forward: stride must be positive");
    TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "maxpool2d_forward: padding must be non-negative");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "maxpool2d_forward: dilation must be positive");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto [Hout, Wout] = get_maxpool_output_size(
        H, W, kernel_size_h, kernel_size_w, stride_h, stride_w,
        padding_h, padding_w, dilation_h, dilation_w, ceil_mode);

    TORCH_CHECK(Hout > 0 && Wout > 0, "maxpool2d_forward: computed output dimensions must be positive");

    auto output = torch::empty({N, C, Hout, Wout}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long sN = static_cast<long long>(x.stride(0));
    long long sC = static_cast<long long>(x.stride(1));
    long long sH = static_cast<long long>(x.stride(2));
    long long sW = static_cast<long long>(x.stride(3));

    maxpool2d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        Hout, Wout,
        kernel_size_h, kernel_size_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        ceil_mode,
        sN, sC, sH, sW,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("maxpool2d_forward: CUDA error: ") + cudaGetErrorString(err));
    }

    return output;
}

torch::Tensor elu_forward_impl(torch::Tensor x, float alpha) {
    TORCH_CHECK(x.is_cuda(), "elu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "elu_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "elu_forward: input must be 4D (N,C,H,W)");

    auto output = torch::empty_like(x, x.options().memory_format(torch::MemoryFormat::Contiguous));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long sN = static_cast<long long>(x.stride(0));
    long long sC = static_cast<long long>(x.stride(1));
    long long sH = static_cast<long long>(x.stride(2));
    long long sW = static_cast<long long>(x.stride(3));

    int numel = static_cast<int>(x.numel());

    elu_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        sN, sC, sH, sW,
        alpha,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("elu_forward: CUDA error: ") + cudaGetErrorString(err));
    }

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("maxpool2d_forward", [](py::args args, py::kwargs kwargs) -> torch::Tensor {
        if (args.size() != 6) {
            throw std::runtime_error("maxpool2d_forward: expected 6 positional arguments");
        }
        torch::Tensor x = args[0].cast<torch::Tensor>();
        py::object kernel_obj = args[1];
        py::object stride_obj = args[2];
        py::object padding_obj = args[3];
        py::object dilation_obj = args[4];
        bool ceil_mode = args[5].cast<bool>();

        int kh, kw;
        if (py::isinstance<py::tuple>(kernel_obj) || py::isinstance<py::list>(kernel_obj)) {
            auto seq = kernel_obj.cast<py::sequence>();
            if (seq.size() != 2) throw std::runtime_error("kernel_size must have 2 elements");
            kh = seq[0].cast<int>();
            kw = seq[1].cast<int>();
        } else {
            int k = kernel_obj.cast<int>();
            kh = kw = k;
        }

        int sh, sw;
        if (py::isinstance<py::tuple>(stride_obj) || py::isinstance<py::list>(stride_obj)) {
            auto seq = stride_obj.cast<py::sequence>();
            if (seq.size() != 2) throw std::runtime_error("stride must have 2 elements");
            sh = seq[0].cast<int>();
            sw = seq[1].cast<int>();
        } else {
            int s = stride_obj.cast<int>();
            sh = sw = s;
        }

        int ph, pw;
        if (py::isinstance<py::tuple>(padding_obj) || py::isinstance<py::list>(padding_obj)) {
            auto seq = padding_obj.cast<py::sequence>();
            if (seq.size() != 2) throw std::runtime_error("padding must have 2 elements");
            ph = seq[0].cast<int>();
            pw = seq[1].cast<int>();
        } else {
            int p = padding_obj.cast<int>();
            ph = pw = p;
        }

        int dh, dw;
        if (py::isinstance<py::tuple>(dilation_obj) || py::isinstance<py::list>(dilation_obj)) {
            auto seq = dilation_obj.cast<py::sequence>();
            if (seq.size() != 2) throw std::runtime_error("dilation must have 2 elements");
            dh = seq[0].cast<int>();
            dw = seq[1].cast<int>();
        } else {
            int d = dilation_obj.cast<int>();
            dh = dw = d;
        }

        return maxpool2d_forward_impl(x, kh, kw, sh, sw, ph, pw, dh, dw, ceil_mode);
    });

    m.def("elu_forward", [](py::args args, py::kwargs kwargs) -> torch::Tensor {
        if (args.size() != 2) {
            throw std::runtime_error("elu_forward: expected 2 positional arguments");
        }
        torch::Tensor x = args[0].cast<torch::Tensor>();
        float alpha = args[1].cast<float>();
        return elu_forward_impl(x, alpha);
    });
}

REGISTER_BINDING(custom_ops, register_custom_ops);