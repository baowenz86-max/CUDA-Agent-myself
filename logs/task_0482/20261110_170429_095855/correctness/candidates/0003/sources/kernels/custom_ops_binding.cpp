#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <../binding_registry.h>

// Declare launchers from custom_ops.cu
extern "C" void sum_dim1_keepdim_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_d,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_d,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream
);

extern "C" void maxpool3d_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kd, int kh, int kw,
    int sd, int sh, int sw,
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_d,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_c,
    long long out_stride_d,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream
);

torch::Tensor sum_dim1_keepdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (batch, channels, depth, height, width)");
    TORCH_CHECK(x.size(1) > 0, "Channel dimension must be positive");

    int batch = x.size(0);
    int channels = x.size(1);
    int depth = x.size(2);
    int height = x.size(3);
    int width = x.size(4);

    auto output = torch::empty({batch, 1, depth, height, width}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long in_stride_b = x.stride(0);
    long long in_stride_c = x.stride(1);
    long long in_stride_d = x.stride(2);
    long long in_stride_h = x.stride(3);
    long long in_stride_w = x.stride(4);

    long long out_stride_b = output.stride(0);
    long long out_stride_d = output.stride(2);
    long long out_stride_h = output.stride(3);
    long long out_stride_w = output.stride(4);

    sum_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, depth, height, width,
        in_stride_b, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
        out_stride_b, out_stride_d, out_stride_h, out_stride_w,
        stream
    );

    return output;
}

torch::Tensor maxpool3d(torch::Tensor x, pybind11::object kernel_size, pybind11::object stride) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (batch, channels, depth, height, width)");

    int64_t k = pybind11::cast<int64_t>(kernel_size);
    int64_t s = pybind11::cast<int64_t>(stride);
    TORCH_CHECK(k > 0, "kernel_size must be positive");
    TORCH_CHECK(s > 0, "stride must be positive");

    int batch = x.size(0);
    int channels = x.size(1);
    int in_depth = x.size(2);
    int in_height = x.size(3);
    int in_width = x.size(4);

    int kd = static_cast<int>(k);
    int kh = static_cast<int>(k);
    int kw = static_cast<int>(k);
    int sd = static_cast<int>(s);
    int sh = static_cast<int>(s);
    int sw = static_cast<int>(s);

    int out_depth = (in_depth - kd) / sd + 1;
    int out_height = (in_height - kh) / sh + 1;
    int out_width = (in_width - kw) / sw + 1;

    TORCH_CHECK(out_depth > 0 && out_height > 0 && out_width > 0,
                "Output spatial dimensions must be positive");

    auto output = torch::empty({batch, channels, out_depth, out_height, out_width}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long in_stride_b = x.stride(0);
    long long in_stride_c = x.stride(1);
    long long in_stride_d = x.stride(2);
    long long in_stride_h = x.stride(3);
    long long in_stride_w = x.stride(4);

    long long out_stride_b = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_d = output.stride(2);
    long long out_stride_h = output.stride(3);
    long long out_stride_w = output.stride(4);

    maxpool3d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kd, kh, kw,
        sd, sh, sw,
        in_stride_b, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("sum_dim1_keepdim", &sum_dim1_keepdim, "Sum over channel dimension with keepdim",
          pybind11::arg("x"));
    m.def("maxpool3d", &maxpool3d, "3D max pooling with scalar kernel and stride",
          pybind11::arg("x"), pybind11::arg("kernel_size"), pybind11::arg("stride"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
