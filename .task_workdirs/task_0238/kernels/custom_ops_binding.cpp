#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// External declarations from custom_ops.cu
extern "C" void clamp_log2_launcher(const float* x, float* y, int64_t numel, int64_t ndim,
                                    const int64_t* shape, const int64_t* strides, float min_val, cudaStream_t stream);
extern "C" void mse_loss_launcher(const float* x, const float* target, float* output, int64_t numel,
                                  int64_t ndim, const int64_t* x_shape, const int64_t* x_strides,
                                  const int64_t* target_shape, const int64_t* target_strides, cudaStream_t stream);
extern "C" void conv_transpose1d_launcher(const float* x, const float* weight, const float* bias,
                                          float* y, int N, int C_in, int C_out, int L_in, int L_out,
                                          int kernel_size, int stride, int padding,
                                          int64_t ndim, const int64_t* x_shape, const int64_t* x_strides,
                                          cudaStream_t stream);

// ConvTranspose1d forward (custom CUDA kernel)
static torch::Tensor conv_transpose1d_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                                              int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                                              int64_t stride, int64_t padding, int64_t output_padding) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, kernel_size)");
    TORCH_CHECK(weight.size(0) == in_channels, "weight first dim must equal in_channels");
    TORCH_CHECK(weight.size(1) == out_channels, "weight second dim must equal out_channels");
    TORCH_CHECK(weight.size(2) == kernel_size, "weight third dim must equal kernel_size");

    bool has_bias = bias.defined();
    const float* bias_ptr = nullptr;
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_channels, "bias must be 1D of size out_channels");
        bias_ptr = bias.data_ptr<float>();
    }

    int64_t N = x.size(0);
    int64_t L_in = x.size(2);
    int64_t L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "invalid output length");

    // Ensure x is contiguous (weight assumed contiguous as it's a parameter, but check)
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();

    // Allocate output tensor (contiguous)
    auto output = torch::empty({N, out_channels, L_out}, x.options());

    // Get shape/strides of x_contig (since we made it contiguous, but still pass actual dims)
    std::vector<int64_t> x_shape = {N, in_channels, L_in};
    std::vector<int64_t> x_strides = {x_contig.stride(0), x_contig.stride(1), x_contig.stride(2)};
    int64_t ndim = 3;

    // Copy shape/strides to device (for consistency with other kernels)
    int64_t* d_x_shape;
    int64_t* d_x_strides;
    cudaMalloc(&d_x_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_x_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_x_shape, x_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_strides, x_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_launcher(
        x_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        (int)N, (int)in_channels, (int)out_channels, (int)L_in, (int)L_out,
        (int)kernel_size, (int)stride, (int)padding,
        ndim, d_x_shape, d_x_strides,
        stream
    );

    cudaFree(d_x_shape);
    cudaFree(d_x_strides);

    return output;
}

// clamp_log2 forward wrapper
torch::Tensor clamp_log2_forward(torch::Tensor x, double min_val = 1e-7) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    auto output = torch::empty_like(x);

    auto sizes = x.sizes();
    auto strides = x.strides();
    int64_t ndim = x.dim();
    std::vector<int64_t> shape(sizes.begin(), sizes.end());
    std::vector<int64_t> stride_vec(strides.begin(), strides.end());

    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, stride_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clamp_log2_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        ndim,
        d_shape,
        d_strides,
        static_cast<float>(min_val),
        stream
    );

    cudaFree(d_shape);
    cudaFree(d_strides);

    return output;
}

// mse_loss forward wrapper
torch::Tensor mse_loss_forward(torch::Tensor x, torch::Tensor target) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(x.sizes() == target.sizes(), "x and target must have same shape");

    auto output = torch::empty({}, x.options()); // scalar tensor

    auto x_sizes = x.sizes();
    auto x_strides = x.strides();
    auto tgt_sizes = target.sizes();
    auto tgt_strides = target.strides();
    int64_t ndim = x.dim();
    std::vector<int64_t> x_shape(x_sizes.begin(), x_sizes.end());
    std::vector<int64_t> x_stride(x_strides.begin(), x_strides.end());
    std::vector<int64_t> tgt_shape(tgt_sizes.begin(), tgt_sizes.end());
    std::vector<int64_t> tgt_stride(tgt_strides.begin(), tgt_strides.end());

    int64_t* d_x_shape;
    int64_t* d_x_strides;
    int64_t* d_tgt_shape;
    int64_t* d_tgt_strides;
    cudaMalloc(&d_x_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_x_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_tgt_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_tgt_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_x_shape, x_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_strides, x_stride.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt_shape, tgt_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt_strides, tgt_stride.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mse_loss_launcher(
        x.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        ndim,
        d_x_shape,
        d_x_strides,
        d_tgt_shape,
        d_tgt_strides,
        stream
    );

    cudaFree(d_x_shape);
    cudaFree(d_x_strides);
    cudaFree(d_tgt_shape);
    cudaFree(d_tgt_strides);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using custom CUDA kernel",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("clamp_log2_forward", &clamp_log2_forward,
          "clamp and log2 forward",
          py::arg("x"), py::arg("min") = 1e-7);
    m.def("mse_loss_forward", &mse_loss_forward,
          "MSE loss forward",
          py::arg("x"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);