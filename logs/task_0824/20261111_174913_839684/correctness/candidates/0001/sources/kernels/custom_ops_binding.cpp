#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void batch_norm2d_inference_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream);

extern "C" void permute_contiguous_launcher(
    float* output,
    const float* input,
    const int64_t* in_shape,
    const int64_t* in_stride,
    const int64_t* out_shape,
    const int64_t* dims,
    int64_t total_elements,
    cudaStream_t stream);

extern "C" void max_pool1d_launcher(
    const float* input,
    float* output,
    int64_t B,
    int64_t L,
    int64_t C,
    int64_t L_out,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// batch_norm2d_inference implementation
torch::Tensor batch_norm2d_inference(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (B, C, H, W)");

    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kFloat32, "weight must be a CUDA float32 tensor");
    TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32, "bias must be a CUDA float32 tensor");
    TORCH_CHECK(running_mean.is_cuda() && running_mean.dtype() == torch::kFloat32, "running_mean must be a CUDA float32 tensor");
    TORCH_CHECK(running_var.is_cuda() && running_var.dtype() == torch::kFloat32, "running_var must be a CUDA float32 tensor");
    TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous() && running_mean.is_contiguous() && running_var.is_contiguous(),
                "weight, bias, running_mean, running_var must be contiguous");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(weight.numel() == C && bias.numel() == C && running_mean.numel() == C && running_var.numel() == C,
                "parameter sizes must match C");

    auto output = torch::empty_like(x);

    cudaStream_t stream = getCurrentStream();
    batch_norm2d_inference_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        B, C, H, W,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batch_norm2d_inference kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// permute_contiguous implementation
torch::Tensor permute_contiguous(torch::Tensor x, torch::Tensor dims) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(dims.is_cpu(), "dims must be a CPU tensor");
    TORCH_CHECK(dims.dtype() == torch::kInt64, "dims must be int64");
    TORCH_CHECK(dims.is_contiguous(), "dims must be contiguous");
    TORCH_CHECK(dims.numel() == 4, "dims must have 4 elements");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");

    const int64_t* dims_ptr = dims.data_ptr<int64_t>();
    // Validate dims is a permutation of 0..3
    bool seen[4] = {false};
    for (int i = 0; i < 4; ++i) {
        int64_t d = dims_ptr[i];
        TORCH_CHECK(d >= 0 && d < 4 && !seen[d], "dims must be a permutation of 0..3");
        seen[d] = true;
    }

    // Get input shape and stride
    std::vector<int64_t> in_shape(4);
    std::vector<int64_t> in_stride(4);
    for (int i = 0; i < 4; ++i) {
        in_shape[i] = x.size(i);
        in_stride[i] = x.stride(i);
    }

    // Compute output shape (permuted)
    std::vector<int64_t> out_shape(4);
    for (int i = 0; i < 4; ++i) {
        out_shape[i] = in_shape[dims_ptr[i]];
    }

    auto output = torch::empty(out_shape, x.options());

    // Copy in_shape, in_stride, out_shape, dims to device for kernel
    int64_t* d_in_shape;
    int64_t* d_in_stride;
    int64_t* d_out_shape;
    int64_t* d_dims;
    cudaError_t err;

    err = cudaMalloc(&d_in_shape, 4 * sizeof(int64_t));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc d_in_shape failed");
    err = cudaMalloc(&d_in_stride, 4 * sizeof(int64_t));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc d_in_stride failed");
    err = cudaMalloc(&d_out_shape, 4 * sizeof(int64_t));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc d_out_shape failed");
    err = cudaMalloc(&d_dims, 4 * sizeof(int64_t));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc d_dims failed");

    cudaStream_t stream = getCurrentStream();

    // Copy arrays to device asynchronously
    err = cudaMemcpyAsync(d_in_shape, in_shape.data(), 4 * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync d_in_shape failed");
    err = cudaMemcpyAsync(d_in_stride, in_stride.data(), 4 * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync d_in_stride failed");
    err = cudaMemcpyAsync(d_out_shape, out_shape.data(), 4 * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync d_out_shape failed");
    err = cudaMemcpyAsync(d_dims, dims_ptr, 4 * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync d_dims failed");

    int64_t total_elements = x.numel();
    permute_contiguous_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        d_in_shape,
        d_in_stride,
        d_out_shape,
        d_dims,
        total_elements,
        stream);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "permute_contiguous kernel launch failed: ", cudaGetErrorString(err));

    // Free device memory after kernel launch (asynchronously? To be safe, free after stream sync?)
    // Since kernel uses these pointers, we need to ensure kernel completes before freeing.
    // We can use cudaFree after cudaStreamSynchronize, but that blocks CPU. Alternative: use a callback or event.
    // For simplicity, use cudaFree which will be scheduled after kernel due to stream ordering if using cudaFreeAsync?
    // cudaFree is synchronous, causing host wait, but acceptable for correctness.
    cudaFree(d_in_shape);
    cudaFree(d_in_stride);
    cudaFree(d_out_shape);
    cudaFree(d_dims);

    return output;
}

// view implementation
torch::Tensor view(torch::Tensor x, torch::Tensor shape) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous for view");
    TORCH_CHECK(shape.is_cpu(), "shape must be a CPU tensor");
    TORCH_CHECK(shape.dtype() == torch::kInt64, "shape must be int64");
    TORCH_CHECK(shape.is_contiguous(), "shape must be contiguous");

    // Convert shape tensor to IntArrayRef
    std::vector<int64_t> shape_vec(shape.data_ptr<int64_t>(), shape.data_ptr<int64_t>() + shape.numel());
    return x.view(shape_vec);
}

// max_pool1d implementation
torch::Tensor max_pool1d(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (B, L, C)");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0 && dilation > 0, "invalid pooling parameters");

    int64_t B = x.size(0);
    int64_t L = x.size(1);
    int64_t C = x.size(2);

    int64_t L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "output length must be positive");

    auto output = torch::empty({B, L_out, C}, x.options());

    cudaStream_t stream = getCurrentStream();
    max_pool1d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, L, C, L_out,
        kernel_size, stride, padding, dilation,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "max_pool1d kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm2d_inference", &batch_norm2d_inference,
          "BatchNorm2d inference",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps") = 1e-05);
    m.def("permute_contiguous", &permute_contiguous,
          "Permute and contiguous",
          py::arg("x"), py::arg("dims"));
    m.def("view", &view,
          "View",
          py::arg("x"), py::arg("shape"));
    m.def("max_pool1d", &max_pool1d,
          "Max pool 1d",
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
