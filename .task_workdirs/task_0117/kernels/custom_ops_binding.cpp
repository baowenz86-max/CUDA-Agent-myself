#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void avgpool1d_forward_launcher(const float* x,
                                            float* y,
                                            int batch,
                                            int channels,
                                            int length,
                                            int pooled_length,
                                            int kernel_size,
                                            int stride,
                                            int padding,
                                            cudaStream_t stream);

extern "C" void tanh_forward_launcher(const float* x, float* y, int n, cudaStream_t stream);

extern "C" void cross_entropy_loss_forward_launcher(const float* x,
                                                    const long long* target,
                                                    float* loss,
                                                    int batch,
                                                    int classes,
                                                    cudaStream_t stream);

// Wrapper for avgpool1d_forward
torch::Tensor avgpool1d_forward(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (batch, channels, length)");

    int batch = x.size(0);
    int channels = x.size(1);
    int length = x.size(2);

    // Compute pooled length as in PyTorch AvgPool1d: floor((L + 2*padding - kernel_size) / stride) + 1
    int pooled_length = (length + 2 * static_cast<int>(padding) - static_cast<int>(kernel_size)) / static_cast<int>(stride) + 1;

    TORCH_CHECK(pooled_length > 0, "Invalid pooling parameters");

    auto y = torch::empty({batch, channels, pooled_length}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avgpool1d_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        batch,
        channels,
        length,
        pooled_length,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        stream
    );

    return y;
}

// Wrapper for tanh_forward (actually tangent, as per reference)
torch::Tensor tanh_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto y = torch::empty_like(x);
    int n = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    tanh_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n,
        stream
    );

    return y;
}

// Wrapper for cross_entropy_loss_forward
torch::Tensor cross_entropy_loss_forward(torch::Tensor x, torch::Tensor target) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "Target must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "Target must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(target.dtype() == torch::kInt64, "Target must be int64");
    TORCH_CHECK(x.dim() >= 2, "Input x must have at least 2 dimensions");
    TORCH_CHECK(target.dim() == 1, "Target must be 1D");

    int batch = x.size(0);
    int classes = x.numel() / batch;
    TORCH_CHECK(target.size(0) == batch, "Batch size mismatch between x and target");

    // Allocate scalar output on GPU
    auto loss = torch::empty({}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cross_entropy_loss_forward_launcher(
        x.data_ptr<float>(),
        reinterpret_cast<const long long*>(target.data_ptr<int64_t>()),
        loss.data_ptr<float>(),
        batch,
        classes,
        stream
    );

    return loss;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("avgpool1d_forward", &avgpool1d_forward,
          "AvgPool1d forward (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"),
          pybind11::arg("padding"));

    m.def("tanh_forward", &tanh_forward,
          "Element-wise tangent forward (CUDA)",
          pybind11::arg("x"));

    m.def("cross_entropy_loss_forward", &cross_entropy_loss_forward,
          "Cross entropy loss forward (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
