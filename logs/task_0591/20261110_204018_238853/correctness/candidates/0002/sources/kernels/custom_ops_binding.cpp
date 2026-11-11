#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void gelu_forward_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream);

// GELU forward wrapper
torch::Tensor gelu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);
    int64_t numel = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    gelu_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

// Linear forward wrapper using cuBLAS
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "Bias must be float32");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias must be contiguous");
    }

    // Input shape: [batch_size, seq_len, in_features] (contiguous)
    // Weight shape: [out_features, in_features]
    int64_t batch_size = input.size(0);
    int64_t seq_len = input.size(1);
    int64_t in_features = input.size(2);
    int64_t out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features, "Weight shape mismatch");
    TORCH_CHECK(!bias.has_value() || !bias.value().defined() || bias.value().numel() == out_features, "Bias size mismatch");

    auto output = torch::empty({batch_size, seq_len, out_features}, input.options());

    int64_t m = batch_size * seq_len;  // Rows in GEMM
    int64_t n = out_features;          // Columns in GEMM
    int64_t k = in_features;           // Reduction dimension

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major. Our input is row-major [m, k], weight is row-major [n, k].
    // Compute output^T = weight^T * input^T with column-major layout.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                &alpha,
                weight.data_ptr<float>(), static_cast<int>(n),
                input.data_ptr<float>(), static_cast<int>(m),
                &beta,
                output.data_ptr<float>(), static_cast<int>(n));

    // Add bias if provided
    if (bias.has_value() && bias.value().defined()) {
        // Use cuBLAS rank-1 update: output += ones * bias^T
        // Allocate device memory for ones vector of length m
        float* ones_device;
        cudaMalloc(&ones_device, m * sizeof(float));
        // Fill ones vector on host and copy to device
        std::vector<float> ones_host(m, 1.0f);
        cudaMemcpyAsync(ones_device, ones_host.data(), m * sizeof(float), cudaMemcpyHostToDevice, stream);

        const float beta_bias = 1.0f;
        // Column-major: output is n x m, bias is n x 1, ones is 1 x m
        // C = alpha * A * B + beta * C, where A = bias (n x 1), B = ones^T (1 x m)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(n), static_cast<int>(m), 1,
                    &alpha,
                    bias.value().data_ptr<float>(), static_cast<int>(n),
                    ones_device, 1,
                    &beta_bias,
                    output.data_ptr<float>(), static_cast<int>(n));

        cudaFree(ones_device);
    }

    cublasDestroy(handle);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward, "GELU forward", py::arg("input"));
    m.def("linear_forward", &linear_forward, "Linear forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
