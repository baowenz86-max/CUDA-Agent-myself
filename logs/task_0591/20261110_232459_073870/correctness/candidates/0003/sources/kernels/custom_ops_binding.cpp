#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void gelu_forward_launcher(const float* input,
                                     float* output,
                                     int64_t n,
                                     cudaStream_t stream);

extern "C" void bias_add_optional_launcher(const float* input,
                                          const float* bias,
                                          float* output,
                                          int64_t rows,
                                          int64_t cols,
                                          bool has_bias,
                                          cudaStream_t stream);

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gelu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

torch::Tensor linear_forward(torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");

    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    int64_t N = weight.size(0); // out_features
    int64_t K = weight.size(1); // in_features
    TORCH_CHECK(x.size(-1) == K, "x last dim must match weight in_features");

    int64_t M = 1;
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < x.dim() - 1; ++i) {
        M *= x.size(i);
        output_shape.push_back(x.size(i));
    }
    output_shape.push_back(N);

    auto output = torch::empty({M, N}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute output = x @ weight^T using cuBLAS
    cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        (int)N,
        (int)M,
        (int)K,
        &alpha,
        weight.data_ptr<float>(),
        (int)K,
        x.data_ptr<float>(),
        (int)K,
        &beta,
        output.data_ptr<float>(),
        (int)N
    );

    // Add bias if present
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.value().numel() == N, "bias size must match out_features");

        bias_add_optional_launcher(
            output.data_ptr<float>(),
            bias.value().data_ptr<float>(),
            output.data_ptr<float>(),
            M,
            N,
            true,
            stream
        );
    }

    return output.view(output_shape);
}

void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward, "GELU activation forward",
          py::arg("x"));
    m.def("linear_forward", &linear_forward, "Linear forward (cuBLAS + bias)",
          py::arg("x"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);