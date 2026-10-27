#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "../binding_registry.h"

// Extern declarations of launchers from custom_ops.cu
extern "C" void sigmoid_forward_launcher(
    float* output,
    const float* input,
    int numel,
    cudaStream_t stream);

extern "C" void avg_pool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    cudaStream_t stream);

extern "C" void flatten_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int sN, int sC, int sH, int sW,
    cudaStream_t stream);

// Linear forward uses cuBLAS, so no launcher declaration needed.

// Helper to get cuBLAS handle for current device/stream, cached per thread
typedef struct {
    cublasHandle_t handle;
    int device;
    cudaStream_t stream;
} CublasHandleCache;

static thread_local CublasHandleCache cublas_cache = {nullptr, -1, nullptr};

cublasHandle_t get_cublas_handle() {
    int current_device;
    cudaError_t err = cudaGetDevice(&current_device);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed");
    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream().stream();

    if (cublas_cache.handle == nullptr || cublas_cache.device != current_device || cublas_cache.stream != current_stream) {
        if (cublas_cache.handle != nullptr) {
            cublasDestroy(cublas_cache.handle);
        }
        cublasStatus_t status = cublasCreate(&cublas_cache.handle);
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasCreate failed");
        cublas_cache.device = current_device;
        cublas_cache.stream = current_stream;
        status = cublasSetStream(cublas_cache.handle, current_stream);
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSetStream failed");
    }
    return cublas_cache.handle;
}

// sigmoid_forward binding
torch::Tensor sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    int numel = input_contig.numel();
    if (numel == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sigmoid_forward_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        numel,
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sigmoid_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// avg_pool2d_forward binding
torch::Tensor avg_pool2d_forward(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(kernel_size == 2 && stride == 2 && padding == 0,
                "Only kernel_size=2, stride=2, padding=0 is supported");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    auto input_contig = input.contiguous();
    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);

    int64_t outH = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t outW = (W + 2 * padding - kernel_size) / stride + 1;

    TORCH_CHECK(outH > 0 && outW > 0, "Invalid output dimensions");

    auto output = torch::empty({N, C, outH, outW}, input_contig.options());

    if (output.numel() == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    avg_pool2d_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(outH),
        static_cast<int>(outW),
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "avg_pool2d_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// flatten_forward binding
torch::Tensor flatten_forward(torch::Tensor input, int64_t start_dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(start_dim == 1, "Only start_dim=1 is supported");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    int64_t flat_size = C * H * W;
    auto output = torch::empty({N, flat_size}, input.options());

    if (output.numel() == 0) {
        return output;
    }

    auto input_contig = input.contiguous();

    int sN = static_cast<int>(C * H * W);
    int sC = static_cast<int>(H * W);
    int sH = static_cast<int>(W);
    int sW = 1;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    flatten_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        sN, sC, sH, sW,
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "flatten_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// linear_forward binding
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D [N, in_features]");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D [out_features, in_features]");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D [out_features]");
    TORCH_CHECK(weight.size(1) == input.size(1), "Input feature dimension mismatch");
    TORCH_CHECK(weight.size(0) == bias.size(0), "Weight and bias out_features mismatch");

    int64_t N = input.size(0);   // batch size
    int64_t K = input.size(1);   // in_features
    int64_t M = weight.size(0);  // out_features

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    auto output = torch::empty({N, M}, input_contig.options());

    if (output.numel() == 0) {
        return output;
    }

    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
        &alpha,
        input_contig.data_ptr<float>(), static_cast<int>(K),
        weight_contig.data_ptr<float>(), static_cast<int>(K),
        &beta,
        output.data_ptr<float>(), static_cast<int>(N)
    );
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSgemm failed");

    // Add bias: output += ones * bias^T
    auto ones = torch::ones({N}, input_contig.options());

    const float one_alpha = 1.0f;
    status = cublasSger(
        handle,
        static_cast<int>(N), static_cast<int>(M),
        &one_alpha,
        ones.data_ptr<float>(), 1,
        bias_contig.data_ptr<float>(), 1,
        output.data_ptr<float>(), static_cast<int>(N)
    );
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSger (bias) failed");

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("sigmoid_forward", &sigmoid_forward, "Sigmoid forward",
          py::arg("input"));
    m.def("avg_pool2d_forward", &avg_pool2d_forward, "Average pool 2d forward",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
    m.def("flatten_forward", &flatten_forward, "Flatten forward",
          py::arg("input"), py::arg("start_dim"));
    m.def("linear_forward", &linear_forward, "Linear forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
