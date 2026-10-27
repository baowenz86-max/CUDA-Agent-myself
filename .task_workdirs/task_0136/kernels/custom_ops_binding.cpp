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

// For linear, we use cuBLAS, so no launcher from .cu is needed.
// We will implement a helper to call cublasSgemm.

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
    // Input can be non-contiguous, but our kernel assumes contiguous. We can make contiguous if needed.
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

    // Preserve original shape and strides if input was non-contiguous? PyTorch operations return contiguous by default.
    // The reference output is contiguous, so returning a contiguous tensor is fine.
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

    // Flatten from dim 1 to end, always produce contiguous output
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    int64_t flat_size = C * H * W;
    auto output = torch::empty({N, flat_size}, input.options());

    if (output.numel() == 0) {
        return output;
    }

    // Ensure input is contiguous for our flatten kernel, which assumes row-major contiguous indexing.
    auto input_contig = input.contiguous();

    // Get strides (contiguous strides: C*H*W, H*W, W, 1)
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

    // Use cuBLAS: C = alpha * A * B + beta * C
    // We need: output = input @ weight.T + bias
    // In cuBLAS column-major, we can treat row-major A (N x K) as column-major A^T (K x N).
    // And row-major B (M x K) as column-major B^T (K x M).
    // Then C^T (M x N) = B^T (K x M) * A^T (K x N), so C = A * B^T, which is what we want.
    // We set: cublas operation on A (input) as CUBLAS_OP_T, on B (weight) as CUBLAS_OP_N.
    // Wait, let's be careful.
    // Standard cuBLAS: C = op(A) * op(B), with A m x k, B k x n, C m x n.
    // If we treat input as row-major N x K, we can view it as column-major transposed: A_col = input^T, which is K x N.
    // Similarly, weight^T row-major M x K -> column-major transposed: B_col = (weight^T)^T = weight, which is K x M.
    // Then C_col = A_col * B_col, which is (K x N) * (K x M) -> not valid.
    // Alternative: We want C = input (N x K) * weight^T (K x M).
    // Use cublasSgemm with column-major storage: we can pass transa = CUBLAS_OP_T, transb = CUBLAS_OP_N,
    // and set A = input (row-major NxK) but treated as column-major; B = weight (row-major MxK) treated as column-major.
    // Let's derive: In column-major, A (m x k) with op(A) = A^T means we have A^T (k x m).
    // We want m = N, k = K, n = M.
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, input, K, weight, K, &beta, output, N)
    // Here: lda = K (leading dimension of input as column-major, but input is row-major with leading dimension K).
    // ldb = K (leading dimension of weight as column-major, weight is row-major with leading dimension K).
    // ldc = N (leading dimension of output, output row-major leading dimension M, but column-major leading dim = N).
    // Wait, output is row-major N x M, leading dimension = M (number of columns). But cublas expects column-major, so output_col would be M x N? Actually cublas assumes column-major, so if we pass row-major output as column-major, its leading dimension should be number of rows in column-major interpretation, which is N (number of rows of original array). So ldc = N.
    // Let's set: alpha=1, beta=0 first, compute matmul, then add bias.
    // Or use beta=1 and prefill output with bias? No, we'll do separate.

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

    // Add bias: output += bias broadcast across rows
    // We need a kernel for this, or we can use cublasSgemm with beta=1 and pre-filled output? But output is uninitialized.
    // Alternative: Use cublasSgemm with beta=1 and set output to bias repeated? That requires filling.
    // Simpler: Use a custom kernel for bias add, or use cudaMemcpy2D? We can incorporate bias in the .cu linear launcher instead.
    // But we already have a linear_kernel_v2 in .cu that includes bias, though it's not used if we use cuBLAS.
    // To keep things simple and avoid extra kernel, we can call cublasSgemm with beta=1 and set output to bias using a simple CUDA kernel.
    // However, better to implement bias addition using a custom kernel or use cublasSgemm with beta=1 after filling output with bias.
    // We'll implement a small bias addition using a CUDA kernel? But we cannot modify .cu now? The task says only .cpp, but we can add a new kernel to .cu? The task says "只允许修改 model_new.py、kernels/custom_ops.cu，或 kernels/ 下的 .cpp 文件" so we can modify .cu, but the current phase says only generate or modify kernels/ 下的绑定 .cpp 文件. So we cannot modify .cu in this phase. Therefore, we must rely on existing .cu launchers or pure C++ (cuBLAS/cuDNN) in .cpp. Since we cannot add new CUDA kernels in .cpp, we must use cuBLAS for linear and handle bias separately without a kernel.
    // We can use cublasSgemm with beta=0 to get the matmul result, then use cudaMemcpy? Not for bias.
    // Alternatively, we can use cublasSgemm with beta=1 and ensure output is pre-filled with bias by using cudaMemset? No, bias varies per column.
    // Option: Use a second cublas call to add bias? cuBLAS has no direct bias add, but we can do a rank-1 update? The bias is constant across rows, so we can use cublasSger (rank-1 update) to add bias: output = output + ones * bias^T, where ones is a vector of 1s length N.
    // That would require an additional matrix-vector operation, but cuBLAS can do it efficiently. We can first compute matmul, then add bias using cublasSger with alpha=1, x = ones (length N), y = bias (length M), resulting in output += ones * bias^T. Need to allocate a ones vector, but that's a one-time cost? We can allocate a small vector of 1.0f each call or cache it.
    // Or simpler: use cublasSgemm with beta=1 and pre-set output with bias by using a custom approach? There's cublasSetMatrix with custom? Not feasible without kernel.
    // Another option: Use cublasSgemm to compute input @ weight^T, then use cublasSaxpy to add bias to each row? cublasSaxpy operates on vectors, but we need to add bias vector to each row of matrix, which is a rank-1 update as described.
    // Let's implement bias addition via cublasSger (rank-1 update). We need a ones vector of length N (number of rows) and bias vector of length M (columns). Then C = alpha*x*y^T + C. This adds bias to each row.
    // Allocate ones tensor (N) as float, fill with 1.0f. Could cache it or create each call. N is known and small (512), so overhead minimal.
    // We'll create a ones tensor and fill with 1.0f using torch::ones or by memset? torch::ones is allowed? It's a tensor creation, not computation. We can use torch::ones({N}, input.options()) which is acceptable as tensor creation. But it creates a new tensor with ones, which is fine.
    // Then call cublasSger(handle, N, M, &alpha, ones_ptr, 1, bias_ptr, 1, output_ptr, ldc).
    // Note: cublasSger computes A = alpha*x*y^T + A, where A is m x n. x length m, y length n. We want output += bias for each row: output[i][j] += bias[j]. So x = ones (m=N), y = bias (n=M). Good.

    // Create ones tensor
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
