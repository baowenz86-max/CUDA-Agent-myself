#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "../binding_registry.h"

#define CUBLAS_CHECK(condition) \
    do { \
        cublasStatus_t status = (condition); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuBLAS error: ") + cublasGetStatusString(status)); \
        } \
    } while(0)

cublasHandle_t get_cublas_handle() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    return handle;
}

void destroy_cublas_handle(cublasHandle_t handle) {
    cublasDestroy(handle);
}

// Linear layer: y = x @ W^T + b
// Input x: [batch, in_features]
// Weight: [out_features, in_features]
// Bias: [out_features] (may be None)
// Output: [batch, out_features]
torch::Tensor linear_cublas(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    }
    TORCH_CHECK(input.dim() == 2, "Input must be 2D [batch, in_features]");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D [out_features, in_features]");
    TORCH_CHECK(input.size(1) == weight.size(1), "Input feature dimension must match weight in_features");

    int64_t batch = input.size(0);
    int64_t in_features = input.size(1);
    int64_t out_features = weight.size(0);

    TORCH_CHECK(batch > 0 && in_features > 0 && out_features > 0, "Dimensions must be positive");

    // Allocate output tensor (contiguous)
    auto output = torch::empty({batch, out_features}, input.options());

    // cuBLAS handle
    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Perform GEMM: output = input * weight^T + bias
    // cuBLAS uses column-major, so we need to interpret row-major as transposed.
    // We want C (batch x out_features) = A (batch x in_features) * B^T (in_features x out_features)
    // In column-major, C^T = B (out_features x in_features) * A^T (in_features x batch)
    // So we compute C^T = weight * input^T, then transpose result.
    // Alternatively use cublasSgemm with proper transposes: C = alpha * op(A) * op(B) + beta * C
    // If we set op(A) = input^T (in_features x batch), op(B) = weight^T (out_features x in_features)??
    // Let's use the common approach: cuBLAS column-major, but we treat our row-major matrices as transposed.
    // So we call cublasSgemm with transa=CUBLAS_OP_N, transb=CUBLAS_OP_T, m=out_features, n=batch, k=in_features,
    // A = weight (out_features x in_features) leading dimension out_features, B = input (batch x in_features) leading dimension batch,
    // C = output stored column-major meaning we write output as [batch, out_features] but with leading dimension batch.
    // But our output is row-major [batch, out_features] with ld = out_features. We can instead compute output^T (out_features x batch)
    // and then transpose copy, or use cublasSgemmStridedBatched? Simplest: allocate temp of shape [out_features, batch] and then transpose.
    // But we can avoid copy by using beta=0 and computing C = weight * input^T directly into output^T, then use torch::transpose to view.
    // Actually, muze a temporary tensor and then use a custom transpose kernel? Since we cannot add kernels to .cpp, we can use cublasSgeam for transpose or just output.transpose(0,1) after filling correctly.
    // Better: allocate output as [out_features, batch] with column-major interpretation, then return transposed view? But contract says output shape [batch, out_features].
    // Solution: Use cublasSgemm with transa=CUBLAS_OP_N, transb=CUBLAS_OP_T, but set m=out_features, n=batch, k=in_features,
    // A = weight (lda=out_features), B = input (ldb=batch), C = output_temp (ldc=out_features) where output_temp has shape [out_features, batch].
    // Then we transpose output_temp to [batch, out_features] using cublasSgeam (which can transpose).
    // But cublasSgeam is a BLAS-3 transpose operation, but simpler: use cublasSgeam with C = 1*A^T + 0*B.
    // To avoid extra kernel, we can allocate output as [out_features, batch] and return output.transpose(0,1) but that returns a view not a new tensor; but the contract says output contiguous [batch, out_features] with stride [out_features,1], so we need to make a copy.
    // We can use cudaMemcpy2D or a simple elementwise copy in a separate kernel, but .cpp cannot launch kernels. So we use cublasSgeam.
    // cublasSgeam can compute C = alpha * op(A) + beta * op(B). We can set A = output_temp, op(A)=transpose, B = dummy, beta=0, C = output (batch x out_features).
    // Or we can compute directly into output with a trick: treat output row-major as column-major with transposed dimensions: output (batch x out_features) in row-major is equivalent to output^T (out_features x batch) in column-major.
    // So if we set C = output (with shape batch x out_features) in column-major, then its leading dimension is batch, and its actual physical storage is row-major with leading dimension out_features. That mismatch causes wrong strides.
    // Let's keep it simple: allocate a temporary tensor of shape [out_features, batch], compute the GEMM into it, then use cublasSgeam to transpose into the final output.
    auto temp = torch::empty({out_features, batch}, input.options());
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N,       // transa: no transpose
                             CUBLAS_OP_T,       // transb: transpose input
                             out_features,      // m: rows of A and C
                             batch,             // n: columns of B and C
                             in_features,       // k: common dimension
                             &alpha,
                             weight.data_ptr<float>(), out_features,  // A = weight, lda = out_features (row-major)
                             input.data_ptr<float>(), batch,          // B = input, ldb = batch (row-major)
                             &beta,
                             temp.data_ptr<float>(), out_features)); // C = temp, ldc = out_features

    // Now temp is [out_features, batch] but physically stored as column-major? Actually cuBLAS treats matrices as column-major, but we passed row-major pointers with appropriate leading dimensions.
    // The result temp is a column-major matrix of dimension m x n = out_features x batch, i.e., temp is stored in column-major with leading dim out_features.
    // Our tensor temp is allocated row-major with shape [out_features, batch] and strides [batch, 1]. Since we treat it as column-major, the actual memory layout corresponds to a transposed matrix.
    // To get the final output [batch, out_features] row-major, we can simply view temp as [batch, out_features] after transpose? But we need actual data rearranged.
    // Observation: column-major storage of m x n matrix with ld=m is identical to row-major storage of n x m matrix with ld=m. So temp's memory is exactly a row-major matrix of shape [batch, out_features] with stride [out_features, 1].
    // Therefore, we can use temp.view({batch, out_features}) as the output without any copy! But temp's shape is [out_features, batch], so view would be [out_features, batch] not [batch, out_features]. Need to transpose.
    // We can return temp.transpose(0,1).contiguous()? That would copy. But we can instead allocate output as empty [batch, out_features] and directly use it as the C matrix by treating it as column-major.
    // Set m = batch, n = out_features, k = in_features, A = input, B = weight with transposes, then C = output with ldc = batch (since column-major). But output row-major has leading dim out_features, so not compatible.
    // The easiest correct method: compute into temp, then use cublasSgeam to transpose into output.
    // cublasSgeam can compute C = alpha * op(A) + beta * op(B). Set A=temp (out_features x batch), op(A)=transpose, B=unused, beta=0, C=output (batch x out_features).
    // Allocate output as required.
    // We already have output allocated. Use cublasSgeam.
    float alpha_geam = 1.0f, beta_geam = 0.0f;
    CUBLAS_CHECK(cublasSgeam(handle,
                             CUBLAS_OP_T,       // op(A): transpose temp -> batch x out_features
                             CUBLAS_OP_N,       // op(B): not used
                             batch,             // m = rows of A^T and C
                             out_features,      // n = cols of A^T and C
                             &alpha_geam,
                             temp.data_ptr<float>(), out_features,  // A = temp, lda = out_features (since temp is column-major m=out_features, n=batch, ld=out_features)
                             &beta_geam,
                             nullptr, batch,    // B = nullptr, ldb ignored
                             output.data_ptr<float>(), out_features)); // C = output, ldc = out_features

    // Add bias if provided
    if (bias.defined()) {
        // Use cublasSaxpy for each column? Or add bias manually using a separate kernel? Since .cpp cannot launch kernel, use cuBLAS: output + bias using cublasSaxpy on each column.
        for (int64_t i = 0; i < batch; ++i) {
            float* row = output.data_ptr<float>() + i * out_features;
            CUBLAS_CHECK(cublasSaxpy(handle, out_features, &alpha, bias.data_ptr<float>(), 1, row, 1));
        }
    }

    destroy_cublas_handle(handle);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error after linear_cublas: ") + cudaGetErrorString(err));
    }

    return output;
}

void register_linear(pybind11::module& m) {
    m.def("linear_cublas", &linear_cublas,
          "Linear layer using cuBLAS",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"));
}

REGISTER_BINDING(linear, register_linear);
