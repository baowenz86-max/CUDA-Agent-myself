#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

__global__ void bias_add_kernel(float* output, const float* bias, int N, int M) {
    int total = N * M;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        int row = idx / M;
        int col = idx % M;
        output[idx] += bias[col];
    }
}

template <typename T>
__global__ void selu_kernel(const T* input, T* output, int total) {
    const T scale = (T)1.0507009873554804934193349852946;
    const T alpha = (T)1.6732632423543772848170429916717;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        T x = input[idx];
        if (x > 0.0f) {
            output[idx] = scale * x;
        } else {
            output[idx] = scale * alpha * (expf(x) - 1.0f);
        }
    }
}

template <typename T>
__global__ void sigmoid_kernel(const T* input, T* output, int total) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        T x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

extern "C" void selu_forward_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    int max_blocks_per_dim = 65535;
    int actual_grid = grid_size < max_blocks_per_dim ? grid_size : max_blocks_per_dim;
    selu_kernel<float><<<actual_grid, block_size, 0, stream>>>(input, output, total);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void sigmoid_forward_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    int max_blocks_per_dim = 65535;
    int actual_grid = grid_size < max_blocks_per_dim ? grid_size : max_blocks_per_dim;
    sigmoid_kernel<float><<<actual_grid, block_size, 0, stream>>>(input, output, total);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int in_features,
    int out_features,
    cudaStream_t stream)
{
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed");
    }
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    stat = cublasSgemm(handle,
                       CUBLAS_OP_T, CUBLAS_OP_N,
                       out_features, N, in_features,
                       &alpha,
                       weight, in_features,
                       input, in_features,
                       &beta,
                       output, out_features);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        throw std::runtime_error("cublasSgemm failed");
    }

    int total = N * out_features;
    if (total > 0) {
        const int block_size = 256;
        const int grid_size = (total + block_size - 1) / block_size;
        int max_blocks_per_dim = 65535;
        int actual_grid = grid_size < max_blocks_per_dim ? grid_size : max_blocks_per_dim;
        bias_add_kernel<<<actual_grid, block_size, 0, stream>>>(output, bias, N, out_features);
        CUDA_CHECK(cudaGetLastError());
    }

    cublasDestroy(handle);
}
