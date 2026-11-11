#include <cuda_runtime.h>
#include <cstdint>

__global__ void equal_contiguous_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    long long numel,
    int* __restrict__ mismatch_flag)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        if (__ldg(x + i) != __ldg(y + i)) {
            atomicExch(mismatch_flag, 1);
            return;
        }
    }
}

__global__ void l1_sum_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    long long numel,
    double* __restrict__ global_sum)
{
    extern __shared__ double sdata[];
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    double local_sum = 0.0;
    for (long long i = idx; i < numel; i += stride) {
        local_sum += fabs((double)x[i] - (double)y[i]);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

extern "C" bool custom_equal_launcher(
    const float* x,
    const float* y,
    long long numel,
    const long long* x_strides,
    const long long* y_strides,
    int ndim,
    cudaStream_t stream)
{
    if (numel == 0) return true;

    int device;
    cudaGetDevice(&device);
    int* d_flag = nullptr;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemsetAsync(d_flag, 0, sizeof(int), stream);

    int block = 256;
    int grid = (int)std::min((numel + block - 1) / block, (long long)65535);
    equal_contiguous_kernel<<<grid, block, 0, stream>>>(x, y, numel, d_flag);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_flag);
        return false;
    }
    int h_flag = 0;
    cudaMemcpyAsync(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_flag);
    return (h_flag == 0);
}

extern "C" float custom_l1_loss_launcher(
    const float* x,
    const float* y,
    long long numel,
    const long long* x_strides,
    const long long* y_strides,
    int ndim,
    cudaStream_t stream)
{
    if (numel == 0) {
        return 0.0f;
    }

    double* d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemsetAsync(d_sum, 0, sizeof(double), stream);

    const int block = 256;
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    long long max_grid_x = prop.maxGridSize[0];
    long long grid = (numel + block - 1) / block;
    if (grid > max_grid_x) grid = max_grid_x;

    size_t shared_mem = block * sizeof(double);
    l1_sum_kernel<<<(int)grid, block, shared_mem, stream>>>(x, y, numel, d_sum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_sum);
        return 0.0f;
    }

    double h_sum = 0.0;
    cudaMemcpyAsync(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_sum);

    float result = (float)(h_sum / (double)numel);
    return result;
}
