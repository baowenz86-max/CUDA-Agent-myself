#include <cuda_runtime.h>

__global__ void nll_loss_reduce_kernel(
    const float* __restrict__ log_prob,
    const int64_t* __restrict__ target,
    float* __restrict__ partial_sums,
    int batch_size,
    int num_classes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    for (int i = tid; i < batch_size; i += stride) {
        int64_t t = target[i];
        if (t >= 0 && t < num_classes) {
            local_sum += log_prob[i * num_classes + t];
        }
    }

    __shared__ float shared_sum[256];
    int lane = threadIdx.x;
    int warp_lane = lane & 31;
    int warp_id = lane >> 5;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (warp_lane == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < (blockDim.x + 31) / 32) ? shared_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            partial_sums[blockIdx.x] = block_sum;
        }
    }
}

__global__ void final_reduce_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ output,
    int num_blocks,
    int batch_size)
{
    __shared__ float shared_sum[32];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }

    shared_sum[tid] = sum;
    __syncthreads();

    if (tid < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum = shared_sum[tid];
            if (tid + offset < blockDim.x) {
                sum += shared_sum[tid + offset];
            }
            __syncthreads();
            shared_sum[tid] = sum;
            __syncthreads();
        }
    }

    if (tid == 0) {
        output[0] = -shared_sum[0] / batch_size;
    }
}

extern "C" void nll_loss_forward_launcher(
    const float* log_prob,
    const int64_t* target,
    float* output,
    int batch_size,
    int num_classes,
    cudaStream_t stream)
{
    int block_size = 256;
    int max_blocks_per_grid = (batch_size + block_size - 1) / block_size;
    if (max_blocks_per_grid < 1) max_blocks_per_grid = 1;
    int num_blocks = (max_blocks_per_grid > 4096) ? 4096 : max_blocks_per_grid;

    float* partial_sums = nullptr;
    cudaError_t err = cudaMalloc(&partial_sums, num_blocks * sizeof(float));
    if (err != cudaSuccess) {
        return;
    }

    nll_loss_reduce_kernel<<<num_blocks, block_size, 0, stream>>>(
        log_prob, target, partial_sums, batch_size, num_classes);

    int final_block_size = 32;
    final_reduce_kernel<<<1, final_block_size, 0, stream>>>(
        partial_sums, output, num_blocks, batch_size);

    cudaError_t sync_err = cudaStreamSynchronize(stream);
    cudaFree(partial_sums);

    if (sync_err != cudaSuccess) {
        return;
    }
}
