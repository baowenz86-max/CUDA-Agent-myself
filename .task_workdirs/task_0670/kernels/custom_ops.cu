#include <cuda_runtime.h>

// Pad a 4D tensor (N, C, H, W) with constant value on height and width dimensions.
// input: contiguous float32 tensor of shape [N, C, H_in, W_in]
// output: contiguous float32 tensor of shape [N, C, H_in + 2*pad, W_in + 2*pad]
// pad: integer padding amount on each side (top, bottom, left, right)
// value: constant fill value
__global__ void pad2d_constant_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int pad,
    float value)
{
    // Total number of output elements
    long long total = (long long)N * C * (H_in + 2 * pad) * (W_in + 2 * pad);
    // Grid-stride loop for arbitrary size
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x)
    {
        // Decode linear index to (n, c, h_out, w_out)
        int w_out = idx % (W_in + 2 * pad);
        long long tmp = idx / (W_in + 2 * pad);
        int h_out = tmp % (H_in + 2 * pad);
        tmp = tmp / (H_in + 2 * pad);
        int c = tmp % C;
        int n = tmp / C;

        // Determine input coordinates
        int h_in = h_out - pad;
        int w_in = w_out - pad;

        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            // Inside original region: copy from input
            long long input_idx = (((long long)n * C + c) * H_in + h_in) * W_in + w_in;
            output[idx] = input[input_idx];
        } else {
            // Outside: fill with constant value
            output[idx] = value;
        }
    }
}

extern "C" void pad2d_constant_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int pad,
    float value,
    cudaStream_t stream)
{
    int H_out = H_in + 2 * pad;
    int W_out = W_in + 2 * pad;
    long long total = (long long)N * C * H_out * W_out;

    // Choose block size and grid size (capped to reasonable maximum)
    int block = 256;
    int grid = (int)min((total + block - 1) / block, (long long)65535 * 32); // grid-stride handles more
    if (grid < 1) grid = 1;

    pad2d_constant_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, H_in, W_in, pad, value);

    // Error check (no sync)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Cannot throw from extern "C" launcher; caller will handle via error check on stream
        // Optionally print error, but no PyTorch dependency. We assume binding will check.
    }
}

// Sum all elements of a contiguous float32 tensor.
// input: contiguous float32 tensor of arbitrary shape, total number of elements = numel
// output: pointer to a scalar float32 value (device memory)
__global__ void sum_all_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel)
{
    extern __shared__ float warp_sums[];

    // Each thread accumulates a local sum using grid-stride loop
    float local_sum = 0.0f;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += (long long)gridDim.x * blockDim.x)
    {
        local_sum += input[i];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Write warp sum to shared memory
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    if (laneId == 0) {
        warp_sums[warpId] = local_sum;
    }
    __syncthreads();

    // Now, first warp reduces warp_sums (number of warps)
    int numWarps = blockDim.x / 32;
    if (warpId == 0) {
        float val = (laneId < numWarps) ? warp_sums[laneId] : 0.0f;
        // reduce within first warp
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (laneId == 0) {
            // add block sum to global output using atomicAdd
            atomicAdd(output, val);
        }
    }
}

extern "C" void sum_all_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream)
{
    // Ensure output is initialized to zero before kernel launch
    cudaMemsetAsync(output, 0, sizeof(float), stream);

    int block = 256;
    int grid = (int)min((numel + block - 1) / block, (long long)65535 * 32);
    if (grid < 1) grid = 1;

    // Shared memory size for warp_sums: number of warps * float
    int warps_per_block = block / 32; // 8
    size_t shared_mem_size = warps_per_block * sizeof(float);

    sum_all_kernel<<<grid, block, shared_mem_size, stream>>>(
        input, output, numel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error handling left to binding
    }
}
