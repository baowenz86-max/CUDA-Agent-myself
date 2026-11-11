#include <cuda_runtime.h>

// Concatenate two contiguous 4D tensors along dim=1 (channels).
// Each output element is written by one thread.
template <typename scalar_t>
__global__ void custom_cat_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ x1,
    const scalar_t* __restrict__ x2,
    int64_t total_elements,
    int64_t C1,
    int64_t C2,
    int64_t HW,
    int64_t C_out) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t n = i / (C_out * HW);
        int64_t rem = i % (C_out * HW);
        int64_t c_out = rem / HW;
        int64_t hw = rem % HW;

        if (c_out < C1) {
            int64_t src_idx = n * (C1 * HW) + c_out * HW + hw;
            output[i] = x1[src_idx];
        } else {
            int64_t c2 = c_out - C1;
            int64_t src_idx = n * (C2 * HW) + c2 * HW + hw;
            output[i] = x2[src_idx];
        }
    }
}

// L1 loss kernel: sum of absolute differences using double precision accumulation.
// Each block reduces its partial sum in shared memory, then atomically adds to global double accumulator.
template <typename scalar_t>
__global__ void custom_l1_loss_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ target,
    int64_t total_elements,
    double* __restrict__ global_sum) {
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int64_t idx = blockIdx.x * blockDim.x + tid;
    const int64_t stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;
    for (int64_t i = idx; i < total_elements; i += stride) {
        double diff = static_cast<double>(input[i]) - static_cast<double>(target[i]);
        local_sum += fabs(diff);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

// Kernel to convert double sum to float and optionally divide by count for mean.
__global__ void double_to_float_kernel(const double* src, float* dst, double divisor) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[0] = static_cast<float>(src[0] / divisor);
    }
}

// Launcher for custom_cat.
extern "C" void custom_cat_launcher(
    float* output,
    const float* x1,
    const float* x2,
    int64_t total_elements,
    int64_t C1,
    int64_t C2,
    int64_t HW,
    int64_t C_out,
    cudaStream_t stream) {

    const int threads = 256;
    // Use grid-stride loop with a reasonable grid size.
    int64_t blocks64 = (total_elements + threads - 1) / threads;
    // Limit blocks to avoid excessive grid dimensions (max ~2^31-1).
    const int64_t max_blocks = 65535 * 32; // large but safe
    if (blocks64 > max_blocks) blocks64 = max_blocks;
    int blocks = static_cast<int>(blocks64);

    custom_cat_kernel<float><<<blocks, threads, 0, stream>>>(
        output, x1, x2, total_elements, C1, C2, HW, C_out);
}

// Launcher for custom_l1_loss.
extern "C" void custom_l1_loss_launcher(
    const float* input,
    const float* target,
    int64_t total_elements,
    float* output,
    cudaStream_t stream) {

    double* device_sum;
    cudaMalloc(&device_sum, sizeof(double));
    cudaMemset(device_sum, 0, sizeof(double));

    const int threads = 256;
    // Use enough blocks to cover data but not too many due to atomic contention.
    int64_t blocks64 = (total_elements + threads - 1) / threads;
    // Limit to 65535 blocks to allow grid-stride loop and keep atomic overhead manageable.
    if (blocks64 > 65535) blocks64 = 65535;
    int blocks = static_cast<int>(blocks64);

    size_t shared_mem_size = threads * sizeof(double);
    custom_l1_loss_kernel<float><<<blocks, threads, shared_mem_size, stream>>>(
        input, target, total_elements, device_sum);

    // Convert double sum to float and divide by total_elements for mean.
    double divisor = static_cast<double>(total_elements);
    double_to_float_kernel<<<1, 1, 0, stream>>>(device_sum, output, divisor);

    // Free device memory. We assume the launcher completion is not needed before free,
    // because cudaFree is asynchronous with respect to the device stream.
    cudaFree(device_sum);
}
