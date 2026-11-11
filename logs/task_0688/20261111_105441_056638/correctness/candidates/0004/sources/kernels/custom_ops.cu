#include <cuda_runtime.h>
#include <math.h>

// Concatenate x1 and x2 along channel dimension (dim=1)
// Inputs are contiguous [N, C1, H, W] and [N, C2, H, W]
// Output is contiguous [N, C1+C2, H, W]
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
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t n = i / (C_out * HW);
        int64_t rem = i % (C_out * HW);
        int64_t c = rem / HW;
        int64_t hw = rem % HW;
        if (c < C1) {
            output[i] = x1[n * (C1 * HW) + c * HW + hw];
        } else {
            int64_t c2 = c - C1;
            output[i] = x2[n * (C2 * HW) + c2 * HW + hw];
        }
    }
}

// L1 loss reduction kernel using double precision accumulation
template <typename scalar_t>
__global__ void custom_l1_loss_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ target,
    int64_t total_elements,
    double* __restrict__ global_sum) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;
    int64_t stride = gridDim.x * blockDim.x;

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

// Convert double device sum to float scalar output
__global__ void double_to_float_kernel(const double* src, float* dst) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[0] = static_cast<float>(src[0]);
    }
}

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
    // Use grid-stride loop so grid size can be modest and handle large tensors
    int max_blocks = 1024; // reasonable
    int64_t blocks_needed = (total_elements + threads - 1) / threads;
    int blocks = (int)min((int64_t)max_blocks, blocks_needed);
    if (blocks == 0) blocks = 1;

    custom_cat_kernel<float><<<blocks, threads, 0, stream>>>(
        output, x1, x2, total_elements, C1, C2, HW, C_out);
}

extern "C" void custom_l1_loss_launcher(
    const float* input,
    const float* target,
    int64_t total_elements,
    float* output,
    cudaStream_t stream) {

    // Temporary device double for atomic accumulation
    double* device_sum = nullptr;
    cudaMalloc(&device_sum, sizeof(double));
    cudaMemsetAsync(device_sum, 0, sizeof(double), stream);

    const int threads = 256;
    int max_blocks = 1024;
    int64_t blocks_needed = (total_elements + threads - 1) / threads;
    int blocks = (int)min((int64_t)max_blocks, blocks_needed);
    if (blocks == 0) blocks = 1;
    size_t shared_mem_size = threads * sizeof(double);

    custom_l1_loss_kernel<float><<<blocks, threads, shared_mem_size, stream>>>(
        input, target, total_elements, device_sum);

    // Convert double sum to float in output[0]
    double_to_float_kernel<<<1, 1, 0, stream>>>(device_sum, output);

    cudaFreeAsync(device_sum, stream);
}
