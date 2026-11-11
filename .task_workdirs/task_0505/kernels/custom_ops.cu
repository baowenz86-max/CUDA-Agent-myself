#include <cuda_runtime.h>

// GroupNorm forward kernel: one block per (n, group)
// x is assumed contiguous with shape [N, C, HW] where HW = product of spatial dims
// weight, bias are [C] contiguous
template <int BLOCK_SIZE>
__global__ void group_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int C_per_group,
    int HW,
    float eps)
{
    extern __shared__ float smem[];
    float* s_mean = smem;                    // group mean accumulator
    float* s_var = smem + BLOCK_SIZE;        // group variance accumulator (sum of squares)

    int group = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;

    int channels_per_group = C_per_group;
    int total_elems = channels_per_group * HW;

    // local accumulation
    float local_sum = 0.0f;
    float local_sq = 0.0f;

    const float* x_base = x + (long long)n * (channels_per_group * group * HW) + (long long)group * channels_per_group * HW;
    // Actually x layout: [N, C, HW], for group g, channels start at group * channels_per_group
    // The above base is incorrect; fix:
    const float* x_group_base = x + (long long)n * (channels_per_group * (blockIdx.y == 0 ? 0 : 0));
    // Simpler: compute offset
    int C_total = channels_per_group * gridDim.x; // not known here, pass as parameter

    // We'll use loop over elements with grid-stride for accumulation
    for (int i = tid; i < total_elems; i += BLOCK_SIZE) {
        int c_local = i / HW;
        int s = i - c_local * HW;
        int c_global = group * channels_per_group + c_local;
        float val = x[(long long)n * C_total * HW + (long long)c_global * HW + s];
        local_sum += val;
        local_sq += val * val;
    }

    // block reduction
    s_mean[tid] = local_sum;
    s_var[tid] = local_sq;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[tid] += s_mean[tid + stride];
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_mean[0] / total_elems;
    float var = s_var[0] / total_elems - mean * mean;
    float invstd = rsqrtf(var + eps);

    // write output
    for (int i = tid; i < total_elems; i += BLOCK_SIZE) {
        int c_local = i / HW;
        int s = i - c_local * HW;
        int c_global = group * channels_per_group + c_local;
        float val = x[(long long)n * C_total * HW + (long long)c_global * HW + s];
        y[(long long)n * C_total * HW + (long long)c_global * HW + s] =
            (val - mean) * invstd * weight[c_global] + bias[c_global];
    }
}

// Launcher function (extern "C" for binding)
extern "C" void group_norm_forward_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    int64_t N,
    int64_t C,
    int64_t HW,
    int64_t num_groups,
    double eps,
    cudaStream_t stream)
{
    int channels_per_group = (int)(C / num_groups);
    dim3 grid(num_groups, N);
    dim3 block(256); // tune later
    int smem_size = 2 * block.x * sizeof(float);

    group_norm_kernel<256><<<grid, block, smem_size, stream>>>(
        x, y, weight, bias, channels_per_group, (int)HW, (float)eps);
}
