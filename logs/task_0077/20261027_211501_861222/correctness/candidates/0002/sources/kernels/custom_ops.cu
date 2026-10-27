#include <cuda_runtime.h>

// LayerNorm forward kernel: normalizes over the last two dimensions
// Input layout: [N, C, L] where C*L is the normalization size
// Each block handles one sample (one N index)
// Grid dimension: N (number of batch elements)
// Block dimension: blockDim.x threads, shared memory for reduction
__global__ void layer_norm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int C,
    int L,
    float eps
) {
    extern __shared__ float shared[];
    float* s_mean = shared;
    float* s_var = shared + blockDim.x;

    int n = blockIdx.x;
    int tid = threadIdx.x;
    int total = C * L;

    // Each thread processes multiple elements with grid-stride within the sample
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = tid; i < total; i += blockDim.x) {
        // input is [N, C, L] potentially non-contiguous, need strides passed by launcher
        // We'll assume launcher makes input contiguous or passes base pointer with appropriate offsets
        // For safety, launcher should ensure contiguous or handle strides
        float val = input[n * total + i];
        sum += val;
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Store per-warp partials to shared memory
    int warpId = tid / 32;
    int laneId = tid % 32;
    int numWarps = blockDim.x / 32;
    if (laneId == 0) {
        s_mean[warpId] = sum;
        s_var[warpId] = sum_sq;
    }
    __syncthreads();

    // First warp reduces partials
    if (tid < 32) {
        sum = (tid < numWarps) ? s_mean[tid] : 0.0f;
        sum_sq = (tid < numWarps) ? s_var[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (tid == 0) {
            s_mean[0] = sum;
            s_var[0] = sum_sq;
        }
    }
    __syncthreads();

    float mean = s_mean[0] / total;
    float variance = s_var[0] / total - mean * mean;
    float inv_std = rsqrtf(variance + eps);

    // Apply normalization and affine transform
    for (int i = tid; i < total; i += blockDim.x) {
        int idx = n * total + i;
        float val = (input[idx] - mean) * inv_std;
        int c = i / L;
        int l = i % L;
        output[idx] = val * weight[c * L + l] + bias[c * L + l];
    }
}

// ConvTranspose1d forward kernel: computes transposed convolution (also called deconvolution)
// Input: [N, C_in, L_in], weight: [C_in, C_out/groups, K], bias: [C_out]
// Output: [N, C_out, L_out], where L_out = (L_in-1)*stride - 2*padding + dilation*(K-1) + output_padding + 1
// The forward is equivalent to the backward pass of a regular convolution.
// We implement directly as a loop over input positions and kernel positions, accumulating to output.
// Each thread block processes one output element (n, c_out, l_out) or a tile for efficiency.
// Grid: dim3(ceil(L_out / blockDim.x), C_out, N) -> block handles a chunk of L_out for one output channel and batch.
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int L_in,
    int C_out, int K,
    int stride, int padding, int output_padding, int dilation, int groups,
    int L_out
) {
    // blockIdx.x -> output position chunk within a (n, cout) plane
    // blockIdx.y -> output channel (cout)
    // blockIdx.z -> batch index n
    int n = blockIdx.z;
    int cout = blockIdx.y;
    int lout_start = blockIdx.x * blockDim.x + threadIdx.x;
    int lout_stride = blockDim.x * gridDim.x;

    for (int lout = lout_start; lout < L_out; lout += lout_stride) {
        // Determine group and corresponding input channel range
        int group = cout / (C_out / groups);
        int cin_start = group * (C_in / groups);
        int cin_end = (group + 1) * (C_in / groups);

        float acc = 0.0f;
        for (int cin = cin_start; cin < cin_end; ++cin) {
            for (int k = 0; k < K; ++k) {
                // For transposed conv: output position lout corresponds to input position (lout + padding - k*dilation) / stride
                // if that division is integer and the position is valid.
                int lin = lout + padding - k * dilation;
                if (lin % stride != 0) continue;
                lin /= stride;
                if (lin >= 0 && lin < L_in) {
                    // weight shape: [C_in, C_out/groups, K]
                    // for groups, the weight is per group: index [cin, cout % (C_out/groups), k]
                    int cout_in_group = cout - group * (C_out / groups);
                    acc += input[((n * C_in + cin) * L_in) + lin] *
                           weight[((cin * (C_out / groups) + cout_in_group) * K) + k];
                }
            }
        }
        output[((n * C_out + cout) * L_out) + lout] = acc + bias[cout];
    }
}

// Launcher for layer_norm_forward
// Assumes input is contiguous; if not, binding should make contiguous or provide strides.
// We handle only contiguous case as binding ensures contiguity.
void layer_norm_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N,
    int C,
    int L,
    float eps,
    cudaStream_t stream
) {
    int total = C * L;
    // Choose block size: 256 threads
    int threads = 256;
    // Shared memory size: 2 arrays of numWarps floats for reduction
    int numWarps = threads / 32;
    int shared_mem_size = 2 * numWarps * sizeof(float);

    // Grid: one block per sample
    dim3 grid(N);
    dim3 block(threads);

    layer_norm_kernel<<<grid, block, shared_mem_size, stream>>>(
        output, input, weight, bias, C, L, eps
    );
}

// Launcher for conv_transpose1d_forward
// Assumes input is contiguous; if not, binding should make contiguous.
void conv_transpose1d_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int L_in,
    int C_out, int K,
    int stride, int padding, int output_padding, int dilation, int groups,
    int L_out,
    cudaStream_t stream
) {
    // Choose block size and grid configuration
    int threads = 256;
    // Grid: x dimension covers L_out in chunks, y dimension covers C_out, z dimension covers N
    int grid_x = (L_out + threads - 1) / threads;
    if (grid_x < 1) grid_x = 1;
    dim3 grid(grid_x, C_out, N);
    dim3 block(threads);

    conv_transpose1d_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, L_in, C_out, K,
        stride, padding, output_padding, dilation, groups,
        L_out
    );
}
