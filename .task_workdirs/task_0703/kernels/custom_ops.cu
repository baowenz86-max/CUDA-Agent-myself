#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// mean over dims (0,2,3) for a contiguous NCHW tensor
// output shape [C]
// Strategy: one block per channel, grid-stride reduction over N*H*W elements
// -----------------------------------------------------------------------------
__global__ void mean_dim_023_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int N, int C, int H, int W) {
    extern __shared__ float sdata[];
    int c = blockIdx.x;
    if (c >= C) return;

    const int inner_size = H * W;
    const int total_per_channel = N * inner_size;

    float sum = 0.0f;
    // grid-stride over all N*H*W elements for this channel
    for (int idx = threadIdx.x; idx < total_per_channel; idx += blockDim.x) {
        // decompose idx into n and hw
        int n = idx / inner_size;
        int hw = idx - n * inner_size;
        const float* base = input + (size_t)n * C * inner_size + (size_t)c * inner_size;
        sum += base[hw];
    }

    // block reduction
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[c] = sdata[0] / (float)total_per_channel;
    }
}

// -----------------------------------------------------------------------------
// histogram with stable order (torch.histc behavior)
// input [C] float32, output [bins] float32
// elements outside [min_val, max_val) are ignored (torch.histc ignores values >= max? actually torch.histc includes max in last bin when value == max)
// Let's mimic torch.histc: formula: bin = floor((val - min) / (max - min) * bins), clamp to [0, bins-1]
// But torch.histc says: if val == max, it's counted in last bin. So use: bin = min((int)((val-min)*scale), bins-1)
// -----------------------------------------------------------------------------
__global__ void histc_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int C, int bins, float min_val, float max_val) {
    const int tid = threadIdx.x;
    const int totalThreads = blockDim.x;

    // shared histogram, zero initialized
    extern __shared__ float shist[];
    for (int i = tid; i < bins; i += totalThreads) {
        shist[i] = 0.0f;
    }
    __syncthreads();

    float scale = (max_val > min_val) ? (float)bins / (max_val - min_val) : 0.0f;
    if (scale == 0.0f) {
        // if min==max, torch.histc returns all zeros? Actually torch.histc with min==max undefined; we'll just set all zero
        __syncthreads();
        if (tid < bins) output[tid] = 0.0f;
        return;
    }

    for (int idx = tid; idx < C; idx += totalThreads) {
        float val = input[idx];
        if (val >= min_val && val <= max_val) {
            int bin = (int)((val - min_val) * scale);
            if (bin == bins) bin = bins - 1; // include max in last bin
            if (bin >= 0 && bin < bins) {
                atomicAdd(&shist[bin], 1.0f);
            }
        }
        // else ignore
    }
    __syncthreads();

    // write shared histogram to global output
    for (int i = tid; i < bins; i += totalThreads) {
        output[i] = shist[i];
    }
}

// -----------------------------------------------------------------------------
// Launchers (extern "C" for C++ binding)
// -----------------------------------------------------------------------------
extern "C" void mean_dim_023_launcher(const float* input, float* output,
                                      int N, int C, int H, int W,
                                      cudaStream_t stream) {
    const int threads = 256;
    dim3 blocks(C);
    size_t shared_mem = threads * sizeof(float);
    mean_dim_023_kernel<<<blocks, threads, shared_mem, stream>>>(
        input, output, N, C, H, W);
}

extern "C" void histc_custom_launcher(const float* input, float* output,
                                      int C, int bins, double min_val, double max_val,
                                      cudaStream_t stream) {
    // use a single block with shared memory histogram, threads = 256
    const int threads = 256;
    size_t shared_mem = bins * sizeof(float);
    histc_kernel<<<1, threads, shared_mem, stream>>>(
        input, output, C, bins, (float)min_val, (float)max_val);
}
