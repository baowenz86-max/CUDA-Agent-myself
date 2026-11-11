#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

#define SOFTMAX_BLOCK_SIZE 256
#define SORT_BLOCK_SIZE 256

// Softmax along dim=1 for tensor shape [batch, height, width]
// Each block processes one (batch, width) row of length height.
// Uses online softmax with shared memory reduction for numerical stability.
__global__ void softmax_dim1_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int height, int width) {
    extern __shared__ float smem[];
    float* sdata = smem;
    float* s_red = &smem[height];
    int row = blockIdx.x;
    int width_idx = row % width;
    int base = row * height + width_idx;
    int tid = threadIdx.x;
    int numThreads = blockDim.x;

    float local_max = -CUDART_INF_F;
    for (int i = tid; i < height; i += numThreads) {
        float val = input[base + i * width];
        sdata[i] = val;
        local_max = fmaxf(local_max, val);
    }
    __syncthreads();

    int lane = tid & 31;
    int warp = tid >> 5;
    int numWarps = (numThreads + 31) >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    if (lane == 0) {
        s_red[warp] = local_max;
    }
    __syncthreads();

    if (warp == 0) {
        float warp_max = (lane < numWarps) ? s_red[lane] : -CUDART_INF_F;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
        }
        if (lane == 0) {
            s_red[0] = warp_max;
        }
    }
    __syncthreads();

    float m = s_red[0];

    float local_sum = 0.0f;
    for (int i = tid; i < height; i += numThreads) {
        float e = __expf(sdata[i] - m);
        sdata[i] = e;
        local_sum += e;
    }
    __syncthreads();

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) {
        s_red[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float warp_sum = (lane < numWarps) ? s_red[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (lane == 0) {
            s_red[0] = warp_sum;
        }
    }
    __syncthreads();

    float sum = s_red[0];

    for (int i = tid; i < height; i += numThreads) {
        output[base + i * width] = sdata[i] / sum;
    }
}

// Stable merge sort kernel for descending order along dim=1.
// Each block sorts one column of length height.
// Shared memory layout: vals[height], idxs[height], tmp_vals[height], tmp_idxs[height]
__global__ void sort_descending_dim1_kernel(const float* __restrict__ input,
                                            float* __restrict__ values_out,
                                            int64_t* __restrict__ indices_out,
                                            int height, int width) {
    extern __shared__ char shared_raw[];
    float* vals = reinterpret_cast<float*>(shared_raw);
    int* idxs = reinterpret_cast<int*>(&shared_raw[height * sizeof(float)]);
    float* tmp_vals = reinterpret_cast<float*>(&shared_raw[2 * height * sizeof(float)]);
    int* tmp_idxs = reinterpret_cast<int*>(&shared_raw[2 * height * sizeof(float) + height * sizeof(int)]);

    int col = blockIdx.x;
    int width_idx = col % width;
    int base = (col / width) * height * width + width_idx;
    int tid = threadIdx.x;
    int numThreads = blockDim.x;

    for (int i = tid; i < height; i += numThreads) {
        vals[i] = input[base + i * width];
        idxs[i] = i;
    }
    __syncthreads();

    // Sort small segments with insertion sort (stable)
    const int segmentSize = 16;
    int numSegments = (height + segmentSize - 1) / segmentSize;
    for (int seg = tid; seg < numSegments; seg += numThreads) {
        int start = seg * segmentSize;
        int end = min(start + segmentSize, height);
        for (int i = start + 1; i < end; ++i) {
            float key_val = vals[i];
            int key_idx = idxs[i];
            int j = i - 1;
            while (j >= start && (vals[j] < key_val || (vals[j] == key_val && idxs[j] > key_idx))) {
                vals[j + 1] = vals[j];
                idxs[j + 1] = idxs[j];
                --j;
            }
            vals[j + 1] = key_val;
            idxs[j + 1] = key_idx;
        }
    }
    __syncthreads();

    // Merge segments of increasing size
    for (int curSize = segmentSize; curSize < height; curSize <<= 1) {
        int numMerges = (height + 2 * curSize - 1) / (2 * curSize);
        // Each merge is performed by a single thread (sequential merge within that thread)
        if (tid < numMerges) {
            int leftStart = tid * 2 * curSize;
            int mid = min(leftStart + curSize, height);
            int rightEnd = min(leftStart + 2 * curSize, height);
            int i = leftStart, j = mid, k = leftStart;
            while (i < mid && j < rightEnd) {
                if (vals[i] > vals[j] || (vals[i] == vals[j] && idxs[i] <= idxs[j])) {
                    tmp_vals[k] = vals[i];
                    tmp_idxs[k] = idxs[i];
                    ++i;
                } else {
                    tmp_vals[k] = vals[j];
                    tmp_idxs[k] = idxs[j];
                    ++j;
                }
                ++k;
            }
            while (i < mid) {
                tmp_vals[k] = vals[i];
                tmp_idxs[k] = idxs[i];
                ++i; ++k;
            }
            while (j < rightEnd) {
                tmp_vals[k] = vals[j];
                tmp_idxs[k] = idxs[j];
                ++j; ++k;
            }
        }
        __syncthreads();

        // Copy back merged segments
        if (tid < numMerges) {
            int leftStart = tid * 2 * curSize;
            int rightEnd = min(leftStart + 2 * curSize, height);
            for (int idx = leftStart + tid; idx < rightEnd; idx += numThreads) {
                vals[idx] = tmp_vals[idx];
                idxs[idx] = tmp_idxs[idx];
            }
        }
        __syncthreads();
    }

    // Write output
    for (int i = tid; i < height; i += numThreads) {
        values_out[base + i * width] = vals[i];
        indices_out[base + i * width] = (int64_t)idxs[i];
    }
}

extern "C" void softmax_dim1_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int height,
    int width,
    cudaStream_t stream
) {
    int rows = batch * width;
    dim3 grid(rows);
    dim3 block(SOFTMAX_BLOCK_SIZE);
    size_t shared_mem = (height + SOFTMAX_BLOCK_SIZE / 32 + 2) * sizeof(float);
    softmax_dim1_kernel<<<grid, block, shared_mem, stream>>>(input, output, height, width);
}

extern "C" void sort_descending_dim1_forward_launcher(
    const float* input,
    float* values_out,
    int64_t* indices_out,
    int batch,
    int height,
    int width,
    cudaStream_t stream
) {
    int cols = batch * width;
    dim3 grid(cols);
    dim3 block(SORT_BLOCK_SIZE);
    size_t shared_mem = 2 * height * sizeof(float) + 2 * height * sizeof(int);
    sort_descending_dim1_kernel<<<grid, block, shared_mem, stream>>>(input, values_out, indices_out, height, width);
}
