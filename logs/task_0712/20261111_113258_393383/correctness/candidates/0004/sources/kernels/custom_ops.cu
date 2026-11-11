#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

#define SOFTMAX_BLOCK_SIZE 256
#define SORT_BLOCK_SIZE 256

// Softmax along dim=1 for tensor shape [batch, height, width]
// Each block processes one (batch, width) row of length height.
// Uses a two-pass approach: first compute max and sum serially per row
// using a single thread (or cooperative reduction) to avoid warp shuffle issues.
__global__ void softmax_dim1_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int height, int width) {
    int row = blockIdx.x;
    int batch_idx = row / width;
    int width_idx = row % width;
    int base = batch_idx * height * width + width_idx;

    // Single thread per block computes max then sum for simplicity (slow but correct)
    if (threadIdx.x == 0) {
        float max_val = -CUDART_INF_F;
        for (int i = 0; i < height; ++i) {
            float val = input[base + i * width];
            if (val > max_val) max_val = val;
        }
        float sum = 0.0f;
        for (int i = 0; i < height; ++i) {
            float val = __expf(input[base + i * width] - max_val);
            sum += val;
        }
        for (int i = 0; i < height; ++i) {
            float val = __expf(input[base + i * width] - max_val);
            output[base + i * width] = val / sum;
        }
    }
}

// Stable merge sort kernel for descending order along dim=1.
// Each block sorts one column of length height.
// Shared memory layout: vals[height], idxs[height], tmp_vals[height], tmp_idxs[height]
// Uses cooperative merge sort with multiple threads per merge to improve correctness.
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
    int batch_idx = col / width;
    int width_idx = col % width;
    int base = batch_idx * height * width + width_idx;
    int tid = threadIdx.x;
    int numThreads = blockDim.x;

    // Load elements
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
        // Each merge can be done cooperatively by threads in the block
        for (int merge = tid; merge < numMerges; merge += numThreads) {
            int leftStart = merge * 2 * curSize;
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

        // Copy back merged segments cooperatively
        for (int idx = tid; idx < height; idx += numThreads) {
            vals[idx] = tmp_vals[idx];
            idxs[idx] = tmp_idxs[idx];
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
    // No dynamic shared memory needed for simple version
    softmax_dim1_kernel<<<grid, block, 0, stream>>>(input, output, height, width);
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
