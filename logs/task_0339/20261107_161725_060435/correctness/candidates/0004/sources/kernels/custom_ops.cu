// kernels/custom_ops.cu
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Define tile and column constants for median kernel
constexpr int ROWS_PER_TILE = 64;
constexpr int COLS_PER_BLOCK = 32;

// Use CUDA's erff for float precision. It is accurate and fast.
__global__ void erf_kernel(float* __restrict__ out, const float* __restrict__ in,
                           int64_t total_elements, int64_t in_stride0, int64_t in_stride1) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t r = i / (int64_t)in_stride0;  // This is wrong; need cols info. But this kernel is not used; the actual erf_forward_kernel is below.
    }
}

// Actual elementwise erf kernel for 2D tensor with arbitrary strides
__global__ void erf_forward_kernel(const float* __restrict__ input, float* __restrict__ output,
                                   int64_t rows, int64_t cols,
                                   int64_t in_stride0, int64_t in_stride1,
                                   int64_t out_stride0, int64_t out_stride1) {
    int64_t total = rows * cols;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t r = i / cols;
        int64_t c = i - r * cols;
        const float* in_ptr = input + r * in_stride0 + c * in_stride1;
        float* out_ptr = output + r * out_stride0 + c * out_stride1;
        *out_ptr = erff(*in_ptr);
    }
}

// Launcher for erf
void erf_forward_launcher(const float* input, float* output,
                          int64_t rows, int64_t cols,
                          int64_t in_stride0, int64_t in_stride1,
                          int64_t out_stride0, int64_t out_stride1,
                          cudaStream_t stream) {
    int64_t total = rows * cols;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    erf_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, rows, cols,
                                                       in_stride0, in_stride1,
                                                       out_stride0, out_stride1);
}

// ---------------------------------------------------------------------
// Median along dimension 0 for a 2D tensor: each column independently.
// We implement radix select to find the lower median and its original index.
// Constants: ROWS_PER_TILE and COLS_PER_BLOCK defined above.
// ---------------------------------------------------------------------

// Convert float to sortable uint32 for bitwise radix ordering
__device__ inline uint32_t float_to_sortable(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000) ? ~u : (u | 0x80000000);
}

// Convert sortable uint32 back to float
__device__ inline float sortable_to_float(uint32_t key) {
    uint32_t u = (key & 0x80000000) ? (key & 0x7fffffff) : ~key;
    return __uint_as_float(u);
}

__global__ void median_dim0_kernel(const float* __restrict__ input,
                                   float* __restrict__ values_out,
                                   int64_t* __restrict__ indices_out,
                                   int rows, int cols,
                                   int64_t in_stride0, int64_t in_stride1) {
    extern __shared__ char smem[];
    float* tile = reinterpret_cast<float*>(smem); // [ROWS_PER_TILE][COLS_PER_BLOCK]
    int* hist = reinterpret_cast<int*>(tile + ROWS_PER_TILE * COLS_PER_BLOCK); // [COLS_PER_BLOCK][256]
    uint32_t* prefixes = reinterpret_cast<uint32_t*>(hist + COLS_PER_BLOCK * 256);
    int* ranks = reinterpret_cast<int*>(prefixes + COLS_PER_BLOCK);
    int* occ_count = ranks + COLS_PER_BLOCK;
    bool* found = reinterpret_cast<bool*>(occ_count + COLS_PER_BLOCK);
    int64_t* result_indices = reinterpret_cast<int64_t*>(found + COLS_PER_BLOCK);
    float* result_values = reinterpret_cast<float*>(result_indices + COLS_PER_BLOCK);

    int ncols = min(COLS_PER_BLOCK, cols - blockIdx.x * COLS_PER_BLOCK);
    if (ncols <= 0) return;

    int tid = threadIdx.x;

    // Initialize prefixes to 0, ranks to (rows-1)/2, occ_count to 0, found to false, result_indices to -1
    for (int c = tid; c < ncols; c += blockDim.x) {
        prefixes[c] = 0;
        ranks[c] = (rows - 1) / 2;
        occ_count[c] = 0;
        found[c] = false;
        result_indices[c] = -1;
    }
    __syncthreads();

    // Radix select: 4 passes over bytes (MSB to LSB)
    for (int byte_pos = 3; byte_pos >= 0; --byte_pos) {
        // Reset hist for all columns and buckets
        for (int i = tid; i < ncols * 256; i += blockDim.x) {
            int c = i / 256;
            int b = i % 256;
            hist[c * 256 + b] = 0;
        }
        __syncthreads();

        // Process tiles
        for (int tile_start = 0; tile_start < rows; tile_start += ROWS_PER_TILE) {
            int tile_rows = min(ROWS_PER_TILE, rows - tile_start);
            int tile_elems = tile_rows * ncols;

            // Load tile data (coalesced)
            for (int idx = tid; idx < tile_elems; idx += blockDim.x) {
                int c = idx % ncols;
                int r = idx / ncols;
                int global_col = blockIdx.x * COLS_PER_BLOCK + c;
                int global_row = tile_start + r;
                tile[r * COLS_PER_BLOCK + c] = input[global_row * in_stride0 + global_col * in_stride1];
            }
            __syncthreads();

            // Update histogram for this byte
            for (int idx = tid; idx < tile_elems; idx += blockDim.x) {
                int c = idx % ncols;
                int r = idx / ncols;
                float val = tile[r * COLS_PER_BLOCK + c];
                uint32_t key = float_to_sortable(val);
                uint32_t mask = (byte_pos == 3) ? 0 : (0xFFFFFFFF << ((byte_pos + 1) * 8));
                if ((key & mask) == prefixes[c]) {
                    uint8_t byte_val = (key >> (byte_pos * 8)) & 0xFF;
                    atomicAdd(&hist[c * 256 + byte_val], 1);
                }
            }
            __syncthreads();
        }

        // Compute new prefixes and ranks for each column
        if (tid < ncols) {
            int c = tid;
            int target_rank = ranks[c];
            int cumul = 0;
            int selected_bucket = 0;
            int less_before = 0;
            for (int b = 0; b < 256; ++b) {
                int count = hist[c * 256 + b];
                if (target_rank < cumul + count) {
                    selected_bucket = b;
                    less_before = cumul;
                    break;
                }
                cumul += count;
            }
            ranks[c] = target_rank - less_before;
            prefixes[c] |= (selected_bucket << (byte_pos * 8));
        }
        __syncthreads();
    }

    // After radix select, prefixes hold the sortable key of median value.
    // Convert to float and store in result_values.
    if (tid < ncols) {
        int c = tid;
        result_values[c] = sortable_to_float(prefixes[c]);
        occ_count[c] = 0;
        found[c] = false;
        result_indices[c] = -1;
    }
    __syncthreads();

    // Find the original index of the median: scan tiles in order, one thread per column counts occurrences.
    for (int tile_start = 0; tile_start < rows; tile_start += ROWS_PER_TILE) {
        int tile_rows = min(ROWS_PER_TILE, rows - tile_start);
        int tile_elems = tile_rows * ncols;

        // Load tile data
        for (int idx = tid; idx < tile_elems; idx += blockDim.x) {
            int c = idx % ncols;
            int r = idx / ncols;
            int global_col = blockIdx.x * COLS_PER_BLOCK + c;
            int global_row = tile_start + r;
            tile[r * COLS_PER_BLOCK + c] = input[global_row * in_stride0 + global_col * in_stride1];
        }
        __syncthreads();

        // Process with one thread per column (sequential scan within tile)
        if (tid < ncols) {
            int c = tid;
            if (!found[c]) {
                float median_val = result_values[c];
                int target_occ = ranks[c];
                int local_count = occ_count[c];
                for (int r = 0; r < tile_rows; ++r) {
                    if (tile[r * COLS_PER_BLOCK + c] == median_val) {
                        if (local_count == target_occ) {
                            result_indices[c] = tile_start + r;
                            found[c] = true;
                            break;
                        }
                        local_count++;
                    }
                }
                occ_count[c] = local_count;
            }
        }
        __syncthreads();
    }

    // Write results to global memory
    if (tid < ncols) {
        int c = tid;
        int global_col = blockIdx.x * COLS_PER_BLOCK + c;
        values_out[global_col] = result_values[c];
        indices_out[global_col] = result_indices[c];
    }
}

// Launcher for median
void median_dim0_forward_launcher(const float* input, float* values_out, int64_t* indices_out,
                                  int rows, int cols,
                                  int64_t in_stride0, int64_t in_stride1,
                                  cudaStream_t stream) {
    if (cols == 0) return;
    int blocks = (cols + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
    int threads = 256;
    size_t smem_size = ROWS_PER_TILE * COLS_PER_BLOCK * sizeof(float)
                       + COLS_PER_BLOCK * 256 * sizeof(int)
                       + COLS_PER_BLOCK * sizeof(uint32_t)
                       + COLS_PER_BLOCK * sizeof(int)  // ranks
                       + COLS_PER_BLOCK * sizeof(int)  // occ_count
                       + COLS_PER_BLOCK * sizeof(bool) // found
                       + COLS_PER_BLOCK * sizeof(int64_t) // result_indices
                       + COLS_PER_BLOCK * sizeof(float);  // result_values

    median_dim0_kernel<<<blocks, threads, smem_size, stream>>>(
        input, values_out, indices_out, rows, cols, in_stride0, in_stride1);
}
