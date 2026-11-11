#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define ROWS_PER_TILE 64
#define COLS_PER_BLOCK 32

__device__ __forceinline__ uint32_t float_to_sortable(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float sortable_to_float(uint32_t k) {
    uint32_t u = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
    return __uint_as_float(u);
}

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

__global__ void median_dim0_forward_kernel(const float* __restrict__ input,
                                           float* __restrict__ values_out,
                                           int64_t* __restrict__ indices_out,
                                           int rows, int cols,
                                           int64_t in_stride0, int64_t in_stride1) {
    __shared__ float tile[ROWS_PER_TILE][COLS_PER_BLOCK];
    __shared__ int hist[COLS_PER_BLOCK][256];
    __shared__ uint32_t prefixes[COLS_PER_BLOCK];
    __shared__ int ranks[COLS_PER_BLOCK];
    __shared__ int occ_count[COLS_PER_BLOCK];
    __shared__ bool found[COLS_PER_BLOCK];
    __shared__ int64_t result_indices[COLS_PER_BLOCK];
    __shared__ float result_values[COLS_PER_BLOCK];

    int ncols = min(COLS_PER_BLOCK, cols - blockIdx.x * COLS_PER_BLOCK);
    if (ncols <= 0) return;

    // Initialize state
    if (threadIdx.x < ncols) {
        int c = threadIdx.x;
        prefixes[c] = 0;
        ranks[c] = (rows - 1) / 2;
        occ_count[c] = 0;
        found[c] = false;
        result_indices[c] = -1;
        result_values[c] = 0.0f;
    }
    __syncthreads();

    // Radix select loop
    for (int byte_pos = 3; byte_pos >= 0; --byte_pos) {
        // Reset hist
        for (int i = threadIdx.x; i < ncols * 256; i += blockDim.x) {
            int c = i / 256;
            int b = i % 256;
            hist[c][b] = 0;
        }
        __syncthreads();

        // Loop over tiles
        for (int tile_start = 0; tile_start < rows; tile_start += ROWS_PER_TILE) {
            int tile_rows = min(ROWS_PER_TILE, rows - tile_start);
            int tile_elems = tile_rows * ncols;

            // Load tile
            for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
                int c = idx % ncols;
                int r = idx / ncols;
                int global_col = blockIdx.x * COLS_PER_BLOCK + c;
                int global_row = tile_start + r;
                tile[r][c] = input[global_row * in_stride0 + global_col * in_stride1];
            }
            __syncthreads();

            // Update histogram for current byte
            for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
                int c = idx % ncols;
                int r = idx / ncols;
                float val = tile[r][c];
                uint32_t key = float_to_sortable(val);
                uint32_t mask = (byte_pos == 3) ? 0 : (0xFFFFFFFFu << ((byte_pos + 1) * 8));
                if ((key & mask) == prefixes[c]) {
                    uint8_t byte_val = (key >> (byte_pos * 8)) & 0xFF;
                    atomicAdd(&hist[c][byte_val], 1);
                }
            }
            __syncthreads();
        }

        // Update prefixes and ranks
        if (threadIdx.x < ncols) {
            int c = threadIdx.x;
            int cumul = 0;
            int target_rank = ranks[c];
            int selected_bucket = 0;
            int less_before = 0;
            for (int b = 0; b < 256; ++b) {
                int count = hist[c][b];
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

    // Store median values and reset for index finding
    if (threadIdx.x < ncols) {
        int c = threadIdx.x;
        result_values[c] = sortable_to_float(prefixes[c]);
        occ_count[c] = 0;
        found[c] = false;
        result_indices[c] = -1;
    }
    __syncthreads();

    // Find indices
    for (int tile_start = 0; tile_start < rows; tile_start += ROWS_PER_TILE) {
        int tile_rows = min(ROWS_PER_TILE, rows - tile_start);
        int tile_elems = tile_rows * ncols;

        // Load tile
        for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            int c = idx % ncols;
            int r = idx / ncols;
            int global_col = blockIdx.x * COLS_PER_BLOCK + c;
            int global_row = tile_start + r;
            tile[r][c] = input[global_row * in_stride0 + global_col * in_stride1];
        }
        __syncthreads();

        // Scan for target occurrence using one thread per column
        if (threadIdx.x < ncols) {
            int c = threadIdx.x;
            if (!found[c]) {
                float median_val = result_values[c];
                int target_occ = ranks[c];
                int local_count = occ_count[c];
                for (int r = 0; r < tile_rows; ++r) {
                    if (tile[r][c] == median_val) {
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

    // Write results
    if (threadIdx.x < ncols) {
        int c = threadIdx.x;
        int global_col = blockIdx.x * COLS_PER_BLOCK + c;
        values_out[global_col] = result_values[c];
        indices_out[global_col] = result_indices[c];
    }
}

void median_dim0_forward_launcher(const float* input, float* values_out, int64_t* indices_out,
                                  int rows, int cols,
                                  int64_t in_stride0, int64_t in_stride1,
                                  cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;
    int blocks = (cols + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
    median_dim0_forward_kernel<<<blocks, 256, 0, stream>>>(input, values_out, indices_out,
                                                           rows, cols, in_stride0, in_stride1);
}
