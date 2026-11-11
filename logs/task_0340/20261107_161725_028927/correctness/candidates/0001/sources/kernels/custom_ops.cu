#include <cuda_runtime.h>
#include <cmath>

// SELU constants
extern "C" __constant__ float selu_scale = 1.0507009873554804934193349852946f;
extern "C" __constant__ float selu_alpha = 1.6732632423543772848170429916717f;

// Elementwise SELU kernel
// Each thread computes one output element.
// Input and output are strided 2D tensors with shape [rows, cols].
// We flatten the index and map to (row, col) using output strides.
__global__ void selu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    long long input_stride0,
    long long input_stride1,
    long long output_stride0,
    long long output_stride1)
{
    long long total = (long long)rows * cols;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long i = idx; i < total; i += stride) {
        int row = i / cols;
        int col = i % cols;
        float x = input[row * input_stride0 + col * input_stride1];
        float out;
        if (x >= 0.0f) {
            out = selu_scale * x;
        } else {
            out = selu_scale * (selu_alpha * expf(x) - selu_alpha);
        }
        output[row * output_stride0 + col * output_stride1] = out;
    }
}

// Cumulative sum along dimension 0.
// We parallelize over columns: one block per column.
// Within each block, threads stride over rows maintaining a running sum.
// Since rows=20000 and cols=5, this is simple and effective.
// To avoid race conditions, each thread handles a consecutive chunk of rows
// using block-level scan via a single thread? Better: use one block per column,
// threads cooperative scan using shared memory for each chunk.
// Approach: one block per column, threads iterate over rows in chunks,
// using shared memory to store partial sums and perform scan.
// For simplicity and correctness, use a single thread per block to do sequential scan.
// That would be too slow if F is large, but F=5 and N=20000, so 5 threads total,
// which is inefficient but correct. However, grid size must be at least 1.
// Better: use a parallel scan within each column using multiple threads and shared memory.
// We'll implement a block-level scan per column using a two-pass approach:
//   Pass 1: each thread computes sum of its assigned segment, stores in shared array.
//   Pass 2: scan shared array to get offset, then each thread computes prefix within its segment and writes output.
// This is efficient and maintains order.

// Define a template for block size, but we'll just use 256.
#define CUMSUM_BLOCK_SIZE 256

__global__ void cumsum0_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    long long input_stride0,
    long long input_stride1,
    long long output_stride0,
    long long output_stride1)
{
    // One block per column
    int col = blockIdx.x;
    if (col >= cols) return;

    // Shared memory for partial sums of each thread's segment
    __shared__ float partial_sums[CUMSUM_BLOCK_SIZE];
    // Shared memory for scanned offsets (same array reused)
    float* scanned_offsets = partial_sums;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Total number of rows to process
    int n = rows;
    // Number of elements per thread (rounded up)
    int elements_per_thread = (n + block_size - 1) / block_size;

    // Compute start and end row for this thread
    int start_row = tid * elements_per_thread;
    int end_row = min(start_row + elements_per_thread, n);

    // Step 1: compute partial sum of this thread's segment
    float local_sum = 0.0f;
    for (int r = start_row; r < end_row; ++r) {
        local_sum += input[r * input_stride0 + col * input_stride1];
    }

    // Store partial sum in shared memory
    partial_sums[tid] = local_sum;
    __syncthreads();

    // Step 2: exclusive scan of partial_sums to get offset for each thread
    // We'll use a simple work-efficient scan for up to 1024 threads (block_size <= 1024)
    int offset = 0;
    int stride_scan = 1;
    while (stride_scan < block_size) {
        // In-place scan using shared memory: each thread adds element at stride_scan away if exists
        if (tid >= stride_scan) {
            partial_sums[tid] += partial_sums[tid - stride_scan];
        }
        __syncthreads();
        stride_scan <<= 1;
    }
    // After loop, partial_sums[tid] is inclusive scan up to tid.
    // The exclusive offset for thread tid is partial_sums[tid] - local_sum, but we have local_sum in register.
    // However we overwrote partial_sums[tid] with inclusive scan.
    // We can compute offset = (tid == 0) ? 0 : partial_sums[tid-1].
    offset = (tid == 0) ? 0.0f : partial_sums[tid - 1];
    __syncthreads(); // ensure all reads of partial_sums are done before reuse? Actually no reuse now, but safe.

    // Step 3: each thread computes prefix within its segment and writes output
    float running = offset;
    for (int r = start_row; r < end_row; ++r) {
        float val = input[r * input_stride0 + col * input_stride1];
        running += val;
        output[r * output_stride0 + col * output_stride1] = running;
    }
}

// Launcher for SELU
extern "C" void selu_launcher(
    const float* input,
    float* output,
    int rows,
    int cols,
    long long input_stride0,
    long long input_stride1,
    long long output_stride0,
    long long output_stride1,
    cudaStream_t stream)
{
    long long total = (long long)rows * cols;
    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    // Grid-stride approach: can launch fewer blocks but we'll cap to reasonable grid.
    int max_blocks = 65535; // typical max grid dimension
    int blocks = (int)min(blocks_ll, (long long)max_blocks);
    selu_kernel<<<blocks, threads, 0, stream>>>(
        input, output, rows, cols,
        input_stride0, input_stride1,
        output_stride0, output_stride1);
}

// Launcher for cumsum0
extern "C" void cumsum0_launcher(
    const float* input,
    float* output,
    int rows,
    int cols,
    long long input_stride0,
    long long input_stride1,
    long long output_stride0,
    long long output_stride1,
    cudaStream_t stream)
{
    int threads = CUMSUM_BLOCK_SIZE;
    int blocks = cols;
    cumsum0_kernel<<<blocks, threads, 0, stream>>>(
        input, output, rows, cols,
        input_stride0, input_stride1,
        output_stride0, output_stride1);
}
