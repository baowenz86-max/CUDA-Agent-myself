#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for cumulative maximum along dim=0 (columns)
// Each block handles one column, sequential scan using shared memory
template <typename T, typename IndexType = int64_t>
__global__ void cummax_dim0_kernel(const T* __restrict__ input,
                                   T* __restrict__ values,
                                   IndexType* __restrict__ indices,
                                   int N, int M) {
    int col = blockIdx.x;
    if (col >= M) return;
    
    // Load input values for this column into shared memory in chunks
    extern __shared__ char smem[];
    T* s_vals = reinterpret_cast<T*>(smem);
    // Shared memory size: N * sizeof(T) + N * sizeof(IndexType)
    // but we'll compute in chunks to avoid huge shared memory; however for N=20000 and T=float,
    // shared memory per block would be 20000*4 + 20000*8 = 240KB, too much.
    // Instead we use a sequential scan by each thread? Better: one thread per column? No, too slow.
    // Use one warp per column and warp-level scan? N=20000, need sequential scan within each column.
    // Let's use a single thread per column? 50 columns, launch 50 threads only = terrible occupancy.
    // Alternative: parallel scan per column using block of threads.
    // We'll implement a block-level scan for each column using shared memory and work-efficient parallel scan.
    // This is not trivial but doable: each block scans its column.
    // Since N=20000 is much larger than block size, we need hierarchical scan.
    // Simplest correct approach: process each column by a single block, sequentially loop over rows in chunks,
    // each thread handles multiple rows sequentially but maintains running max. But then we don't need shared memory except for the running max per column (one value).
    // However to compute indices we also need running index. We can have each thread process a portion of the column sequentially,
    // but they must be in order. If one thread handles all rows for a column, it would be sequential and slow but correct.
    // With M=50 columns, and N=20000, using one thread per column gives only 50 threads total, very poor.
    // Better: use one block per column, with threads partitioning rows; but we need prefix maximum across partitions.
    // We can use a multi-pass approach: first compute inclusive scan of max with index using shared memory across the block,
    // but that requires block-wide scan; with N=20000, we can have block size 1024 and process in chunks with cross-chunk carry.
    // Let's design a kernel that processes each column using a block of 256 threads.
    // For each column, we loop over rows in tiles of size blockDim.x, maintaining a running max and running index.
    // Each tile: load values into shared memory, compute prefix max within tile, then combine with running max.
    // Actually each tile can be processed independently using a parallel scan within the tile, then combine with running max,
    // but the running max from previous tiles affects all elements.
    // Approach: For each tile, each thread gets its element, we need inclusive prefix max across the tile starting from previous global max.
    // We can use a work-efficient scan (Blelloch) for max and index. But writing a generic scan with index is complex.
    // Simpler: use a single thread to do sequential scan for one column, but launch multiple threads per column by splitting?
    // No, sequential dependency across rows prevents parallelization within a column without a proper scan.
    // Given N=20000, M=50, total elements=1e6, we can afford a simple approach where each thread handles one column? That would be 50 threads.
    // Maybe we can use grid-stride over elements, each thread handles a different row and column, but cummax is along columns so each column is independent, but within a column sequential.
    // We could have each block handle multiple columns? Use block of 256 threads, assign each thread one column? That would need threads >= M, yes 256 >= 50, so each thread can handle one column totally sequentially.
    // That gives parallelism across columns (50 threads per block, maybe multiple blocks). With 50 columns, one block of 256 threads uses 50 threads, leaving many idle, but it's just one block. Could launch a grid of say 1 block, 256 threads, 50 active. That's fine for this small workload. Occupancy is low but computation is not huge; 1e6 elements, sequential per column (20000 iterations per thread) = 20000 * 50 = 1e6 operations, trivial.
    // Actually that might be fast enough. But we can improve by using 50 blocks, one per column, each block with 256 threads? But then only one thread in each block does work. Not good.
    // Alternative: use 50 blocks, each block has 256 threads but within each block we parallelize the scan using shared memory and a proper scan algorithm. That is more efficient for N=20000, but more code.
    // Considering performance target and simplicity, we might implement the sequential per-column thread approach and see if it's fast enough. Likely yes.
    // But we can also implement a block-level scan per column with multiple threads.
    // Let's implement a robust block-level scan per column using shared memory for a tile of up to blockDim.x elements,
    // using a simple two-pass Hillis-Steele scan, but ensuring correct indices and latest-index tie-breaking.
    // We'll launch one block per column. Each block of threads (e.g., 256) processes the column.
    // For each column, we process the entire column using a loop over tiles of blockDim.x.
    // Maintain a global running maximum (globalMax) and its index (globalMaxIdx) computed so far.
    // For each tile: load tile elements, compute inclusive prefix max within tile starting from value=-inf, index=-1, but actually we need to combine with globalMax.
    // We can compute inclusive prefix max of the tile ignoring globalMax, then each element's final max is max(globalMax, prefixMaxWithinTile). For indices, if prefixMax > globalMax, index is prefixMaxIdx, else if prefixMax == globalMax, index is max(globalMaxIdx, prefixMaxIdx) (latest index tie), else index is globalMaxIdx.
    // However prefixMaxWithinTile is computed relative to tile start; for elements before the first max in tile, prefixMax may be less than globalMax, then final max=globalMax.
    // Actually we need to combine correctly: The final value v[i] = max(globalMax, max(input[0..i within this tile]? no, within tile from tile start to current element). Since globalMax is max over all previous elements (before tile start), and within tile we have prefix max of input elements in this tile up to current position. So v[i] = max(globalMax, tilePrefixMax[i]). That is correct.
    // For index: if tilePrefixMax[i] > globalMax, then index = tilePrefixMaxIdx[i]; if tilePrefixMax[i] < globalMax, index = globalMaxIdx; if equal, index = max(globalMaxIdx, tilePrefixMaxIdx[i]) (latest).
    // So we need per-tile inclusive prefix max and index (with latest tie-break) for the tile.
    // We can compute tile prefix max and index using a parallel scan within the tile using shared memory.
    // For block size 256, the tile has 256 elements, we can use a work-efficient scan or a simple sequential scan by one warp? But 256 threads each handle one element; we need scan across 256 threads. Hillis-Steele takes log2(256)=8 steps, each step uses shfl or shared memory. We can implement using shared memory and warp shuffles.
    // Alternatively, we can have each thread handle multiple elements per tile (grid-stride within tile) to reduce number of tiles, but blockDim.x is constant.
    // Let's implement a simple inclusive scan using shared memory arrays and a reduction-like pattern? Actually Hillis-Steele inclusive scan: for offset=1..blockDim.x-1 (doubling), each thread adds element from offset behind, but that's O(n log n) work. However max operation is associative, and we also need index. It's okay for blockDim=256, 8 steps.
    // But we can use Blelloch scan which is O(n) and more efficient. However for simplicity, Hillis-Steele is acceptable for correctness and perhaps performance.
    // Given limited time, maybe implement the very simple one-thread-per-column approach first? That's much less code and likely passes performance if it's within 5%.
    // Let's estimate: 50 columns, each column 20000 elements, sequential loop in one thread -> 1e6 iterations, each iteration does a few operations. A single thread on a modern GPU can do ~1e9 simple ops/sec, so 1e6 ops ~1 microsecond? Actually memory access dominates: each iteration reads one float from global memory, maybe not coalesced (column stride = M, but we process column-major? Input is row-major [N,M], so for a column, elements are strided by M floats (50). That's terrible coalescing: each thread reads a float at interval 50 floats. That would be uncoalesced, 32 threads reading 32 floats spaced 50 floats apart = 32 separate 32-byte transactions, huge slowdown. With 50 threads, memory throughput would be poor. So one-thread-per-column is likely very slow.
    // Better: we need coalesced access. So we should process multiple columns simultaneously, with adjacent threads reading adjacent columns and same row. That gives coalesced reads for that row. For a scan along columns, we need to process rows? No, scan along dim=0 means for each column, prefix along rows. To get coalesced access, we can transpose the problem: process one row at a time for all columns? But scan is along rows, so each column's prefix depends on previous rows. We can process row by row: for each row i, for each column j, compute cumulative max = max(prev_row[j], input[i][j]). That is just a sequential loop over rows; within each row, we need to read input[i][j] for all columns coalesced. That's exactly like a simple loop with one thread per column? Actually we can have one warp process multiple columns per row? Let's think: We can launch a grid of threads such that each thread handles one column, but we iterate rows sequentially; threads aligned by column index. Then when reading row i, thread j reads input[i*M + j], which is coalesced (consecutive threads read consecutive floats). That is perfect! The number of threads must be at least M (50). So we launch one block of at least 64 threads (or multiple blocks), and each thread handles one column if threadIdx.x < M. For M=50, block of 64 works. Then each thread maintains running max and index in registers, and iterates over rows. This is very efficient, coalesced reads, all threads active if we choose block size multiple of warp (64). Great! This will be much faster than per-column blocks with shared memory scan. This is essentially a parallel over columns with sequential over rows, which is exactly what the problem is. Since M=50, we use one block of 64 threads, 50 active. That should be fine. If M were large, we could launch multiple blocks, each handling a subset of columns.
    // So final design: kernel cummax_dim0_kernel(input, values, indices, N, M).
    // Grid: launch enough blocks to cover M columns, each block has blockDim.x threads (e.g., 256), each thread handles columns with stride gridDim.x * blockDim.x.
    // Each thread: col = blockIdx.x * blockDim.x + threadIdx.x; if col >= M return.
    // Then that thread sequentially processes all N rows for that column, maintaining running_max and running_idx.
    // For each row i: value = input[i*M + col]; if value > running_max or (value == running_max and i > running_idx) then running_max=value, running_idx=i (latest tie). Actually torch.cummax returns latest index, so on tie update to current i.
    // Write values[i*M+col]=running_max, indices[i*M+col]=running_idx.
    // This is simple and correct. Memory access: for fixed thread, reads input[i*M+col] for varying i. Across threads at a given i, addresses are contiguous (i*M + col) with col consecutive for threads in a warp (assuming threadIdx.x consecutive). So within a warp, for each row i, threads read a contiguous segment of M floats, but M=50, which is not a multiple of 32, so warp reads 50 floats, some threads inactive? If blockDim.x=64, the warp has 32 threads, threads 0-31 handle columns 0-31, threads 32-49 handle columns 32-49, threads 50-63 inactive. So two warps in block. The first warp reads 32 consecutive floats for row i, the second warp reads 18 (since 50-32=18) consecutive floats, with 14 inactive. Coalescing is good: each warp's active threads read consecutive floats (first warp 32 floats fully coalesced, second warp 18 floats coalesced but underutilized). Fine.
    // We could choose blockDim.x=50, but that's not a multiple of 32; better 64.
    // We launch one block with 64 threads (or maybe 256 and grid=1? No need).
    // This will be extremely fast. No shared memory needed, no manual scan.
    // This is equivalent to a simple for loop over rows, parallel over columns. Perfect for this small column count.
    // So we'll implement this straightforward version.
    // We need to handle N and M possibly zero? M=50, N=20000.
    // Since input is contiguous, we can use linear indexing: input[i*M + col].
    // We'll add a check for empty tensors: if N==0 or M==0 return.
    // For int64 indices, use long long (int64_t).
    // Launcher: extern "C" void cummax_forward_launcher(const float* input, float* values, int64_t* indices, int64_t N, int64_t M, cudaStream_t stream).
    // Binding will call this with data_ptr and dim (but dim is always 0; we ignore dim parameter, but contract includes dim int; we can assert dim==0 in binding or ignore since only dim=0 used). In kernel we only handle dim=0.
    // We'll declare a launcher function with signature that includes dim to match binding contract: but .cu cannot include torch types, so launcher takes raw dim as int64_t.
    // We'll name it cummax_forward_launcher and it takes dim, checks if dim==0, otherwise returns error? Since binding will ensure dim=0 by contract, but we can still check.
    // We'll implement as:
    // extern "C" void cummax_forward_launcher(const float* input, float* values, int64_t* indices, int64_t N, int64_t M, int64_t dim, cudaStream_t stream) { ... }
    // In kernel launch, we ignore dim except to verify.
    // We'll use grid of one block with 64 threads? But to be safe, if M could be larger in future, we can use grid-stride: blocks = (M + blockDim.x -1)/blockDim.x, each block handles a chunk. For current M=50, one block of 64 threads is enough. We'll set blockDim=256 and launch 1 block? Actually 256 threads, 50 active; fine, but 256 threads with only 50 active reduces occupancy slightly? But one block is fine; still okay. Better to use 64 threads. We'll use 64 threads, one block.
    // But if M > 64, we need multiple blocks; we'll compute grid = (M + 63)/64.
    // Kernel code as described.
    // We also need to ensure indices are written as int64_t, using long long.
    // We'll include <cstdint>.
    // No torch headers.
}

// Actually the above template is overly complex; we'll write a non-template kernel with fixed float and int64_t.

__global__ void cummax_dim0_kernel(const float* __restrict__ input,
                                   float* __restrict__ values,
                                   int64_t* __restrict__ indices,
                                   int64_t N, int64_t M) {
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= M) return;
    
    float running_max = -__FLT_MAX__; // or -inf
    int64_t running_idx = -1;
    
    for (int64_t row = 0; row < N; ++row) {
        float val = input[row * M + col];
        if (val > running_max || (val == running_max && row > running_idx)) {
            running_max = val;
            running_idx = row;
        }
        values[row * M + col] = running_max;
        indices[row * M + col] = running_idx;
    }
}

// Launcher
extern "C" void cummax_forward_launcher(const float* input,
                                         float* values,
                                         int64_t* indices,
                                         int64_t N,
                                         int64_t M,
                                         int64_t dim,
                                         cudaStream_t stream) {
    // Only dim=0 supported per contract
    if (dim != 0 || N <= 0 || M <= 0) {
        return;
    }
    
    const int block_size = 64;
    int64_t grid_size = (M + block_size - 1) / block_size;
    cummax_dim0_kernel<<<grid_size, block_size, 0, stream>>>(input, values, indices, N, M);
}
