#include <cuda_runtime.h>

// Softmax along dim=1 for tensor shape [batch, height, width]
// Each thread block processes one (batch, width) row of length height.
// Uses online softmax with shared memory reduction for numerical stability.
__global__ void softmax_dim1_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int height, int width) {
    int row = blockIdx.x;          // row index = batch * width + width_idx
    int width_idx = threadIdx.x;   // width index within the row
    
    extern __shared__ float smem[]; // [height] for values, [1] for max, [1] for sum
    float* values = smem;
    float* max_val = &smem[height];
    float* sum_val = &smem[height + 1];
    
    int base_idx = row * height + width_idx; // input index for first element of this column
    
    // Load values into shared memory, compute thread-local max first
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < height; i += blockDim.x) {
        float v = input[base_idx + i * width];
        values[i] = v;
        local_max = fmaxf(local_max, v);
    }
    __syncthreads();
    
    // Reduce max across threads
    if (threadIdx.x == 0) {
        float global_max = local_max;
        for (int i = 1; i < blockDim.x; ++i) {
            global_max = fmaxf(global_max, smem[i]); // can't directly access other thread local max; need re-read
        }
        // Recompute global max from shared memory
        global_max = -INFINITY;
        for (int i = 0; i < height; ++i) {
            global_max = fmaxf(global_max, values[i]);
        }
        max_val[0] = global_max;
    }
    __syncthreads();
    
    float m = max_val[0];
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < height; i += blockDim.x) {
        float e = __expf(values[i] - m);
        values[i] = e;
        local_sum += e;
    }
    __syncthreads();
    
    // Reduce sum across threads using shared memory
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < height; ++i) {
            total += values[i];
        }
        sum_val[0] = total;
    }
    __syncthreads();
    
    float sum = sum_val[0];
    
    // Write normalized output
    for (int i = threadIdx.x; i < height; i += blockDim.x) {
        output[base_idx + i * width] = values[i] / sum;
    }
}

// Stable descending sort along dim=1 for tensor shape [batch, height, width]
// Each thread block processes one (batch, width) column (length height).
// Input is softmax output (positive values), output values sorted descending, indices int64.
__global__ void sort_descending_dim1_kernel(const float* __restrict__ input,
                                            float* __restrict__ values_out,
                                            long long* __restrict__ indices_out,
                                            int height, int width) {
    int col = blockIdx.x;          // column index = batch * width + width_idx
    int width_idx = threadIdx.x;   // width index (not actually used per thread below)
    
    extern __shared__ float smem[]; // space for values (height) and original indices (height)
    float* vals = smem;
    int* orig_idx = reinterpret_cast<int*>(&smem[height]);
    
    int base_idx = col * height + width_idx; // wrong: need separate handling per thread
    // Actually, each block handles a column along height; threads cooperatively load and sort.
    // Use dynamic parallelism? No, use shared memory for whole column.
    
    // Load column into shared memory
    for (int i = threadIdx.x; i < height; i += blockDim.x) {
        vals[i] = input[col * height + i * width + width_idx];
        orig_idx[i] = i;
    }
    __syncthreads();
    
    // Simple bubble sort (or insertion sort) for small height? height=2048, too large for shared memory per block.
    // Need different approach: sort entire column of length 2048 with one block? Shared memory limit 48KB, so 2048 floats = 8KB, plus indices 8KB, total 16KB okay.
    // But sorting 2048 elements with bubble sort O(n^2) is slow. Better use bitonic sort or cub? Since we cannot use thrust, implement bitonic sort in shared memory.
    // However, stable sort required. Bitonic sort is not stable. The reference torch.sort is stable. Need stable sort.
    // Alternative: use radix sort on the float keys? Could be complex. Since softmax outputs are small positive numbers and we need stable descending sorted indices, we can use a stable counting sort? No, floats not integers.
    // Simpler: use insertion sort for each column, but with 2048 elements and 128*32=4096 columns, O(n^2) per column is unacceptable.
    // Better: use a merge sort network or implement a stable sort using a comparison-based algorithm that preserves original order. Since keys are floats, we can sort pairs (value, index) using a stable sorting algorithm.
    // CUDA has no standard library sort. We can use a bitonic sort and then break ties by index, but bitonic sort is not stable. However, since we sort pairs (value, index) with value descending and index ascending, the total ordering is deterministic and matches stable sort if we compare (value, index) lexicographically. Because if values are equal, we compare indices; but bitonic sort may reorder equal elements. To ensure stability, we must ensure that when values are equal, we never swap; we can modify comparison to only swap when values differ or (values equal and index order is already ascending). But bitonic sort's compare-exchange operations may not preserve the original order if we don't handle equal case carefully. If we define comparator: if a.value != b.value then compare values, else compare indices (ascending). Then the sorted order will be descending value, and for equal values ascending index. This gives a unique total order, and if we sort all elements, the result is the same as stable sort (because stable sort also sorts by value descending and preserves original index order, which is ascending since original indices are 0..height-1). So any correct comparison-based sort that sorts according to this comparator will produce the same output as stable sort. So we can use bitonic sort with this comparator, as long as the sort network is correct for this total order. The network doesn't need to be stable per se, as long as it sorts the keys according to the comparator. So we can implement bitonic sort on (value, index) pairs.
    // But bitonic sort requires power-of-two length. height=2048 which is power of two (2^11). Good. If not power of two, we'd need padding. We can pad with sentinel values (-inf or +inf) to next power of two. For descending, we want smallest values at end, so pad with -inf. For ascending index tiebreak, we set index = height (larger) so they go last? Actually we just need a total order; padded elements should be less than any real element, so value=-inf, index=height. Then after sort, they will be at the end. We can then ignore them.
    // Since height=2048, no padding needed.
    
    // Implement bitonic sort in shared memory using a single block with 2048 threads? A block can have up to 1024 threads, so not enough to hold all elements in registers. We can use a smaller number of threads and perform the sort in shared memory using multiple passes, each pass using all elements in shared memory. Standard bitonic sort can be implemented with blockDim.x threads and loops over elements.
    
    // However, shared memory size limit: 2048 * (4 bytes float + 4 bytes int) = 16KB, which is fine. But we also need enough threads to do the compare-exchange operations. We can set blockDim.x = 256 or 512. Each thread handles multiple elements in shared memory per stage.
    
    // For simplicity and performance, we can use a known CUDA bitonic sort kernel pattern adapted for shared memory.
    // But this comment section is getting long; I'll implement a straightforward shared memory bitonic sort.
    
    // Edge cases: height may not be power of two. We need to handle gracefully. We'll implement a kernel that pads to next power of two using -inf, then sorts.
    
    // The launcher will determine block size. For height=2048, we can use blockDim.x = 256 or 512. Each block processes one column. gridDim.x = batch * width = 128*32 = 4096.
    
    // For better performance, we might process multiple columns per block, but for now, one column per block is fine given the large number of blocks.
    
    // Note: The softmax kernel also uses one block per row, but height=2048 vs width=32. For softmax, each block has width=32 threads? Actually row length is height=2048, so blockDim.x could be 256 or 512, and each block processes one (batch,width) row. In softmax kernel above, I set blockDim.x = width threads? That would be 32 threads, which is insufficient for reduction over 2048 efficiently. Better to redesign softmax kernel to use blockDim.x = 256 or 512, with gridDim.x = batch*width. Then each block handles one row, threads cooperate to reduce over height. I should fix that.
    
    // I'll write a clean implementation below with separate functions.
}

// Helper for bitonic sort (to be implemented)

// Launcher functions

extern "C" void softmax_dim1_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int height,
    int width,
    cudaStream_t stream
) {
    int rows = batch * width;
    // Use block size 256, grid size rows
    dim3 grid(rows);
    dim3 block(256);
    size_t shared_mem = (height + 2) * sizeof(float); // flexible but need explicit allocation
    // Actually extern shared array size must be known at launch. We can set dynamic shared memory size per block.
    // For height=2048, we need 2050 floats per block ~ 8.2KB, okay.
    shared_mem = (height + 2) * sizeof(float);
    softmax_dim1_kernel<<<grid, block, shared_mem, stream>>>(input, output, height, width);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In binding we can check, but here just ignore; binding will check after call.
    }
}

extern "C" void sort_descending_dim1_forward_launcher(
    const float* input,
    float* values_out,
    long long* indices_out,
    int batch,
    int height,
    int width,
    cudaStream_t stream
) {
    int cols = batch * width;
    // Use block size 256, grid size cols
    dim3 grid(cols);
    dim3 block(256);
    size_t shared_mem = (height + 1) * sizeof(float) + height * sizeof(int); // vals and orig_idx
    // For height=2048, shared memory = 2048*4 + 2048*4 = 16KB, plus maybe padding = fine.
    // Launch kernel (to be implemented).
    // We'll implement a bitonic sort kernel with proper shared memory allocation.
    // For now, use a placeholder? No, must be complete.
    // I'll implement the full kernel in final code.
}

// Actually, I need to provide complete implementation in the file. Let me write the final code now.

// Final implementation below:

#include <cuda_runtime.h>
#include <math_constants.h>

template<int BLOCK_SIZE>
__global__ void softmax_dim1_kernel_opt(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int height, int width) {
    int row = blockIdx.x;
    int col_offset = blockIdx.x % width; // width index
    // Actually row = blockIdx.x corresponds to batch*width + width_idx, so col_offset = row % width.
    extern __shared__ float smem[];
    float* sdata = smem;       // height elements
    float* s_max = &sdata[height];
    float* s_sum = &sdata[height + 1];
    
    int base = row * height + col_offset;
    
    // Load and compute max
    float local_max = -CUDART_INF_F;
    for (int i = threadIdx.x; i < height; i += BLOCK_SIZE) {
        float val = input[base + i * width];
        sdata[i] = val;
        local_max = fmaxf(local_max, val);
    }
    __syncthreads();
    
    // Reduce max using shared memory (block reduction)
    // Since height may be > BLOCK_SIZE, we need to reduce over all height elements.
    // We can do a two-level reduction: first each thread computes max over its strided elements, then reduce across threads using shared memory.
    // But we already have all values in sdata, so we can reduce over sdata using a standard tree reduction with BLOCK_SIZE threads.
    // For simplicity, perform reduction with a loop over successive halves.
    // Initialize s_max with local_max from each thread? Actually we need global max. We can use a parallel reduction over sdata.
    // We'll use a shared memory reduction pattern: each thread holds a partial max over its strided elements (already in local_max). Write local_max to a temporary array of size BLOCK_SIZE, then reduce that.
    // But we can directly use a tree reduction over sdata with BLOCK_SIZE threads, where each thread handles multiple elements in each step if height > BLOCK_SIZE.
    // Below is a simple but correct reduction.
    
    // Clear s_max for reduction
    if (threadIdx.x == 0) {
        s_max[0] = -CUDART_INF_F;
    }
    __syncthreads();
    
    // Strided loop over height, each thread accumulates max into its slot
    // Actually we can use atomicMax on shared memory? No.
    // Alternative: Use warp shuffle reduction for local_max, then write to s_max, then reduce s_max.
    // Since local_max per thread is over height/BLOCK_SIZE elements, we can reduce those across threads.
    
    // Let's do warp-level reduction of local_max within each warp, then store per-warp max in shared memory, then reduce across warps.
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int numWarps = (BLOCK_SIZE + 31) >> 5;
    
    // Warp reduction of local_max
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    if (lane == 0) {
        s_max[warp] = local_max;
    }
    __syncthreads();
    
    // Reduce per-warp maxima
    if (warp == 0) {
        float warp_max = (lane < numWarps) ? s_max[lane] : -CUDART_INF_F;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
        }
        if (lane == 0) {
            s_max[0] = warp_max;
        }
    }
    __syncthreads();
    
    float m = s_max[0];
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < height; i += BLOCK_SIZE) {
        float e = __expf(sdata[i] - m);
        sdata[i] = e;
        local_sum += e;
    }
    __syncthreads();
    
    // Reduce sum similarly to max
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) {
        s_sum[warp] = local_sum;
    }
    __syncthreads();
    
    if (warp == 0) {
        float warp_sum = (lane < numWarps) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (lane == 0) {
            s_sum[0] = warp_sum;
        }
    }
    __syncthreads();
    
    float sum = s_sum[0];
    
    // Write output
    for (int i = threadIdx.x; i < height; i += BLOCK_SIZE) {
        output[base + i * width] = sdata[i] / sum;
    }
}

// Bitonic sort kernel for stable descending sort along dim=1
// Each block sorts one column of length n (height), n power of two up to 4096.
// Shared memory: values (n floats) and indices (n ints).
// Comparator: if values a < b, swap? We want descending order, so if a < b, we need to move b before a.
// For bitonic sort, we need to define direction. We'll implement standard bitonic sort ascending on (value, index) with value ascending? We can sort ascending by values, then reverse to get descending. But easier: sort descending directly by using appropriate direction in comparators.
// We'll create a function to compare two elements: if (val1 < val2) or (val1 == val2 and idx1 > idx2), then element1 should come after element2 in descending order (since descending: larger value first, for equal value smaller index first). So we need to swap if val1 < val2 or (val1 == val2 and idx1 > idx2).
// In bitonic sort, we have stages with increasing size and direction (ascending/descending). We'll implement with a comparator based on desired order.

template<int BLOCK_SIZE>
__global__ void sort_descending_dim1_kernel_opt(
    const float* __restrict__ input,
    float* __restrict__ values_out,
    long long* __restrict__ indices_out,
    int height, int width) {
    int col = blockIdx.x;
    int width_idx = col % width;
    int base = (col / width) * height * width + width_idx;
    
    extern __shared__ char shared_raw[];
    float* vals = reinterpret_cast<float*>(shared_raw);
    int* idxs = reinterpret_cast<int*>(&shared_raw[height * sizeof(float)]);
    
    // Load column into shared memory
    for (int i = threadIdx.x; i < height; i += BLOCK_SIZE) {
        vals[i] = input[base + i * width];
        idxs[i] = i;
    }
    __syncthreads();
    
    int n = height;
    // Bitonic sort: for size = 2,4,8,...,n
    for (int size = 2; size <= n; size <<= 1) {
        // Direction: we want descending order overall, so for the final merge, direction is descending.
        // For bitonic sort, each sub-block has a direction, but we can simplify by using a comparator that always enforces descending order in each merge step.
        // Actually standard bitonic sort sorts ascending, but we can adjust comparator to descending by swapping if a < b.
        // We'll implement with a single direction (descending) for all stages, which still sorts correctly? Not exactly; bitonic sort requires alternating directions for sub-sequences. However, if we always use descending direction in all merges, the result is still a sorted sequence? It may not be correct. The standard bitonic sort requires specific directions per block. To avoid confusion, we can sort ascending using comparators, then after sort, reverse the order in shared memory before writing out. That's simpler.
        
        // So we'll implement ascending bitonic sort on (value, index) where comparison is: if (vals[j] > vals[i]) or (vals[j] == vals[i] and idxs[j] < idxs[i]) then swap (so smaller value at i, equal value lower index). This sorts ascending by value, and for equal values, ascending by index.
        // Then after sorting, we reverse to make descending by value, and for equal values, we need descending by index? Actually for stable descending sort, equal values should preserve original order, which is ascending index. If we sort ascending by value and then reverse the array, the order for equal values becomes descending index, which is wrong. So we need to handle equal values carefully. Instead, we can sort descending by value directly using a comparator that ensures correct order, and use bitonic sort with correct directions per stage. But implementing direction per stage is doable.
        
        // Let's implement standard bitonic sort as described in many examples, but adapt comparator to descending order and use the standard direction pattern (ascending/descending for sub-sequences).
        // The direction pattern: for each sub-block, directions alternate based on bit. We can follow the typical implementation:
        // for (int k = 2; k <= n; k <<= 1)
        //   for (int j = k >> 1; j > 0; j >>= 1)
        //     for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        //       int ixj = i ^ j;
        //       if (ixj > i) {
        //         if ((i & k) == 0 && vals[i] > vals[ixj]) swap;  // ascending
        //         if ((i & k) != 0 && vals[i] < vals[ixj]) swap;  // descending
        //       }
        //     }
        // This standard pattern sorts ascending overall. To sort descending, we can invert the comparisons: (i & k)==0 -> vals[i] < vals[ixj] swap; (i & k)!=0 -> vals[i] > vals[ixj] swap.
        // But we also need to handle equal values with index tiebreak. We'll define a function to decide swap based on (value, index). We want final order: descending value, and for equal value, ascending index (stable). So for the ascending pattern reversed (descending), we can define comparator:
        // returns true if a should come before b in the final descending order.
        // a should come before b if a.val > b.val or (a.val == b.val and a.idx < b.idx).
        // In the bitonic sort, at any merge step, we compare elements and swap if they are out of order relative to the desired direction for that sub-block.
        // To avoid complexity, we can sort using a simple insertion sort for each column, but height=2048 would be O(4M) per column, too slow.
        
        // Given time constraints, I'll implement a correct but perhaps not optimal sort using a stable merge sort via bitonic sort with tiebreak, using the standard pattern but with a custom comparator that ensures global descending + index ascending order. Since the total order is total, any correct sorting network will produce the same result as stable sort. The standard bitonic sort network is correct for any total order, regardless of direction pattern, as long as the compare-exchange operations respect the total order according to the network's requirements. If we use the standard ascending network but with a comparator that defines the order we want, then the network will sort according to that comparator, yielding the desired sequence. That is, the bitonic sort network is independent of the comparator; it sorts according to a comparator that defines a strict weak ordering. So we can use the standard ascending pattern and define comparator such that "a < b" in the comparator means a should come after b in our desired descending order? Wait, need to map correctly.
        
        // Let's define comparator less(a,b) returning true if a should appear before b in the final sorted order (as we want it). For descending by value and ascending by index, we have less(a,b) = (a.val > b.val) or (a.val == b.val and a.idx < b.idx).
        // In the bitonic sort ascending pattern, the compare-exchange usually does: if ((i & k) == 0 && vals[i] > vals[ixj]) swap; else if ((i & k) != 0 && vals[i] < vals[ixj]) swap. This effectively sorts in ascending order using the natural '<' on values. To sort according to our custom comparator, we can replace the condition with: if ((i & k) == 0 && !less(vals[i], vals[ixj], idxs[i], idxs[ixj])) swap; etc. But the direction pattern in the original is designed for natural ascending order; using a different comparator may not be valid because the network's direction pattern assumes the comparator is consistent with the directions. However, as long as the comparator is a strict weak ordering, the bitonic sort network with the same direction pattern will still sort correctly? I believe the bitonic sort network is correct for any total order if the direction pattern is chosen appropriately for the specific order. If we use the standard pattern but change the comparator, the directions may not align, causing incorrect result. So we need to adjust directions accordingly. To be safe, we can implement a sorting network that is correct for any comparator, like odd-even mergesort, but that is more complex.
        
        // Given the small height (2048) and the number of columns (4096), maybe a simpler approach like using a per-thread merge sort with shared memory would be faster to implement correctly but may still be O(n log^2 n) with bitonic. We can implement bitonic sort with directions derived from the desired order: for descending order, we want the final bitonic sort to be descending. The standard bitonic sort can be adapted by using the standard pattern but with a comparator that flips the order based on direction. Actually, the standard pattern sorts in ascending order; to sort in descending order, we can just apply the same pattern but with comparisons reversed. But the tiebreak remains a challenge.
        
        // Another approach: since height=2048 is not huge, we can use a block-level radix sort on the float keys? But stable sort requirement makes it tricky.
        
        // Given the time, I'll implement a stable counting sort? No, floats.
        
        // Let's step back: maybe we can rely on cub? But cannot use third-party libraries except cuBLAS/cuDNN. So no cub.
        
        // I'll implement a simple merge sort in shared memory using a parallel merge sort algorithm: each block sorts its column using a bottom-up merge sort. The merge sort can be stable if we merge carefully. For n=2048, we can use a single block with 256 threads, and implement merge sort using shared memory and a temporary array. This is doable and stable. Complexity O(n log n), and each merge can be done by multiple threads. Let's implement that.
        
        // Merge sort algorithm:
        // 1. Initially, each thread sorts a small segment (e.g., 8 elements) using insertion sort.
        // 2. Then merge segments of increasing size using parallel merge.
        // This is a common approach and can be stable.
        
        // I'll implement a simpler version: use a cooperative groups? Not necessary.
        
        // For bitonic sort, I can use the known CUDA sample and adapt for stability by including index in comparison. Since the standard bitonic sort is not stable, but if we sort pairs (value, index) with a comparator that orders by value desc and index asc, and the sort network is correct for that total order, then the output will be exactly the stable sorted order. The network's correctness does not depend on the comparator as long as the comparator defines a total order. The direction pattern in bitonic sort is based on the natural order, but if we change the comparator, the direction pattern may not match; however, we can use a sorting network that is independent of direction, like the one in the CUDA sample "bitonicSort" which sorts in ascending order using a fixed pattern. If we want descending, we can sort ascending and then reverse. For stable descending, we can sort ascending by (value, index) with direction bits set appropriately, then reverse the array to get descending by value and descending by index, which is not stable. To get stable descending, we need to sort ascending by value, and for equal values ascending by index, then reverse; that gives descending value but descending index, which is not stable. So that doesn't work.
        
        // Alternatively, sort descending directly with proper direction bits. The standard bitonic sort has direction bits: for each sub-block, if the bit is 0, sort ascending; if 1, sort descending. To sort descending overall, we can set all direction bits to 1 (descending). Then the comparator simply swaps if vals[i] < vals[ixj] (or equal with index condition). This will produce descending order by value, but for equal values, the order is determined by the network, which may not preserve original index order. However, if we include index in the comparison such that for equal values we want lower index to come first (because descending overall? Actually for stable descending, equal values should appear in original order, which is ascending index if original indices are 0..n-1. In descending order, if we have equal values, we want the one with smaller original index to appear first, which is the opposite of what a simple descending comparison would do (default would put larger index first). So we need to define comparator: a should come before b if a.val > b.val or (a.val == b.val and a.idx < b.idx). This is a total order. Then if we sort using a network that sorts according to this comparator, we get the desired stable descending order. The network direction pattern must be consistent with this comparator; we can set directions all to "ascending" relative to this comparator, i.e., in the standard pattern, use "less" as the condition for swapping in ascending direction, and "greater" for descending direction. But since our comparator defines a total order, we can just use the standard ascending pattern with the custom comparator as the "less than" function, and then the result will be sorted in ascending order according to that comparator, which is exactly our desired descending order (since the comparator defines a < b as a should come before b). Then we don't need to reverse. The standard ascending pattern uses conditions like if ((i & k) == 0 && vals[i] > vals[ixj]) swap; else if ((i & k) != 0 && vals[i] < vals[ixj]) swap. Here, comparisons are on values only, but we need to compare using our comparator. So we can replace those conditions with a function that returns true if a is "less than" b according to our comparator (i.e., a should come after b in the final order). But the standard pattern's direction alternation is based on natural numeric order; we need to adapt the direction bits to sort according to our comparator. Since our comparator is just a modification of numeric order (value first, then index), and we want descending value, the direction for each sub-block might be different from natural. However, if we use the standard pattern with our comparator as the "less" function, but keep the direction bits as in the original (alternating based on bit), will it sort correctly? The standard bitonic sort works for any total order if the direction bits are chosen according to the desired final order. For natural ascending order, the direction bits are 0 for all sub-blocks? Actually in the standard pattern, the direction alternates within each merge stage: for k=2, no direction bits; for k=4, the lower half is ascending and upper half descending? Let's recall the original code from CUDA sample:
        // for (unsigned int k = 2; k <= n; k <<= 1) {
        //   for (unsigned int j = k >> 1; j > 0; j >>= 1) {
        //     unsigned int ixj = i ^ j;
        //     if (ixj > i) {
        //       if ((i & k) == 0 && a[i] > a[ixj]) swap;
        //       if ((i & k) != 0 && a[i] < a[ixj]) swap;
        //     }
        //   }
        // }
        // This sorts in ascending order (as shown in many examples). To sort in descending order, we can invert the condition: if ((i & k) == 0 && a[i] < a[ixj]) swap; if ((i & k) != 0 && a[i] > a[ixj]) swap. This works for natural order. But for our custom comparator, we need to define the condition based on the desired order within each sub-block. Since we want descending order overall (according to our comparator), we can use the descending version of the standard pattern and replace comparisons with our comparator. That is, use:
        // if ((i & k) == 0 && comparator_less_than(a[ixj], a[i])) swap; // because for descending, a[i] should be before a[ixj] if a[i] > a[ixj] in natural, but we use custom
        // Actually, let's derive: In the standard ascending code, the swap happens when a[i] > a[ixj] for (i&k)==0, meaning a[i] should be after a[ixj] in ascending order. For descending code, we swap when a[i] < a[ixj] for (i&k)==0. So if we define a comparator `precedes(a,b)` meaning a should come before b in final order, then for descending order, `precedes(a,b) = a > b` (natural). So the condition for swap (i&k)==0 is when !precedes(a[i], a[ixj]). For (i&k)!=0, we swap when precedes(a[i], a[ixj])? Let's check descending pattern: for (i&k)!=0, swap when a[i] > a[ixj]. That is when !precedes(a[i], a[ixj])? Since precedes is >, !precedes is <=. So not consistent. But standard descending pattern is: if ((i & k) == 0 && a[i] < a[ixj]) swap; if ((i & k) != 0 && a[i] > a[ixj]) swap. That sorts descending. So we can use that pattern and replace `a[i] < a[ixj]` with a custom comparison that determines if a[i] should come after a[ixj] in our final order. Let's define `should_swap_descending(a, b)` returning true if a should be placed after b in descending order. In our desired order (value descending, index ascending), a should be after b if a.val < b.val or (a.val == b.val and a.idx > b.idx). So for the descending pattern, for (i&k)==0 we swap if should_swap_descending(a[i], a[ixj]) is true (i.e., a[i] should be after a[ixj]), and for (i&k)!=0 we swap if !should_swap_descending(a[i], a[ixj]) (since that block is expected to be in descending order, but the pattern assumes different direction). Actually, I'm getting confused.
        
        // Simpler: Use the ascending pattern and a comparator that defines our desired order as "less than". If our desired final order is descending by value and ascending by index, then we can define comparator `less(a,b)` = !should_swap_descending(a,b) = (a.val > b.val) or (a.val == b.val and a.idx < b.idx). Then we sort ascending according to this comparator. The standard ascending pattern with natural `<` works; if we replace the comparison with our `less`, will the network produce sorted order according to `less`? I'm not 100% sure because the direction bits are designed for natural ordering, but the bitonic sort network is a fixed sequence of compare-exchange operations that sorts any sequence if the compare-exchange operations are consistent with the comparator (i.e., if the comparator is a total order). The direction pattern in the standard code is one particular way to set directions to achieve sorting. I think if we use the same direction pattern but with a different total order, the network will still sort correctly as long as the direction pattern is updated accordingly. But we can avoid this issue by using an odd-even transposition sort or a bitonic sort with explicit direction setting based on the comparator. However, given that our comparator is just a lexicographic order on (value desc, index asc), we can transform the problem: we can negate the values and treat index as part of key, then sort ascending by (neg_value, index), and then we can use the standard ascending bitonic sort with comparison on these transformed keys. But we need to sort pairs in shared memory.
        
        // Given the complexity, I'll implement a stable merge sort using a single block per column. It is straightforward and guaranteed stable. Height=2048, block 256, we can do merge sort in shared memory with a temp buffer. Performance may be acceptable because we have many blocks and shared memory usage is high but okay.
        
        // I'll implement merge sort with bottom-up approach:
        // Each thread sorts a small chunk (e.g., 8 elements) using insertion sort.
        // Then merge chunks of increasing size.
        // Shared memory layout: vals[height], idxs[height], temp_vals[height], temp_idxs[height]. That is 4*height*4/8 = 32KB for height=2048? Actually 4*2048*4 = 32KB, plus maybe padding, okay under 48KB. We'll use dynamic shared memory and split accordingly.
        
        // Let's implement.
        
        // For stability, when merging, if values are equal, we take from left half first.
        
        // I'll now write the final code below, replacing the unfinished kernel above.
}

// Final code:

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

#define SOFTMAX_BLOCK_SIZE 256
#define SORT_BLOCK_SIZE 256

// Softmax kernel
__global__ void softmax_dim1_kernel(const float* input, float* output, int height, int width) {
    extern __shared__ float smem[];
    float* sdata = smem;
    float* s_red = &smem[height]; // space for reduction (two floats per warp? we'll use two levels)
    int row = blockIdx.x;
    int width_idx = row % width;
    int base = row * height + width_idx;
    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    
    // Load data and compute local max
    float local_max = -CUDART_INF_F;
    for (int i = tid; i < height; i += numThreads) {
        float val = input[base + i * width];
        sdata[i] = val;
        local_max = fmaxf(local_max, val);
    }
    __syncthreads();
    
    // Reduce max across block using shared memory reduction
    // Use warp shuffle and then cross-warp
    int lane = tid & 31;
    int warp = tid >> 5;
    int numWarps = (numThreads + 31) >> 5;
    
    // Warp reduce local_max
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    // Write per-warp max to shared memory
    if (lane == 0) {
        s_red[warp] = local_max;
    }
    __syncthreads();
    
    // Reduce per-warp maxima by first warp
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
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < height; i += numThreads) {
        float e = __expf(sdata[i] - m);
        sdata[i] = e;
        local_sum += e;
    }
    __syncthreads();
    
    // Reduce sum similarly
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
    
    // Write output
    for (int i = tid; i < height; i += numThreads) {
        output[base + i * width] = sdata[i] / sum;
    }
}

// Stable merge sort kernel for descending order along dim=1
// Each block sorts one column of length height.
// Shared memory layout: vals[height], idxs[height], tmp_vals[height], tmp_idxs[height]
__global__ void sort_descending_dim1_kernel(const float* input,
                                            float* values_out,
                                            int64_t* indices_out,
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
    
    // Load column into shared memory
    for (int i = tid; i < height; i += numThreads) {
        vals[i] = input[base + i * width];
        idxs[i] = i;
    }
    __syncthreads();
    
    // Insertion sort for segments of size 16 per thread initially
    int segmentSize = 16;
    int numSegments = (height + segmentSize - 1) / segmentSize;
    // Each thread handles one segment
    if (tid < numSegments) {
        int start = tid * segmentSize;
        int end = min(start + segmentSize, height);
        // insertion sort (stable)
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
        // Each merge is handled by all threads cooperatively: but we can assign one merge per thread block? Simpler: use a single thread per merge? That would be O(n log n) but serial, not good. However, we have many blocks (4096), and each block can have 256 threads, so we can parallelize each merge using multiple threads.
        // We'll implement parallel merge: for each merge, we need to merge two sorted sequences of length curSize each (or less at boundaries). We can have a single thread perform the merge (sequential) while others idle? That would be slow. Better: use a parallel merge algorithm where each thread finds its position via binary search and copies elements.
        // Simplified: since height=2048 and block has 256 threads, we can use a simple approach: for each merge step, we can assign each thread a portion of the output and use binary search to find corresponding input ranges. We'll implement a standard parallel merge using co-ranking.
        // However, for simplicity and to ensure correctness, I'll use a single-thread merge per merge operation but assign each merge to a different thread if there are fewer merges than threads? In early stages, numMerges is large, so many threads can work on different merges. In later stages, numMerges is small, and only a few threads active. That's acceptable because total work is O(n log n) and we have many blocks. For height=2048, segmentSize=16, stages: 16->32,32->64,64->128,128->256,256->512,512->1024,1024->2048. At start, numMerges=128, threads=256, so each merge handled by one thread, good. At later stages, numMerges shrinks, but the work per merge grows. We could still use one thread per merge, but performance degrades. We can instead parallelize each merge using multiple threads. I'll implement a parallel merge using a simple approach: for each merge, we can use all threads in the block to perform the merge, but we need to coordinate. Since there are many merges per stage initially, we can have each thread group handle one merge. At final stages, we can use all threads for one merge.
        // To keep code simpler, I'll implement a sequential merge per merge operation, but assign each merge to a different warp? Actually, let's use a cooperative approach: for each merge, we can have a single thread perform the merge, but we'll have each thread handle one merge; when numMerges < numThreads, only some threads work, others idle, but that's okay because the total number of merge operations across stages is O(n log n / segmentSize) ~ 2048*7/16 ~ 896 per column, and each merge takes O(curSize). Total serial work per column ~ sum over stages of numMerges * curSize = (height/2 + height/2 + ...) = O(height log height) ~ 2048*11 ~ 22k operations per column, times 4096 columns ~ 90 million operations, which is okay on GPU if each operation is simple. With 256 threads per block, and many blocks running concurrently, this may be acceptable for initial implementation.
        int mergeIdx = tid;
        if (mergeIdx < numMerges) {
            int leftStart = mergeIdx * 2 * curSize;
            int mid = min(leftStart + curSize, height);
            int rightEnd = min(leftStart + 2 * curSize, height);
            // Merge vals[leftStart:mid] and vals[mid:rightEnd] into tmp
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
        // Copy back from tmp to vals for the merged segment
        // Since only the threads that performed the merge know the boundaries, we need to copy back the entire affected area. Simpler: after all merges are done for this stage, copy the entire tmp array back to vals? But tmp only has valid data for the merged portions; unmerged portions should remain as they are (from previous stage). We can have all threads copy back the segments that were merged.
        // For each merge, we need to copy from tmp to vals for range [leftStart, rightEnd). We can have threads do that.
        // We'll assign each thread to copy elements from tmp to vals for the ranges that were merged.
        // Since we have the mergeIdx assignment, we can let each thread that performed a merge also copy back its merged segment.
        if (mergeIdx < numMerges) {
            int leftStart = mergeIdx * 2 * curSize;
            int rightEnd = min(leftStart + 2 * curSize, height);
            for (int idx = leftStart + tid; idx < rightEnd; idx += numThreads) {
                vals[idx] = tmp_vals[idx];
                idxs[idx] = tmp_idxs[idx];
            }
        }
        __syncthreads();
    }
    
    // Write output: vals are in descending order (stable), idxs are original indices
    for (int i = tid; i < height; i += numThreads) {
        values_out[base + i * width] = vals[i];
        indices_out[base + i * width] = (int64_t)idxs[i];
    }
}

// Launchers

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
    size_t shared_mem = (height + SOFTMAX_BLOCK_SIZE / 32 + 2) * sizeof(float); // enough for sdata and reduction
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
    size_t shared_mem = 4 * height * (sizeof(float) + sizeof(int)); // vals, idxs, tmp_vals, tmp_idxs (actually we need 2 arrays of float and 2 of int, total 4*height*4? No, 2*height*float + 2*height*int = height*(8+8)=16*height? Actually float 4, int 4, so 2*4*height + 2*4*height = 16*height bytes. For height=2048, 32KB, ok.
    shared_mem = 4 * height * sizeof(float); // overkill but safe? Need exact: we allocate 2 float arrays + 2 int arrays. Use dynamic size from launcher.
    // We'll pass explicit size.
    shared_mem = (2 * height * sizeof(float)) + (2 * height * sizeof(int));
    sort_descending_dim1_kernel<<<grid, block, shared_mem, stream>>>(input, values_out, indices_out, height, width);
}

// End of file
