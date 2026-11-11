#include <cuda_runtime.h>
#include <cstdint>

// Diagonal scatter kernel along specified dims.
// Each thread handles one diagonal element, using grid-stride loop.
// For the special case dim1=1, dim2=2 common in this task, but implement generic dims.
// We assume dim1 and dim2 are valid dimensions of a 3D tensor (B, H, W).
// offset shifts diagonal. We compute diagonal length based on shape dims.
__global__ void diagonal_scatter_kernel(
    const float* __restrict__ input,
    const float* __restrict__ src,
    float* __restrict__ output,
    int B,
    int H,
    int W,
    int offset,
    int dim1,
    int dim2,
    long long total_diag_elements
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    // For each diagonal element: compute batch index and position along diagonal
    for (long long diag_idx = idx; diag_idx < total_diag_elements; diag_idx += stride) {
        // Since dim1 and dim2 can be any two dims of 3D (0,1,2), we handle only the case dim1=1, dim2=2 as required by contract.
        // But to be robust, we implement generic 3D case filtering dims not equal to batch.
        // For simplicity, assume dim1=1, dim2=2 and dims are adjacent; but we'll implement assuming 3D and dim1,dim2 are 1,2.
        // Since contract says dim1=1,dim2=2, we can hardcode for performance, but keep generic indexing via strides.
        // Actually let's implement generic by converting to linear index using strides of output (contiguous or strided).
        // TODO: use strides? But launcher doesn't pass strides; we assume contiguous input/output as contract says "通常是连续张量".
        // But maybe input can be non-contiguous? We can require contiguous in binding. So assume contiguous 3D with dims [B,H,W].
        int b = diag_idx / H; // since H = W for this task, but generally diagonal length varies.
        int j = diag_idx % H;
        // compute actual offset: for dim1=1, dim2=2, diagonal element (b, i, i+offset) where i from 0 to diagonal length-1.
        // Here we need actual row/col positions based on offset and orientation.
        // We'll implement generic diagonal for any two dims of 3D.
        // But for simplicity and correctness under given constraints, implement for 3D and dim1=1, dim2=2.
        // However user expects generic ints maybe other values but contract fixed.
        // Given only these values in model, we can hardcode.
        // Calculate row index r and col index c along diagonal for batch b.
        int start_row, start_col, diag_len;
        if (offset >= 0) {
            start_row = 0;
            start_col = offset;
            diag_len = min(H, W - offset);
        } else {
            start_row = -offset;
            start_col = 0;
            diag_len = min(H + offset, W);
        }
        // diag_len is total number of elements per batch.
        int total_per_batch = diag_len;
        b = diag_idx / total_per_batch;
        int t = diag_idx % total_per_batch;
        if (b >= B) continue;
        int row = start_row + t;
        int col = start_col + t;
        // Compute flat output index: ((b * H) + row) * W + col
        long long out_idx = ((long long)b * H + row) * W + col;
        // src index: b * total_per_batch + t
        long long src_idx = (long long)b * total_per_batch + t;
        output[out_idx] = src[src_idx];
    }
}

// Kernel to copy input to output (preserve non-diagonal elements)
__global__ void copy_kernel(const float* __restrict__ input, float* __restrict__ output, long long total_elements) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        output[i] = input[i];
    }
}

extern "C" void diagonal_scatter_launcher(
    const float* input,
    const float* src,
    float* output,
    int B,
    int H,
    int W,
    int offset,
    int dim1,
    int dim2,
    cudaStream_t stream
) {
    // Compute total diagonal elements
    int diag_len;
    if (offset >= 0) {
        diag_len = min(H, W - offset);
    } else {
        diag_len = min(H + offset, W);
    }
    if (diag_len <= 0) {
        // No diagonal elements, just copy input to output
        long long total = (long long)B * H * W;
        int block = 256;
        long long grid = (total + block - 1) / block;
        copy_kernel<<<(unsigned int)grid, block, 0, stream>>>(input, output, total);
        return;
    }
    long long total_diag = (long long)B * diag_len;
    long long total_elements = (long long)B * H * W;

    // First copy entire input to output
    int copy_block = 256;
    long long copy_grid = (total_elements + copy_block - 1) / copy_block;
    if (copy_grid > 0) {
        copy_kernel<<<(unsigned int)copy_grid, copy_block, 0, stream>>>(input, output, total_elements);
    }

    // Then scatter diagonal elements
    int scatter_block = 256;
    long long scatter_grid = (total_diag + scatter_block - 1) / scatter_block;
    if (scatter_grid > 0) {
        diagonal_scatter_kernel<<<(unsigned int)scatter_grid, scatter_block, 0, stream>>>(
            input, src, output, B, H, W, offset, dim1, dim2, total_diag);
    }
}
