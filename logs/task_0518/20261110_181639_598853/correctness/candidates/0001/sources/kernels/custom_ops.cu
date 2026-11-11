#include <cuda_runtime.h>
#include <math.h>

// Stable softmin kernel: softmin(x, dim) = softmax(-x, dim)
// Input x is contiguous of shape [num_rows, feat_dim]
// Each thread computes one row's softmin.
__global__ void softmin_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int64_t num_rows,
                               int64_t feat_dim) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t r = row; r < num_rows; r += stride) {
        const float* row_in = input + r * feat_dim;
        float* row_out = output + r * feat_dim;

        // Find max of -x for numerical stability
        float max_val = -row_in[0];
        for (int64_t j = 1; j < feat_dim; ++j) {
            float val = -row_in[j];
            max_val = fmaxf(max_val, val);
        }

        // Compute exp(-x - max) and sum
        float sum = 0.0f;
        for (int64_t j = 0; j < feat_dim; ++j) {
            float val = expf(-row_in[j] - max_val);
            row_out[j] = val;  // temporarily store
            sum += val;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (int64_t j = 0; j < feat_dim; ++j) {
            row_out[j] *= inv_sum;
        }
    }
}

// Launcher: assumes contiguous float32 input/output, feat_dim > 0.
// num_rows = total number of rows (e.g., batch * seq_len).
extern "C" void softmin_forward_launcher(const float* input,
                                         float* output,
                                         int64_t num_rows,
                                         int64_t feat_dim,
                                         cudaStream_t stream) {
    if (num_rows == 0 || feat_dim == 0) return;

    const int block_size = 256;
    // Grid size: enough to cover rows, but not exceed max grid dimension (65535 in x)
    int64_t grid_size = (num_rows + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;

    softmin_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, num_rows, feat_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, we would throw; here we ignore since no exception support.
        // The binding layer should check cudaGetLastError after launcher returns.
    }
}
