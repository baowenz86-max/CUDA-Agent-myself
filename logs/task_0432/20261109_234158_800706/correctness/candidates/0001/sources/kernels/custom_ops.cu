#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// max_dim1_indices_launcher:
//   Computes values and indices of maximum along dimension 1 (channels),
//   keeping the dimension. Input x is 4D float32 tensor of shape [N, C, H, W]
//   with arbitrary strides. Outputs:
//     values: float32 [N, 1, H, W]
//     indices: int64 [N, 1, H, W]
//   The implementation uses one thread per output element (n, 0, h, w).
//   Each thread iterates over all channels and finds max value and its index.
//   The index is in [0, C-1].
//   Memory accesses may be non-coalesced depending on strides, but this matches
//   PyTorch semantics. The launcher assumes x is on current CUDA device and
//   pointers are valid.
// -----------------------------------------------------------------------------
__global__ void max_dim1_indices_kernel(
    const float* __restrict__ input,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_h,
    int64_t stride_w,
    int64_t out_stride_n,
    int64_t out_stride_h,
    int64_t out_stride_w)
{
    int64_t total = N * H * W;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        // Decode n, h, w from flattened index
        int64_t tmp = idx;
        int64_t w = tmp % W;
        tmp /= W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t n = tmp;

        const float* base = input + n * stride_n + h * stride_h + w * stride_w;
        float max_val = base[0];
        int64_t max_idx = 0;

        for (int64_t c = 1; c < C; ++c) {
            float val = base[c * stride_c];
            // PyTorch's torch.max returns the first maximal index when there are ties.
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }

        // Output layout: [N, 1, H, W] contiguous as produced by torch.max with keepdim=True.
        int64_t out_idx = n * out_stride_n + h * out_stride_h + w * out_stride_w;
        values[out_idx] = max_val;
        indices[out_idx] = max_idx;
    }
}

// Launcher for max_dim1_indices. Input and output pointers are device pointers.
// All strides are element counts, not bytes.
extern "C" void max_dim1_indices_launcher(
    const float* input,
    float* values,
    int64_t* indices,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t input_stride_n,
    int64_t input_stride_c,
    int64_t input_stride_h,
    int64_t input_stride_w,
    int64_t output_stride_n,
    int64_t output_stride_h,
    int64_t output_stride_w,
    cudaStream_t stream)
{
    int64_t total = N * H * W;
    // Use a reasonable number of blocks. Grid-stride loop handles remainder.
    int block_size = 256;
    int64_t blocks = (total + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535; // grid cap
    if (blocks == 0) return; // no output elements
    max_dim1_indices_kernel<<<blocks, block_size, 0, stream>>>(
        input, values, indices, N, C, H, W,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_h, output_stride_w);
}

// -----------------------------------------------------------------------------
// one_hot_from_indices_launcher:
//   Converts indices tensor of shape [N, 1, H, W] (int64) to one-hot tensor of
//   shape [N, 1, H, W, num_classes] (int64). The indices input is assumed to
//   have exactly 4 dimensions. The output is a new contiguous tensor with same
//   batch/spatial dimensions plus an extra class dimension at the end.
//   Each thread writes one entire one-hot vector for one spatial position,
//   filling all num_classes entries to 0 except the entry at the class index.
//   The output layout is contiguous: [N, 1, H, W, num_classes] with strides
//   (1*H*W*num_classes, H*W*num_classes, W*num_classes, num_classes, 1).
//   If indices contain out-of-range values (>= num_classes), PyTorch's
//   F.one_hot raises error; we assume valid input due to earlier validation.
// -----------------------------------------------------------------------------
__global__ void one_hot_from_indices_kernel(
    const int64_t* __restrict__ indices,
    int64_t* __restrict__ output,
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t num_classes,
    int64_t indices_stride_n,
    int64_t indices_stride_h,
    int64_t indices_stride_w)
{
    int64_t spatial_total = N * H * W;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < spatial_total; idx += blockDim.x * gridDim.x) {
        int64_t tmp = idx;
        int64_t w = tmp % W;
        tmp /= W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t n = tmp;

        // Read the class index from indices tensor. Indices shape [N,1,H,W].
        int64_t class_idx = indices[n * indices_stride_n + h * indices_stride_h + w * indices_stride_w];

        // Output base pointer for this (n,h,w) position.
        // Output is contiguous [N,1,H,W,num_classes], flatten to [N*H*W*num_classes].
        int64_t out_base = ((n * H + h) * W + w) * num_classes;

        // Initialize all classes to 0, then set one position to 1.
        for (int64_t c = 0; c < num_classes; ++c) {
            output[out_base + c] = (c == class_idx) ? 1 : 0;
        }
    }
}

extern "C" void one_hot_from_indices_launcher(
    const int64_t* indices,
    int64_t* output,
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t num_classes,
    int64_t indices_stride_n,
    int64_t indices_stride_h,
    int64_t indices_stride_w,
    cudaStream_t stream)
{
    int64_t spatial_total = N * H * W;
    int block_size = 256;
    int64_t blocks = (spatial_total + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return;
    one_hot_from_indices_kernel<<<blocks, block_size, 0, stream>>>(
        indices, output, N, H, W, num_classes,
        indices_stride_n, indices_stride_h, indices_stride_w);
}
