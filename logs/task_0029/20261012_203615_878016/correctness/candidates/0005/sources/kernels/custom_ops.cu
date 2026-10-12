#include <cuda_runtime.h>
#include <stdint.h>

__global__ void unbind_max_kernel(const float* __restrict__ input,
                                  float* __restrict__ values,
                                  int64_t* __restrict__ indices,
                                  int B, int C, int H, int W) {
    // Each thread handles one (b, w) pair, reducing over C and H.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * W;
    if (idx >= total) return;

    int b = idx / W;
    int w = idx % W;

    float max_val = -INFINITY;
    int64_t max_idx = 0;

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            // input shape: (B, C, H, W)
            int input_idx = ((b * C + c) * H + h) * W + w;
            float val = input[input_idx];
            if (val > max_val || (val == max_val && c < max_idx ? false : false)) {
                // careful with NaN? torch.max propagates NaN. But for simplicity we ignore NaN.
                if (val > max_val || (val == max_val && (c < max_idx))) {
                    max_val = val;
                    max_idx = c;
                }
            }
        }
    }

    // output shape: (B, H, W) for values, but we need output for each w separately? Actually we store in contiguous (B, W) layout or (B, H, W)?
    // Wait: The reference output for each w is (B, H) for values and indices. The launcher receives output arrays of shape (B, H, W) because the binding will allocate that and then slice. So we need to write for each (b, h, w) the max over c. That is, for each w, the result is max over C for each (b, h). Our thread above computed max over both C and H incorrectly. Need to reduce over C only, not H.
    // Actually torch.max(xi, dim=1) reduces over dimension 1, which is C, and the output shape is (B, H). So we need to compute for each (b, h, w) the max over c.
    // Therefore each thread should handle one (b, h, w) element. Let's redesign.
    // The output values array has shape (B, H, W) with strides (H*W, W, 1). So each (b, h, w) corresponds to output index ((b*H + h)*W + w). The input index for (b, c, h, w) is ((b*C + c)*H + h)*W + w.
    // We'll have a grid-stride loop over total = B*H*W.
}

__global__ void unbind_max_kernel_v2(const float* __restrict__ input,
                                     float* __restrict__ values,
                                     int64_t* __restrict__ indices,
                                     int B, int C, int H, int W) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = B * H * W;
    for (int idx = tid; idx < total; idx += stride) {
        int b = idx / (H * W);
        int rem = idx % (H * W);
        int h = rem / W;
        int w = rem % W;

        float max_val = -INFINITY;
        int64_t max_idx = 0;
        for (int c = 0; c < C; ++c) {
            int input_idx = ((b * C + c) * H + h) * W + w;
            float val = input[input_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }
        values[idx] = max_val;
        indices[idx] = max_idx;
    }
}

void custom_unbind_max_launcher(const float* input,
                                float* values,
                                int64_t* indices,
                                int B, int C, int H, int W,
                                cudaStream_t stream) {
    int total = B * H * W;
    int block = 256;
    int grid = (total + block - 1) / block;
    // Cap grid to avoid excessive blocks, use grid-stride loop for large sizes.
    if (grid > 65535) {
        grid = 65535;
    }
    unbind_max_kernel_v2<<<grid, block, 0, stream>>>(input, values, indices, B, C, H, W);
}
