#include <cuda_runtime.h>
#include <cstdint>

__global__ void fused_pad_rotate_conv_maskinf_addbias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ add_bias,
    float* __restrict__ output,
    int N,
    int C,
    int H,
    int W,
    int K,
    int R,
    int S,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right,
    int outH,
    int outW,
    float pad_value
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int n = bz / K;
    int k = bz % K;

    int ow = bx * blockDim.x + tx;
    int oh = by * blockDim.y + ty;

    if (ow >= outW || oh >= outH) {
        return;
    }

    float sum = 0.0f;

    extern __shared__ float smem[];

    int ow_start = bx * blockDim.x;
    int ow_end = min(ow_start + blockDim.x - 1, outW - 1);
    int oh_start = by * blockDim.y;
    int oh_end = min(oh_start + blockDim.y - 1, outH - 1);

    int W_pad = W + pad_left + pad_right;
    int H_pad = H + pad_top + pad_bottom;

    int w_pad_min = ow_start;
    int w_pad_max = ow_end + S - 1;
    if (w_pad_max > W_pad - 1) w_pad_max = W_pad - 1;
    if (w_pad_min > W_pad - 1) w_pad_min = W_pad - 1;

    int h_pad_max = H_pad - 1 - oh_start;
    int h_pad_min = H_pad - 1 - (oh_end + R - 1);
    if (h_pad_max > H_pad - 1) h_pad_max = H_pad - 1;
    if (h_pad_min < 0) h_pad_min = 0;
    if (h_pad_min > H_pad - 1) h_pad_min = H_pad - 1;
    if (h_pad_max < 0) h_pad_max = 0;

    int tile_h = h_pad_max - h_pad_min + 1;
    int tile_w = w_pad_max - w_pad_min + 1;

    int max_tile_h = blockDim.y + R - 1;
    int max_tile_w = blockDim.x + S - 1;
    int smem_stride = max_tile_w;
    int smem_plane = max_tile_h * max_tile_w;

    int total_load = C * tile_h * tile_w;
    int tid = ty * blockDim.x + tx;
    int nthreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < total_load; idx += nthreads) {
        int c = idx / (tile_h * tile_w);
        int rem = idx % (tile_h * tile_w);
        int local_h = rem / tile_w;
        int local_w = rem % tile_w;
        int h_idx = h_pad_min + local_h;
        int w_idx = w_pad_min + local_w;

        float val;
        if (h_idx < pad_top || h_idx >= pad_top + H || w_idx < pad_left || w_idx >= pad_left + W) {
            val = pad_value;
        } else {
            int h_in = h_idx - pad_top;
            int w_in = w_idx - pad_left;
            val = input[((n * C + c) * H + h_in) * W + w_in];
        }
        smem[c * smem_plane + local_h * smem_stride + local_w] = val;
    }

    __syncthreads();

    sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            int h_idx_rot = H_pad - 1 - (oh + r);
            int local_h = h_idx_rot - h_pad_min;
            for (int s = 0; s < S; ++s) {
                int w_idx_rot = ow + s;
                int local_w = w_idx_rot - w_pad_min;
                float input_val = smem[c * smem_plane + local_h * smem_stride + local_w];
                float w_val = weight[((k * C + c) * R + r) * S + s];
                sum += input_val * w_val;
            }
        }
    }

    sum += conv_bias[k];

    if (isinf(sum)) {
        sum = 0.0f;
    }

    sum += add_bias[k];

    output[((n * K + k) * outH + oh) * outW + ow] = sum;
}

extern "C" void fused_pad_rotate_conv_maskinf_addbias_launcher(
    const float* input,
    const float* weight,
    const float* conv_bias,
    const float* add_bias,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int K,
    int R,
    int S,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right,
    float pad_value,
    cudaStream_t stream
) {
    int outH = H + pad_top + pad_bottom - R + 1;
    int outW = W + pad_left + pad_right - S + 1;

    const int BLOCK_X = 16;
    const int BLOCK_Y = 16;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((outW + BLOCK_X - 1) / BLOCK_X,
              (outH + BLOCK_Y - 1) / BLOCK_Y,
              N * K);

    int max_tile_h = BLOCK_Y + R - 1;
    int max_tile_w = BLOCK_X + S - 1;
    size_t smem_size = (size_t)C * max_tile_h * max_tile_w * sizeof(float);

    fused_pad_rotate_conv_maskinf_addbias_kernel<<<grid, block, smem_size, stream>>>(
        input, weight, conv_bias, add_bias, output,
        N, C, H, W, K, R, S,
        pad_top, pad_bottom, pad_left, pad_right,
        outH, outW, pad_value
    );
}
