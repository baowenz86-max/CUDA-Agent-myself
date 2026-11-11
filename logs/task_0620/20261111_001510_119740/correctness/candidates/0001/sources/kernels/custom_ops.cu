// kernels/custom_ops.cu
// CUDA kernels for hardtanh and avg_pool3d
// This file contains only CUDA kernels and C-style launchers with raw pointers,
// no PyTorch/ATen dependencies.
#include <cuda_runtime.h>
#include <stdint.h>

// Elementwise hardtanh: out = min(max(x, min_val), max_val)
__global__ void hardtanh_kernel(float* output, const float* input, int64_t numel, float min_val, float max_val) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_total = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride_total) {
        float val = input[i];
        val = fminf(fmaxf(val, min_val), max_val);
        output[i] = val;
    }
}

// AvgPool3d kernel with arbitrary strides.
// input_strides and output_strides are in elements, length 5: [N_stride, C_stride, D_stride, H_stride, W_stride]
// We assume contiguous or arbitrary but valid strides.
__global__ void avg_pool3d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t outD, int64_t outH, int64_t outW,
    int64_t kernel_size, int64_t stride, int64_t padding,
    int64_t* __restrict__ input_strides,  // length 5
    int64_t* __restrict__ output_strides  // length 5
) {
    int64_t total = N * C * outD * outH * outW;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_total = blockDim.x * gridDim.x;
    for (int64_t linear_idx = idx; linear_idx < total; linear_idx += stride_total) {
        // Decode linear index to output coordinates
        int64_t ow = linear_idx % outW;
        int64_t oh = (linear_idx / outW) % outH;
        int64_t od = (linear_idx / (outW * outH)) % outD;
        int64_t c  = (linear_idx / (outW * outH * outD)) % C;
        int64_t n  = linear_idx / (outW * outH * outD * C);

        // Base input coordinates corresponding to output (od, oh, ow) with padding shift
        int64_t in_d_start = od * stride - padding;
        int64_t in_h_start = oh * stride - padding;
        int64_t in_w_start = ow * stride - padding;

        float sum = 0.0f;
        int count = 0;
        for (int64_t kd = 0; kd < kernel_size; ++kd) {
            int64_t id = in_d_start + kd;
            if (id < 0 || id >= D) continue;
            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                int64_t ih = in_h_start + kh;
                if (ih < 0 || ih >= H) continue;
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t iw = in_w_start + kw;
                    if (iw < 0 || iw >= W) continue;
                    // Compute linear offset using input_strides
                    int64_t offset = n * input_strides[0] + c * input_strides[1]
                                   + id * input_strides[2] + ih * input_strides[3] + iw * input_strides[4];
                    sum += input[offset];
                    count++;
                }
            }
        }
        // Since padding=0 in this contract, count should always equal kernel_size^3.
        // But to be safe, handle count > 0; divisibility guaranteed by PyTorch semantics.
        float avg = (count > 0) ? (sum / (float)count) : 0.0f;
        // Write output at linear offset computed using output_strides
        int64_t out_offset = n * output_strides[0] + c * output_strides[1]
                            + od * output_strides[2] + oh * output_strides[3] + ow * output_strides[4];
        output[out_offset] = avg;
    }
}

// Launcher for hardtanh
extern "C" void hardtanh_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream
) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Grid-stride loop allows any grid size, but cap at some sane limit to avoid too many blocks.
    if (grid_size > 65535) grid_size = 65535;
    hardtanh_kernel<<<grid_size, block_size, 0, stream>>>(output, input, numel, min_val, max_val);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In launcher we cannot throw; but binding will check after call.
        // We just ignore or could print. Better to propagate through return? 
        // We'll leave as is; binding should check cudaGetLastError after calling launcher.
    }
}

// Launcher for avg_pool3d
extern "C" void avg_pool3d_launcher(
    float* output,
    const float* input,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t outD, int64_t outH, int64_t outW,
    int64_t kernel_size, int64_t stride, int64_t padding,
    int64_t* input_strides,
    int64_t* output_strides,
    cudaStream_t stream
) {
    int64_t total = N * C * outD * outH * outW;
    if (total <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    avg_pool3d_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input,
        N, C, D, H, W,
        outD, outH, outW,
        kernel_size, stride, padding,
        input_strides, output_strides
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // binding will check
    }
}
