#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Softmax along dimension 1 of a 3D tensor with shape [outer, inner, innermost]
// For input tensor x with strides sx0, sx1, sx2 and dim=1, compute softmax over the inner dimension.
// For general dim support, this kernel assumes dim=1 and 3D, but can be adapted.
// The tensor is contiguous in most cases, but we support non-contiguous strides.
// kernel processes each row (outer*innermost rows) individually.

__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int outer,
                               int inner,
                               int innermost,
                               long long stride0,
                               long long stride1,
                               long long stride2)
{
    // total number of rows = outer * innermost
    int total_rows = outer * innermost;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int row = tid; row < total_rows; row += stride) {
        int o = row / innermost;
        int w = row % innermost;

        // base offset for this row at innermost index w
        const float* row_ptr = input + o * stride0 + w * stride2;
        float* out_row_ptr = output + o * stride0 + w * stride2;

        // Find max in this row (over dim=1)
        float max_val = -1e30f;
        for (int h = 0; h < inner; ++h) {
            float val = row_ptr[h * stride1];
            if (val > max_val) max_val = val;
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int h = 0; h < inner; ++h) {
            float val = __expf(row_ptr[h * stride1] - max_val);
            sum += val;
            // we can store exp values temporarily? Not needed, but to avoid recompute later,
            // we can write to output and then normalize after sum, but that requires another pass.
            // Since we have to read input again anyway, recompute exp for normalization.
            // Alternatively, use shared memory for temporary exp, but rows are independent.
            // For performance, we can keep recompute: outer loop for max, then exp+sum, then another exp+divide.
            // That's 3 passes over the row; acceptable for inner=2048.
            // We'll store exp in output temporarily ? Not safe unless we then overwrite all elements.
            // Simpler: recompute exp in third pass.
        }

        // Second pass to compute sum (we need sum before writing normalized)
        // Actually we can combine: first find max, second compute sum using exp, third write normalized.
        // We need sum before writing, so third pass required.
        sum = 0.0f;
        for (int h = 0; h < inner; ++h) {
            sum += __expf(row_ptr[h * stride1] - max_val);
        }

        // Third pass: write normalized values
        float inv_sum = 1.0f / sum;
        for (int h = 0; h < inner; ++h) {
            out_row_ptr[h * stride1] = __expf(row_ptr[h * stride1] - max_val) * inv_sum;
        }
    }
}

// Sub scalar kernel: output = input - scalar, elementwise
__global__ void sub_scalar_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  long long total_elements,
                                  float scalar)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total_elements; idx += stride) {
        output[idx] = input[idx] - scalar;
    }
}

// Launcher for custom_softmax
// Parameters:
//   input: pointer to input tensor data (float)
//   output: pointer to output tensor data (float, allocated by caller)
//   outer, inner, innermost: dimensions of 3D tensor (contiguous logical)
//   stride0, stride1, stride2: strides in elements (type long long)
//   stream: CUDA stream
// Contract: dim is assumed to be 1 (as used in model_new.py).
extern "C" void custom_softmax_launcher(const float* input,
                                        float* output,
                                        int outer,
                                        int inner,
                                        int innermost,
                                        long long stride0,
                                        long long stride1,
                                        long long stride2,
                                        cudaStream_t stream)
{
    int total_rows = outer * innermost;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    // Limit blocks to avoid excessive grid size (max 65535 in x dimension)
    if (blocks > 65535) blocks = 65535;
    // If total_rows could be huge, grid-stride loop handles it.
    softmax_kernel<<<blocks, threads, 0, stream>>>(input, output, outer, inner, innermost,
                                                  stride0, stride1, stride2);
}

// Launcher for custom_sub_scalar
// Parameters:
//   input: pointer to input tensor data (float)
//   output: pointer to output tensor data (float, allocated by caller)
//   total_elements: total number of elements
//   scalar: value to subtract (float)
//   stream: CUDA stream
extern "C" void custom_sub_scalar_launcher(const float* input,
                                           float* output,
                                           long long total_elements,
                                           float scalar,
                                           cudaStream_t stream)
{
    int threads = 256;
    long long blocks_ll = (total_elements + threads - 1) / threads;
    int blocks = (int)min((long long)65535, blocks_ll);
    sub_scalar_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements, scalar);
}

// Note: For simplicity, the launchers do not include error checking here;
// the binding code can call cudaGetLastError after launch. But since we are only
// producing .cu, we can add cudaGetLastError and cudaStreamSynchronize? Not necessary.
// We will assume the binding wrapper handles error checking.
