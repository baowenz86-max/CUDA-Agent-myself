#include <cuda_runtime.h>
#include <float.h>

// Utility to get maximum of two floats
template<typename T>
__device__ inline T device_max(T a, T b) { return a > b ? a : b; }

// Softmax kernel: one block per slice (batch, width). For each block we reduce over the height dimension.
// grid: (batch * width) blocks, each block has blockDim.x threads (<= 1024, typically 256).
// We loop over height dimension, compute max, then exponentials and sum.
// Input and output use the provided strides; for contiguous tensors, input_stride_height = width, input_stride_width = 1, etc.
// The softmax is over dimension dim, which is assumed to be 1 (height). General dim is not implemented; launcher only supports dim==1.
__global__ void softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int height,
    int width,
    int input_stride_batch,
    int input_stride_height,
    int input_stride_width,
    int output_stride_batch,
    int output_stride_height,
    int output_stride_width) {

    int slice_idx = blockIdx.x; // 0 .. batch_size*width-1
    if (slice_idx >= batch_size * width) return;

    int b = slice_idx / width;
    int w = slice_idx - b * width;

    const float* input_slice = input + b * input_stride_batch + w * input_stride_width;
    float* output_slice = output + b * output_stride_batch + w * output_stride_width;

    extern __shared__ float shared[]; // size = blockDim.x * sizeof(float)
    float* smem = shared;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Step 1: find max value along height dimension
    float local_max = -FLT_MAX;
    for (int h = tid; h < height; h += block_size) {
        float val = input_slice[h * input_stride_height];
        local_max = device_max(local_max, val);
    }
    smem[tid] = local_max;
    __syncthreads();

    // Reduce max within block
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = device_max(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = smem[0];

    // Step 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int h = tid; h < height; h += block_size) {
        float val = __expf(input_slice[h * input_stride_height] - max_val);
        local_sum += val;
        output_slice[h * output_stride_height] = val; // store exp(x-max) temporarily
    }
    smem[tid] = local_sum;
    __syncthreads();

    // Reduce sum within block
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float sum = smem[0];

    // Step 3: normalize by sum
    float inv_sum = 1.0f / sum;
    for (int h = tid; h < height; h += block_size) {
        output_slice[h * output_stride_height] *= inv_sum;
    }
}

// Elementwise exp kernel: grid-stride loop
__global__ void exp_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = __expf(input[i]);
    }
}

// Launcher for softmax only (no exp)
extern "C" void softmax_forward_launcher(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t height,
    int64_t width,
    int64_t input_stride_batch,
    int64_t input_stride_height,
    int64_t input_stride_width,
    int64_t output_stride_batch,
    int64_t output_stride_height,
    int64_t output_stride_width,
    int64_t dim,
    cudaStream_t stream) {

    // Only support softmax over dim=1 for this model
    if (dim != 1) {
        // For now, no implementation; binding should ensure dim=1
        return;
    }

    int num_slices = batch_size * width;
    // Choose block size: use 256 threads for good occupancy and reduce over shared memory
    const int block_size = 256;
    int shared_mem_size = block_size * sizeof(float);
    dim3 grid(num_slices);
    softmax_forward_kernel<<<grid, block_size, shared_mem_size, stream>>>(
        input, output,
        (int)batch_size, (int)height, (int)width,
        (int)input_stride_batch, (int)input_stride_height, (int)input_stride_width,
        (int)output_stride_batch, (int)output_stride_height, (int)output_stride_width);
}

// Launcher for softmax followed by exp (fused)
extern "C" void softmax_exp_forward_launcher(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t height,
    int64_t width,
    int64_t input_stride_batch,
    int64_t input_stride_height,
    int64_t input_stride_width,
    int64_t output_stride_batch,
    int64_t output_stride_height,
    int64_t output_stride_width,
    int64_t dim,
    cudaStream_t stream) {

    // First compute softmax into output, then apply exp in-place using same buffer.
    // Since output is contiguous and we are only doing elementwise exp, we can reuse output as input for exp.
    softmax_forward_launcher(input, output, batch_size, height, width,
                             input_stride_batch, input_stride_height, input_stride_width,
                             output_stride_batch, output_stride_height, output_stride_width,
                             dim, stream);

    int64_t numel = batch_size * height * width;
    int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    exp_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        output, output, numel);
}

// Launcher for elementwise exp only
extern "C" void exp_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream) {

    int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    exp_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, numel);
}
