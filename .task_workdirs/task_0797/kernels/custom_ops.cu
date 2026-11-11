#include <cuda_runtime.h>

// isin_custom kernel: output[i] = true if x[i] matches any element in elements
__global__ void isin_kernel(const float* __restrict__ x, const float* __restrict__ elements,
                            bool* __restrict__ output, int64_t numel, int64_t elements_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = x[i];
        bool found = false;
        for (int64_t j = 0; j < elements_size; ++j) {
            if (val == elements[j]) {
                found = true;
                break;
            }
        }
        output[i] = found;
    }
}

// split_with_sizes kernel: copy input block to output, assumes dim=1 contiguous chunks
// For general case we use 3D indexing: (batch, channel, spatial)
// Here we hardcode for dim=1 split, but we can keep generic enough for the contract
// Since dim is always 1 in the contract, we assume x shape [N, C, H, W]
// and split along C. Each output is [N, split_size, H, W] and is contiguous.
// We copy N * split_size * H * W elements from input offset.
__global__ void split_copy_kernel(const float* __restrict__ input, float* __restrict__ output,
                                   int64_t total_copy, int64_t input_offset) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_copy; i += stride) {
        output[i] = input[input_offset + i];
    }
}

// Launcher for isin_custom
// x: input tensor pointer, contiguous or not (we assume contiguous as per contract?)
// elements: elements tensor pointer (contiguous)
// output: output bool tensor pointer (same shape as x)
// numel: total number of elements in x
// elements_size: number of elements in elements
// stream: CUDA stream
extern "C" void isin_custom_launcher(const float* x, const float* elements, bool* output,
                                    int64_t numel, int64_t elements_size, cudaStream_t stream) {
    if (numel <= 0) return;
    const int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid limit (not strictly needed, but safe)
    isin_kernel<<<blocks, threads, 0, stream>>>(x, elements, output, numel, elements_size);
}

// Launcher for split_with_sizes_custom
// We receive x pointer, and we need to copy into multiple output tensors.
// Since binding layer will allocate outputs, we receive pointers to each output
// and their sizes. We provide a vector of pointers and offsets.
// For simplicity with raw pointers, we design a launcher that takes:
//   const float* x
//   int64_t num_outputs
//   const float** outputs  (array of output pointers)
//   const int64_t* output_numels
//   const int64_t* input_offsets
//   int64_t total_elements_per_split? Actually we need split sizes, but we can pass offsets
// We'll define a C++ struct? But .cu cannot include vector? We can use raw arrays.
// Instead, we can have a simpler launcher that takes a single output pointer and copy size?
// However, binding must call multiple times or we can define a launcher that handles all outputs.
// Since contract says split_sizes is int64_t (list), we need to pass list via C-style array.
// Use raw pointers to arrays.
extern "C" void split_with_sizes_custom_launcher(
    const float* x,
    int64_t num_outputs,
    float** outputs,
    int64_t* output_numels,
    int64_t* input_offsets,
    cudaStream_t stream) {
    if (num_outputs <= 0) return;
    const int threads = 256;
    for (int64_t out_idx = 0; out_idx < num_outputs; ++out_idx) {
        int64_t total = output_numels[out_idx];
        if (total <= 0) continue;
        int blocks = (int)((total + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;
        split_copy_kernel<<<blocks, threads, 0, stream>>>(
            x, outputs[out_idx], total, input_offsets[out_idx]);
    }
}
