#include <cuda_runtime.h>

// Elementwise leaky ReLU kernel.
// Each thread processes one element using a grid-stride loop.
__global__ void leaky_relu_kernel(float* output, const float* input, float negative_slope, long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        float val = input[i];
        output[i] = (val > 0.0f) ? val : (val * negative_slope);
    }
}

// Add bias kernel with broadcasting: output[n,c,h,w] = input[n,c,h,w] + bias[c].
// The input is assumed to be a 4D tensor with dimensions [N, C, H, W].
// The bias is of shape [C, 1, 1].
__global__ void add_bias_kernel(float* output, const float* input, const float* bias, int C, int HW, long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        // Determine channel index from linear index.
        // total_elements = N*C*H*W, but we use channel = (i / HW) % C.
        int c = (int)((i / HW) % C);
        output[i] = input[i] + bias[c];
    }
}

// Launcher for leaky_relu_forward.
// Parameters: output, input pointers, negative_slope, total number of elements, stream.
extern "C" void leaky_relu_launcher(float* output, const float* input, float negative_slope, long long total_elements, cudaStream_t stream) {
    const int block_size = 256;
    long long blocks = (total_elements + block_size - 1) / block_size;
    // Cap blocks to 65535 for grid dimension limit; grid-stride loop handles the rest.
    if (blocks > 65535) blocks = 65535;
    leaky_relu_kernel<<<(int)blocks, block_size, 0, stream>>>(output, input, negative_slope, total_elements);
}

// Launcher for add_bias_forward.
// Parameters: output, input, bias pointers, C, HW (height*width), total_elements, stream.
extern "C" void add_bias_launcher(float* output, const float* input, const float* bias, int C, int HW, long long total_elements, cudaStream_t stream) {
    const int block_size = 256;
    long long blocks = (total_elements + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    add_bias_kernel<<<(int)blocks, block_size, 0, stream>>>(output, input, bias, C, HW, total_elements);
}
