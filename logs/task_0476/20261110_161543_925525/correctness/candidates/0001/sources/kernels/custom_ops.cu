#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// CUDA kernel for ReflectionPad2d
// grid: (out_numel + block - 1) / block
template <typename T>
__global__ void reflection_pad2d_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int64_t pad_left, int64_t pad_right, int64_t pad_top, int64_t pad_bottom,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;
    if (idx >= total) return;

    // decompose idx into n,c,ho,wo
    int64_t tmp = idx;
    int64_t wo = tmp % W_out;
    tmp /= W_out;
    int64_t ho = tmp % H_out;
    tmp /= H_out;
    int64_t c = tmp % C;
    tmp /= C;
    int64_t n = tmp;

    // reflect without repeating edge: if pad >= dimension, need special handling
    int64_t hi = ho - pad_top;
    if (hi < 0) {
        // reflect: hi = -hi - 1 if pad_top <= H_in - 1? Actually PyTorch reflection pad repeats edge when pad > dim-1
        // But contract says pad < dimension, so simple reflect works
        // General robust: repeat edge if pad >= dimension
        if (pad_top >= H_in) {
            hi = -hi - 1; // still reflects into negative? We'll clamp
        }
        hi = -hi - 1;
        if (hi < 0) hi = 0;
        else if (hi >= H_in) hi = H_in - 1;
    } else if (hi >= H_in) {
        hi = 2 * H_in - hi - 2;
        if (hi < 0) hi = 0;
        else if (hi >= H_in) hi = H_in - 1;
    }

    int64_t wi = wo - pad_left;
    if (wi < 0) {
        wi = -wi - 1;
        if (wi < 0) wi = 0;
        else if (wi >= W_in) wi = W_in - 1;
    } else if (wi >= W_in) {
        wi = 2 * W_in - wi - 2;
        if (wi < 0) wi = 0;
        else if (wi >= W_in) wi = W_in - 1;
    }

    const T* src = input + n * in_stride_n + c * in_stride_c + hi * in_stride_h + wi * in_stride_w;
    output[idx] = *src;
}

// CUDA kernel for roll
// grid: (numel + block - 1) / block
template <typename T>
__global__ void roll_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    int64_t numel,
    int64_t ndim,
    int64_t* __restrict__ dims,   // array of dims (length dims_size)
    int64_t* __restrict__ shifts, // array of shifts
    int64_t dims_size,
    int64_t* __restrict__ shape,  // shape of tensor
    int64_t* __restrict__ in_strides,
    int64_t* __restrict__ out_strides) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // compute source index
    int64_t src_idx = 0;
    int64_t out_idx = idx;
    // for each dim, compute input coordinate using roll shift
    // We have dims to roll, others keep same coordinate.
    // We'll compute source coordinates from output linear index by iterating dimensions.
    
    // Better: iterate over dimensions from last to first to decompose idx into coords, then apply roll to get source coords.
    int64_t coords[8]; // assume up to 4 dims, but support more via dynamic? For now up to 4
    // Since tensor is 4D, use 4.
    int64_t dim = ndim - 1;
    int64_t remaining = idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        int64_t size = shape[d];
        int64_t coord = remaining % size;
        remaining /= size;
        coords[d] = coord;
    }

    // apply roll shifts to relevant dims
    for (int64_t i = 0; i < dims_size; ++i) {
        int64_t d = dims[i];
        int64_t shift = shifts[i];
        int64_t size = shape[d];
        // normalize shift to [0, size-1]
        shift = ((shift % size) + size) % size;
        // input coordinate = (output_coord - shift + size) % size
        coords[d] = (coords[d] - shift + size) % size;
    }

    // compute source linear index using in_strides
    src_idx = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        src_idx += coords[d] * in_strides[d];
    }

    output[idx] = input[src_idx];
}

extern "C" void custom_reflection_pad2d_launcher(
    float* output,
    const float* input,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int64_t pad_left, int64_t pad_right, int64_t pad_top, int64_t pad_bottom,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    cudaStream_t stream) {
    
    int64_t total = N * C * H_out * W_out;
    const int block = 256;
    int grid = (total + block - 1) / block;
    reflection_pad2d_kernel<float><<<grid, block, 0, stream>>>(
        output, input, N, C, H_in, W_in, H_out, W_out,
        pad_left, pad_right, pad_top, pad_bottom,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w);
}

extern "C" void custom_roll_launcher(
    float* output,
    const float* input,
    int64_t numel,
    int64_t ndim,
    int64_t* dims_device,
    int64_t* shifts_device,
    int64_t dims_size,
    int64_t* shape_device,
    int64_t* in_strides_device,
    int64_t* out_strides_device,
    cudaStream_t stream) {
    
    const int block = 256;
    int grid = (numel + block - 1) / block;
    roll_kernel<float><<<grid, block, 0, stream>>>(
        output, input, numel, ndim,
        dims_device, shifts_device, dims_size,
        shape_device, in_strides_device, out_strides_device);
}

// Helper to report CUDA errors (optional)
static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
    }
}
