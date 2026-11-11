#include <cuda_runtime.h>
#include <cstdint>

// Kernel: element-wise absolute value for float32, supports arbitrary stride via
// linear index? Actually input is given as pointer and we assume contiguous layout
// as per contract stride [163840, 16384, 128, 1] which is contiguous. We will
// use a simple 1D grid-stride loop over total elements, but handle possible
// non-contiguity by mapping linear index to 4D indices if needed. However,
// abs_forward contract says supports strides, but likely contiguous. To be safe,
// we can launch with total numel and use index into flat pointer assuming input
// is contiguous; if not, caller should make contiguous in binding. The binding
// will check contiguity and call .contiguous() if needed. Our launcher just
// accepts raw pointer and total count.
__global__ void abs_kernel(float* __restrict__ out, const float* __restrict__ in, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        out[i] = fabsf(in[i]);
    }
}

// Kernel: argmax over dim=1 (size classes) and one-hot encode.
// Input: x of shape [batch, classes, height, width] (float)
// Output: one_hot of shape [batch, height, width, num_classes] (int64)
// We assume x is contiguous, but we can handle strides by passing strides.
// However, to keep launcher simple, we pass total batch, classes, height, width,
// and assume x is contiguous. The binding can ensure contiguity. Output will be
// contiguous as allocated by torch::empty_like? Actually we need shape
// [batch, height, width, num_classes]. We create output in binding and pass pointer,
// so it's contiguous. We compute argmax and write 1 at position [b, h, w, argmax],
// all other positions zero. We can either write zeros then set one, or each thread
// writes only its one position; but output may not be initialized, so we must ensure
// every element gets written. We can do two kernels: one to zero, one to set ones,
// or single kernel where each thread writes multiple elements. Simpler: single kernel
// where each thread handles one output element and checks if its class index equals
// argmax for that (b,h,w) location. That would require recomputing argmax for each
// class, which is inefficient. Better: each thread handles one (b,h,w) position,
// computes argmax, then writes num_classes elements (zero all then one). That still
// writes num_classes per thread, but output has batch*height*width*num_classes elements
// total, and we have batch*height*width threads, so exactly matches total writes.
// So one kernel, each thread handles one spatial location, loops over classes to find
// argmax, then loops again to write zeros and one. That writes num_classes int64 values
// per thread sequentially. This is acceptable for num_classes=10.
__global__ void argmax_onehot_kernel(const float* __restrict__ x,
                                     int64_t* __restrict__ out,
                                     int64_t batch,
                                     int64_t classes,
                                     int64_t height,
                                     int64_t width,
                                     int64_t num_classes) {
    int64_t total = batch * height * width;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total; linear += stride) {
        // Decompose linear index into (b, h, w)
        int64_t w = linear % width;
        int64_t tmp = linear / width;
        int64_t h = tmp % height;
        int64_t b = tmp / height;

        // Find argmax over classes.
        // Input x layout: [batch, classes, height, width]
        // Index: ((b * classes + c) * height + h) * width + w
        int64_t argmax = 0;
        float max_val = x[((b * classes + 0) * height + h) * width + w];
        for (int64_t c = 1; c < classes; ++c) {
            float val = x[((b * classes + c) * height + h) * width + w];
            if (val > max_val) {
                max_val = val;
                argmax = c;
            }
        }

        // Write one-hot output at position [b, h, w, :]
        // Output layout: [batch, height, width, num_classes]
        // Index: ((b * height + h) * width + w) * num_classes + c
        int64_t base_out = ((b * height + h) * width + w) * num_classes;
        for (int64_t c = 0; c < num_classes; ++c) {
            out[base_out + c] = (c == argmax) ? 1 : 0;
        }
    }
}

// Launcher for abs
extern "C" void abs_forward_launcher(float* out, const float* in, int64_t total, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // max grid dim
    if (total > 0) {
        abs_kernel<<<blocks, threads, 0, stream>>>(out, in, total);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Cannot throw, but we can maybe printf? We'll just ignore for now.
    }
}

// Launcher for argmax_onehot
extern "C" void argmax_onehot_forward_launcher(const float* x,
                                               int64_t* out,
                                               int64_t batch,
                                               int64_t classes,
                                               int64_t height,
                                               int64_t width,
                                               int64_t num_classes,
                                               cudaStream_t stream) {
    int64_t total = batch * height * width;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (total > 0) {
        argmax_onehot_kernel<<<blocks, threads, 0, stream>>>(x, out, batch, classes, height, width, num_classes);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
}
