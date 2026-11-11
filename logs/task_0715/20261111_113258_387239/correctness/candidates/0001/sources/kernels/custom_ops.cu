#include <cuda_runtime.h>

// Elementwise absolute value kernel for float32.
// Handles arbitrary strides by taking base pointer, total elements,
// and then using a generic index mapping from linear index to multidimensional indices.
// For simplicity, we assume input is contiguous or at least has standard strides;
// but we implement indexing generation from strides passed as int64_t*.
// However, to avoid copying strides arrays to device, we compute the linear index mapping on host?
// Better: use a 1D kernel that reads x and writes y assuming both are contiguous.
// The binding layer will ensure contiguity if needed, but contract says 'supports strides',
// so we must handle non-contiguous? Actually 'supports strides' might mean we need to respect arbitrary strides.
// We can implement a generic kernel that takes strides and shape as device pointers or constants.
// But passing arrays of strides is messy. Instead, we can use a simpler approach:
// Since the input is likely to be contiguous in practice (from torch.randn),
// and binding can call .contiguous() if needed, but contract says 'supports strides' and 
// 'kernel must handle non-contiguous views safely', we should implement correct stride handling.
// A robust way: launcher receives tensor metadata (shape, strides) and computes 
// linear index to multidimensional index on the fly using precomputed constants in constant memory or passed as scalars.
// Given shape [16,64,256,512], dims=4, we can specialize but better to be generic.

// Kernel with shape and strides passed as parameters (copied to constant memory? but that's per kernel launch).
// We'll use a templated kernel with dims as template parameter (4) for this specific case, but to be generic we can pass shape and strides as kernel arguments.
// Passing arrays by value from host to device kernel arguments is supported: you can pass struct containing arrays.
// We'll define a struct TensorMeta with shape[4] and strides[4], pass by value.

struct TensorMeta4D {
    long long shape[4];
    long long strides[4];
};

__global__ void abs_kernel_4d(float* __restrict__ out, const float* __restrict__ in, TensorMeta4D meta, long long numel) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        // compute multi-index from linear i
        long long rem = i;
        long long out_idx = 0;
        // general 4D index computation
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            long long coord = rem % meta.shape[d];
            rem /= meta.shape[d];
            out_idx += coord * meta.strides[d];
        }
        out[i] = fabsf(in[out_idx]);
    }
}

// PReLU kernel: channel-wise weight applies to channel index c.
// Input is assumed 4D NCHW. We need to read weight[c] for each element.
// We'll use similar indexing. Weight is passed as device pointer, shape[1] channels.
// For each element, compute channel index as (linear_index / (H*W)) % C, but we compute from multi-index.

__global__ void prelu_kernel_4d(float* __restrict__ out, const float* __restrict__ in, const float* __restrict__ weight, TensorMeta4D meta, long long numel) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        long long rem = i;
        long long in_idx = 0;
        long long channel = 0;
        // compute multi-index and channel index efficiently
        // For 4D NCHW: dims order n,c,h,w
        // rem for n: rem / (C*H*W)
        // c: (rem / (H*W)) % C
        // but using divisions in kernel is okay for 134M elements? might be slow but acceptable.
        // More efficient: we can precompute inner_size = H*W and channels = C, then channel = (i / inner_size) % C.
        // However shape/strides might be non-standard? We'll still use generic method but optimize for contiguous case.
        // Since input likely contiguous, we can assume strides are standard and use faster method.
        // But contract mandates support arbitrary strides, so we must compute index correctly.
        // We'll use a device-side function to compute multi-index from linear index using meta.
        long long linear = i;
        long long coord[4];
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            coord[d] = linear % meta.shape[d];
            linear /= meta.shape[d];
        }
        // channel is coord[1] if NCHW.
        channel = coord[1];
        // compute input offset
        long long offset = 0;
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            offset += coord[d] * meta.strides[d];
        }
        float val = in[offset];
        float w = weight[channel];
        out[i] = (val >= 0.0f) ? val : w * val;
    }
}

// Launcher functions with extern "C"

extern "C" void abs_forward_launcher(float* out, const float* in, const long long* shape, const long long* strides, long long numel, cudaStream_t stream) {
    // shape and strides are on host; we need to pass them to kernel by value as TensorMeta4D.
    // We'll copy to a local struct and pass.
    TensorMeta4D meta;
    for (int i = 0; i < 4; ++i) {
        meta.shape[i] = shape[i];
        meta.strides[i] = strides[i];
    }
    int threads = 256;
    long long blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // max grid dim in x
    abs_kernel_4d<<<blocks, threads, 0, stream>>>(out, in, meta, numel);
}

extern "C" void prelu_forward_launcher(float* out, const float* in, const float* weight, const long long* shape, const long long* strides, long long numel, cudaStream_t stream) {
    TensorMeta4D meta;
    for (int i = 0; i < 4; ++i) {
        meta.shape[i] = shape[i];
        meta.strides[i] = strides[i];
    }
    int threads = 256;
    long long blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    prelu_kernel_4d<<<blocks, threads, 0, stream>>>(out, in, weight, meta, numel);
}
