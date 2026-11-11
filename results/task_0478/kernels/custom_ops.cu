#include <cuda_runtime.h>
#include <math.h>

// ---------- Softmax along dim=1 (H dimension) for shape [N, H, W] ----------
// Each (b, :, w) slice is normalized independently.
// Input/output are contiguous float32 tensors. N*H*W total elements.

// Kernel to compute per-slice max along H dimension (dim=1)
template <int BLOCK_DIM_Y>
__global__ void softmax_max_kernel(const float* __restrict__ input,
                                   float* __restrict__ max_vals,
                                   int N, int H, int W) {
    const int b = blockIdx.x; // batch index
    const int w = blockIdx.y; // width index
    extern __shared__ float smem[]; // size BLOCK_DIM_Y floats
    const int tid = threadIdx.x; // thread along H dimension
    const int start_idx = ((b * H) + 0) * W + w;

    float thread_max = -INFINITY;
    // stride over H dimension
    for (int h = tid; h < H; h += blockDim.x) {
        float val = input[(((b * H + h) * W) + w)];
        thread_max = fmaxf(thread_max, val);
    }

    smem[tid] = thread_max;
    __syncthreads();

    // Block reduction within BLOCK_DIM_Y threads (BLOCK_DIM_Y is blockDim.x)
    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_vals[b * W + w] = smem[0];
    }
}

// Kernel to compute per-slice sum of exp(x - max) along H dimension
template <int BLOCK_DIM_Y>
__global__ void softmax_sum_kernel(const float* __restrict__ input,
                                   const float* __restrict__ max_vals,
                                   float* __restrict__ sum_vals,
                                   int N, int H, int W) {
    const int b = blockIdx.x;
    const int w = blockIdx.y;
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const float max_val = max_vals[b * W + w];

    float thread_sum = 0.0f;
    for (int h = tid; h < H; h += blockDim.x) {
        float val = input[(((b * H + h) * W) + w)];
        thread_sum += __expf(val - max_val);
    }

    smem[tid] = thread_sum;
    __syncthreads();

    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum_vals[b * W + w] = smem[0];
    }
}

// Kernel to compute final softmax output = exp(x - max) / sum
template <int BLOCK_DIM_Y>
__global__ void softmax_apply_kernel(const float* __restrict__ input,
                                     const float* __restrict__ max_vals,
                                     const float* __restrict__ sum_vals,
                                     float* __restrict__ output,
                                     int N, int H, int W) {
    const int b = blockIdx.x;
    const int w = blockIdx.y;
    const int tid = threadIdx.x;
    const float max_val = max_vals[b * W + w];
    const float inv_sum = 1.0f / sum_vals[b * W + w];

    for (int h = tid; h < H; h += blockDim.x) {
        int idx = ((b * H + h) * W) + w;
        float val = input[idx];
        output[idx] = __expf(val - max_val) * inv_sum;
    }
}

// Launcher: computes softmax along dim=1 for [N,H,W] contiguous float32
// output must be allocated with same shape.
extern "C" void custom_softmax_lastdim_v2_launcher(
    float* output,
    const float* input,
    int N,
    int H,
    int W,
    cudaStream_t stream) {

    if (N <= 0 || H <= 0 || W <= 0) {
        // handle empty or degenerate tensor: nothing to compute, but output already allocated (maybe empty)
        return;
    }

    // Choose block dimension along H. We need a power-of-two blockDim.x <= 1024.
    // For performance choose 256 if H is sufficiently large, else next power of two >= min(H, 1024), max 1024.
    int block_y = 256;
    if (H < block_y) {
        // find next power of two >= H, but at least 1 (handled below)
        block_y = 1;
        while (block_y < H) block_y <<= 1;
        if (block_y < 1) block_y = 1;
        if (block_y > 1024) block_y = 1024;
    } else {
        block_y = 256;
    }
    // Avoid launching with blockDim.x > 1024
    if (block_y > 1024) block_y = 1024;

    // grid dims: (N, W)
    dim3 grid(N, W);
    dim3 block(block_y);

    // Shared memory for reduction: block_y floats
    size_t shared_mem = block_y * sizeof(float);

    // Temporary buffers for per-slice max and sum (each size N*W)
    // We'll allocate using cudaMalloc (caller frees)
    float* max_vals = nullptr;
    float* sum_vals = nullptr;
    cudaMalloc(&max_vals, (size_t)N * W * sizeof(float));
    cudaMalloc(&sum_vals, (size_t)N * W * sizeof(float));
    if (!max_vals || !sum_vals) {
        // allocation failed; can't proceed, but shouldn't happen in valid case
        if (max_vals) cudaFree(max_vals);
        if (sum_vals) cudaFree(sum_vals);
        return;
    }

    // Launch max kernel
    if (block_y == 256) {
        softmax_max_kernel<256><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W);
    } else if (block_y == 512) {
        softmax_max_kernel<512><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W);
    } else if (block_y == 1024) {
        softmax_max_kernel<1024><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W);
    } else {
        // fallback: use a generic kernel that can handle any block size via a template parameter runtime? 
        // We'll instantiate for common powers of two up to 128 for small H.
        switch (block_y) {
            case 1: softmax_max_kernel<1><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W); break;
            case 2: softmax_max_kernel<2><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W); break;
            case 4: softmax_max_kernel<4><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W); break;
            case 8: softmax_max_kernel<8><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W); break;
            case 16: softmax_max_kernel<16><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W); break;
            case 32: softmax_max_kernel<32><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W); break;
            case 64: softmax_max_kernel<64><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W); break;
            case 128: softmax_max_kernel<128><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W); break;
            default: 
                // should never happen; for safety instantiate 256
                softmax_max_kernel<256><<<grid, block, shared_mem, stream>>>(input, max_vals, N, H, W);
        }
    }

    // Launch sum kernel similarly
    if (block_y == 256) {
        softmax_sum_kernel<256><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W);
    } else if (block_y == 512) {
        softmax_sum_kernel<512><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W);
    } else if (block_y == 1024) {
        softmax_sum_kernel<1024><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W);
    } else {
        switch (block_y) {
            case 1: softmax_sum_kernel<1><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W); break;
            case 2: softmax_sum_kernel<2><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W); break;
            case 4: softmax_sum_kernel<4><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W); break;
            case 8: softmax_sum_kernel<8><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W); break;
            case 16: softmax_sum_kernel<16><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W); break;
            case 32: softmax_sum_kernel<32><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W); break;
            case 64: softmax_sum_kernel<64><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W); break;
            case 128: softmax_sum_kernel<128><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W); break;
            default: softmax_sum_kernel<256><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, N, H, W);
        }
    }

    // Launch apply kernel
    if (block_y == 256) {
        softmax_apply_kernel<256><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W);
    } else if (block_y == 512) {
        softmax_apply_kernel<512><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W);
    } else if (block_y == 1024) {
        softmax_apply_kernel<1024><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W);
    } else {
        switch (block_y) {
            case 1: softmax_apply_kernel<1><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W); break;
            case 2: softmax_apply_kernel<2><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W); break;
            case 4: softmax_apply_kernel<4><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W); break;
            case 8: softmax_apply_kernel<8><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W); break;
            case 16: softmax_apply_kernel<16><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W); break;
            case 32: softmax_apply_kernel<32><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W); break;
            case 64: softmax_apply_kernel<64><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W); break;
            case 128: softmax_apply_kernel<128><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W); break;
            default: softmax_apply_kernel<256><<<grid, block, shared_mem, stream>>>(input, max_vals, sum_vals, output, N, H, W);
        }
    }

    // free temporary buffers
    cudaFree(max_vals);
    cudaFree(sum_vals);
}

// ---------- Mish activation: x * tanh(log(1 + exp(x))) ----------
// Elementwise, contiguous input/output float32.
__global__ void mish_kernel(const float* __restrict__ input, float* __restrict__ output, long total) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = idx; i < total; i += stride) {
        float x = input[i];
        // stable softplus: log1p(exp(x)) when x < 20, else x
        float sp;
        if (x > 20.0f) {
            sp = x;
        } else {
            // Use expf and log1pf. expf(x) may overflow for x>88 but x<=20 so fine.
            sp = log1pf(expf(x));
        }
        // tanh(sp)
        float th = tanhf(sp);
        output[i] = x * th;
    }
}

extern "C" void custom_mish_launcher(
    float* output,
    const float* input,
    long total,
    cudaStream_t stream) {
    if (total <= 0) return;
    const int threads = 256;
    long blocks = (total + threads - 1) / threads;
    if (blocks > 65535) {
        blocks = 65535; // grid-stride handles more
    }
    mish_kernel<<<(int)blocks, threads, 0, stream>>>(input, output, total);
}
