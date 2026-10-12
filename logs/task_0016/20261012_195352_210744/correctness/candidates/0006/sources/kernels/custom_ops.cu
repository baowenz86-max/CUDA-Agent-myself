#include <cuda_runtime.h>
#include <float.h>

// -----------------------------------------------------------------------------
// Kernel 1: AddMM forward
// Computes C = bias + A * B^T
// A is [M, K], B is [N, K] (i.e., weight.t() from [K, N])
// Output C is [M, N]
// bias is [N]
// All tensors are float32, contiguous.
// -----------------------------------------------------------------------------
__global__ void addmm_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             const float* __restrict__ bias,
                             float* __restrict__ C,
                             int M, int N, int K) {
    int total = M * N;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = tid; idx < total; idx += stride) {
        int row = idx / N;
        int col = idx % N;
        float sum = bias[col];
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[idx] = sum;
    }
}

// Launcher for addmm
// Inputs: bias [N], A [M,K], B [N,K]; Output: C [M,N]
// All pointers are raw device pointers to float32 data.
// M, N, K are integers; stream is the CUDA stream.
extern "C" void addmm_forward_launcher(const float* bias,
                                       const float* A,
                                       const float* B,
                                       float* C,
                                       int M, int N, int K,
                                       cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    int total = M * N;
    const int threads = 256;
    // Choose grid size based on occupancy, cap at max grid (2^31-1) but use grid-stride loop.
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // sensible cap, grid-stride loop handles rest
    addmm_kernel<<<blocks, threads, 0, stream>>>(A, B, bias, C, M, N, K);
}

// -----------------------------------------------------------------------------
// Kernel 2: Unsqueeze forward (adds a dimension at specified position, but here only dim=0 is used)
// Input shape: [M, N], output shape: [1, M, N]. Data is copied.
// The kernel treats the tensor as flat contiguous memory, so effectively a copy.
// We provide a generic unsqueeze kernel that handles any dim, but the launcher is
// specialized for dim=0 for the current use case (since model_new.py always uses dim=0).
// For safety, implement a simple memcpy-like copy.
// -----------------------------------------------------------------------------
__global__ void unsqueeze_copy_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int total_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < total_elements; i += stride) {
        output[i] = input[i];
    }
}

// Launcher for unsqueeze
// We ignore the dim parameter because it doesn't affect the underlying data layout
// for a contiguous tensor when adding a leading dimension (dim=0).
// If dim is not 0, the binding layer must handle it, but for our contract dim=0 always.
extern "C" void unsqueeze_forward_launcher(const float* input,
                                           float* output,
                                           int total_elements,
                                           int dim,
                                           cudaStream_t stream) {
    if (total_elements <= 0) return;
    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    unsqueeze_copy_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}

// -----------------------------------------------------------------------------
// Kernel 3: Linear interpolation for 3D input [1, M, N] with scale_factor=2
// Output shape [1, M, 2*N]
// Align_corners=False: out_x = (in_x + 0.5) * scale - 0.5; clamp to [0, N-1]
// Here scale = output_width / input_width = 2.0 (since scale_factor=2 and input is [1,M,N])
// Actually we compute scale_factor as output_size / input_size, where output_size
// is always 2*N (since scale_factor=2). We assume only scale_factor=2 is used.
// In general, we can compute using scale_factor parameter given to launcher, but
// model_new.py passes self.scale_factor which is 2.0. We'll implement for arbitrary
// positive scale_factor by computing output width as (int)(input_width * scale_factor).
// However, to be safe, we'll implement using the computed output width.
// The kernel handles each output element: for each (b=0, m, out_x), find source coordinate.
// -----------------------------------------------------------------------------
__global__ void interpolate_linear_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int input_height,   // M
                                          int input_width,    // N
                                          int output_width,   // out_N
                                          float scale) {
    int total = input_height * output_width; // batch=1
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = tid; idx < total; idx += stride) {
        int m = idx / output_width;
        int out_x = idx % output_width;
        // Compute source coordinate for align_corners=False
        float src_x_f = (static_cast<float>(out_x) + 0.5f) * scale - 0.5f;
        // Clamp to [0, input_width-1]
        src_x_f = fmaxf(0.0f, fminf(src_x_f, static_cast<float>(input_width - 1)));
        int x0 = static_cast<int>(floorf(src_x_f));
        int x1 = min(x0 + 1, input_width - 1);
        float dx = src_x_f - static_cast<float>(x0);
        // Linear interpolation along width
        const float* row_ptr = input + m * input_width;
        float val = row_ptr[x0] * (1.0f - dx) + row_ptr[x1] * dx;
        output[idx] = val;
    }
}

// Launcher for interpolate_linear
// Input shape: batch=1, channels=M, length=N (we call height and width generically)
// Output width = (int)(N * scale_factor). For our case scale_factor=2, so output_width=2*N.
// We assume contiguous float32 tensor.
// Note: The binding layer must compute output_width and pass it.
extern "C" void interpolate_linear_forward_launcher(const float* input,
                                                    float* output,
                                                    int input_height,
                                                    int input_width,
                                                    int output_width,
                                                    float scale_factor,
                                                    cudaStream_t stream) {
    if (input_height <= 0 || input_width <= 0 || output_width <= 0) return;
    int total = input_height * output_width;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    // scale = (float)input_width / (float)output_width (align_corners=False)
    float scale = static_cast<float>(input_width) / static_cast<float>(output_width);
    interpolate_linear_kernel<<<blocks, threads, 0, stream>>>(
        input, output, input_height, input_width, output_width, scale);
}
