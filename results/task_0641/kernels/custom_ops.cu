#include <cuda_runtime.h>

// Helper to convert linear index to multidimensional coordinates given strides.
// strides in elements, shape in elements.
__device__ __forceinline__ void linear_to_coords(
    long long linear,
    const int ndim,
    const long long* __restrict__ shape,
    const long long* __restrict__ strides, // strides in elements (not bytes)
    long long* __restrict__ coords)
{
    for (int d = ndim - 1; d >= 0; --d) {
        coords[d] = linear % shape[d];
        linear /= shape[d];
    }
}

// Kernel for reduce_sum along dimension dim.
// output_shape = shape with dim removed.
// output linear index corresponds to all coords except coord[dim].
// Each thread accumulates sum over all elements along dim for its output position.
template <typename T>
__global__ void reduce_sum_dim_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int64_t* __restrict__ shape,   // full input shape
    const int64_t* __restrict__ strides, // input strides in elements
    const int64_t* __restrict__ out_shape, // output shape (dim removed)
    const int64_t* __restrict__ out_strides, // output strides in elements
    int ndim,
    int dim,
    long long total_output)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total_output; idx += stride) {
        // decompose output linear index into output coords
        long long rem = idx;
        long long out_coords[8];
        int out_ndim = ndim - 1;
        for (int d = out_ndim - 1; d >= 0; --d) {
            out_coords[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }

        // construct full coords: insert 0 at dim initially
        long long full_coords[8];
        for (int d = 0, i = 0; d < ndim; ++d) {
            if (d == dim) {
                full_coords[d] = 0;
            } else {
                full_coords[d] = out_coords[i++];
            }
        }

        // base linear index = sum(full_coords[d]*strides[d])
        long long base = 0;
        for (int d = 0; d < ndim; ++d) {
            base += full_coords[d] * strides[d];
        }

        // sum over dim
        T sum = 0;
        long long dim_size = shape[dim];
        long long dim_stride = strides[dim];
        for (long long k = 0; k < dim_size; ++k) {
            sum += input[base + k * dim_stride];
        }
        output[idx] = sum;
    }
}

// Kernel for count_nonzero along dimension dim.
template <typename T>
__global__ void count_nonzero_dim_kernel(
    const T* __restrict__ input,
    int64_t* __restrict__ output,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ strides,
    const int64_t* __restrict__ out_shape,
    const int64_t* __restrict__ out_strides,
    int ndim,
    int dim,
    long long total_output)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total_output; idx += stride) {
        long long rem = idx;
        long long out_coords[8];
        int out_ndim = ndim - 1;
        for (int d = out_ndim - 1; d >= 0; --d) {
            out_coords[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }

        long long full_coords[8];
        for (int d = 0, i = 0; d < ndim; ++d) {
            if (d == dim) {
                full_coords[d] = 0;
            } else {
                full_coords[d] = out_coords[i++];
            }
        }

        long long base = 0;
        for (int d = 0; d < ndim; ++d) {
            base += full_coords[d] * strides[d];
        }

        long long count = 0;
        long long dim_size = shape[dim];
        long long dim_stride = strides[dim];
        for (long long k = 0; k < dim_size; ++k) {
            T val = input[base + k * dim_stride];
            if (val != static_cast<T>(0)) {
                ++count;
            }
        }
        output[idx] = count;
    }
}

// Launcher for sum reduction.
// input and output must be valid device pointers.
// shape, strides, out_shape, out_strides are host arrays copied to device internally.
extern "C" void reduce_sum_dim_launcher(
    const float* input,
    float* output,
    const int64_t* shape,
    const int64_t* strides, // in elements, assuming contiguous? We'll use actual strides from binding (maybe in elements)
    const int64_t* out_shape,
    const int64_t* out_strides,
    int ndim,
    int dim,
    long long total_output,
    cudaStream_t stream)
{
    // Copy shape/strides to device
    int64_t* d_shape;
    int64_t* d_strides;
    int64_t* d_out_shape;
    int64_t* d_out_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_out_shape, (ndim-1) * sizeof(int64_t));
    cudaMalloc(&d_out_strides, (ndim-1) * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_strides, strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_shape, out_shape, (ndim-1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_strides, out_strides, (ndim-1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    long long blocks_ll = (total_output + threads - 1) / threads;
    int blocks = (int)min((long long)blocks_ll, (long long)2147483647);
    if (blocks == 0) blocks = 1;

    reduce_sum_dim_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, d_shape, d_strides, d_out_shape, d_out_strides,
        ndim, dim, total_output);

    cudaFree(d_shape);
    cudaFree(d_strides);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
}

// Launcher for count_nonzero reduction.
extern "C" void count_nonzero_dim_launcher(
    const float* input,
    int64_t* output,
    const int64_t* shape,
    const int64_t* strides,
    const int64_t* out_shape,
    const int64_t* out_strides,
    int ndim,
    int dim,
    long long total_output,
    cudaStream_t stream)
{
    int64_t* d_shape;
    int64_t* d_strides;
    int64_t* d_out_shape;
    int64_t* d_out_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_out_shape, (ndim-1) * sizeof(int64_t));
    cudaMalloc(&d_out_strides, (ndim-1) * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_strides, strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_shape, out_shape, (ndim-1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_strides, out_strides, (ndim-1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    long long blocks_ll = (total_output + threads - 1) / threads;
    int blocks = (int)min((long long)blocks_ll, (long long)2147483647);
    if (blocks == 0) blocks = 1;

    count_nonzero_dim_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, d_shape, d_strides, d_out_shape, d_out_strides,
        ndim, dim, total_output);

    cudaFree(d_shape);
    cudaFree(d_strides);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
}
