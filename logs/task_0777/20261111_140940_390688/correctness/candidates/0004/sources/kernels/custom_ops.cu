#include <cuda_runtime.h>
#include <cfloat>

// 沿着指定维度 dim 求均值
// x: 输入数据指针, 支持任意 stride
// out: 输出连续数据指针
// shape: 输入形状数组（维度数 ndim）
// strides: 输入步长数组（元素个数）
// out_shape: 输出形状数组（紧凑，不包含 dim 维度）
// ndim: 维度数
// dim: 要归约的维度
// reduce_size: 归约维度的长度
// total_out: 输出元素总数
__global__ void mean_kernel(const float* __restrict__ x,
                            float* __restrict__ out,
                            const int64_t* __restrict__ shape,
                            const int64_t* __restrict__ strides,
                            const int64_t* __restrict__ out_shape,
                            int ndim,
                            int dim,
                            int64_t reduce_size,
                            int64_t total_out) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total_out; linear += stride) {
        // 将线性索引映射到输出多维坐标（跳过 dim 维度）
        int64_t tmp = linear;
        int64_t out_coords[8];  // 最多支持 8 维
        int64_t in_coords[8];
        int out_ndim = 0;
        for (int i = 0; i < ndim; ++i) {
            if (i == dim) continue;
            int64_t size = out_shape[out_ndim];
            out_coords[i] = tmp % size;
            tmp /= size;
            ++out_ndim;
        }
        // 填充 in_coords: 非归约维度复制，归约维度置 0 起始
        for (int i = 0; i < ndim; ++i) {
            if (i == dim) in_coords[i] = 0;
            else in_coords[i] = out_coords[i];
        }
        int64_t base_offset = 0;
        for (int i = 0; i < ndim; ++i) {
            base_offset += in_coords[i] * strides[i];
        }
        float sum = 0.0f;
        for (int64_t k = 0; k < reduce_size; ++k) {
            sum += x[base_offset + k * strides[dim]];
        }
        out[linear] = sum / reduce_size;
    }
}

// 沿指定维度计算稳定 softmax
// x: 输入数据指针, 支持任意 stride
// out: 输出连续数据指针
// shape: 输入形状数组
// strides: 输入步长数组
// out_shape: 输出形状数组（与 shape 相同）
// ndim: 维度数
// dim: softmax 维度
// dim_size: softmax 维度长度
__global__ void softmax_kernel(const float* __restrict__ x,
                               float* __restrict__ out,
                               const int64_t* __restrict__ shape,
                               const int64_t* __restrict__ strides,
                               const int64_t* __restrict__ out_shape,
                               int ndim,
                               int dim,
                               int64_t dim_size) {
    int64_t total_out = 1;
    for (int i = 0; i < ndim; ++i) total_out *= out_shape[i];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total_out; linear += stride) {
        // 将线性索引映射到输出多维坐标
        int64_t tmp = linear;
        int64_t coords[8];
        for (int i = ndim - 1; i >= 0; --i) {
            coords[i] = tmp % out_shape[i];
            tmp /= out_shape[i];
        }
        // 计算 base_offset 和 stride for softmax dim
        int64_t base_offset = 0;
        for (int i = 0; i < ndim; ++i) {
            base_offset += coords[i] * strides[i];
        }
        // 减去 dim 维度的步长贡献，因为我们要沿该维度迭代
        base_offset -= coords[dim] * strides[dim];

        // 找最大值（数值稳定性）
        float max_val = -FLT_MAX;
        for (int64_t k = 0; k < dim_size; ++k) {
            float val = x[base_offset + k * strides[dim]];
            if (val > max_val) max_val = val;
        }
        // 计算分母
        float sum = 0.0f;
        for (int64_t k = 0; k < dim_size; ++k) {
            float val = __expf(x[base_offset + k * strides[dim]] - max_val);
            sum += val;
        }
        // 写回 softmax 值
        for (int64_t k = 0; k < dim_size; ++k) {
            int64_t write_offset = base_offset + k * strides[dim];
            out[write_offset] = __expf(x[write_offset] - max_val) / sum;
        }
    }
}

// Launcher for custom_mean: 沿 dim 求均值
// 输入: x (float*, 任意 stride), shape, strides, ndim, dim
// 输出: out (float*, 连续)
// 假设 out_shape 已由调用者计算，且 out 内存已分配
// 使用当前 stream
extern "C" void custom_mean_launcher(
    const float* x,
    float* out,
    const int64_t* shape,
    const int64_t* strides,
    const int64_t* out_shape,
    int ndim,
    int64_t dim,
    cudaStream_t stream) {
    int64_t reduce_size = shape[dim];
    int64_t total_out = 1;
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) total_out *= shape[i];
    }
    int threads = 256;
    int blocks = (int)((total_out + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    mean_kernel<<<blocks, threads, 0, stream>>>(x, out, shape, strides, out_shape, ndim, (int)dim, reduce_size, total_out);
}

// Launcher for custom_softmax: 沿 dim 计算 softmax
// 输入: x (float*, 任意 stride), shape, strides, ndim, dim
// 输出: out (float*, 连续)
// out_shape 与 shape 相同，但保留参数
extern "C" void custom_softmax_launcher(
    const float* x,
    float* out,
    const int64_t* shape,
    const int64_t* strides,
    const int64_t* out_shape,
    int ndim,
    int64_t dim,
    cudaStream_t stream) {
    int64_t dim_size = shape[dim];
    int64_t total_out = 1;
    for (int i = 0; i < ndim; ++i) total_out *= out_shape[i];
    int threads = 256;
    int blocks = (int)((total_out + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    softmax_kernel<<<blocks, threads, 0, stream>>>(x, out, shape, strides, out_shape, ndim, (int)dim, dim_size);
}
