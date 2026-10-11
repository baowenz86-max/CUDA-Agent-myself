#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include "attention_launchers.h"

static void check_blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS failed with status " + std::to_string(static_cast<int>(status)));
}

class BlasApi {
    void* library;

    template<class Function>
    Function symbol(const char* name) {
        auto address = dlsym(library, name);
        if (!address)
            throw std::runtime_error(std::string("Missing cuBLAS symbol: ") + name);
        return reinterpret_cast<Function>(address);
    }

public:
    decltype(&cublasCreate) create;
    decltype(&cublasDestroy) destroy;
    decltype(&cublasSetStream) set_stream;
    decltype(&cublasSetMathMode) set_math;
    decltype(&cublasSgemm) gemm;
    decltype(&cublasSgemmStridedBatched) batched_gemm;

    BlasApi() {
        library = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
        if (!library)
            library = dlopen("libcublas.so.13", RTLD_NOW | RTLD_LOCAL);
        if (!library)
            library = dlopen("libcublas.so", RTLD_NOW | RTLD_LOCAL);
        if (!library)
            throw std::runtime_error("Cannot load cuBLAS; add its directory to LD_LIBRARY_PATH");
        create = symbol<decltype(create)>("cublasCreate_v2");
        destroy = symbol<decltype(destroy)>("cublasDestroy_v2");
        set_stream = symbol<decltype(set_stream)>("cublasSetStream_v2");
        set_math = symbol<decltype(set_math)>("cublasSetMathMode");
        gemm = symbol<decltype(gemm)>("cublasSgemm_v2");
        batched_gemm = symbol<decltype(batched_gemm)>("cublasSgemmStridedBatched");
    }
};

static BlasApi& blas_api() {
    static BlasApi api;
    return api;
}

struct HandleCache {
    std::map<int, cublasHandle_t> handles;

    ~HandleCache() {
        int original_device = 0;
        cudaGetDevice(&original_device);
        for (auto& entry : handles) {
            cudaSetDevice(entry.first);
            blas_api().destroy(entry.second);
        }
        cudaSetDevice(original_device);
    }
};

static cublasHandle_t blas_handle(cudaStream_t stream) {
    thread_local HandleCache cache;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        throw std::runtime_error("Cannot get current CUDA device");
    auto found = cache.handles.find(device);
    if (found == cache.handles.end()) {
        cublasHandle_t handle;
        check_blas(blas_api().create(&handle));
        check_blas(blas_api().set_math(handle, CUBLAS_PEDANTIC_MATH));
        found = cache.handles.emplace(device, handle).first;
    }
    check_blas(blas_api().set_stream(found->second, stream));
    return found->second;
}

void attention_gemm(const float* input, const float* weight, float* output,
                    int rows, int input_features, int output_features, cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = 0.0f;
    check_blas(blas_api().gemm(blas_handle(stream), CUBLAS_OP_T, CUBLAS_OP_N,
        output_features, rows, input_features, &alpha, weight, input_features,
        input, input_features, &beta, output, output_features));
}

void attention_scores(const float* query, const float* key, float* scores,
                      int batches, int queries, int keys, int depth, cudaStream_t stream) {
    float alpha = 1.0f / std::sqrt(static_cast<float>(depth));
    float beta = 0.0f;
    check_blas(blas_api().batched_gemm(blas_handle(stream), CUBLAS_OP_T, CUBLAS_OP_N,
        keys, queries, depth, &alpha, key, depth, static_cast<long long>(keys) * depth,
        query, depth, static_cast<long long>(queries) * depth, &beta, scores, keys,
        static_cast<long long>(queries) * keys, batches));
}

void attention_values(const float* probabilities, const float* value, float* context,
                      int batches, int queries, int keys, int depth, cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = 0.0f;
    check_blas(blas_api().batched_gemm(blas_handle(stream), CUBLAS_OP_N, CUBLAS_OP_N,
        depth, queries, keys, &alpha, value, depth, static_cast<long long>(keys) * depth,
        probabilities, keys, static_cast<long long>(queries) * keys, &beta, context, depth,
        static_cast<long long>(queries) * depth, batches));
}
