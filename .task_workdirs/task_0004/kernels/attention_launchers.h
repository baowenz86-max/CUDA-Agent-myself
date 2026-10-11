#pragma once
#include <cuda_runtime.h>
#include <cstdint>

void attention_pack(const float*, const float*, float*, int, int, int, int, bool, cudaStream_t);
void attention_softmax(float*, int64_t, int, cudaStream_t);
void attention_merge(const float*, float*, int, int, int, int, bool, cudaStream_t);
void attention_epilogue(const float*, const float*, float*, int, int, int, int, float, bool, cudaStream_t);
void attention_gemm(const float*, const float*, float*, int, int, int, cudaStream_t);
void attention_scores(const float*, const float*, float*, int, int, int, int, cudaStream_t);
void attention_values(const float*, const float*, float*, int, int, int, int, cudaStream_t);
