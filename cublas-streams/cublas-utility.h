/*
 * cublas-utility.h
 * Copyright (C) 2019 Nicolas Farabegoli <nicolas.farabegoli@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CUBLAS_UTILITY_H
#define CUBLAS_UTILITY_H

#include <cublas_v2.h>
#include <stdio.h>
#include <assert.h>

const char* cublasGetErrorString(cublasStatus_t status)
{
  switch(status)
  {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "unknown error";
}

inline cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}


void mma_batched(cublasHandle_t handle, int m, int n, int k, void * const Aarrya[], void * const Barray[], void * const Carray[], int batchCount)
{
  cublasStatus_t stat;
  half alpha = __float2half(1.0f);
  half beta = __float2half(0.0f);
  stat = cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
      Aarrya, CUDA_R_16F, k,
      Barray, CUDA_R_16F, n, &beta,
      Carray, CUDA_R_16F, n, batchCount, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  checkCublas(stat);
}

#endif /* !CUBLAS_UTILITY_H */
