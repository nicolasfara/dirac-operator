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


void mma_batched_tcu(cublasHandle_t handle, int m, int n, int k, const void * const * Aarrya, const void * const * const Barray, void * const * const Carray, int batchCount)
{
  cublasStatus_t stat;
  //cublasSetStream(handle, stream);
  half alpha = __float2half(1.0f);
  half beta = __float2half(0.0f);
  stat = cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
      Aarrya, CUDA_R_16F, k,
      Barray, CUDA_R_16F, n, &beta,
      Carray, CUDA_R_16F, n, batchCount, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  checkCublas(stat);
}

void mma_batched(cublasHandle_t handle, int m, int n, int k, const half * const * Aarray, const half * const * const Barray, half * const * const Carray, int batchCount)
{
  cublasStatus_t stat;
  //cublasSetStream(handle, stream[0]);
  half alpha = __float2half(1.0f);
  half beta = __float2half(0.0f);
  stat = cublasHgemmBatched(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  m, n, k,
                                  &alpha,
                                  Aarray, k,
                                  Barray, n,
                                  &beta,
                                  Carray, n,
                                  batchCount);
  checkCublas(stat);
}

__global__ void mat_vec_mul(cuDoubleComplex *matrix, cuDoubleComplex *in_vect, cuDoubleComplex *out_vect)
{
  cuDoubleComplex vec0 = in_vect[0];
  cuDoubleComplex vec1 = in_vect[1];
  cuDoubleComplex vec2 = in_vect[2];

  cuDoubleComplex mat00 = matrix[0];
  cuDoubleComplex mat01 = matrix[1];
  cuDoubleComplex mat02 = matrix[2];

  cuDoubleComplex mat10 = matrix[3];
  cuDoubleComplex mat11 = matrix[4];
  cuDoubleComplex mat12 = matrix[5];

  cuDoubleComplex mat20 = matrix[6];
  cuDoubleComplex mat21 = matrix[7];
  cuDoubleComplex mat22 = matrix[8];

  out_vect[0] = cuCadd( cuCadd( cuCmul( mat00, vec0 ),
                             cuCmul( mat01, vec1 )),
                             cuCmul( mat02, vec2 ));

  out_vect[1] = cuCadd( cuCadd( cuCmul( mat10, vec0 ),
                             cuCmul( mat11, vec1 )),
                             cuCmul( mat12, vec2 ));

  out_vect[2] = cuCadd( cuCadd( cuCmul( mat20, vec0 ),
                                 cuCmul( mat21, vec1 )),
                                 cuCmul( mat22, vec2 ));
}

void test_3x3matvec(cuDoubleComplex *matrix, cuDoubleComplex *in_vec, cuDoubleComplex *out_vec, int batch)
{
  for (unsigned i = 0; i < batch; i++) {
    mat_vec_mul<<<1, 1>>>(matrix, in_vec, out_vec);
  }
}

#endif /* !CUBLAS_UTILITY_H */
