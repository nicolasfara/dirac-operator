#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuComplex.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLKSIZE 1024
#define ALIGN 128

__device__ bool isValid;

using namespace std;

void inline checkError(cublasStatus_t status, const char *msg)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    printf("%s", msg);
    exit(EXIT_FAILURE);
  }
}

__global__ void fillMatrix(__restrict cuFloatComplex * const mat, const unsigned size)
{
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < size) {
    mat[i] = make_cuFloatComplex((float) i, 3.0f);
  }
}

__global__ void checkMatrix(const __restrict cuFloatComplex * const m1, const __restrict cuFloatComplex * const m2, const unsigned size, bool *isValid)
{

  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  *isValid = true;

  if (i < size) {
    if (fabsf(cuCrealf(m1[i]) - cuCrealf(m2[i]) > 1e-3)) {
      *isValid = false;
    }

    if (fabsf(cuCimagf(m1[i]) - cuCimagf(m2[i]) > 1e-3)) {
      *isValid = false;
    }
  }

}

int main(int argc, char **argv)
{
  // cuBLAS initializzation
  cublasHandle_t handle;
  cublasStatus_t stat;
  checkError(cublasCreate(&handle), "cublasCreate() error!\n");
  checkError(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode() error!\n");

  // Cuda event setup
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed;

  size_t M = 2048;
  size_t K = 4096;
  size_t N = 2048;
  size_t A_size = M * K;
  size_t B_size = K * N;
  size_t C_size = M * N;

  cuFloatComplex *A_h;
  cuFloatComplex *B_h;
  cuFloatComplex *C_h;

  // Host allocation matrix
  posix_memalign((void **)&A_h, ALIGN, sizeof(cuFloatComplex) * A_size);
  posix_memalign((void **)&B_h, ALIGN, sizeof(cuFloatComplex) * B_size);
  posix_memalign((void **)&C_h, ALIGN, sizeof(cuFloatComplex) * C_size);

  cuFloatComplex *A_d;
  cuFloatComplex *B_d;
  cuFloatComplex *C1_d;
  cuFloatComplex *C2_d;

  // Device allocation matrix
  cudaMalloc((void **)&A_d, sizeof(cuFloatComplex) * A_size);
  cudaMalloc((void **)&B_d, sizeof(cuFloatComplex) * B_size);
  cudaMalloc((void **)&C1_d, sizeof(cuFloatComplex) * C_size);
  cudaMalloc((void **)&C2_d, sizeof(cuFloatComplex) * C_size);

  dim3 gridA((A_size + BLKSIZE - 1) / BLKSIZE);
  dim3 gridB((B_size + BLKSIZE - 1) / BLKSIZE);
  dim3 gridC((C_size + BLKSIZE - 1) / BLKSIZE);
  dim3 block(BLKSIZE);

  cudaEventRecord(start, 0);

  fillMatrix<<<gridA, block>>>(A_d, A_size);
  fillMatrix<<<gridB, block>>>(B_d, B_size);
  fillMatrix<<<gridC, block>>>(C1_d, C_size);
  fillMatrix<<<gridC, block>>>(C2_d, C_size);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1000.0f;

  printf("[DEBUG] Allocation time: %fs\n", elapsed);

  cuComplex alpha = make_cuFloatComplex(1.0f, 1.0f);
  cuComplex beta  = make_cuFloatComplex(0.0f, 0.0f);

  cudaEventRecord(start, 0);

  stat = cublasCgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
      &alpha, A_d, K,
      B_d, N, &beta,
      C1_d, N);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1000.0f;

  printf("cublasCgemm() taken: %fs\n", elapsed);

  checkError(stat, "\ncublasCgemm() failed!\n");

  //// cublasGemmEX (Tensor Core) ////

  cudaEventRecord(start, 0);

  stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
      &alpha, A_d, CUDA_C_32F, K,
      B_d, CUDA_C_32F, N, &beta,
      C2_d, CUDA_C_32F, N, CUDA_C_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1000.0f;

  printf("cublasGemmEx() taken: %fs\n\n", elapsed);

  //// end Tensor Core section ////

  //// Validation section ////
  checkMatrix<<<gridC, block>>>(C1_d, C2_d, C_size, &isValid);
  bool isValidH;
  cudaMemcpyFromSymbol(&isValid, &isValidH, sizeof(bool));

  if (isValidH) {
    printf("Matrix C1 and C2 are equal\n\n");
  } else {
    printf("Matrix C1 and C2 are not equal\n\n");
  }

  // Deallocation

  cublasDestroy(handle);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  free(A_h);
  free(B_h);
  free(C_h);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C1_d);
  cudaFree(C2_d);

  printf("End success");

  return EXIT_SUCCESS;
}
