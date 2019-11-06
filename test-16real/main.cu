#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include "wmma-common.h"
#include "matrix-utility.h"

#define TCU_MAT 1920
#define RUN     10
#define BLKSIZE 1024
#define MAT_PER_BLOCK 160

int main(int argc, char **argv)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed;

  half *h_a_tcu;
  half *h_b_tcu;
  half *h_c_tcu;
  cpuAllocTCUMatrixHalf(&h_a_tcu, TCU_MAT);
  cpuAllocTCUMatrixHalf(&h_b_tcu, TCU_MAT);
  cpuAllocTCUMatrixHalf(&h_c_tcu, TCU_MAT);
  fillZeroTCUMatrixHalf(h_a_tcu, TCU_MAT);
  fillZeroTCUMatrixHalf(h_b_tcu, TCU_MAT);
  fillZeroTCUMatrixHalf(h_c_tcu, TCU_MAT);
  fillTCUMatrixHalf(h_a_tcu, TCU_MAT);
  fillTCUMatrixHalf(h_b_tcu, TCU_MAT);

  half *d_a_tcu;
  half *d_b_tcu;
  half *d_c_tcu;
  gpuAllocTCUMatrixHalf((void **)&d_a_tcu, TCU_MAT);
  gpuAllocTCUMatrixHalf((void **)&d_b_tcu, TCU_MAT);
  gpuAllocTCUMatrixHalf((void **)&d_c_tcu, TCU_MAT);
  copyHDTCUMatrixHalf(d_a_tcu, h_a_tcu, TCU_MAT);
  copyHDTCUMatrixHalf(d_b_tcu, h_b_tcu, TCU_MAT);
  copyHDTCUMatrixHalf(d_c_tcu, h_c_tcu, TCU_MAT);

  cudaEventRecord(start, 0);

  dim3 grid_tcu(TCU_MAT/MAT_PER_BLOCK);
  dim3 block_tcu(BLKSIZE);

  for (unsigned i = 0; i < RUN; i++) {
    dot_wmma16x16<<<grid_tcu, block_tcu>>>(d_a_tcu, d_b_tcu, d_c_tcu, TCU_MAT);
    cudaDeviceSynchronize();
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1000.0f;
  printf("TCU Version: %fs\n", elapsed/RUN);

  copyDHTCUMatrixHalf(h_c_tcu, d_c_tcu, TCU_MAT);

  for (unsigned i = 0; i < 16; i++) {
    for (unsigned j = 0; j < 16; j++) {
      printf("%.1f\t", __half2float(h_a_tcu[j+i*16]));
    }
    printf("\n");
  }

  printf("\n Second\n");

  for (unsigned i = 0; i < 16; i++) {
    for (unsigned j = 0; j < 16; j++) {
      printf("%.1f\t", __half2float((h_a_tcu+256*383)[j+i*16]));
    }
    printf("\n");
  }
  cudaFree(d_a_tcu);
  cudaFree(d_b_tcu);
  cudaFree(d_c_tcu);
  free(h_a_tcu);
  free(h_b_tcu);
  free(h_c_tcu);


  ////////// End TCU version////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////

  half *h_a;
  half *h_b;
  half *h_c;
  cpuAllocMatrixHalf(&h_a, 3, 3, TCU_MAT);
  cpuAllocMatrixHalf(&h_b, 3, 1, TCU_MAT);
  cpuAllocMatrixHalf(&h_c, 3, 1, TCU_MAT);
  fillMatrixHalf(h_a, 3, 3, TCU_MAT);
  fillMatrixHalf(h_b, 3, 1, TCU_MAT);
  fillMatrixHalf(h_c, 3, 1, TCU_MAT);

  //printf("Before\n");
  //for (unsigned i = 0; i < 12; i++) {
  //  if (i % 3 == 0) printf("\n");
  //  printf("%f ", __half2float(h_c[i]));
  //}

  half *d_a;
  half *d_b;
  half *d_c;

  gpuAllocMatrixHalf((void **)&d_a, 3, 3, TCU_MAT);
  gpuAllocMatrixHalf((void **)&d_b, 3, 1, TCU_MAT);
  gpuAllocMatrixHalf((void **)&d_c, 3, 1, TCU_MAT);
  copyHDMatrixHalf(d_a, h_a, 3, 3, TCU_MAT);
  copyHDMatrixHalf(d_b, h_b, 3, 1, TCU_MAT);
  copyHDMatrixHalf(d_c, h_c, 3, 1, TCU_MAT);

  cudaEventRecord(start, 0);

  dim3 grid((TCU_MAT+BLKSIZE-1)/BLKSIZE);
  dim3 block(BLKSIZE);

  for (unsigned i = 0; i < RUN; i++) {
    mat_vec_mul<<<grid, block>>>(d_a, d_b, d_c, TCU_MAT);
    cudaDeviceSynchronize();
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1000.0f;
  printf("Normal Version: %fs\n", elapsed/RUN);

  copyDHMatrixHalf(h_c, d_c, 3, 1, TCU_MAT);

  //printf("After:\n");
  //for (unsigned i = 0; i < 12; i++) {
  //  if (i % 3 == 0) printf("\n");
  //  printf("%f ", __half2float(h_c[i]));
  //}

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  //////////////////////// Test cublas full matrix /////////////////////////
  half *d_a_c;
  half *d_b_c;
  half *d_c_c;
  cudaMalloc((void **)&d_a_c, sizeof(half)*5760*5760);
  cudaMalloc((void **)&d_b_c, sizeof(half)*5760*5760);
  cudaMalloc((void **)&d_c_c, sizeof(half)*5760*5760);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  half alpha = __float2half(1.0f);
  half beta = __float2half(0.0f);

  cudaEventRecord(start, 0);

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 5760, 5760, 5760, &alpha, d_a_c, CUDA_R_16F, 5760, d_b_c, CUDA_R_16F, 5760, &beta, d_c_c, CUDA_R_16F, 5760, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1000.0f;
  printf("Cublas Version: %fs", elapsed);

  cudaFree(d_a_c);
  cudaFree(d_b_c);
  cudaFree(d_c_c);
  return EXIT_SUCCESS;
}
