#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include "wmma-common.h"
#include "matrix-utility.h"

#define TCU_MAT 1024

int main(int argc, char **argv)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed;
  
  //half *h_a_tcu;
  //half *h_b_tcu;
  //half *h_c_tcu;
  //allocate_cpu_side_tcu_matrix_half(&h_a_tcu, TCU_MAT);
  //allocate_cpu_side_tcu_matrix_half(&h_b_tcu, TCU_MAT);
  //allocate_cpu_side_tcu_matrix_half(&h_c_tcu, TCU_MAT);
  //fill_zero_tcu_matrix(h_a_tcu, TCU_MAT);
  //fill_zero_tcu_matrix(h_b_tcu, TCU_MAT);
  //fill_zero_tcu_matrix(h_c_tcu, TCU_MAT);
  //fill_tcu_matrix_half(h_a_tcu, TCU_MAT);
  //fill_tcu_matrix_half(h_b_tcu, TCU_MAT);

  ////for (unsigned i = 0; i < 16; i++) {
  ////  for (unsigned j = 0; j < 16; j++) {
  ////    printf("%.1f\t", __half2float(h_a_tcu[j+i*16]));
  ////  }
  ////  printf("\n");
  ////}

  //half *d_a_tcu;
  //half *d_b_tcu;
  //half *d_c_tcu;
  //allocate_gpu_side_tcu_matrix_half((void **)&d_a_tcu, TCU_MAT);
  //allocate_gpu_side_tcu_matrix_half((void **)&d_b_tcu, TCU_MAT);
  //allocate_gpu_side_tcu_matrix_half((void **)&d_c_tcu, TCU_MAT);
  //copy_tcu_matrix_to_gpu_half(d_a_tcu, h_a_tcu, TCU_MAT);
  //copy_tcu_matrix_to_gpu_half(d_b_tcu, h_b_tcu, TCU_MAT);
  //copy_tcu_matrix_to_gpu_half(d_c_tcu, h_c_tcu, TCU_MAT);

  //cudaStream_t streams[TCU_MAT/5];
  //for (unsigned i = 0; i < TCU_MAT/5; i++)
  //  cudaStreamCreate(&streams[i]);
 
  //cudaEventRecord(start, 0);

  //for (unsigned i = 0; i < TCU_MAT/5; i++)
  //  dot_wmma16x16<<<1, 32, 0, streams[i]>>>(d_a_tcu+i*256, d_b_tcu+i*256, d_c_tcu+i*256);
  //  //dot_wmma16x16<<<1, 32>>>(d_a_tcu+i*256, d_b_tcu+i*256, d_c_tcu+i*256);

  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&elapsed, start, stop);
  //elapsed /= 1000.0f;
  //printf("TCU Version: %fs\n", elapsed);

  //copy_tcu_matrix_to_cpu_half(h_c_tcu, d_c_tcu, TCU_MAT);

  ////////// End TCU version////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////

  half *h_a;
  half *h_b;
  half *h_c;
  allocate_cpu_side_matrix_half(&h_a, 3, 3, TCU_MAT);
  allocate_cpu_side_matrix_half(&h_b, 3, 1, TCU_MAT);
  allocate_cpu_side_matrix_half(&h_c, 3, 1, TCU_MAT);
  
  half *d_a;
  half *d_b;
  half *d_c;

  allocate_gpu_side_matrix_half((void **)&d_a, 3, 3, TCU_MAT);
  allocate_gpu_side_matrix_half((void **)&d_b, 3, 1, TCU_MAT);
  allocate_gpu_side_matrix_half((void **)&d_c, 3, 1, TCU_MAT);
  copy_matrix_to_gpu_half(d_a, h_a, 3, 3, TCU_MAT);
  copy_matrix_to_gpu_half(d_b, h_b, 3, 1, TCU_MAT);
  copy_matrix_to_gpu_half(d_c, h_c, 3, 1, TCU_MAT);

  cudaEventRecord(start, 0);

  //for (unsigned i = 0; i < TCU_MAT; i++)
  mat_vec_mul<<<1, TCU_MAT>>>(d_a, d_b, d_c);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1000.0f;
  printf("Normal Version: %fs\n", elapsed);


  //////////////////////// Test cublas full matrix /////////////////////////
  half *d_a_c; 
  half *d_b_c; 
  half *d_c_c; 
  cudaMalloc((void **)&d_a_c, sizeof(half)*3072*3072);
  cudaMalloc((void **)&d_b_c, sizeof(half)*3072*3072);
  cudaMalloc((void **)&d_c_c, sizeof(half)*3072*3072);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  half alpha = __float2half(1.0f);
  half beta = __float2half(0.0f);

  cudaEventRecord(start, 0);

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3072, 3072, 3072, &alpha, d_a_c, CUDA_R_16F, 3072, d_b_c, CUDA_R_16F, 3072, &beta, d_c_c, CUDA_R_16F, 3072, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1000.0f;
  printf("Cublas Version: %fs", elapsed);

  //free(h_a_tcu);
  //free(h_b_tcu);
  //free(h_c_tcu);
  //cudaFree(d_a_tcu);
  //cudaFree(d_b_tcu);
  //cudaFree(d_c_tcu);
  return EXIT_SUCCESS;
}
