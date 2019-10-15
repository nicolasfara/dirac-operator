/*
 * cuda_utility.h
 * Copyright (C) 2019 Nicolas Farabegoli <nicolas.farabegoli@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CUDA_UTILITY_H
#define CUDA_UTILITY_H

#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_fp16.h>

#define BLKSIZE 1024
#define BLKDIM  32
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void allocate_matrix(void **mat_ptr, size_t size)
{
  checkCudaErrors(cudaMalloc(mat_ptr, size));
  checkCudaErrors(cudaGetLastError());
}

void copy_H2D(void *d_ptr, void *h_ptr, size_t size)
{
  checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaGetLastError());
}

void copy_D2H(void *h_ptr, void *d_ptr, size_t size)
{
  checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaGetLastError());
}

__global__ void kernel_fill_matrix(half *d_ptr, size_t size)
{
  const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size)
    d_ptr[i] = __float2half((float) i);
}

void fill_matrix(void *d_ptr, size_t rows, size_t cols)
{
  dim3 grid((rows*cols + BLKSIZE-1) / BLKSIZE);
  dim3 block(BLKSIZE);
  kernel_fill_matrix<<<grid, block>>>((half *)d_ptr, rows*cols);
  checkCudaErrors(cudaGetLastError());
}

void display_matrix(void *d_ptr, size_t m, size_t n)
{
  half *h_ptr = (half *) malloc(sizeof(half) * m * n);
  assert(h_ptr);
  copy_D2H(h_ptr, d_ptr, m*n*sizeof(half));
  printf("\n[\t");
  for (unsigned i = 0; i < m*n; i++){
    if (i % m == 0 && i != 0) printf("\n\t");
    printf("%.2f\t", __half2float(h_ptr[i]));
  }
  printf("]\n\n");
  free(h_ptr);
}

#endif /* !CUDA_UTILITY_H */
