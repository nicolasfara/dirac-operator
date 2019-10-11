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

__global__ void kernel_fill_matrix(half *d_ptr, size_t mat_size)
{
  const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < mat_size)
    d_ptr[i] = __float2half((float) i);
}

void fill_matrix(half *d_ptr, size_t mat_size)
{
  dim3 grid((mat_size + BLKSIZE-1) / BLKSIZE);
  dim3 block(BLKSIZE);
  kernel_fill_matrix<<<grid, block>>>(d_ptr, mat_size);
  checkCudaErrors(cudaGetLastError());
}

void display_matrix(half *d_ptr, size_t m, size_t n)
{
  half *h_ptr = (half *) malloc(sizeof(half) * m * n);
  assert(h_ptr);
  copy_D2H(h_ptr, d_ptr, m*n*sizeof(half));
  printf("\n");
  for (unsigned i = 0; i < m*n; i++){
    if (i % m == 0 && i != 0) printf("\n");
    printf("%.2f\t", __half2float(h_ptr[i]));
  }
  free(h_ptr);
}

#endif /* !CUDA_UTILITY_H */
