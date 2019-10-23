/*
 * matrix-utility.h
 * Copyright (C) 2019 Nicolas Farabegoli <nicolas.farabegoli@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef MATRIX_UTILITY_H
#define MATRIX_UTILITY_H

#include <helper_cuda.h>
#include <helper_functions.h>

void allocate_cpu_side_matrix_half(half **h_ptr, const size_t rows, const size_t cols, const unsigned matrix_count)
{
  // Allocate on CPU the matrix with allignment
  *h_ptr = (half *) malloc(rows*cols*matrix_count*sizeof(half));
}

void allocate_gpu_side_matrix_half(void **d_ptr, const size_t rows, const size_t cols, const unsigned matrix_count)
{
  checkCudaErrors(cudaMalloc(d_ptr, rows*cols*matrix_count*sizeof(half)));
  checkCudaErrors(cudaGetLastError());
}

void copy_matrix_to_cpu_half(half *h_ptr, half *d_ptr, unsigned rows, unsigned cols, const unsigned matrix_count)
{
  checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, rows*cols*matrix_count*sizeof(half), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaGetLastError());
}

void copy_matrix_to_gpu_half(half *d_ptr, half *h_ptr, unsigned rows, unsigned cols, const unsigned matrix_count)
{
  checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, rows*cols*matrix_count*sizeof(half), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaGetLastError());
}

void fill_matrix_half(half *h_ptr, const size_t rows, const size_t cols, const unsigned matrix_count)
{
  for (unsigned z = 0; z < matrix_count; z++) {
    for (unsigned i = 0; i < rows; i++)
      for (unsigned j = 0; j < cols; j++) {
        unsigned index = (rows*cols*z)+(j+i*cols);
        h_ptr[index] = __float2half(j+i*cols);
      }
  }
}

////////////////////////////////// TCU Version //////////////////////////////////////////////


void allocate_cpu_side_tcu_matrix_half(half **h_ptr, const unsigned matrix_count)
{
  const unsigned mat_16x16 = 16*16;
  const unsigned mat_16x16_count = matrix_count/5; //On each 16x16 matrix we could insert at most 5 matrix 3x3 on the diagonal of the big one.
  *h_ptr = (half *) malloc(mat_16x16*mat_16x16_count*sizeof(half));
}

void allocate_gpu_side_tcu_matrix_half(void **d_ptr, const unsigned matrix_count)
{
  const unsigned mat_16x16 = 16*16;
  const unsigned mat_16x16_count = matrix_count/5;
  checkCudaErrors(cudaMalloc(d_ptr, mat_16x16*mat_16x16_count*sizeof(half)));
  checkCudaErrors(cudaGetLastError());
}

void copy_tcu_matrix_to_cpu_half(half *h_ptr, half *d_ptr, const unsigned matrix_count)
{
  const unsigned mat_16x16 = 16*16;
  const unsigned mat_16x16_count = matrix_count/5;
  checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, mat_16x16*mat_16x16_count*sizeof(half), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaGetLastError());
}

void copy_tcu_matrix_to_gpu_half(half *d_ptr, half *h_ptr, const unsigned matrix_count)
{
  const unsigned mat_16x16 = 16*16;
  const unsigned mat_16x16_count = matrix_count/5;
  checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, mat_16x16*mat_16x16_count*sizeof(half), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaGetLastError());
}

void fill_zero_tcu_matrix(half *h_ptr, const unsigned matrix_count)
{
  const unsigned mat_16x16 = 16*16;
  const unsigned mat_16x16_count = matrix_count/5; //On each 16x16 matrix we could insert at most 5 matrix 3x3 on the diagonal of the big one.
  for (unsigned i = 0; i < mat_16x16*mat_16x16_count; i++)
    h_ptr[i] = __float2half(0.0f);
}

void fill_tcu_matrix_half(half *h_ptr, const unsigned matrix_count)
{
  for (unsigned z = 0; z < matrix_count; z++) {
    for (unsigned i = 0; i < 5; i++) {
      for (unsigned j = 0; j < 3; j++) {
        for (unsigned y = 0; y < 3; y++) {
          unsigned offset = (256*z)+(51*i);
          h_ptr[offset+j*16+y] = __float2half(y+j*3);
        }
      }
    }
  }
}


///////////////////////////////////// Kernels //////////////////////////////////////////

__global__ void mat_vec_mul(half *matrix, half *in_vect, half *out_vect)
{
  const unsigned i = threadIdx.x + blockIdx.x*blockDim.x;
  half vec0 = (in_vect+i*3)[0];
  half vec1 = (in_vect+i*3)[1];
  half vec2 = (in_vect+i*3)[2];

  half mat00 = (matrix+i*9)[0];
  half mat01 = (matrix+i*9)[1];
  half mat02 = (matrix+i*9)[2];

  half mat10 = (matrix+i*9)[3];
  half mat11 = (matrix+i*9)[4];
  half mat12 = (matrix+i*9)[5];

  half mat20 = (matrix+i*9)[6];
  half mat21 = (matrix+i*9)[7];
  half mat22 = (matrix+i*9)[8];

//Multiply 3rd row by eta
  //mat20 = make_cuDoubleComplex(cuCreal(mat20)*eta, cuCimag(mat20)*eta);
  //mat21 = make_cuDoubleComplex(cuCreal(mat21)*eta, cuCimag(mat21)*eta);
  //mat22 = make_cuDoubleComplex(cuCreal(mat22)*eta, cuCimag(mat22)*eta);

  (out_vect+i*3)[0] = mat00*vec0 + mat01*vec1 + mat02*vec2;

  (out_vect+i*3)[1] = mat10*vec0 + mat11*vec1 + mat12*vec2;

  (out_vect+i*3)[2] = mat20*vec0 + mat21*vec1 + mat22*vec2;
}

__global__ void dot_wmma16x16(half *a, half *b, half *c)
{
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
  wmma::load_matrix_sync(a_frag, a, 16);
  wmma::load_matrix_sync(b_frag, b, 16);
  wmma::fill_fragment(c_frag, 0.0f);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}


#endif /* !MATRIX_UTILITY_H */
