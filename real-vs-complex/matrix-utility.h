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

void cpuAllocMatrixHalf(half **h_ptr, const size_t rows, const size_t cols, const unsigned matrix_count)
{
  // Allocate on CPU the matrix with allignment
  *h_ptr = (half *) malloc(rows*cols*matrix_count*sizeof(half));
}

void gpuAllocMatrixHalf(void **d_ptr, const size_t rows, const size_t cols, const unsigned matrix_count)
{
  checkCudaErrors(cudaMalloc(d_ptr, rows*cols*matrix_count*sizeof(half)));
  checkCudaErrors(cudaGetLastError());
}

void copyDHMatrixHalf(half *h_ptr, half *d_ptr, unsigned rows, unsigned cols, const unsigned matrix_count)
{
  checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, rows*cols*matrix_count*sizeof(half), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaGetLastError());
}

void copyHDMatrixHalf(half *d_ptr, half *h_ptr, unsigned rows, unsigned cols, const unsigned matrix_count)
{
  checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, rows*cols*matrix_count*sizeof(half), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaGetLastError());
}

void fillMatrixHalf(half *h_ptr, const size_t rows, const size_t cols, const unsigned matrix_count)
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


void cpuAllocTCUMatrixHalf(half **h_ptr, const unsigned matrix_count)
{
  const unsigned mat_16x16 = 16*16;
  const unsigned mat_16x16_count = matrix_count/5; //On each 16x16 matrix we could insert at most 5 matrix 3x3 on the diagonal of the big one.
  *h_ptr = (half *) malloc(mat_16x16*mat_16x16_count*sizeof(half));
}

void gpuAllocTCUMatrixHalf(void **d_ptr, const unsigned matrix_count)
{
  const unsigned mat_16x16 = 16*16;
  const unsigned mat_16x16_count = matrix_count/5;
  checkCudaErrors(cudaMalloc(d_ptr, mat_16x16*mat_16x16_count*sizeof(half)));
  checkCudaErrors(cudaGetLastError());
}

void copyDHTCUMatrixHalf(half *h_ptr, half *d_ptr, const unsigned matrix_count)
{
  const unsigned mat_16x16 = 16*16;
  const unsigned mat_16x16_count = matrix_count/5;
  checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, mat_16x16*mat_16x16_count*sizeof(half), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaGetLastError());
}

void copyHDTCUMatrixHalf(half *d_ptr, half *h_ptr, const unsigned matrix_count)
{
  const unsigned mat_16x16 = 16*16;
  const unsigned mat_16x16_count = matrix_count/5;
  checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, mat_16x16*mat_16x16_count*sizeof(half), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaGetLastError());
}

void fillZeroTCUMatrixHalf(half *h_ptr, const unsigned matrix_count)
{
  const unsigned mat_16x16 = 16*16;
  const unsigned mat_16x16_count = matrix_count/5; //On each 16x16 matrix we could insert at most 5 matrix 3x3 on the diagonal of the big one.
  for (unsigned i = 0; i < mat_16x16*mat_16x16_count; i++)
    h_ptr[i] = __float2half(0.0f);
}

void fillTCUMatrixHalf(half *h_ptr, const unsigned matrix_count)
{
  const unsigned lut[] = { 0, 4, 8, 12, 64, 68, 72, 76, 128, 132, 136, 140, 192, 196, 200, 204 };
  for (unsigned z = 0; z < matrix_count/16; z++) {
    for (unsigned i = 0; i < 16; i++) {
      for (unsigned j = 0; j < 3; j++) {
        for (unsigned y = 0; y < 3; y++) {
          unsigned offset = (256*z)+lut[i];
          h_ptr[offset+j*16+y] = __float2half(y+j*3);
        }
      }
    }
  }
}

void fillTCUVectorHalf(half *h_ptr, const unsigned matrix_count)
{
  const unsigned lut[] = { 0, 1, 64, 65, 4, 5, 68, 69, 8, 9, 73, 73, 12, 13, 76, 77, 130, 131, 194, 195, 134, 135, 198, 199, 138, 139, 202, 203, 142, 143, 206, 207};
  for (unsigned z = 0; z < matrix_count/16; z++) {
    for (unsigned i = 0; i < 32; i++) {
      for (unsigned j = 0; j < 3; j++) {
        unsigned offset = (256*z)+lut[i];
          h_ptr[offset+j*16] = __float2half(j);
      }
    }
  }

  const unsigned lut2[] = { 64, 68, 72, 76, 194, 198, 202, 206 };
  for (unsigned i = 0; i < 8; i++) {
    for (unsigned j = 0; j < 3; j++) {
      float neg = __half2float(h_ptr[lut2[i]+j*16])*-1;
      h_ptr[lut2[i]+j*16] = __float2half(neg);
      //h_ptr[lut2[i]+j*16] = __float2half(99);
    }
  }
}


///////////////////////////////////// Kernels //////////////////////////////////////////

__global__ void mat_vec_mul(half *matrix, half *in_vect, half *out_vect, const unsigned matrix_count)
{
  const unsigned i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < matrix_count) {
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
}

__global__ void mat_vec_mul(cuDoubleComplex *matrix, cuDoubleComplex *in_vect, cuDoubleComplex *out_vect, const unsigned matrix_count)
{
  const unsigned i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < matrix_count) {

    cuDoubleComplex vec0 = (in_vect+i*3)[0];
    cuDoubleComplex vec1 = (in_vect+i*3)[1];
    cuDoubleComplex vec2 = (in_vect+i*3)[2];

    cuDoubleComplex mat00 = (matrix+i*9)[0];
    cuDoubleComplex mat01 = (matrix+i*9)[1];
    cuDoubleComplex mat02 = (matrix+i*9)[2];

    cuDoubleComplex mat10 = (matrix+i*9)[3];
    cuDoubleComplex mat11 = (matrix+i*9)[4];
    cuDoubleComplex mat12 = (matrix+i*9)[5];

    cuDoubleComplex mat20 = (matrix+i*9)[6];
    cuDoubleComplex mat21 = (matrix+i*9)[7];
    cuDoubleComplex mat22 = (matrix+i*9)[8];

    //Multiply 3rd row by eta
    //mat20 = make_cuDoubleComplex(cuCreal(mat20)*eta, cuCimag(mat20)*eta);
    //mat21 = make_cuDoubleComplex(cuCreal(mat21)*eta, cuCimag(mat21)*eta);
    //mat22 = make_cuDoubleComplex(cuCreal(mat22)*eta, cuCimag(mat22)*eta);

    (out_vect+i*3)[0] = cuCadd( cuCadd( cuCmul( mat00, vec0 ),
        cuCmul( mat01, vec1 )),
        cuCmul( mat02, vec2 ));

    (out_vect+i*3)[1] = cuCadd( cuCadd( cuCmul( mat10, vec0 ),
        cuCmul( mat11, vec1 )),
        cuCmul( mat12, vec2 ));

    (out_vect+i*3)[2] = cuCadd( cuCadd( cuCmul( mat20, vec0 ),
        cuCmul( mat21, vec1 )),
        cuCmul( mat22, vec2 ));
  }
}

__global__ void dot_wmma16x16(half *a, half *b, half *c, const unsigned matrix_count)
{
  unsigned offset = (threadIdx.x + blockIdx.x * blockDim.x)/32; //32 -> warp for WMMA
  const unsigned index_offset = offset*256; // 16x16=256 element for each matrix
  if (offset < matrix_count/16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::load_matrix_sync(a_frag, a+index_offset, 16);
    wmma::load_matrix_sync(b_frag, b+index_offset, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(c+index_offset, c_frag, 16, wmma::mem_row_major);
  }
}


#endif /* !MATRIX_UTILITY_H */
