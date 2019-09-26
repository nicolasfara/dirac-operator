#include <iostream>
#include <stdio.h>
#include <mma.h>
#include <cuComplex.h>

#define WARP_SIZE 32
#define BLKSIZE 1024

using namespace nvcuda;

__global__ void dot_wmma16x16(half *a, half *b, float *c)
{
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::load_matrix_sync(a_frag, a, 16);
  wmma::load_matrix_sync(b_frag, b, 16);
  wmma::fill_fragment(c_frag, 0.0f);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

__global__ void mat_sub(half *a, half *b, half *res, const unsigned size)
{
  const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    res[i] = a[i] - b[i];
  }
}

__global__ void mat_add(half *a, half *b, half *res, const unsigned size)
{
  const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    res[i] = a[i] + b[i];
  }
}

/*__global__ void matrix_padding_16x16(cuDoubleComplex *a, half *re, half *im)
{
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < 16 && j < 16) {
    if (i < 3 && j < 3){
      re[j + 3*i] = __float2half((float) cuCreal(a[j + 3*i]));
      im[j + 3*i] = __float2half((float) cuCimag(a[j + 3*i]));
    } else {
      re[j + 3*i] = __float2half(0.0f);
      im[j + 3*i] = __float2half(0.0f);
    }
  }
}

__global__ void vector_padding_16x16(cuDoubleComplex *a, half *re, half *im)
{
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < 16 && j < 16) {
    if (i < 3 && j < 1) {
      re[j + 3*i] = __float2half((float) cuCreal(a[j + 3*i]));
      im[j + 3*i] = __float2half((float) cuCimag(a[j + 3*i]));
    } else {
      re[j + 3*i] = __float2half(0.0f);
      im[j + 3*i] = __float2half(0.0f);
    }
  }
}*/

__host__ __device__ static inline half* IDX(half *m, unsigned i, unsigned j, unsigned size)
{
  return m + (j + i*size);
}

__global__ void fill_zero(half *re, half *im, float *c)
{
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < 16 && j < 16) {
    re[j + 16*i] = __float2half(0.0f);
    im[j + 16*i] = __float2half(0.0f);
    c[j + 16*i] = 0.0f;
  }
}

__host__ __device__ static inline void _sub_matrix_real(half *m, cuDoubleComplex *a)
{
#pragma unroll
  for (unsigned i = 0; i < 3; i++)
#pragma unroll
    for (unsigned j = 0; j < 3; j++) {
      *IDX(m, i, j, 3) = __float2half((float) cuCreal(a[j + i*3]));
    }
}

__host__ __device__ static inline void _sub_matrix_imag(half *m, cuDoubleComplex *a)
{
#pragma unroll
  for (unsigned i = 0; i < 3; i++)
#pragma unroll
    for (unsigned j = 0; j < 3; j++) {
      *IDX(m, i, j, 3) = __float2half((float) cuCimag(a[j + i*3]));
    }
}

__host__ __device__ static inline void _sub_vec_real(half *v, cuDoubleComplex *a)
{
#pragma unroll
  for (unsigned i = 0; i < 3; i++)
    *IDX(v, i, 0, 3) = __float2half((float) cuCreal(a[i]));
}

__host__ __device__ static inline void _sub_vec_imag(half *v, cuDoubleComplex *a)
{
#pragma unroll
  for (unsigned i = 0; i < 3; i++)
    *IDX(v, i, 0, 3) = __float2half((float) cuCimag(a[i]));
}

__global__ void compose_matrix(half *t_mat, half *t_vec, cuDoubleComplex *mat, cuDoubleComplex *vec)
{
  const unsigned gi = threadIdx.x + blockIdx.x * blockDim.x;
  if (gi == 0)   _sub_matrix_real(t_mat, mat);
  if (gi == 51)  _sub_matrix_imag(t_mat + 51, mat);
  if (gi == 86)  _sub_matrix_real(t_mat + 86, mat);
  if (gi == 137) _sub_matrix_imag(t_mat + 137, mat);

  if (gi == 0)   _sub_vec_real(t_vec, vec);
  if (gi == 51)  _sub_vec_imag(t_vec + 51, vec);
  if (gi == 86)  _sub_vec_real(t_vec + 86, vec);
  if (gi == 137) _sub_vec_imag(t_vec + 137, vec);
}

void complex_mma(cuDoubleComplex *mat, cuDoubleComplex *vec, const unsigned size)
{
  const size_t mat_size = 16 * 16;

  half *t_mat;
  half *t_vec;
  float *t_res;

  cudaMalloc((void **)& t_mat, sizeof(half) * mat_size);
  cudaMalloc((void **)& t_vec, sizeof(half) * mat_size);
  cudaMalloc((void **)& t_res, sizeof(float) * mat_size);

  dim3 blockDim(BLKSIZE, BLKSIZE);
  dim3 gridDim((16 + BLKSIZE - 1) / 16, (16 + BLKSIZE - 1) / 16);

  fill_zero<<<gridDim, blockDim>>>(t_mat, t_vec, t_res);
  compose_matrix<<<1, mat_size>>>(t_mat, t_vec, mat, vec);
  dot_wmma16x16<<<1, WARP_SIZE>>>(t_mat, t_vec, t_res);

  float *p_res = (float *) malloc(sizeof(float) * mat_size);
  cudaMemcpy(p_res, t_res, sizeof(float) * mat_size, cudaMemcpyDeviceToHost);

  for (unsigned i = 0; i < 16; i++){
    for (unsigned j = 0; j < 16; j++){
      printf("%.2f ", p_res[j + 16*i]);
    }
    printf("\n");
  }

  // Padding matrix to be used with tensor core
  //matrix_padding_16x16<<<gridDim, blockDim>>>(a, a_re, a_im);
  //vector_padding_16x16<<<gridDim, blockDim>>>(b, b_re, b_im);

  cudaFree(t_mat);
  cudaFree(t_vec);
  cudaFree(t_res);
}

__global__ void fill_matrix(half *m, const unsigned size)
{
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size) {
    m[i] = __float2half((float) i);
  }
}

int main(int argc, char **argv)
{

  size_t mat_size = 16 * 16;

  half *a_h;
  half *b_h;
  half *c_h;

  //posix_memalign((void **)&a_h, 128, mat_size * sizeof(half));
  //posix_memalign((void **)&b_h, 128, mat_size * sizeof(half));
  //posix_memalign((void **)&c_h, 128, mat_size * sizeof(half));

  a_h = (half *) malloc(sizeof(half) * mat_size);
  b_h = (half *) malloc(sizeof(half) * mat_size);
  c_h = (half *) malloc(sizeof(half) * mat_size);

  for (unsigned i = 0; i < mat_size; i++) {
    a_h[i] = __float2half((float) i);
    printf("%f ", __half2float(a_h[i]));
  }
  printf("\n\n");
  for (unsigned i = 0; i < mat_size; i++) {
    b_h[i] = __float2half((float)  1);
    printf("%f ", __half2float(b_h[i]));
  }
  printf("\n\n");
  for (unsigned i = 0; i < mat_size; i++) {
    c_h[i] = __float2half((float) 0.0f);
    printf("%f ", __half2float(c_h[i]));
  }
  printf("\n\n");

  half *a_d;
  half *b_d;
  half *c_d;

  cudaMalloc((void **)&a_d, sizeof(half) * mat_size);
  cudaMalloc((void **)&b_d, sizeof(half) * mat_size);
  cudaMalloc((void **)&c_d, sizeof(half) * mat_size);

  cudaMemcpy(a_d, a_h, sizeof(half) * mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, sizeof(half) * mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c_h, sizeof(half) * mat_size, cudaMemcpyHostToDevice);

  dot_wmma4x4<<<1, 32>>>(a_d, b_d, c_d);

  cudaMemcpy(c_h, c_d, sizeof(half) * mat_size, cudaMemcpyDeviceToHost);

  printf("\n\n");
  for (unsigned i = 0; i < mat_size; i++) {
    printf("%f ", __half2float(c_h[i]));
  }

  return EXIT_SUCCESS;
}
