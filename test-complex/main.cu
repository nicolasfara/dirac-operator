#include <iostream>
#include <stdio.h>
#include <mma.h>

#define WARP_SIZE 32

using namespace nvcuda;

__global__ void dot_wmma4x4(half *a, half *b, half *c)
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
