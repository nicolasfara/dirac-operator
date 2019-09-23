#include <iostream>
#include <mma.h>

//using namespace nvcuda;

/*__global__ void dot_wmma4x4(half *a, half *b, half *c)
{
  wmma::fragment<wmma::matrix_a, 4, 4, 4, half, col_major> a_frag;
  wmma::fragment<wmma::matrix_b, 4, 4, 4, half, row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
  wmma::load_matrix_sync(a_frag, a, 4);
  wmma::load_matrix_sync(b_frag, b, 4);
  wmma::fill_fragment(c_frag, 0.0f);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix(c, c_frag, 4, row_major);
}*/

__global__ void fill_matrix(half *m, const unsigned size)
{
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size) {
    m[i] = __float2half((float) i);
  }
}

int main(int argc, char **argv)
{

  size_t mat_size = 4 * 4;

  half *a_h;
  half *b_h;
  half *c_h;

  posix_memalign((void **)&a_h, 128, mat_size * sizeof(half));
  posix_memalign((void **)&b_h, 128, mat_size * sizeof(half));
  posix_memalign((void **)&c_h, 128, mat_size * sizeof(half));

  for (unsigned i = 0; i < mat_size; i++) {
    a_h[i] = __float2half((float) i);
    printf("%f ", __half2float(a_h[i]));
  }
  printf("\n\n");
  for (unsigned i = 0; i < mat_size; i++) {
    b_h[i] = __float2half((float) 2 * i);
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

  dot_wmma4x4<<<1, 1>>>(a_d, b_d, c_d);

  cudaMemcpy(c_h, c_d, sizeof(half) * mat_size, cudaMemcpyDeviceToHost);

  printf("\n\n");
  for (unsigned i = 0; i < mat_size; i++) {
    c_h[i] = __float2half((float) 0.0f);
    printf("%f ", __half2float(c_h[i]));
  }


  return EXIT_SUCCESS;
}
