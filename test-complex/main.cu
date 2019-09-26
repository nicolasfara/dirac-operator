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

__global__ void mat_sub(float *a, float *b, float *res, const unsigned size)
{
  const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    res[i] = a[i] - b[i];
  }
}

__global__ void mat_add(float *a, float *b, float *res, const unsigned size)
{
  const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    res[i] = a[i] + b[i];
  }
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
  *m      = __float2half((float) cuCreal(a[0]));
  *(m+1)  = __float2half((float) cuCreal(a[1]));
  *(m+2)  = __float2half((float) cuCreal(a[2]));

  *(m+16) = __float2half((float) cuCreal(a[3]));
  *(m+17) = __float2half((float) cuCreal(a[4]));
  *(m+18) = __float2half((float) cuCreal(a[5]));
  
  *(m+32) = __float2half((float) cuCreal(a[6]));
  *(m+33) = __float2half((float) cuCreal(a[7]));
  *(m+34) = __float2half((float) cuCreal(a[8]));
}

__host__ __device__ static inline void _sub_matrix_imag(half *m, cuDoubleComplex *a)
{
  *m      = __float2half((float) cuCimag(a[0]));
  *(m+1)  = __float2half((float) cuCimag(a[1]));
  *(m+2)  = __float2half((float) cuCimag(a[2]));

  *(m+16) = __float2half((float) cuCimag(a[3]));
  *(m+17) = __float2half((float) cuCimag(a[4]));
  *(m+18) = __float2half((float) cuCimag(a[5]));
  
  *(m+32) = __float2half((float) cuCimag(a[6]));
  *(m+33) = __float2half((float) cuCimag(a[7]));
  *(m+34) = __float2half((float) cuCimag(a[8]));
}

__host__ __device__ static inline void _sub_vec_real(half *v, cuDoubleComplex *a)
{
  *v      = __float2half((float) cuCreal(a[0]));
  *(v+16) = __float2half((float) cuCreal(a[1]));
  *(v+32) = __float2half((float) cuCreal(a[2]));
}

__host__ __device__ static inline void _sub_vec_imag(half *v, cuDoubleComplex *a)
{
  *v      = __float2half((float) cuCimag(a[0]));
  *(v+16) = __float2half((float) cuCimag(a[1]));
  *(v+32) = __float2half((float) cuCimag(a[2]));
}

__global__ void compose_matrix(half *t_mat, half *t_vec, cuDoubleComplex *mat, cuDoubleComplex *vec)
{
  const unsigned gi = threadIdx.x + blockIdx.x * blockDim.x;
  if (gi == 0)   _sub_matrix_real(t_mat, mat);
  if (gi == 51)  _sub_matrix_imag(t_mat + 51, mat);
  if (gi == 102)  _sub_matrix_real(t_mat + 102, mat);
  if (gi == 153) _sub_matrix_imag(t_mat + 153, mat);

  if (gi == 0)   _sub_vec_real(t_vec, vec);
  if (gi == 51)  _sub_vec_imag(t_vec + 51, vec);
  if (gi == 102)  _sub_vec_real(t_vec + 102, vec);
  if (gi == 153) _sub_vec_imag(t_vec + 153, vec);
}

__host__ __device__ static inline void _get_vec(float *mat, float *t)
{
  t[0] = *mat;
  t[1] = *(mat+16);
  t[2] = *(mat+32);
}

__global__ void isolate_vec(float *mat, float *t1, float *t2, float *t3, float *t4)
{
  const unsigned gi = threadIdx.x + blockIdx.x * blockDim.x;
  if (gi == 0) _get_vec(mat, t1); 
  if (gi == 51) _get_vec(mat+51, t2); 
  if (gi == 102) _get_vec(mat+102, t3); 
  if (gi == 153) _get_vec(mat+153, t4); 
}

__global__ void add_sub_vec(float *t1, float *t2, float *t3, float *t4, float *t_re, float *t_im)
{
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < 3) {
    t_re[i] = t1[i] - t2[i];
    t_im[i] = t3[i] + t4[i];
  }
}

__global__ void combine(float *t_re, float *t_im, cuDoubleComplex *res)
{
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < 3) {
    res[i] = make_cuDoubleComplex((double) t_re[i], (double) t_im[i]);
  }
}

void complex_mma(__restrict cuDoubleComplex * const mat, __restrict cuDoubleComplex * const vec, __restrict cuDoubleComplex * const res)
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

  float *t1, *t2, *t3, *t4;
  cudaMalloc((void **)&t1, sizeof(float) * 3);
  cudaMalloc((void **)&t2, sizeof(float) * 3);
  cudaMalloc((void **)&t3, sizeof(float) * 3);
  cudaMalloc((void **)&t4, sizeof(float) * 3);

  isolate_vec<<<1, mat_size>>>(t_res, t1, t2, t3, t4);

  float *t_re, *t_im;
  cudaMalloc((void **)&t_re, sizeof(float) * 3);
  cudaMalloc((void **)&t_im, sizeof(float) * 3);
  
  add_sub_vec<<<1, 3>>>(t1, t2, t3, t4, t_re, t_im);
  combine<<<1, 3>>>(t_re, t_im, res);

  cudaFree(t_mat);
  cudaFree(t_vec);
  cudaFree(t_res);
  cudaFree(t1);
  cudaFree(t2);
  cudaFree(t3);
  cudaFree(t4);
  cudaFree(t_re);
  cudaFree(t_im);
}

__global__ void fill_matrix(__restrict half * const m, const unsigned size)
{
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size) {
    m[i] = __float2half((float) i);
  }
}


/////////////////////// TESTING PURPOSE ONLY ////////////////////////////////
__global__ void mat_vec_mul(cuDoubleComplex *matrix, cuDoubleComplex *in_vect, cuDoubleComplex *out_vect)
{
  cuDoubleComplex vec0 = in_vect[0];
  cuDoubleComplex vec1 = in_vect[1];
  cuDoubleComplex vec2 = in_vect[2];

  cuDoubleComplex mat00 = matrix[0];
  cuDoubleComplex mat01 = matrix[1];
  cuDoubleComplex mat02 = matrix[2];

  cuDoubleComplex mat10 = matrix[3];
  cuDoubleComplex mat11 = matrix[4];
  cuDoubleComplex mat12 = matrix[5];

  cuDoubleComplex mat20 = matrix[6];
  cuDoubleComplex mat21 = matrix[7];
  cuDoubleComplex mat22 = matrix[8];

//Multiply 3rd row by eta
  //mat20 = make_cuDoubleComplex(cuCreal(mat20)*eta, cuCimag(mat20)*eta);
  //mat21 = make_cuDoubleComplex(cuCreal(mat21)*eta, cuCimag(mat21)*eta);
  //mat22 = make_cuDoubleComplex(cuCreal(mat22)*eta, cuCimag(mat22)*eta);

  out_vect[0] = cuCadd( cuCadd( cuCmul( mat00, vec0 ),
                             cuCmul( mat01, vec1 )),
                             cuCmul( mat02, vec2 ));

  out_vect[1] = cuCadd( cuCadd( cuCmul( mat10, vec0 ),
                             cuCmul( mat11, vec1 )),
                             cuCmul( mat12, vec2 ));

  out_vect[2] = cuCadd( cuCadd( cuCmul( mat20, vec0 ),
                                 cuCmul( mat21, vec1 )),
                                 cuCmul( mat22, vec2 ));
}

int main(int argc, char **argv)
{

  size_t mat_size = 16 * 16;

  half *a_h;
  half *b_h;
  float *c_h;

  //posix_memalign((void **)&a_h, 128, mat_size * sizeof(half));
  //posix_memalign((void **)&b_h, 128, mat_size * sizeof(half));
  //posix_memalign((void **)&c_h, 128, mat_size * sizeof(half));

  a_h = (half *) malloc(sizeof(half) * mat_size);
  b_h = (half *) malloc(sizeof(half) * mat_size);
  c_h = (float *) malloc(sizeof(float) * mat_size);

#ifdef DBUG
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
    c_h[i] =  0.0f;
    printf("%f ", c_h[i]);
  }
  printf("\n\n");
#endif

  half *a_d;
  half *b_d;
  float *c_d;

  cudaMalloc((void **)&a_d, sizeof(half) * mat_size);
  cudaMalloc((void **)&b_d, sizeof(half) * mat_size);
  cudaMalloc((void **)&c_d, sizeof(float) * mat_size);

  cudaMemcpy(a_d, a_h, sizeof(half) * mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, sizeof(half) * mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c_h, sizeof(float) * mat_size, cudaMemcpyHostToDevice);

  dot_wmma16x16<<<1, 32>>>(a_d, b_d, c_d);

  cudaMemcpy(c_h, c_d, sizeof(float) * mat_size, cudaMemcpyDeviceToHost);

#ifdef DEBUG
  printf("\n\n");
  for (unsigned i = 0; i < mat_size; i++) {
    printf("%f ", __half2float(c_h[i]));
  }
#endif

  printf("\n\nLets cmon\n\n");

  cuDoubleComplex *mat = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * 9);
  cuDoubleComplex *vec = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * 3);
  cuDoubleComplex *res = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * 3);

  for (unsigned i = 0; i < 9; i++)
    mat[i] = make_cuDoubleComplex((double) 1, (double) 1);

  for (unsigned i = 0; i < 3; i++)
    vec[i] = make_cuDoubleComplex((double) 1, (double) 1);

  cuDoubleComplex *d_mat, *d_vec, *d_res;
  cudaMalloc((void **)&d_mat, sizeof(cuDoubleComplex) * 9);
  cudaMalloc((void **)&d_vec, sizeof(cuDoubleComplex) * 3);
  cudaMalloc((void **)&d_res, sizeof(cuDoubleComplex) * 3);
  cudaMemcpy(d_mat, mat, sizeof(mat[0]) * 9, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec, vec, sizeof(vec[0]) * 3, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed;

  printf("compute...\n\n");

  cudaEventRecord(start, 0);

  complex_mma(d_mat, d_vec, d_res);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1000.0f;

  printf("Time elapsed: %f\n\n", elapsed);

  cudaMemcpy(res, d_res, sizeof(d_res[0]) * 3, cudaMemcpyDeviceToHost);

  printf("\n\n\n Final  Result\n\n");

  for (unsigned i = 0; i < 3; i++) {
    printf("IDX %d R: %.1f - I: %.1f\n", i, cuCreal(res[i]), cuCimag(res[i]));
  }

  printf("\n\nLegacy mode\n\n");

  cudaEventRecord(start, 0);

  mat_vec_mul<<<1, 1>>>(d_mat, d_vec, d_res);
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1000.0f;

  printf("Time elapsed: %f\n\n", elapsed);

  cudaMemcpy(res, d_res, sizeof(d_res[0]) * 3, cudaMemcpyDeviceToHost);
  for (unsigned i = 0; i < 3; i++) {
    printf("IDX %d R: %.1f - I: %.1f\n", i, cuCreal(res[i]), cuCimag(res[i]));
  }

  free(c_h);
  free(mat);
  free(vec);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  cudaFree(d_mat);
  cudaFree(d_vec);

  return EXIT_SUCCESS;
}
