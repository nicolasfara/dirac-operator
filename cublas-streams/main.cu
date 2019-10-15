#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas-utility.h"
#include "cuda_utility.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

using namespace std;

int main(int argc, char **argv)
{
  /* Parse input args */
  int mat_side = 3;
  int batch_complex = 10;
  int batch = batch_complex*4;

  if (argc == 3) {
    mat_side = atoi(argv[1]); // Matrix side
    batch_complex = atoi(argv[2]);    // Number of MMA
    batch = batch_complex*4;
  } else {
    fprintf(stderr, "./%s SIDE BATCH", argv[0]);
    return EXIT_FAILURE;
  }

  printf("Processing input args: %d side, %d batch\n\n", mat_side, batch_complex);

  cublasHandle_t handle;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  checkCublas(cublasCreate(&handle));
  checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  half **devPtrA = (half **)malloc(batch_complex * sizeof(*devPtrA));
  half **devPtrB = (half **)malloc(batch_complex * sizeof(*devPtrB));
  half **devPtrC = (half **)malloc(batch_complex * sizeof(*devPtrC));
  half **devPtrA_dev, **devPtrB_dev, **devPtrC_dev;

  for (int i = 0; i < batch ; i++) {
    allocate_matrix((void **)&devPtrA[i], mat_side * mat_side * sizeof(devPtrA[0][0]));
    allocate_matrix((void **)&devPtrB[i], mat_side * mat_side * sizeof(devPtrB[0][0]));
    allocate_matrix((void **)&devPtrC[i], mat_side * mat_side * sizeof(devPtrC[0][0]));
  }

  checkCudaErrors(cudaMalloc((void **)&devPtrA_dev, batch_complex * sizeof(*devPtrA)));
  checkCudaErrors(cudaMalloc((void **)&devPtrB_dev, batch_complex * sizeof(*devPtrB)));
  checkCudaErrors(cudaMalloc((void **)&devPtrC_dev, batch_complex * sizeof(*devPtrC)));
  checkCudaErrors(cudaMemcpy(devPtrA_dev, devPtrA, batch_complex * sizeof(*devPtrA), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devPtrB_dev, devPtrB, batch_complex * sizeof(*devPtrB), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devPtrC_dev, devPtrC, batch_complex * sizeof(*devPtrC), cudaMemcpyHostToDevice));

  fill_matrix(devPtrA[0], mat_side, mat_side);
  fill_matrix(devPtrB[0], mat_side, mat_side);
  fill_matrix(devPtrC[0], mat_side, mat_side);

  /****************** Test without tcu ****************************************/

  checkCudaErrors(cudaEventRecord(start, 0));

  mma_batched(handle, mat_side, mat_side, mat_side, devPtrA_dev, devPtrB_dev, devPtrC_dev, batch_complex);
  display_matrix(devPtrC[0], mat_side, mat_side);

  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  float elapsed;
  checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
  elapsed /= 1000.0f;
  printf("Elapsed WITHOUT TCU:\t %fs\n", elapsed);

  /*************************** Test with TCU **********************************/

  checkCudaErrors(cudaEventRecord(start, 0));

  mma_batched_tcu(handle, mat_side, mat_side, mat_side, (void **)devPtrA_dev, (void **)devPtrB_dev, (void **)devPtrC_dev, batch_complex);
  display_matrix(devPtrC[0], mat_side, mat_side);

  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
  elapsed /= 1000.0f;
  printf("Elapsed WITH TCU:\t %fs\n", elapsed);

  /************************* COMPLEX SECTION **********************************/

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

  checkCudaErrors(cudaEventRecord(start, 0));

  test_3x3matvec(d_mat, d_vec, d_res, batch);

  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
  elapsed /= 1000.0f;
  printf("Elapsed complex:\t %fs\n", elapsed);

  return EXIT_SUCCESS;
}
