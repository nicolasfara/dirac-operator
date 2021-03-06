#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_utility.h"
#include "cuda_utility.h"

#define RUN 10

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

  half **devPtrA = (half **)malloc(batch * sizeof(*devPtrA));
  half **devPtrB = (half **)malloc(batch * sizeof(*devPtrB));
  half **devPtrC = (half **)malloc(batch * sizeof(*devPtrC));
  half **devPtrA_dev, **devPtrB_dev, **devPtrC_dev;

  for (int i = 0; i < batch ; i++) {
    allocate_matrix((void **)&devPtrA[i], mat_side * mat_side * sizeof(devPtrA[0][0]));
    allocate_matrix((void **)&devPtrB[i], mat_side * mat_side * sizeof(devPtrB[0][0]));
    allocate_matrix((void **)&devPtrC[i], mat_side * mat_side * sizeof(devPtrC[0][0]));
  }

  checkCudaErrors(cudaMalloc((void **)&devPtrA_dev, batch * sizeof(*devPtrA)));
  checkCudaErrors(cudaMalloc((void **)&devPtrB_dev, batch * sizeof(*devPtrB)));
  checkCudaErrors(cudaMalloc((void **)&devPtrC_dev, batch * sizeof(*devPtrC)));
  checkCudaErrors(cudaMemcpy(devPtrA_dev, devPtrA, batch * sizeof(*devPtrA), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devPtrB_dev, devPtrB, batch * sizeof(*devPtrB), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devPtrC_dev, devPtrC, batch * sizeof(*devPtrC), cudaMemcpyHostToDevice));

  fill_matrix(devPtrA[0], mat_side, mat_side);
  fill_matrix(devPtrB[0], mat_side, mat_side);
  fill_matrix(devPtrC[0], mat_side, mat_side);

  float elapsed = 0;
  float sum_elapsed = 0;

  /****************** Test without tcu ****************************************/


  for (unsigned i = 0; i < RUN; i++) {
    checkCudaErrors(cudaEventRecord(start, 0));

    mma_batched(handle, mat_side, mat_side, mat_side, devPtrA_dev, devPtrB_dev, devPtrC_dev, batch);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= 1000.0f;
    sum_elapsed += elapsed;
  }

  //display_matrix(devPtrC[0], mat_side, mat_side);
  printf("Elapsed WITHOUT TCU:\t %fs\n", sum_elapsed/RUN);

  /*************************** Test with TCU **********************************/

  sum_elapsed = 0.0f;

  for (unsigned i = 0; i < RUN; i++) {
    checkCudaErrors(cudaEventRecord(start, 0));

    mma_batched_tcu(handle, mat_side, mat_side, mat_side, (void **)devPtrA_dev, (void **)devPtrB_dev, (void **)devPtrC_dev, batch);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= 1000.0f;
    sum_elapsed += elapsed;
  }

  //display_matrix(devPtrC[0], mat_side, mat_side);
  printf("Elapsed WITH TCU:\t %fs\n", sum_elapsed/RUN);

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

  sum_elapsed = 0.0f;

  for (unsigned i = 0; i < RUN; i++) {
    checkCudaErrors(cudaEventRecord(start, 0));

    test_3x3matvec(d_mat, d_vec, d_res, batch_complex);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= 1000.0f;
    sum_elapsed += elapsed;
  }
  printf("Elapsed complex:\t %fs\n", sum_elapsed/RUN);

  /***************** Clean the system *****************************/
  for (int i = 0; i < batch ; i++) {
    cudaFree(devPtrA[i]);
    cudaFree(devPtrB[i]);
    cudaFree(devPtrC[i]);
  }
  cudaFree(devPtrA_dev);
  cudaFree(devPtrB_dev);
  cudaFree(devPtrC_dev);

  free(devPtrA);
  free(devPtrB);
  free(devPtrC);

  return EXIT_SUCCESS;
}
