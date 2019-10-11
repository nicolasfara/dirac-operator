#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas-utility.h"
#include "cuda_utility.h"

using namespace std;

int main(int argc, char **argv)
{
  /* Parse input args */
  int mat_side = 3;
  int batch = 10;

  if (argc == 3) {
    mat_side = atoi(argv[1]); // Matrix side
    batch = atoi(argv[2]);    // Number of MMA
  } else {
    fprintf(stderr, "./%s SIDE BATCH", argv[0]);
    return EXIT_FAILURE;
  }

  printf("Processing input args: %d side, %d batch", mat_side, batch);

  cublasHandle_t handle;

  checkCublas(cublasCreate(&handle));
  checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  half *dA_mat[batch];
  half *dB_mat[batch];
  half *dC_mat[batch];

  for (unsigned i = 0; i < batch; i++){
    allocate_matrix((void **)&dA_mat[i], sizeof(half)*mat_side*mat_side);
    allocate_matrix((void **)&dB_mat[i], sizeof(half)*mat_side*mat_side);
    allocate_matrix((void **)&dC_mat[i], sizeof(half)*mat_side*mat_side);
  }

  for (unsigned i = 0; i < batch; i++) {
    fill_matrix(dA_mat[i], mat_side*mat_side);
    fill_matrix(dB_mat[i], mat_side*mat_side);
    fill_matrix(dC_mat[i], mat_side*mat_side);
  }

  mma_batched(handle, mat_side, mat_side, mat_side, dA_mat, dB_mat, dC_mat, batch);

  display_matrix(dA_mat[0], mat_side, mat_side);

  return EXIT_SUCCESS;
}
