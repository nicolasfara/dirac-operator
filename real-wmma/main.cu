#include <stdio.h>
#include <stdlib.h>
#include "wmma-common.h"
#include "matrix-utility.h"

#define TCU_MAT 10

int main(int argc, char **argv)
{
  half *h_a;
  half *h_b;
  half *h_c;
  allocate_cpu_side_tcu_matrix_half((void **)&h_a, TCU_MAT);
  allocate_cpu_side_tcu_matrix_half((void **)&h_b, TCU_MAT);
  allocate_cpu_side_tcu_matrix_half((void **)&h_c, TCU_MAT);
  fill_tcu_matrix_half(h_a, TCU_MAT);
  fill_tcu_matrix_half(h_b, TCU_MAT);
  fill_zero_tcu_matrix(h_c, TCU_MAT);

  half *d_a;
  half *d_b;
  half *d_c;
  allocate_gpu_side_matrix_half((void **)&d_a, TCU_MAT);
  allocate_gpu_side_matrix_half((void **)&d_b, TCU_MAT);
  allocate_gpu_side_matrix_half((void **)&d_c, TCU_MAT);
  copy_matrix_to_gpu_half(d_a, h_a, TCU_MAT);
  copy_matrix_to_gpu_half(d_b, h_b, TCU_MAT);
  copy_matrix_to_gpu_half(d_c, h_c, TCU_MAT);


  return EXIT_SUCCESS;
}
