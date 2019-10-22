#include <stdio.h>
#include <stdlib.h>
#include "wmma-common.h"
#include "matrix-utility.h"


int main(int argc, char **argv)
{
  half *h_a;
  allocate_cpu_side_tcu_matrix_half((void **)&h_a, 5);
  fill_tcu_matrix_half(h_a, 5);

  for (unsigned i = 0; i < 16; i++) {
    for (unsigned j = 0; j < 16; j++) {
      printf("%.1f\t", __half2float(h_a[j+i*16]));
    }
    printf("\n");
  }

  return EXIT_SUCCESS;
}
