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

void allocate_cpu_side_matrix_half(void **h_ptr, const size_t rows, const size_t cols, const unsigned matrix_count)
{
  // Allocate on CPU the matrix with allignment
  posix_memalign(h_ptr, 128, rows*cols*matrix_count*sizeof(half));
}

void fill_matrix_half(half *h_ptr, const size_t rows, const size_t cols, const unsigned matrix_count)
{
  for (unsigned z = 0; z < matrix_count; z++) {
    for (unsigned i = 0; i < rows; i++)
      for (unsigned j = 0; j < cols; j++) {
        unsigned index = (rows*cols*z)+(j+i*cols);
        h_ptr[index] = __float2half(j+i*cols);
      }
  }
}

void allocate_cpu_side_tcu_matrix_half(void **h_ptr, const size_t rows, const size_t cols, const unsigned matrix_count)
{
  const unsigned mat_16x16 = 16*16;
  const unsigned mat_16x16_count = matrix_count/5; //On each 16x16 matrix we could insert at most 5 matrix 3x3 on the diagonal of the big one.
  posix_memalign(h_ptr, 128, mat_16x16*mat_16x16_count*sizeof(half));
}

void fill_tcu_matrix_half(half *h_ptr, const size_t rows, const size_t cols, const unsigned matrix_count)
{
  for (unsigned z = 0; z < matrix_count; z++) {
    //TODO: compose the matrix on the diagonal
  }
}

#endif /* !MATRIX_UTILITY_H */
