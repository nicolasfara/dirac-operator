#pragma once

#include <mma.h>
//#include "common-cuda.h"

using namespace nvcuda;

#define TENSOR_MAT_SIZE 256

__device__ static __inline__ void su3Mapper(const su3_soa * const in_mat, int in_mat_idx[8], half * const out_mat)
{
    //Collasso due valori adiacenti nello stesso valore. esempio: 0-1 -> 0, 2-3 -> 1 ecc.
    //Serve per indicizzare nel su3_soa la matrice di interesse e siccome some due valori (numero complesso) l'indice pari agisce sulla parte reale, quello dispari 
    //nella parte immaginaria
    int su3_idx = (threadIdx.x/2) - (threadIdx.x & 0x1);
    const unsigned lut[] = { 0,4,8,12,64,68,72,78,128,132,136,140,192,196,200,204 };

    int half_idx = lut[threadIdx.x]; //threadIdx.x va da 0-15

    if ((threadIdx.x & 0x1) == 0) { //Parte reale, controllo che threadIdx.x sia pari
        out_mat[half_idx]    = __float2half(cuCreal(in_mat->r0.c0[in_mat_idx[su3_idx]]));
        out_mat[half_idx+1]  = __float2half(cuCreal(in_mat->r0.c1[in_mat_idx[su3_idx]]));
        out_mat[half_idx+2]  = __float2half(cuCreal(in_mat->r0.c2[in_mat_idx[su3_idx]]));

        out_mat[half_idx+16] = __float2half(cuCreal(in_mat->r1.c0[in_mat_idx[su3_idx]]));
        out_mat[half_idx+17] = __float2half(cuCreal(in_mat->r1.c1[in_mat_idx[su3_idx]]));
        out_mat[half_idx+18] = __float2half(cuCreal(in_mat->r1.c2[in_mat_idx[su3_idx]]));
        
        out_mat[half_idx+32] = __float2half(cuCreal(in_mat->r2.c0[in_mat_idx[su3_idx]]));
        out_mat[half_idx+33] = __float2half(cuCreal(in_mat->r2.c1[in_mat_idx[su3_idx]]));
        out_mat[half_idx+34] = __float2half(cuCreal(in_mat->r2.c2[in_mat_idx[su3_idx]]));
    } else { // Parte immaginaria
        out_mat[half_idx]    = __float2half(cuCimag(in_mat->r0.c0[in_mat_idx[su3_idx]]));
        out_mat[half_idx+1]  = __float2half(cuCimag(in_mat->r0.c1[in_mat_idx[su3_idx]]));
        out_mat[half_idx+2]  = __float2half(cuCimag(in_mat->r0.c2[in_mat_idx[su3_idx]]));

        out_mat[half_idx+16] = __float2half(cuCimag(in_mat->r1.c0[in_mat_idx[su3_idx]]));
        out_mat[half_idx+17] = __float2half(cuCimag(in_mat->r1.c1[in_mat_idx[su3_idx]]));
        out_mat[half_idx+18] = __float2half(cuCimag(in_mat->r1.c2[in_mat_idx[su3_idx]]));
        
        out_mat[half_idx+32] = __float2half(cuCimag(in_mat->r2.c0[in_mat_idx[su3_idx]]));
        out_mat[half_idx+33] = __float2half(cuCimag(in_mat->r2.c1[in_mat_idx[su3_idx]]));
        out_mat[half_idx+34] = __float2half(cuCimag(in_mat->r2.c2[in_mat_idx[su3_idx]]));
    }
}

__device__ static __inline__ void fermionMapper(const vec3_soa * const in_vec, int in_vec_idx[8], half * const out_vec)
{
    //Collasso due valori adiacenti nello stesso valore. esempio: 0-1 -> 0, 2-3 -> 1 ecc.
    //Serve per indicizzare nel su3_soa la matrice di interesse e siccome some due valori (numero complesso) l'indice pari agisce sulla parte reale, quello dispari 
    //nella parte immaginaria
    int vec3_idx = (threadIdx.x/2) - (threadIdx.x & 0x1);
    const unsigned lut[] = { 0,1,130,131,4,5,134,135,8,4,138,139,10,11,142,143 };

    int half_idx = lut[threadIdx.x - 16]; //threadIdx va da 16-31

    if (threadIdx.x & 0x1 == 0) { // Controllo che threadIdx.x sia pari
        out_vec[half_idx]    = __float2half(cuCreal(in_vec->c0[in_vec_idx[vec3_idx]]));
        out_vec[half_idx+16] = __float2half(cuCreal(in_vec->c1[in_vec_idx[vec3_idx]]));
        out_vec[half_idx+32] = __float2half(cuCreal(in_vec->c2[in_vec_idx[vec3_idx]]));

        out_vec[half_idx+65] = __float2half(cuCreal(in_vec->c0[in_vec_idx[vec3_idx]]));
        out_vec[half_idx+81] = __float2half(cuCreal(in_vec->c1[in_vec_idx[vec3_idx]]));
        out_vec[half_idx+97] = __float2half(cuCreal(in_vec->c2[in_vec_idx[vec3_idx]]));

    } else {
        out_vec[half_idx]    = __float2half(cuCimag(in_vec->c0[in_vec_idx[vec3_idx]]));
        out_vec[half_idx+16] = __float2half(cuCimag(in_vec->c1[in_vec_idx[vec3_idx]]));
        out_vec[half_idx+32] = __float2half(cuCimag(in_vec->c2[in_vec_idx[vec3_idx]]));

        //Parte immaginaria con segno opposto
        out_vec[half_idx+63] = __float2half(-cuCimag(in_vec->c0[in_vec_idx[vec3_idx]]));
        out_vec[half_idx+79] = __float2half(-cuCimag(in_vec->c1[in_vec_idx[vec3_idx]]));
        out_vec[half_idx+95] = __float2half(-cuCimag(in_vec->c2[in_vec_idx[vec3_idx]]));
    }    
}

__device__ static __inline__ void tensorToVec(const half * const res_vec, vec3 * const out_vec)
{
    const unsigned lut[] = { 0,1,2,3,68,69,70,71,136,137,138,139,204,205,206,207 };
    
    int half_idx = lut[threadIdx.x];
    int lidx = threadIdx.x/2;

    if ((threadIdx.x & 0x1) == 0) {
        out_vec[lidx].c0 = make_cuDoubleComplex(0.0f, 0.0f);//make_cuDoubleComplex(__half2float(res_vec[half_idx]),    cuCimag(out_vec[lidx]->c0));
        out_vec[lidx].c1 = make_cuDoubleComplex(0.0f, 0.0f);//make_cuDoubleComplex(__half2float(res_vec[half_idx+16]), cuCimag(out_vec[lidx]->c1));
        out_vec[lidx].c2 = make_cuDoubleComplex(0.0f, 0.0f);//make_cuDoubleComplex(__half2float(res_vec[half_idx+32]), cuCimag(out_vec[lidx]->c2));
    } else {
        out_vec[lidx].c0 = make_cuDoubleComplex(0.0f, 0.0f);//make_cuDoubleComplex(cuCreal(out_vec[lidx]->c0), __half2float(res_vec[half_idx]));
        out_vec[lidx].c1 = make_cuDoubleComplex(0.0f, 0.0f);//make_cuDoubleComplex(cuCreal(out_vec[lidx]->c1), __half2float(res_vec[half_idx+16]));
        out_vec[lidx].c2 = make_cuDoubleComplex(0.0f, 0.0f);//make_cuDoubleComplex(cuCreal(out_vec[lidx]->c2), __half2float(res_vec[half_idx+32]));
    }
}

__device__ static __inline__ void tensor_mat_vec_mul( const su3_soa * const mat,
                                                        int idx_mat[8],
                                                        const vec3_soa * in_vec,
                                                        int idx_vec[8],
                                                        vec3 * const out_vec )
{
    __shared__ half t_mat[TENSOR_MAT_SIZE];
    __shared__ half t_in_vec[TENSOR_MAT_SIZE];
    __shared__ half t_out_vec[TENSOR_MAT_SIZE];

    //Use function to map vec3_soa and su3_soa to half array
    //if (threadIdx.x < 16)
    //    su3Mapper(mat, idx_mat, t_mat); //thread 0-15
    //else
    //    fermionMapper(in_vec, idx_vec, t_in_vec); //thread 16-31

       // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, t_mat, 16);
    wmma::load_matrix_sync(b_frag, t_in_vec, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(t_out_vec, c_frag, 16, wmma::mem_row_major);


    //Solo 16 thread convertono i dati in vettori
    //if (threadIdx.x < 16) tensorToVec(t_out_vec, out_vec);
}
