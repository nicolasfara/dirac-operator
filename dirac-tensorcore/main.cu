#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "common-cuda.h"

__host__ __device__ static __inline__ void mat_vec_mul( const __restrict su3_soa * const matrix,
                                                        const int idx_mat,
                                                        const int eta,
                                                        const __restrict vec3_soa * const in_vect,
                                                        const int idx_vect,
                                                        __restrict vec3 * const out_vect) {

  cuDoubleComplex vec0 = in_vect->c0[idx_vect];
  cuDoubleComplex vec1 = in_vect->c1[idx_vect];
  cuDoubleComplex vec2 = in_vect->c2[idx_vect];

  cuDoubleComplex mat00 = matrix->r0.c0[idx_mat];
  cuDoubleComplex mat01 = matrix->r0.c1[idx_mat];
  cuDoubleComplex mat02 = matrix->r0.c2[idx_mat];

  cuDoubleComplex mat10 = matrix->r1.c0[idx_mat];
  cuDoubleComplex mat11 = matrix->r1.c1[idx_mat];
  cuDoubleComplex mat12 = matrix->r1.c2[idx_mat];

#ifdef READROW3
// Load 3rd matrix row from global memory
  cuDoubleComplex mat20 = matrix->r2.c0[idx_mat];
  cuDoubleComplex mat21 = matrix->r2.c1[idx_mat];
  cuDoubleComplex mat22 = matrix->r2.c2[idx_mat];
#else
//Compute 3rd matrix row from the first two
  cuDoubleComplex mat20 = cuConj( cuCsub( cuCmul( mat01, mat12 ), cuCmul( mat02, mat11) ) );
  cuDoubleComplex mat21 = cuConj( cuCsub( cuCmul( mat02, mat10 ), cuCmul( mat00, mat12) ) ); 
  cuDoubleComplex mat22 = cuConj( cuCsub( cuCmul( mat00, mat11 ), cuCmul( mat01, mat10) ) );
#endif

//Multiply 3rd row by eta
  mat20 = make_cuDoubleComplex(cuCreal(mat20)*eta, cuCimag(mat20)*eta);
  mat21 = make_cuDoubleComplex(cuCreal(mat21)*eta, cuCimag(mat21)*eta);
  mat22 = make_cuDoubleComplex(cuCreal(mat22)*eta, cuCimag(mat22)*eta);

  out_vect->c0 = cuCadd( cuCadd( cuCmul( mat00, vec0 ),
                                 cuCmul( mat01, vec1 )),
                                 cuCmul( mat02, vec2 ));

  out_vect->c1 = cuCadd( cuCadd( cuCmul( mat10, vec0 ),
                                 cuCmul( mat11, vec1 )),
                                 cuCmul( mat12, vec2 ));

  out_vect->c2 = cuCadd( cuCadd( cuCmul( mat20, vec0 ),
                                 cuCmul( mat21, vec1 )),
                                 cuCmul( mat22, vec2 ));

}

__host__ __device__ static __inline__ void conjmat_vec_mul( const __restrict su3_soa * const matrix,
                                                            const int idx_mat,
                                                            const int eta,
                                                            const __restrict vec3_soa * const in_vect,
                                                            const int idx_vect,
                                                            __restrict vec3 * const out_vect) {

  cuDoubleComplex vec0 = in_vect->c0[idx_vect];
  cuDoubleComplex vec1 = in_vect->c1[idx_vect];
  cuDoubleComplex vec2 = in_vect->c2[idx_vect];

  cuDoubleComplex mat00 = matrix->r0.c0[idx_mat];
  cuDoubleComplex mat01 = matrix->r0.c1[idx_mat];
  cuDoubleComplex mat02 = matrix->r0.c2[idx_mat];

  cuDoubleComplex mat10 = matrix->r1.c0[idx_mat];
  cuDoubleComplex mat11 = matrix->r1.c1[idx_mat];
  cuDoubleComplex mat12 = matrix->r1.c2[idx_mat];

#ifdef READROW3
// Load 3rd matrix row from global memory
//  cuDoubleComplex mat20 = matrix->r2.c0[idx_mat];
//  cuDoubleComplex mat21 = matrix->r2.c1[idx_mat];
//  cuDoubleComplex mat22 = matrix->r2.c2[idx_mat];
#else
//Compute 3rd matrix row from the first two
  cuDoubleComplex mat20 = cuConj( cuCsub( cuCmul( mat01, mat12 ), cuCmul( mat02, mat11) ) );
  cuDoubleComplex mat21 = cuConj( cuCsub( cuCmul( mat02, mat10 ), cuCmul( mat00, mat12) ) );
  cuDoubleComplex mat22 = cuConj( cuCsub( cuCmul( mat00, mat11 ), cuCmul( mat01, mat10) ) );
#endif

//Multiply 3rd row by eta
  mat20 = make_cuDoubleComplex(cuCreal(mat20)*eta, cuCimag(mat20)*eta);
  mat21 = make_cuDoubleComplex(cuCreal(mat21)*eta, cuCimag(mat21)*eta);
  mat22 = make_cuDoubleComplex(cuCreal(mat22)*eta, cuCimag(mat22)*eta);

  out_vect->c0 = cuCadd( cuCadd( cuCmul( cuConj(mat00), vec0 ),
                                 cuCmul( cuConj(mat10), vec1 )),
                                 cuCmul( cuConj(mat20), vec2 ));

  out_vect->c1 = cuCadd( cuCadd( cuCmul( cuConj(mat01), vec0 ),
                                 cuCmul( cuConj(mat11), vec1 )),
                                 cuCmul( cuConj(mat21), vec2 ));

  out_vect->c2 = cuCadd( cuCadd( cuCmul( cuConj(mat02), vec0 ),
                                 cuCmul( cuConj(mat12), vec1 )),
                                 cuCmul( cuConj(mat22), vec2 ));

}

__host__ __device__ static __inline__ vec3 sumResult ( vec3 aux, vec3 aux_tmp) {

  aux.c0 = cuCadd ( aux.c0, aux_tmp.c0);
  aux.c1 = cuCadd ( aux.c1, aux_tmp.c1);
  aux.c2 = cuCadd ( aux.c2, aux_tmp.c2);

  return aux;

}

__host__ __device__ static __inline__ vec3 subResult ( vec3 aux, vec3 aux_tmp) {

  aux.c0 = cuCsub ( aux.c0, aux_tmp.c0);
  aux.c1 = cuCsub ( aux.c1, aux_tmp.c1);
  aux.c2 = cuCsub ( aux.c2, aux_tmp.c2);

  return aux;

}

__global__ void Deo(const __restrict su3_soa * const u, __restrict vec3_soa * const out, const __restrict vec3_soa * const in) {

  int x, y, z, t, xm, ym, zm, tm, xp, yp, zp, tp, idxh, eta; //, idx;

  vec3 aux_tmp;
  vec3 aux;         

  idxh = ((blockIdx.z * blockDim.z + threadIdx.z) * nxh * ny)                                                             
       + ((blockIdx.y * blockDim.y + threadIdx.y) * nxh)                                                                 
       +  (blockIdx.x * blockDim.x + threadIdx.x); // idxh = snum(x,y,z,t)   

//  idx = 2*idxh;
//  t = (idx / vol3) % nt;
//  z = (idx / vol2) % nz;
//  y =   (blockIdx.y * blockDim.y + threadIdx.y);
//  x = 2*(blockIdx.x * blockDim.x + threadIdx.x) + ((y+z+t) % 2);

  t =   (blockIdx.z * blockDim.z + threadIdx.z) / nz;
  z =   (blockIdx.z * blockDim.z + threadIdx.z) % nz;
  y =   (blockIdx.y * blockDim.y + threadIdx.y);
  x = 2*(blockIdx.x * blockDim.x + threadIdx.x) + ((y+z+t) & 0x1);

  xm = x - 1;
  xm = xm + (((xm >> 31) & 0x1) * nx);
  ym = y -1;
  ym = ym + (((ym >> 31) & 0x1) * ny);
  zm = z -1;
  zm = zm + (((zm >> 31) & 0x1) * nz);
  tm = t -1;
  tm = tm + (((tm >> 31) & 0x1) * nt);

  xp = (x+1);
  xp *= (((xp-nx) >> 31) & 0x1);
  yp = (y+1);
  yp *= (((yp-ny) >> 31) & 0x1);
  zp = (z+1);
  zp *= (((zp-nz) >> 31) & 0x1);
  tp = (t+1);
  tp *= (((tp-nt) >> 31) & 0x1);

  eta = 1;
// mat_vec_mul( &(u_work[snum(x,y,z,t)       ]), &(in[snum(xp,y,z,t)]), &aux_tmp );
  mat_vec_mul( &u[0], idxh, eta, in, snum(xp,y,z,t), &aux_tmp );
  aux = aux_tmp;

  eta = 1 - ( 2*(x & 0x1) ); // if (x % 2 = 0) eta = 1 else -1
// mat_vec_mul( &(u_work[snum(x,y,z,t) + size ]), &(in[snum(x,yp,z,t)]), &aux_tmp );
  mat_vec_mul( &u[2], idxh, eta, in, snum(x,yp,z,t), &aux_tmp );
  aux = sumResult(aux, aux_tmp);

  eta = 1 - ( 2*((x+y) & 0x1) );
// mat_vec_mul( &(u_work[snum(x,y,z,t) + size2]), &(in[snum(x,y,zp,t)]), &aux_tmp );
  mat_vec_mul( &u[4], idxh, eta, in, snum(x,y,zp,t), &aux_tmp);
  aux = sumResult(aux, aux_tmp);

  eta = 1 - ( 2*((x+y+z) & 0x1) );
// mat_vec_mul( &(u_work[snum(x,y,z,t) + size3]), &(in[snum(x,y,z,tp)]), &aux_tmp );
  mat_vec_mul( &u[6], idxh, eta, in, snum(x,y,z,tp), &aux_tmp );
  aux = sumResult(aux, aux_tmp);

//////////////////////////////////////////////////////////////////////////////////////////////
    
  eta = 1;
// conjmat_vec_mul( &(u_work[sizeh + snum(xm,y,z,t)      ]), &(in[ snum(xm,y,z,t) ]), &aux_tmp );
  conjmat_vec_mul( &u[1], snum(xm,y,z,t), eta, in, snum(xm,y,z,t), &aux_tmp );
  aux = subResult(aux, aux_tmp);

  eta = 1 - ( 2*(x & 0x1) );
// conjmat_vec_mul( &(u_work[sizeh + snum(x,ym,z,t) + size ]), &(in[ snum(x,ym,z,t) ]), &aux_tmp );
  conjmat_vec_mul( &u[3], snum(x,ym,z,t), eta, in, snum(x,ym,z,t), &aux_tmp );
  aux = subResult(aux, aux_tmp);

  eta = 1 - ( 2*((x+y) & 0x1) );
// conjmat_vec_mul( &(u_work[sizeh + snum(x,y,zm,t) + size2]), &(in[ snum(x,y,zm,t) ]), &aux_tmp );
  conjmat_vec_mul( &u[5], snum(x,y,zm,t), eta, in, snum(x,y,zm,t), &aux_tmp );
  aux = subResult(aux, aux_tmp);

  eta = 1 - ( 2*((x+y+z) & 0x1) );
// conjmat_vec_mul( &(u_work[sizeh + snum(x,y,z,tm) + size3]), &(in[ snum(x,y,z,tm) ]), &aux_tmp );
  conjmat_vec_mul( &u[7], snum(x,y,z,tm), eta, in, snum(x,y,z,tm), &aux_tmp );
  aux = subResult(aux, aux_tmp);

//////////////////////////////////////////////////////////////////////////////////////////////

  out->c0[idxh] = make_cuDoubleComplex(cuCreal(aux.c0)*0.5, cuCimag(aux.c0)*0.5);
  out->c1[idxh] = make_cuDoubleComplex(cuCreal(aux.c1)*0.5, cuCimag(aux.c1)*0.5);
  out->c2[idxh] = make_cuDoubleComplex(cuCreal(aux.c2)*0.5, cuCimag(aux.c2)*0.5);

}

__global__ void Doe(const __restrict su3_soa * const u, __restrict vec3_soa * const out, const __restrict vec3_soa * const in) {

  int x, y, z, t, xm, ym, zm, tm, xp, yp, zp, tp, idxh, eta; //, idx;

  vec3 aux_tmp;
  vec3 aux;

  idxh = ((blockIdx.z * blockDim.z + threadIdx.z) * nxh * ny)                                                             
       + ((blockIdx.y * blockDim.y + threadIdx.y) * nxh)                                                                 
       +  (blockIdx.x * blockDim.x + threadIdx.x); // idxh = snum(x,y,z,t)   

//  idx = 2*idxh;
//  t = (idx / vol3) % nt;
//  z = (idx / vol2) % nz;
//  y =   (blockIdx.y * blockDim.y + threadIdx.y);
//  x = 2*(blockIdx.x * blockDim.x + threadIdx.x) + ((y+z+t+1) % 2);

  t =   (blockIdx.z * blockDim.z + threadIdx.z) / nz;
  z =   (blockIdx.z * blockDim.z + threadIdx.z) % nz;
  y =   (blockIdx.y * blockDim.y + threadIdx.y);
  x = 2*(blockIdx.x * blockDim.x + threadIdx.x) + ((y+z+t+1) & 0x1);

  xm = x - 1;
  xm = xm + (((xm >> 31) & 0x1) * nx);
  ym = y -1;
  ym = ym + (((ym >> 31) & 0x1) * ny);
  zm = z -1;
  zm = zm + (((zm >> 31) & 0x1) * nz);
  tm = t -1;
  tm = tm + (((tm >> 31) & 0x1) * nt);

  xp = (x+1);
  xp *= (((xp-nx) >> 31) & 0x1);
  yp = (y+1);
  yp *= (((yp-ny) >> 31) & 0x1);
  zp = (z+1);
  zp *= (((zp-nz) >> 31) & 0x1);
  tp = (t+1);
  tp *= (((tp-nt) >> 31) & 0x1);

  eta = 1;
// mat_vec_mul( &(u_work[snum(x,y,z,t) + sizeh      ]), &(in[ snum(xp,y,z,t) ]), &aux_tmp );
  mat_vec_mul( &u[1], idxh, eta, in, snum(xp,y,z,t), &aux_tmp );
  aux = aux_tmp;

  eta = 1 - ( 2*(x & 0x1) );
// mat_vec_mul( &(u_work[snum(x,y,z,t) + sizeh + size ]), &(in[ snum(x,yp,z,t) ]), &aux_tmp );
  mat_vec_mul( &u[3], idxh, eta, in, snum(x,yp,z,t), &aux_tmp );
  aux = sumResult(aux, aux_tmp);

  eta = 1 - ( 2*((x+y) & 0x1) );
// mat_vec_mul( &( u_work[snum(x,y,z,t) + sizeh + size2]), &(in[ snum(x,y,zp,t) ]), &aux_tmp );
  mat_vec_mul( &u[5], idxh, eta, in, snum(x,y,zp,t), &aux_tmp );
  aux = sumResult(aux, aux_tmp);

  eta = 1 - ( 2*((x+y+z) & 0x1) );
// mat_vec_mul( &(u_work[snum(x,y,z,t) + sizeh + size3]), &(in[ snum(x,y,z,tp) ]), &aux_tmp );
  mat_vec_mul( &u[7], idxh, eta, in, snum(x,y,z,tp), &aux_tmp );
  aux = sumResult(aux, aux_tmp);

//////////////////////////////////////////////////////////////////////////////////////////////

  eta = 1;
// conjmat_vec_mul( &(u_work[snum(xm,y,z,t)      ]), &(in[ snum(xm,y,z,t) ]), &aux_tmp );
  conjmat_vec_mul( &u[0], snum(xm,y,z,t), eta, in, snum(xm,y,z,t), &aux_tmp );
  aux = subResult(aux, aux_tmp);

  eta = 1 - ( 2*(x & 0x1) );
// conjmat_vec_mul( &(u_work[snum(x,ym,z,t) + size ]), &(in[ snum(x,ym,z,t) ]), &aux_tmp );
  conjmat_vec_mul( &u[2], snum(x,ym,z,t), eta, in, snum(x,ym,z,t), &aux_tmp );
  aux = subResult(aux, aux_tmp);

  eta = 1 - ( 2*((x+y) & 0x1) );
// conjmat_vec_mul( &(u_work[snum(x,y,zm,t) + size2]), &(in[ snum(x,y,zm,t) ]), &aux_tmp );
  conjmat_vec_mul( &u[4], snum(x,y,zm,t), eta, in, snum(x,y,zm,t), &aux_tmp );
  aux = subResult(aux, aux_tmp);

  eta = 1 - ( 2*((x+y+z) & 0x1) );
// conjmat_vec_mul( &(u_work[snum(x,y,z,tm) + size3]), &(in[ snum(x,y,z,tm) ]), &aux_tmp );
  conjmat_vec_mul( &u[6], snum(x,y,z,tm), eta, in, snum(x,y,z,tm), &aux_tmp );
  aux = subResult(aux, aux_tmp);

//////////////////////////////////////////////////////////////////////////////////////////////

  out->c0[idxh] = make_cuDoubleComplex(cuCreal(aux.c0)*0.5, cuCimag(aux.c0)*0.5);
  out->c1[idxh] = make_cuDoubleComplex(cuCreal(aux.c1)*0.5, cuCimag(aux.c1)*0.5);
  out->c2[idxh] = make_cuDoubleComplex(cuCreal(aux.c2)*0.5, cuCimag(aux.c2)*0.5);

}


int main() {

  int i;
  struct timeval t0, t1;
  double dt_tot = 0.0;

  dim3 dimBlockK1 (DIM_BLOCK_X, DIM_BLOCK_Y, DIM_BLOCK_Z);
//  dim3 dimGridK1  ((nx*ny*nz*nt/2)/DIM_BLOCK_X, 1, 1 );
  dim3 dimGridK1  ((nx/2)/DIM_BLOCK_X, ny/DIM_BLOCK_Y, (nz*nt)/DIM_BLOCK_Z );

  if ( ((nx % 2) != 0) || (((nx/2) % DIM_BLOCK_X) != 0) ) {
    fprintf(stderr, "ERROR: nx should be even and nx/2 should be divisible by DIM_BLOCK_X.");
    return -1;
  }

  su3_soa * u_h;
  half * u_ht[8];
  vec3_soa * fermion1_h;
  half * fermion1_ht;
  vec3_soa * fermion2_h;
  half * fermion2_ht;

  // 8 = number of directions times 2 (even/odd)
  // no_links = sizeh * 8
  posix_memalign((void **)&u_h,        ALIGN, 8*sizeof(su3_soa));
  for (unsigned i=0; i<8; i++)
    posix_memalign((void **)&u_ht[i],       ALIGN, (sizeh/8)*256*sizeof(half));
  posix_memalign((void **)&fermion1_h, ALIGN, sizeof(vec3_soa));
  posix_memalign((void **)&fermion1_ht,ALIGN, (sizeh/8)*256*sizeof(half) );
  posix_memalign((void **)&fermion2_h, ALIGN, sizeof(vec3_soa));
  posix_memalign((void **)&fermion2_ht,ALIGN, (sizeh/8)*256*sizeof(half));

//  printf("Sizeof su3_soa   is: %d \n", sizeof(su3_soa));
//  printf("Sizeof su3_soa_d is: %d \n", sizeof(su3_soa_d));

  su3_soa * u_d;
  half * u_dt[8];
  vec3_soa * fermion1_d;
  half * fermion1_dt;
  vec3_soa * fermion2_d;
  half * fermion2_dt;

  cudaMalloc ((void**)&u_d, 8*sizeof(su3_soa));
  checkCUDAError("Allocating u_d");
  for (unsigned i=0; i < 8; i++)
    cudaMalloc ((void**)&u_dt[i], (sizeh/8)*256*sizeof(half));
  checkCUDAError("Allocating u_dt");
  cudaMalloc ((void**)&fermion1_d, sizeof(vec3_soa));
  checkCUDAError("Allocating fermion1_d");
  cudaMalloc ((void**)&fermion1_dt, (sizeh/8)*256*sizeof(half));
  checkCUDAError("Allocating fermion1_dt");
  cudaMalloc ((void**)&fermion2_d, sizeof(vec3_soa));
  checkCUDAError("Allocating fermion2_d");
  cudaMalloc ((void**)&fermion2_dt, (sizeh/8)*256*sizeof(half));
  checkCUDAError("Allocating fermion2_dt");


if ((nx == 32) && (ny == 32) && (nz == 32) && (nt == 32)) {
  loadSu3FromFileNew( u_h, "gaugeconf_save_32_4");
  loadFermionFromFileNew(fermion1_h, "test_fermion_32_4");
} else if ((nx == 16) && (ny == 16) && (nz == 16) && (nt == 16)) {
  loadSu3FromFile( u_h, "TestConf_16_4.cnf");
  for (unsigned i=0; i<8; i++)
    Su3Mapper(u_h[i], u_ht[i]);
  printf("Mapper Su3_soa\n");
  loadFermionFromFile(fermion1_h, "StartFermion_16_4.fer");
  fermionMapper(fermion1_h, fermion1_ht);
  printf("fermion mapper\n");
} else {
  fprintf(stdout, "Lattice not available... \n");
  exit(1);
}

  printMappedSu3_soa(u_ht[0], u_h[0]);
  printMappedVec3_soa(fermion1_ht, *fermion1_h);

  cudaMemcpy( u_d, u_h, 8*sizeof(su3_soa), cudaMemcpyHostToDevice );
  checkCUDAError("Copying u_d to device");
  cudaMemcpy( fermion1_d, fermion1_h, sizeof(vec3_soa), cudaMemcpyHostToDevice );
  checkCUDAError("Copying fermion1_d to device");
  cudaMemcpy( fermion2_d, fermion2_h, sizeof(vec3_soa), cudaMemcpyHostToDevice );
  checkCUDAError("Copying fermion2_d to device");

  // Prefer larger L1 cache than shared mem
   cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  // Prefer larger shared mem than L1 cache
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  gettimeofday ( &t0, NULL );

  for (i = 0; i < NITER; i++) {
    Deo<<< dimGridK1, dimBlockK1 >>>( u_d, fermion2_d, fermion1_d);
    checkCUDAError("Running kernel Deo");
    //cudaDeviceSynchronize();
    //checkCUDAError("Cuda synch after Deo");
    Doe<<< dimGridK1, dimBlockK1 >>>( u_d, fermion1_d, fermion2_d);
    checkCUDAError("Running kernel Doe");
    //cudaDeviceSynchronize();
    //checkCUDAError("Cuda synch after Doe");
  }

  cudaDeviceSynchronize();
  gettimeofday ( &t1, NULL );

//  cudaMemcpy( fermion1_h, fermion2_d, sizeof(vec3_soa), cudaMemcpyDeviceToHost );
  cudaMemcpy( fermion1_h, fermion1_d, sizeof(vec3_soa), cudaMemcpyDeviceToHost );
  checkCUDAError("Copying fermion1_d to host");

  dt_tot = (double)(t1.tv_sec - t0.tv_sec) + ((double)(t1.tv_usec - t0.tv_usec)/1.0e6);

  printf("TOTAL Exec time:          Tot time: % 3.2f sec    Avg: % 3.02f ms   Avg/site: % 3.02f ns\n",
          dt_tot, \
          (dt_tot/NITER)*(1.0e3),
          ((dt_tot/NITER)/size)*(1.0e9) );

  writeFermionToFile(fermion1_h, "EndFermion.fer");

  free(u_h);
  for (unsigned i=0; i<8; i++)
    free(u_ht[i]);
  free(fermion1_h);
  free(fermion1_ht);
  free(fermion2_h);
  free(fermion2_ht);

  return 0;

}
