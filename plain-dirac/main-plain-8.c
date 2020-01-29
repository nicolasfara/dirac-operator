#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "common-acc.h"

static inline vec3 mat_vec_mul( const su3_soa * const matrix,
                                const int idx_mat,
                                const int eta,
                                const vec3_soa * const in_vect,
                                const int idx_vect) {

  vec3 out_vect;

  d_complex vec0 = in_vect->c0[idx_vect];
  d_complex vec1 = in_vect->c1[idx_vect];
  d_complex vec2 = in_vect->c2[idx_vect];

  d_complex mat00 = matrix->r0.c0[idx_mat];
  d_complex mat01 = matrix->r0.c1[idx_mat];
  d_complex mat02 = matrix->r0.c2[idx_mat];

  d_complex mat10 = matrix->r1.c0[idx_mat];
  d_complex mat11 = matrix->r1.c1[idx_mat];
  d_complex mat12 = matrix->r1.c2[idx_mat];

// Load 3rd matrix row from global memory
//  d_complex mat20 = matrix->r2.c0[idx_mat];
//  d_complex mat21 = matrix->r2.c1[idx_mat];
//  d_complex mat22 = matrix->r2.c2[idx_mat];

//Compute 3rd matrix row from the first two
  d_complex mat20 = conj( ( mat01 * mat12 ) - ( mat02 * mat11) ) ;
  d_complex mat21 = conj( ( mat02 * mat10 ) - ( mat00 * mat12) ) ; 
  d_complex mat22 = conj( ( mat00 * mat11 ) - ( mat01 * mat10) ) ;
//Multiply 3rd row by eta
  mat20 = (mat20)*eta;
  mat21 = (mat21)*eta;
  mat22 = (mat22)*eta;

  out_vect.c0 = ( mat00 * vec0 ) + ( mat01 * vec1 ) + ( mat02 * vec2 );
  out_vect.c1 = ( mat10 * vec0 ) + ( mat11 * vec1 ) + ( mat12 * vec2 );
  out_vect.c2 = ( mat20 * vec0 ) + ( mat21 * vec1 ) + ( mat22 * vec2 );

  return out_vect;

}

static inline vec3 conjmat_vec_mul( const su3_soa * const matrix,
                                    const int idx_mat,
                                    const int eta,
                                    const vec3_soa * const in_vect,
                                    const int idx_vect) {

  vec3 out_vect;

  d_complex vec0 = in_vect->c0[idx_vect];
  d_complex vec1 = in_vect->c1[idx_vect];
  d_complex vec2 = in_vect->c2[idx_vect];

  d_complex mat00 = matrix->r0.c0[idx_mat];
  d_complex mat01 = matrix->r0.c1[idx_mat];
  d_complex mat02 = matrix->r0.c2[idx_mat];

  d_complex mat10 = matrix->r1.c0[idx_mat];
  d_complex mat11 = matrix->r1.c1[idx_mat];
  d_complex mat12 = matrix->r1.c2[idx_mat];

// Load 3rd matrix row from global memory
//  d_complex mat20 = matrix->r2.c0[idx_mat];
//  d_complex mat21 = matrix->r2.c1[idx_mat];
//  d_complex mat22 = matrix->r2.c2[idx_mat];

//Compute 3rd matrix row from the first two
  d_complex mat20 = conj( ( mat01 * mat12 ) - ( mat02 * mat11) );
  d_complex mat21 = conj( ( mat02 * mat10 ) - ( mat00 * mat12) );
  d_complex mat22 = conj( ( mat00 * mat11 ) - ( mat01 * mat10) );
//Multiply 3rd row by eta
  mat20 = (mat20)*eta;
  mat21 = (mat21)*eta;
  mat22 = (mat22)*eta;

  out_vect.c0 = ( conj(mat00) * vec0 ) + ( conj(mat10) * vec1 ) + ( conj(mat20) * vec2 );
  out_vect.c1 = ( conj(mat01) * vec0 ) + ( conj(mat11) * vec1 ) + ( conj(mat21) * vec2 );
  out_vect.c2 = ( conj(mat02) * vec0 ) + ( conj(mat12) * vec1 ) + ( conj(mat22) * vec2 );

  return out_vect;

}

static inline vec3 sumResult ( vec3 aux, vec3 aux_tmp) {

  aux.c0 += aux_tmp.c0;
  aux.c1 += aux_tmp.c1;
  aux.c2 += aux_tmp.c2;

  return aux;

}

static inline vec3 subResult ( vec3 aux, vec3 aux_tmp) {

  aux.c0 -= aux_tmp.c0;
  aux.c1 -= aux_tmp.c1;
  aux.c2 -= aux_tmp.c2;

  return aux;

}

void Deo(const su3_soa * const u, vec3_soa * const out, const vec3_soa * const in) {

  int hx, y, z, t; //, idx;

  for(t=0; t<nt; t++) {
    for(z=0; z<nz; z++) {
      for(y=0; y<ny; y++) {
        for(hx=0; hx < nxh; hx+=8) {

          int x, xm[8], ym, zm, tm, xp[8], yp, zp, tp, idxh[8], eta;
          vec3 aux[8];         

          x = 2*hx + ((y+z+t) & 0x1);

          for (unsigned i=0; i<16; i+=2) {
            idxh[i/2] = snum(x+i,y,z,t); 
          }

          // Coordinate +1, if on the end of the boundary: xm = nx-1. (the same for other coordinate).
          for (unsigned i=0; i<16; i+=2) {
            xm[i/2] = x+i -1;
            xm[i/2] = CHECK_PERIODIC_BOUNDARY_SUB(xm[i/2], nx);
          }
          ym = y-1;
          ym = CHECK_PERIODIC_BOUNDARY_SUB(ym, ny);
          zm = z-1;
          zm = CHECK_PERIODIC_BOUNDARY_SUB(zm, nz);
          tm = t-1;
          tm = CHECK_PERIODIC_BOUNDARY_SUB(tm, nt);

          // Coordinate -1, if on the end of the boundary: xm = 0. (the same for other coordinate).
          for (unsigned i=0; i<16; i+=2) {
            xp[i/2] = x+i +1;
            xp[i/2] *= CHECK_PERIODIC_BOUNDARY_ADD(xp[i/2], nx);
          }
          yp = y+1;
          yp *= CHECK_PERIODIC_BOUNDARY_ADD(yp, ny);
          zp = z+1;
          zp *= CHECK_PERIODIC_BOUNDARY_ADD(zp, nz);
          tp = t+1;
          tp *= CHECK_PERIODIC_BOUNDARY_ADD(tp, nt);

          eta = 1;
// mat_vec_mul( &(u_work[snum(x,y,z,t)       ]), &(in[snum(xp,y,z,t)]), &aux_tmp );
          for (unsigned i=0; i < 16; i+=2) {
            aux[i/2] = mat_vec_mul( &u[0], idxh[i/2], eta, in, snum(xp[i/2],y,z,t) );
          }

          //eta = 1 - ( 2*(x & 0x1) ); // if (x % 2 = 0) eta = 1 else -1
// mat_vec_mul( &(u_work[snum(x,y,z,t) + size ]), &(in[snum(x,yp,z,t)]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = sumResult(aux[i/2], mat_vec_mul( &u[2], idxh[i/2], eta, in, snum(x+i,yp,z,t) ) );
          }

          //eta = 1 - ( 2*((x+y) & 0x1) );
// mat_vec_mul( &(u_work[snum(x,y,z,t) + size2]), &(in[snum(x,y,zp,t)]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = sumResult(aux[i/2], mat_vec_mul( &u[4], idxh[i/2], eta, in, snum(x+i,y,zp,t) ) );
          }

          //eta = 1 - ( 2*((x+y+z) & 0x1) );
// mat_vec_mul( &(u_work[snum(x,y,z,t) + size3]), &(in[snum(x,y,z,tp)]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = sumResult(aux[i/2], mat_vec_mul( &u[6], idxh[i/2], eta, in, snum(x+i,y,z,tp) ) );
          }

//////////////////////////////////////////////////////////////////////////////////////////////
   
          eta = 1;
// conjmat_vec_mul( &(u_work[sizeh + snum(xm,y,z,t)      ]), &(in[ snum(xm,y,z,t) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = subResult(aux[i/2], conjmat_vec_mul( &u[1], snum(xm[i/2],y,z,t), eta, in, snum(xm[i/2],y,z,t) ) );
          }

          //eta = 1 - ( 2*(x & 0x1) );
// conjmat_vec_mul( &(u_work[sizeh + snum(x,ym,z,t) + size ]), &(in[ snum(x,ym,z,t) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = subResult(aux[i/2], conjmat_vec_mul( &u[3], snum(x+i,ym,z,t), eta, in, snum(x+i,ym,z,t) ) );
          }

          //eta = 1 - ( 2*((x+y) & 0x1) );
// conjmat_vec_mul( &(u_work[sizeh + snum(x,y,zm,t) + size2]), &(in[ snum(x,y,zm,t) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = subResult(aux[i/2], conjmat_vec_mul( &u[5], snum(x+i,y,zm,t), eta, in, snum(x+i,y,zm,t) ) );
          }

          //eta = 1 - ( 2*((x+y+z) & 0x1) );
// conjmat_vec_mul( &(u_work[sizeh + snum(x,y,z,tm) + size3]), &(in[ snum(x,y,z,tm) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = subResult(aux[i/2], conjmat_vec_mul( &u[7], snum(x+i,y,z,tm), eta, in, snum(x+i,y,z,tm) ) );
          }

//////////////////////////////////////////////////////////////////////////////////////////////

          for (unsigned i=0; i<8; i++) {
            out->c0[idxh[i]] = (aux[i].c0)*0.5;
            out->c1[idxh[i]] = (aux[i].c1)*0.5;
            out->c2[idxh[i]] = (aux[i].c2)*0.5;
          }

        } // Loop over nxh
      } // Loop over ny
    } // Loop over nz
  } // Loop over nt

}


void Doe(const su3_soa * const u, vec3_soa * const out, const vec3_soa * const in) {

  int hx, y, z, t;

  for(t=0; t<nt; t++) {
    for(z=0; z<nz; z++) {
      for(y=0; y<ny; y++) {
        for(hx=0; hx < nxh; hx+=8) {
          int x, xm[8], ym, zm, tm, xp[8], yp, zp, tp, idxh[8], eta;
          vec3 aux[8];

          x = 2*hx + ((y+z+t+1) & 0x1);

          for (unsigned i=0; i<16; i+=2) {
            idxh[i/2] = snum(x+i,y,z,t);
            //printf("idxh: %d\n", idxh[i/2]);
          }


         // Coordinate +1, if on the end of the boundary: xm = nx-1. (the same for other coordinate).
          for (unsigned i=0; i<16; i+=2) {
            xm[i/2] = x+i -1;
            xm[i/2] = CHECK_PERIODIC_BOUNDARY_SUB(xm[i/2], nx);
          }
          ym = y-1;
          ym = CHECK_PERIODIC_BOUNDARY_SUB(ym, ny);
          zm = z-1;
          zm = CHECK_PERIODIC_BOUNDARY_SUB(zm, nz);
          tm = t-1;
          tm = CHECK_PERIODIC_BOUNDARY_SUB(tm, nt);

          // Coordinate -1, if on the end of the boundary: xm = 0. (the same for other coordinate).
          for (unsigned i=0; i<16; i+=2) {
            xp[i/2] = x+i +1;
            xp[i/2] *= CHECK_PERIODIC_BOUNDARY_ADD(xp[i/2], nx);
            //printf("xp: %d\n", xp[i/2]);
          }
          yp = y+1;
          yp *= CHECK_PERIODIC_BOUNDARY_ADD(yp, ny);
          zp = z+1;
          zp *= CHECK_PERIODIC_BOUNDARY_ADD(zp, nz);
          tp = t+1;
          tp *= CHECK_PERIODIC_BOUNDARY_ADD(tp, nt);

          eta = 1;
// mat_vec_mul( &(u_work[snum(x,y,z,t) + sizeh      ]), &(in[ snum(xp,y,z,t) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = mat_vec_mul( &u[1], idxh[i/2], eta, in, snum(xp[i/2],y,z,t));
            printf("u[1] first idxh: %d, snum: %d\n", idxh[i/2], snum(xp[i/2],y,z,t));
          }

          //eta = 1 - ( 2*(x & 0x1) );
// mat_vec_mul( &(u_work[snum(x,y,z,t) + sizeh + size ]), &(in[ snum(x,yp,z,t) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = sumResult(aux[i/2], mat_vec_mul( &u[3], idxh[i/2], eta, in, snum(x+i,yp,z,t)) );
            printf("u[3] first idxh: %d, snum: %d\n", idxh[i/2], snum(x+i,yp,z,t));
          }

          //eta = 1 - ( 2*((x+y) & 0x1) );
// mat_vec_mul( &( u_work[snum(x,y,z,t) + sizeh + size2]), &(in[ snum(x,y,zp,t) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = sumResult(aux[i/2], mat_vec_mul( &u[5], idxh[i/2], eta, in, snum(x+i,y,zp,t)) );
            printf("u[5] first idxh: %d, snum: %d\n", idxh[i/2], snum(x+i,y,zp,t));
          }

          //eta = 1 - ( 2*((x+y+z) & 0x1) );
// mat_vec_mul( &(u_work[snum(x,y,z,t) + sizeh + size3]), &(in[ snum(x,y,z,tp) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = sumResult(aux[i/2], mat_vec_mul( &u[7], idxh[i/2], eta, in, snum(x+i,y,z,tp)) );
            printf("u[7] first idxh: %d, snum: %d\n", idxh[i/2], snum(x+i,y,z,tp));
          }

//////////////////////////////////////////////////////////////////////////////////////////////

          eta = 1;
// conjmat_vec_mul( &(u_work[snum(xm,y,z,t)      ]), &(in[ snum(xm,y,z,t) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = subResult(aux[i/2], conjmat_vec_mul( &u[0], snum(xm[i/2],y,z,t), eta, in, snum(xm[i/2],y,z,t)) );
            printf("u[0] first idxh: %d, snum: %d\n", snum(xm[i/2],y,z,t), snum(xm[i/2],y,z,t));
          }

          //eta = 1 - ( 2*(x & 0x1) );
// conjmat_vec_mul( &(u_work[snum(x,ym,z,t) + size ]), &(in[ snum(x,ym,z,t) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = subResult(aux[i/2], conjmat_vec_mul( &u[2], snum(x+i,ym,z,t), eta, in, snum(x+i,ym,z,t)) );
            printf("u[2] first idxh: %d, snum: %d\n", snum(x+i,ym,z,t), snum(x+i,ym,z,t));
          }

          //eta = 1 - ( 2*((x+y) & 0x1) );
// conjmat_vec_mul( &(u_work[snum(x,y,zm,t) + size2]), &(in[ snum(x,y,zm,t) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = subResult(aux[i/2], conjmat_vec_mul( &u[4], snum(x+i,y,zm,t), eta, in, snum(x+i,y,zm,t)) );
            printf("u[4] first idxh: %d, snum: %d\n", snum(x+i,y,zm,t), snum(x+i,y,zm,t));
          }

          //eta = 1 - ( 2*((x+y+z) & 0x1) );
// conjmat_vec_mul( &(u_work[snum(x,y,z,tm) + size3]), &(in[ snum(x,y,z,tm) ]), &aux_tmp );
          for (unsigned i=0; i<16; i+=2) {
            aux[i/2] = subResult(aux[i/2], conjmat_vec_mul( &u[6], snum(x+i,y,z,tm), eta, in, snum(x+i,y,z,tm)) );
            printf("u[6] first idxh: %d, snum: %d\n", snum(x+i,y,z,tm), snum(x+i,y,z,tm));
          }

//////////////////////////////////////////////////////////////////////////////////////////////


          for (unsigned i=0; i<8; i++) {
            out->c0[idxh[i]] = aux[i].c0*0.5;
            out->c1[idxh[i]] = aux[i].c1*0.5;
            out->c2[idxh[i]] = aux[i].c2*0.5;
          }

          //printf("\n\n");
        } // Loop over nxh
      } // Loop over ny
    } // Loop over nz
  } // Loop over nt

}


int main() {

  int i;
  struct timeval t0, t1, t2;
  double dt_tot = 0.0;
  double dt_test = 0.0;

  if ( ((nx % 2) != 0) || (((nx/2) % DIM_BLOCK_X) != 0) ) {
    fprintf(stderr, "ERROR: nx should be even and nx/2 should be divisible by DIM_BLOCK_X.\n");
    return -1;
  }

  su3_soa * u_h;
  vec3_soa * fermion1_h;
  vec3_soa * fermion2_h;

  // 8 = number of directions times 2 (even/odd)
  // no_links = sizeh * 8
  posix_memalign((void **)&u_h, ALIGN, 8*sizeof(su3_soa));
  posix_memalign((void **)&fermion1_h, ALIGN, sizeof(vec3_soa));
  posix_memalign((void **)&fermion2_h, ALIGN, sizeof(vec3_soa));

  loadSu3FromFile( u_h, "TestConf_16_4.cnf");

  loadFermionFromFile(fermion1_h, "StartFermion_16_4.fer");

  // Prefer larger L1 cache than shared mem
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  // Prefer larger shared mem than L1 cache
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  fprintf(stderr, "Cpying data in the device... \n");

  { // Copy input data in the device

    fprintf(stderr, "Starting computations on the device... \n");

    gettimeofday ( &t0, NULL );

    for (i = 0; i < NITER; i++) {
//      fprintf(stderr, "Starting deo... \n");
      Deo(u_h, fermion2_h, fermion1_h);
//      gettimeofday ( &t1, NULL );
//      fprintf(stderr, "Starting doe... \n");
      Doe(u_h, fermion1_h, fermion2_h);
//      acc_async_wait_all( ); // This should not be needed... just to be sure
    }

//    acc_async_wait_all( ); // This should not be needed... just to be sure
    gettimeofday ( &t2, NULL );

  } // Copy data back to the host

//  dt_test = (double)(t1.tv_sec - t0.tv_sec) + ((double)(t1.tv_usec - t0.tv_usec)/1.0e6);
//
//  printf("first function exec time: % 3.2f [ms] \n", dt_test*(1.0e3));


  dt_tot = (double)(t2.tv_sec - t0.tv_sec) + ((double)(t2.tv_usec - t0.tv_usec)/1.0e6);

  printf("TOTAL Exec time:          Tot time: % 3.2f sec    Avg: % 3.02f ms   Avg/site: % 3.02f ns\n",
          dt_tot, \
          (dt_tot/NITER)*(1.0e3),
          ((dt_tot/NITER)/size)*(1.0e9) );

  writeFermionToFile(fermion1_h, "EndFermion.fer");
//  writeFermionToFile(fermion2_h, "PartialFermion.fer");

  free(u_h);
  free(fermion1_h);
  free(fermion2_h);

  return 0;

}
