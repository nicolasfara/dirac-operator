#pragma once

#include <math.h>
#include <complex.h>
#include <cuda.h>
#include <cuComplex.h>
#include <mma.h>

// lattice dimensions
// defined in the Makefile
//#define nx 48 
//#define ny 48
//#define nz 48
//#define nt 48

// defined in the Makefile
//#define DIM_BLOCK_X 12 // This should divide (nx/2)
//#define DIM_BLOCK_Y 4  // This should divide ny
//#define DIM_BLOCK_Z 1  // This should divide nz*nt

// Number of iterations
#define NITER 10

//Decomment this to allocate and store the 3rd su3 matrix line
//#define ALLOCROW3
//Decomment this to store and read the 3rd su3 matrix line
#define READROW3
//If we want to read it we should also allocate it
#ifdef READROW3 
  #define ALLOCROW3
#endif

#define vol1 nx
#define vol2 (ny * vol1)
#define vol3 (nz * vol2)
#define vol4 (nt * vol3)

#define nxh (nx >> 1) // nx/2
#define nyh (ny >> 1)
#define nzh (nz >> 1)
#define nth (nt >> 1)


#define size vol4
#define size2 (2*size)
#define size3 (3*size)

#define sizeh (size / 2)
#define no_links (4 * vol4)

#define ALIGN 128

#define CHECK_PERIODIC_BOUNDARY_SUB(coord, domain)    coord + (((coord >> 31) & 0x1) * domain)
#define CHECK_PERIODIC_BOUNDARY_ADD(coord, domain)    (((coord-domain) >> 31) & 0x1)
#define ETA_UPDATE(coord)                             1 - ( 2*((coord) & 0x1))    

typedef cuDoubleComplex d_complex;

typedef struct vec3_soa_t {
  d_complex c0[sizeh];
  d_complex c1[sizeh];
  d_complex c2[sizeh];
} vec3_soa;

typedef struct su3_soa_t {
  vec3_soa r0;
  vec3_soa r1;
#ifdef ALLOCROW3
  vec3_soa r2;
#endif
} su3_soa;

typedef struct vec3_t {
  d_complex c0;
  d_complex c1;
  d_complex c2;
} vec3;


// Common functions

inline void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if ( cudaSuccess != err) {
    fprintf(stderr, "ERROR: %s %s.\n", msg, cudaGetErrorString( err) );
    exit(-1);
  }
}

__host__ __device__ static __inline__ uint snum(uint x, uint y, uint z, uint t) {

  uint ris;

  ris = x + (y*vol1) + (z*vol2) + (t*vol3);

  return ris/2;   // <---  /2 Pay attention to even/odd  (see init_geo) 

}

__host__ __device__ static __inline__ d_complex myConj(d_complex a) {

    d_complex res;

    res.x = a.x;
    res.y = -a.y;

    return res;

}

void loadFermionFromFile( vec3_soa * fermion, const char *filename) {

  FILE *fp;
  double ar, ai, br, bi, cr, ci;
  int i = 0;
  int error =0;

  fp = fopen(filename, "rt");

  if (fp == NULL) {
    printf("Could not open file %s \n", filename);
    exit(-1);
  }

  while ( (i < sizeh) && (!error) ) {
    if (fscanf(fp, "(%lf,%lf) (%lf,%lf) (%lf,%lf) \n", &ar, &ai, &br, &bi, &cr, &ci) == 6) {
//      fermion->c0[i] = (ar + ai * I);
//      fermion->c1[i] = (br + bi * I);
//      fermion->c2[i] = (cr + ci * I);
      fermion->c0[i] = make_cuDoubleComplex(ar, ai);
      fermion->c1[i] = make_cuDoubleComplex(br, bi);
      fermion->c2[i] = make_cuDoubleComplex(cr, ci);
    } else {
      printf("Read error... \n");
      error = 1;
    }
    //printf("Read line: (%lf,%lf) (%lf,%lf) (%lf,%lf) \n", ar, ai, br, bi, cr, ci);
    i++;
  }

  printf("Read %d fermions from file %s \n", i, filename);

  fclose(fp);

}


void loadSu3FromFile(su3_soa * u, const char *filename){

  FILE *fp;
  int nx_l, ny_l, nz_l, nt_l, update_iterations;
  double beta_l, mass_l, no_flavours_l;
  double ar, ai, br, bi, cr, ci;
  int idx;
  int i = 0;
  int j = 0;
  int error = 0;

  fp = fopen(filename, "rt");

  if (fp == NULL) {
    printf("Could not open file %s \n", filename);
    exit(-1);
  }


  fscanf(fp, "%d %d %d %d %lf %lf %lf %d \n", &nx_l, &ny_l, &nz_l, &nt_l, 
                                              &beta_l, &mass_l, &no_flavours_l, 
                                              &update_iterations);

  printf("Reading configuration file with header: \n");
  printf("nx_l: %d, ny_l: %d, nz_l: %d, nt_l: %d \n", nx_l, ny_l, nz_l, nt_l); 
  printf("beta_l: %lf, mass_l: %lf, no_flavours_l: %lf \n", beta_l, mass_l, no_flavours_l);
  printf("update_iterations: %d \n", update_iterations);
 
//  while ( (i < no_links) && (!error) ) {

    while ( (i < sizeh*8) && (!error) ) {
    
      j = i / sizeh;
      idx = i % sizeh;

      if (fscanf(fp, "(%lf,%lf) (%lf,%lf) (%lf,%lf) \n", &ar, &ai, &br, &bi, &cr, &ci) == 6) {
        //u[j].r0.c0[idx] = (ar + ai * I);
        //u[j].r0.c1[idx] = (br + bi * I);
        //u[j].r0.c2[idx] = (cr + ci * I);
        u[j].r0.c0[idx] = make_cuDoubleComplex(ar, ai);
        u[j].r0.c1[idx] = make_cuDoubleComplex(br, bi);
        u[j].r0.c2[idx] = make_cuDoubleComplex(cr, ci);
      } else {
        printf("Read error... ");
        error = 1;
      }

      if (fscanf(fp, "(%lf,%lf) (%lf,%lf) (%lf,%lf) \n", &ar, &ai, &br, &bi, &cr, &ci) == 6) {
        //u[j].r1.c0[idx] = (ar + ai * I);
        //u[j].r1.c1[idx] = (br + bi * I);
        //u[j].r1.c2[idx] = (cr + ci * I);
        u[j].r1.c0[idx] = make_cuDoubleComplex(ar, ai);
        u[j].r1.c1[idx] = make_cuDoubleComplex(br, bi);
        u[j].r1.c2[idx] = make_cuDoubleComplex(cr, ci);
      } else {
        printf("Read error... ");
        error = 1;
      }

      if (fscanf(fp, "(%lf,%lf) (%lf,%lf) (%lf,%lf) \n", &ar, &ai, &br, &bi, &cr, &ci) == 6) {
#ifdef ALLOCROW3
        //u[j].r2.c0[idx] = (ar + ai * I);
        //u[j].r2.c1[idx] = (br + bi * I);
        //u[j].r2.c2[idx] = (cr + ci * I);
        u[j].r2.c0[idx] = make_cuDoubleComplex(ar, ai);
        u[j].r2.c1[idx] = make_cuDoubleComplex(br, bi);
        u[j].r2.c2[idx] = make_cuDoubleComplex(cr, ci);
#endif
      } else {
        printf("Read error... ");
        error = 1;
      }
    
    i++;

  }

  printf("Read %d matrices from file %s \n", i*j, filename);

  fclose(fp);

}

void loadFermionFromFileNew( vec3_soa * fermion, const char *filename) {

  FILE *fp;
  double ar, ai, br, bi, cr, ci;
  int i = 0;
  int error =0;

  fp = fopen(filename, "rt");

  if (fp == NULL) {
    printf("Could not open file %s \n", filename);
    exit(-1);
  }

  while ( (i < sizeh) && (!error) ) {
    if (fscanf(fp, "%lf %lf\n%lf %lf\n%lf %lf\n", &ar, &ai, &br, &bi, &cr, &ci) == 6) {
//      fermion->c0[i] = (ar + ai * I);
//      fermion->c1[i] = (br + bi * I);
//      fermion->c2[i] = (cr + ci * I);
      fermion->c0[i] = make_cuDoubleComplex(ar, ai);
      fermion->c1[i] = make_cuDoubleComplex(br, bi);
      fermion->c2[i] = make_cuDoubleComplex(cr, ci);
    } else {
      printf("Read error... \n");
      error = 1;
    }
    //printf("Read line: (%lf,%lf) (%lf,%lf) (%lf,%lf) \n", ar, ai, br, bi, cr, ci);
    i++;
  }

  printf("Read %d fermions from file %s \n", i, filename);

  fclose(fp);

}


void loadSu3FromFileNew(su3_soa * u, const char *filename){

  FILE *fp;
  int nx_l, ny_l, nz_l, nt_l, update_iterations;
  double ar, ai, br, bi, cr, ci;
  int idx;
  int i = 0;
  int j = 0;
  int error = 0;

  fp = fopen(filename, "rt");

  if (fp == NULL) {
    printf("Could not open file %s \n", filename);
    exit(-1);
  }


  fscanf(fp, "%d %d %d %d %d \n", &nx_l, &ny_l, &nz_l, &nt_l,&update_iterations);

  printf("Reading configuration file with header: \n");
  printf("nx_l: %d, ny_l: %d, nz_l: %d, nt_l: %d \n", nx_l, ny_l, nz_l, nt_l); 
  printf("update_iterations: %d \n", update_iterations);
 
//  while ( (i < no_links) && (!error) ) {

    while ( (i < sizeh*8) && (!error) ) {
    
      j = i / sizeh;
      idx = i % sizeh;

      if (fscanf(fp, "%lf %lf\n%lf %lf\n%lf %lf\n", &ar, &ai, &br, &bi, &cr, &ci) == 6) {
        //u[j].r0.c0[idx] = (ar + ai * I);
        //u[j].r0.c1[idx] = (br + bi * I);
        //u[j].r0.c2[idx] = (cr + ci * I);
        u[j].r0.c0[idx] = make_cuDoubleComplex(ar, ai);
        u[j].r0.c1[idx] = make_cuDoubleComplex(br, bi);
        u[j].r0.c2[idx] = make_cuDoubleComplex(cr, ci);
      } else {
        printf("Read error... ");
        error = 1;
      }

      if (fscanf(fp, "%lf %lf\n%lf %lf\n%lf %lf\n", &ar, &ai, &br, &bi, &cr, &ci) == 6) {
        //u[j].r1.c0[idx] = (ar + ai * I);
        //u[j].r1.c1[idx] = (br + bi * I);
        //u[j].r1.c2[idx] = (cr + ci * I);
        u[j].r1.c0[idx] = make_cuDoubleComplex(ar, ai);
        u[j].r1.c1[idx] = make_cuDoubleComplex(br, bi);
        u[j].r1.c2[idx] = make_cuDoubleComplex(cr, ci);

      } else {
        printf("Read error... ");
        error = 1;
      }

      if (fscanf(fp, "%lf %lf\n%lf %lf\n%lf %lf\n", &ar, &ai, &br, &bi, &cr, &ci) == 6) {
#ifdef ALLOCROW3
        //u[j].r2.c0[idx] = (ar + ai * I);
        //u[j].r2.c1[idx] = (br + bi * I);
        //u[j].r2.c2[idx] = (cr + ci * I);
        u[j].r2.c0[idx] = make_cuDoubleComplex(ar, ai);
        u[j].r2.c1[idx] = make_cuDoubleComplex(br, bi);
        u[j].r2.c2[idx] = make_cuDoubleComplex(cr, ci);
#endif
      } else {
        printf("Read error... ");
        error = 1;
      }
    
    i++;

  }

  printf("Read %d matrices from file %s \n", i, filename);

  fclose(fp);

}



void writeFermionToFile(vec3_soa * fermion, const char *filename){

  FILE *fp;
  int i = 0;
  int error = 0;

  fp = fopen(filename, "w");

  if (fp == NULL) {
    printf("Could not open file %s \n", filename);
    exit(-1);
  }

  while ( (i < sizeh) && (!error) ) {

//    if (fprintf(fp, "(%lf,%lf) (%lf,%lf) (%lf,%lf) \n", cuCreal(fermion->c0[i]), cuCimag(fermion->c0[i]), 
//                                                        cuCreal(fermion->c1[i]), cuCimag(fermion->c1[i]),
//                                                        cuCreal(fermion->c2[i]), cuCimag(fermion->c2[i])) < 0) {

    if (fprintf(fp, "%lf %lf\n%lf %lf\n%lf %lf\n", cuCreal(fermion->c0[i]), cuCimag(fermion->c0[i]),
                                                   cuCreal(fermion->c1[i]), cuCimag(fermion->c1[i]),
                                                   cuCreal(fermion->c2[i]), cuCimag(fermion->c2[i])) < 0) {

      printf("Write error... ");

      error = 1;

    }

    i++;

  }
  
  printf("Wrote %d fermions from file %s \n", i, filename);

  fclose(fp);

}






// Just for debugging:
void showbits(unsigned int x) {

  int i;

  for(i=(sizeof(int)*8)-1; i>=0; i--)
    (x&(1<<i))?putchar('1'):putchar('0');

  printf("\n");

}




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

__device__ static __inline__ void tensorToVec(const float * const res_vec, vec3 * const out_vec)
{
    const unsigned lut[] = { 0,1,2,3,68,69,70,71,136,137,138,139,204,205,206,207 };
    
    int half_idx = lut[threadIdx.x];
    int lidx = threadIdx.x/2;

    if ((threadIdx.x & 0x1) == 0) {
        out_vec[lidx].c0 = make_cuDoubleComplex(res_vec[half_idx],    cuCimag(out_vec[lidx].c0));
        out_vec[lidx].c1 = make_cuDoubleComplex(res_vec[half_idx+16], cuCimag(out_vec[lidx].c1));
        out_vec[lidx].c2 = make_cuDoubleComplex(res_vec[half_idx+32], cuCimag(out_vec[lidx].c2));
    } else {
        out_vec[lidx].c0 = make_cuDoubleComplex(cuCreal(out_vec[lidx].c0), res_vec[half_idx]);
        out_vec[lidx].c1 = make_cuDoubleComplex(cuCreal(out_vec[lidx].c1), res_vec[half_idx+16]);
        out_vec[lidx].c2 = make_cuDoubleComplex(cuCreal(out_vec[lidx].c2), res_vec[half_idx+32]);
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
    __shared__ float t_out_vec[TENSOR_MAT_SIZE];

    //Use function to map vec3_soa and su3_soa to half array
    if (threadIdx.x < 16)
        su3Mapper(mat, idx_mat, t_mat); //thread 0-15
    else
        fermionMapper(in_vec, idx_vec, t_in_vec); //thread 16-31

       // Declare the fragments
   nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> a_frag;
   nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
   nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   nvcuda::wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   nvcuda::wmma::load_matrix_sync(a_frag, t_mat, 16);
   nvcuda::wmma::load_matrix_sync(b_frag, t_in_vec, 16);  

   // Perform the matrix multiplication
   nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   nvcuda::wmma::store_matrix_sync(t_out_vec, c_frag, 16, nvcuda::wmma::mem_row_major);


    //Solo 16 thread convertono i dati in vettori
    if (threadIdx.x < 16) tensorToVec(t_out_vec, out_vec);
}

