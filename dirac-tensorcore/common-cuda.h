#ifndef COMMON_CUDA_H
#define COMMON_CUDA_H

#include <math.h>
#include <complex.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cuda_fp16.h>

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
//#define READROW3
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

// Define for tensor core
#define L_ROW0  0
#define L_ROW1  1
#define L_ROW2  2
#define L_ROW3  3
#define L_ROW4  4
#define L_ROW5  5
#define L_ROW6  6
#define L_ROW7  7
#define L_COL0  0
#define L_COL1  1
#define L_COL2  2
#define L_COL3  3

static inline unsigned int IDX_REAL_MAT(unsigned rows, unsigned cols)
{
  return cols + rows*16;
}

static inline unsigned int IDX_IMAG_MAT(unsigned rows, unsigned cols)
{
  return cols + rows*16 + 4;
}

static inline unsigned int IDX_REAL_VEC(unsigned rows)
{
  return rows*16;
}

static inline unsigned int IDX_IMAG_VEC(unsigned rows)
{
  return rows*16 + 1;
}


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

// Mapper functions

// La lettura dal file di input rivela che esistono 8 strutture su3_soa,
// se quindi in fase di allocazione della mia struttura dati, alloco
// un array di 8 elementi ognuno dei quali e' un puntatore (*half) alle matrici
// "impacchettate" per l'uso con tensor core (?).
// Se cio' fosse corretto, allora dovrei invocare questa funzione 8 volte e
// passare come argomento l'elemento del vettore di puntatori (Vettore di 8 elemeti)
void Su3Mapper(su3_soa in, half *out)
{
  const unsigned lut[] = { 0, 8, 64, 72, 128, 136, 192, 200 };
  for (unsigned i=0; i<sizeh; i++) {
    unsigned mat16_16 = i/8; //index of bigger matrix
    unsigned lindex = lut[i%8]; //local index for 3x3 matrix
    unsigned gindex = mat16_16*256 + lindex;

    out[gindex + IDX_REAL_MAT(L_ROW0, L_COL0)] = __float2half(cuCreal(in.r0.c0[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW0, L_COL0)] = __float2half(cuCimag(in.r0.c0[i]));
    out[gindex + IDX_REAL_MAT(L_ROW0, L_COL1)] = __float2half(cuCreal(in.r0.c1[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW0, L_COL1)] = __float2half(cuCimag(in.r0.c1[i]));
    out[gindex + IDX_REAL_MAT(L_ROW0, L_COL2)] = __float2half(cuCreal(in.r0.c2[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW0, L_COL2)] = __float2half(cuCimag(in.r0.c2[i]));
    out[gindex + IDX_REAL_MAT(L_ROW0, L_COL3)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW0, L_COL3)] = __float2half(0.0f);

    out[gindex + IDX_REAL_MAT(L_ROW1, L_COL0)] = __float2half(cuCreal(in.r1.c0[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW1, L_COL0)] = __float2half(cuCimag(in.r1.c0[i]));
    out[gindex + IDX_REAL_MAT(L_ROW1, L_COL1)] = __float2half(cuCreal(in.r1.c1[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW1, L_COL1)] = __float2half(cuCimag(in.r1.c1[i]));
    out[gindex + IDX_REAL_MAT(L_ROW1, L_COL2)] = __float2half(cuCreal(in.r1.c2[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW1, L_COL2)] = __float2half(cuCimag(in.r1.c2[i]));
    out[gindex + IDX_REAL_MAT(L_ROW1, L_COL3)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW1, L_COL3)] = __float2half(0.0f);

#ifdef ALLOCROW3
    out[gindex + IDX_REAL_MAT(L_ROW2, L_COL0)] = __float2half(cuCreal(in.r2.c0[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW2, L_COL0)] = __float2half(cuCimag(in.r2.c0[i]));
    out[gindex + IDX_REAL_MAT(L_ROW2, L_COL1)] = __float2half(cuCreal(in.r2.c1[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW2, L_COL1)] = __float2half(cuCimag(in.r2.c1[i]));
    out[gindex + IDX_REAL_MAT(L_ROW2, L_COL2)] = __float2half(cuCreal(in.r2.c2[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW2, L_COL2)] = __float2half(cuCimag(in.r2.c2[i]));
    out[gindex + IDX_REAL_MAT(L_ROW2, L_COL3)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW2, L_COL3)] = __float2half(0.0f);
#endif

    out[gindex + IDX_REAL_MAT(L_ROW3, L_COL0)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW3, L_COL0)] = __float2half(0.0f);
    out[gindex + IDX_REAL_MAT(L_ROW3, L_COL1)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW3, L_COL1)] = __float2half(0.0f);
    out[gindex + IDX_REAL_MAT(L_ROW3, L_COL2)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW3, L_COL2)] = __float2half(0.0f);
    out[gindex + IDX_REAL_MAT(L_ROW3, L_COL3)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW3, L_COL3)] = __float2half(0.0f);
  }

}

void Su3MapperConj(su3_soa in, half *out)
{
  const unsigned lut[] = { 0, 8, 64, 72, 128, 136, 192, 200 };
  for (unsigned i=0; i<sizeh; i++) {
    unsigned mat16_16 = i/8; //index of bigger matrix
    unsigned lindex = lut[i%8]; //local index for 3x3 matrix
    unsigned gindex = mat16_16*256 + lindex;

    out[gindex + IDX_REAL_MAT(L_ROW0, L_COL0)] = __float2half(cuCreal(in.r0.c0[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW0, L_COL0)] = __float2half(-cuCimag(in.r0.c0[i]));
    out[gindex + IDX_REAL_MAT(L_ROW0, L_COL1)] = __float2half(cuCreal(in.r0.c1[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW0, L_COL1)] = __float2half(-cuCimag(in.r0.c1[i]));
    out[gindex + IDX_REAL_MAT(L_ROW0, L_COL2)] = __float2half(cuCreal(in.r0.c2[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW0, L_COL2)] = __float2half(-cuCimag(in.r0.c2[i]));
    out[gindex + IDX_REAL_MAT(L_ROW0, L_COL3)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW0, L_COL3)] = __float2half(0.0f);

    out[gindex + IDX_REAL_MAT(L_ROW1, L_COL0)] = __float2half(cuCreal(in.r1.c0[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW1, L_COL0)] = __float2half(-cuCimag(in.r1.c0[i]));
    out[gindex + IDX_REAL_MAT(L_ROW1, L_COL1)] = __float2half(cuCreal(in.r1.c1[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW1, L_COL1)] = __float2half(-cuCimag(in.r1.c1[i]));
    out[gindex + IDX_REAL_MAT(L_ROW1, L_COL2)] = __float2half(cuCreal(in.r1.c2[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW1, L_COL2)] = __float2half(-cuCimag(in.r1.c2[i]));
    out[gindex + IDX_REAL_MAT(L_ROW1, L_COL3)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW1, L_COL3)] = __float2half(0.0f);

#ifdef ALLOCROW3
    out[gindex + IDX_REAL_MAT(L_ROW2, L_COL0)] = __float2half(cuCreal(in.r2.c0[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW2, L_COL0)] = __float2half(-cuCimag(in.r2.c0[i]));
    out[gindex + IDX_REAL_MAT(L_ROW2, L_COL1)] = __float2half(cuCreal(in.r2.c1[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW2, L_COL1)] = __float2half(-cuCimag(in.r2.c1[i]));
    out[gindex + IDX_REAL_MAT(L_ROW2, L_COL2)] = __float2half(cuCreal(in.r2.c2[i]));
    out[gindex + IDX_IMAG_MAT(L_ROW2, L_COL2)] = __float2half(-cuCimag(in.r2.c2[i]));
    out[gindex + IDX_REAL_MAT(L_ROW2, L_COL3)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW2, L_COL3)] = __float2half(0.0f);
#endif

    out[gindex + IDX_REAL_MAT(L_ROW3, L_COL0)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW3, L_COL0)] = __float2half(0.0f);
    out[gindex + IDX_REAL_MAT(L_ROW3, L_COL1)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW3, L_COL1)] = __float2half(0.0f);
    out[gindex + IDX_REAL_MAT(L_ROW3, L_COL2)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW3, L_COL2)] = __float2half(0.0f);
    out[gindex + IDX_REAL_MAT(L_ROW3, L_COL3)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_MAT(L_ROW3, L_COL3)] = __float2half(0.0f);
  }

}

void fermionMapper(vec3_soa *in, half *out)
{
  const unsigned lut[] = { 0, 4, 8, 12, 130, 134, 138, 142 };

  for (unsigned i=0; i < sizeh; i++) {
    unsigned mat16_16 = i/8; //index of bigger matrix
    unsigned lindex = lut[i%8]; //local index for 3x3 matrix
    unsigned gindex = mat16_16*256 + lindex;

    out[gindex + IDX_REAL_VEC(L_ROW0)] = __float2half(cuCreal(in->c0[i]));
    out[gindex + IDX_IMAG_VEC(L_ROW0)] = __float2half(cuCimag(in->c0[i]));
    out[gindex + IDX_REAL_VEC(L_ROW1)] = __float2half(cuCreal(in->c1[i]));
    out[gindex + IDX_IMAG_VEC(L_ROW1)] = __float2half(cuCimag(in->c1[i]));
    out[gindex + IDX_REAL_VEC(L_ROW2)] = __float2half(cuCreal(in->c2[i]));
    out[gindex + IDX_IMAG_VEC(L_ROW2)] = __float2half(cuCimag(in->c2[i]));
    out[gindex + IDX_REAL_VEC(L_ROW3)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_VEC(L_ROW3)] = __float2half(0.0f);


    out[gindex + IDX_REAL_VEC(L_ROW4)] = __float2half(-cuCimag(in->c0[i]));
    out[gindex + IDX_IMAG_VEC(L_ROW4)] = __float2half(cuCreal(in->c0[i]));
    out[gindex + IDX_REAL_VEC(L_ROW5)] = __float2half(-cuCimag(in->c1[i]));
    out[gindex + IDX_IMAG_VEC(L_ROW5)] = __float2half(cuCreal(in->c1[i]));
    out[gindex + IDX_REAL_VEC(L_ROW6)] = __float2half(-cuCimag(in->c2[i]));
    out[gindex + IDX_IMAG_VEC(L_ROW6)] = __float2half(cuCreal(in->c2[i]));
    out[gindex + IDX_REAL_VEC(L_ROW7)] = __float2half(0.0f);
    out[gindex + IDX_IMAG_VEC(L_ROW7)] = __float2half(0.0f);
  }

}

// Debug prupose
void printMappedSu3_soa(half *in, su3_soa a)
{
  FILE *fp;
  FILE *fpa;
  if ((fp=fopen("su3_soa.log", "w")) == NULL) {
    fprintf(stderr, "Unable to create su3_soa log file\n");
  }

  if ((fpa=fopen("su3_soa_fpa.log", "w")) == NULL) {
    fprintf(stderr, "Unable to create su3_soa log file\n");
  }

  for (unsigned i=0; i<sizeh; i++) {
    fprintf(fpa, "(%f %f)\t", cuCreal(a.r0.c0[i]), cuCimag(a.r0.c0[i]));
    fprintf(fpa, "\n");
  }

  for (unsigned i=0; i < sizeh/8; i++) {
    for (unsigned j=0; j < 256; j++) {
      if (j % 16 == 0 && j != 0) fprintf(fp, "\n");
      fprintf(fp, "%f\t", __half2float(in[i*256 + j]));
    }
    fprintf(fp, "\n\n");
  }

  fclose(fp);
}

void printMappedVec3_soa(half *in, vec3_soa a)
{
  FILE *fp;
  FILE *fpa;
  if ((fp=fopen("vec3_soa.log", "w")) == NULL) {
    fprintf(stderr, "Unable to create su3_soa log file\n");
  }

  if ((fpa=fopen("vec3_soa_fpa.log", "w")) == NULL) {
    fprintf(stderr, "Unable to create su3_soa log file\n");
  }

  for (unsigned i=0; i<sizeh; i++) {
    fprintf(fpa, "(%f %f), (%f %f), (%f %f)\t", cuCreal(a.c0[i]), cuCimag(a.c0[i]), cuCreal(a.c1[i]), cuCimag(a.c1[i]), cuCreal(a.c2[i]), cuCimag(a.c2[i]));
    fprintf(fpa, "\n");
  }

  for (unsigned i=0; i < sizeh/8; i++) {
    for (unsigned j=0; j < 256; j++) {
      if (j % 16 == 0 && j != 0) fprintf(fp, "\n");
      fprintf(fp, "%f\t", __half2float(in[i*256 + j]));
    }
    fprintf(fp, "\n\n");
  }

  fclose(fp);
}


// Just for debugging:
void showbits(unsigned int x) {

  int i;

  for(i=(sizeof(int)*8)-1; i>=0; i--)
    (x&(1<<i))?putchar('1'):putchar('0');

  printf("\n");

}

#endif
