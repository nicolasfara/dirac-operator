#include <math.h>
#include <complex.h>
//#include <openacc.h>

// lattice dimensions
//#define nx 32 
//#define ny 32
//#define nz 32
//#define nt 32

#define NITER 10

#define DIM_BLOCK_X 8  // This should divide (nx/2)
#define DIM_BLOCK_Y 8  // This should divide ny
#define DIM_BLOCK_Z 8  // This should divide nz*nt

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

// Type definitions

typedef double complex d_complex;

typedef struct vec3_soa_t {
  d_complex c0[sizeh];
  d_complex c1[sizeh];
  d_complex c2[sizeh];
} vec3_soa;

typedef struct su3_soa_t {
  vec3_soa r0;
  vec3_soa r1;
  vec3_soa r2;
} su3_soa;

typedef struct vec3_t {
  d_complex c0;
  d_complex c1;
  d_complex c2;
} vec3;


// Common functions

static inline int snum(int x, int y, int z, int t) {

  int ris;

  ris = x + (y*vol1) + (z*vol2) + (t*vol3);

  return ris/2;   // <---  /2 Pay attention to even/odd  (see init_geo) 

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
      fermion->c0[i] = (ar + ai * I);
      fermion->c1[i] = (br + bi * I);
      fermion->c2[i] = (cr + ci * I);
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
        u[j].r0.c0[idx] = (ar + ai * I);
        u[j].r0.c1[idx] = (br + bi * I);
        u[j].r0.c2[idx] = (cr + ci * I);
      } else {
        printf("Read error... ");
        error = 1;
      }

      if (fscanf(fp, "(%lf,%lf) (%lf,%lf) (%lf,%lf) \n", &ar, &ai, &br, &bi, &cr, &ci) == 6) {
        u[j].r1.c0[idx] = (ar + ai * I);
        u[j].r1.c1[idx] = (br + bi * I);
        u[j].r1.c2[idx] = (cr + ci * I);
      } else {
        printf("Read error... ");
        error = 1;
      }

      if (fscanf(fp, "(%lf,%lf) (%lf,%lf) (%lf,%lf) \n", &ar, &ai, &br, &bi, &cr, &ci) == 6) {
        u[j].r2.c0[idx] = (ar + ai * I);
        u[j].r2.c1[idx] = (br + bi * I);
        u[j].r2.c2[idx] = (cr + ci * I);
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

    if (fprintf(fp, "(%lf,%lf) (%lf,%lf) (%lf,%lf) \n", creal(fermion->c0[i]), cimag(fermion->c0[i]), 
                                                        creal(fermion->c1[i]), cimag(fermion->c1[i]),
                                                        creal(fermion->c2[i]), cimag(fermion->c2[i])) < 0) {

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

