#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
extern void __device_stub__Z13dot_wmma16x16P6__halfS0_Pf(half *, half *, float *);
extern void __device_stub__Z7mat_subP6__halfS0_S0_j(half *, half *, half *, const unsigned);
extern void __device_stub__Z7mat_addP6__halfS0_S0_j(half *, half *, half *, const unsigned);
extern void __device_stub__Z9fill_zeroP6__halfS0_Pf(half *, half *, float *);
extern void __device_stub__Z14compose_matrixP6__halfS0_P7double2S2_(half *, half *, cuDoubleComplex *, cuDoubleComplex *);
extern void __device_stub__Z11fill_matrixP6__halfj(half *, const unsigned);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z13dot_wmma16x16P6__halfS0_Pf(half *__par0, half *__par1, float *__par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(half *, half *, float *))dot_wmma16x16)));}
# 11 "main.cu"
void dot_wmma16x16( half *__cuda_0,half *__cuda_1,float *__cuda_2)
# 12 "main.cu"
{__device_stub__Z13dot_wmma16x16P6__halfS0_Pf( __cuda_0,__cuda_1,__cuda_2);
# 21 "main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z7mat_subP6__halfS0_S0_j( half *__par0,  half *__par1,  half *__par2,  const unsigned __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(half *, half *, half *, const unsigned))mat_sub))); }
# 23 "main.cu"
void mat_sub( half *__cuda_0,half *__cuda_1,half *__cuda_2,const unsigned __cuda_3)
# 24 "main.cu"
{__device_stub__Z7mat_subP6__halfS0_S0_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3);




}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z7mat_addP6__halfS0_S0_j( half *__par0,  half *__par1,  half *__par2,  const unsigned __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(half *, half *, half *, const unsigned))mat_add))); }
# 31 "main.cu"
void mat_add( half *__cuda_0,half *__cuda_1,half *__cuda_2,const unsigned __cuda_3)
# 32 "main.cu"
{__device_stub__Z7mat_addP6__halfS0_S0_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3);




}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z9fill_zeroP6__halfS0_Pf( half *__par0,  half *__par1,  float *__par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(half *, half *, float *))fill_zero))); }
# 74 "main.cu"
void fill_zero( half *__cuda_0,half *__cuda_1,float *__cuda_2)
# 75 "main.cu"
{__device_stub__Z9fill_zeroP6__halfS0_Pf( __cuda_0,__cuda_1,__cuda_2);
# 84 "main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z14compose_matrixP6__halfS0_P7double2S2_( half *__par0,  half *__par1,  cuDoubleComplex *__par2,  cuDoubleComplex *__par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(half *, half *, cuDoubleComplex *, cuDoubleComplex *))compose_matrix))); }
# 130 "main.cu"
void compose_matrix( half *__cuda_0,half *__cuda_1,cuDoubleComplex *__cuda_2,cuDoubleComplex *__cuda_3)
# 131 "main.cu"
{__device_stub__Z14compose_matrixP6__halfS0_P7double2S2_( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 142 "main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z11fill_matrixP6__halfj( half *__par0,  const unsigned __par1) {  __cudaLaunchPrologue(2); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaLaunch(((char *)((void ( *)(half *, const unsigned))fill_matrix))); }
# 194 "main.cu"
void fill_matrix( half *__cuda_0,const unsigned __cuda_1)
# 195 "main.cu"
{__device_stub__Z11fill_matrixP6__halfj( __cuda_0,__cuda_1);




}
# 1 "main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T49) {  __nv_dummy_param_ref(__T49); __nv_save_fatbinhandle_for_managed_rt(__T49); __cudaRegisterEntry(__T49, ((void ( *)(half *, const unsigned))fill_matrix), _Z11fill_matrixP6__halfj, (-1)); __cudaRegisterEntry(__T49, ((void ( *)(half *, half *, cuDoubleComplex *, cuDoubleComplex *))compose_matrix), _Z14compose_matrixP6__halfS0_P7double2S2_, (-1)); __cudaRegisterEntry(__T49, ((void ( *)(half *, half *, float *))fill_zero), _Z9fill_zeroP6__halfS0_Pf, (-1)); __cudaRegisterEntry(__T49, ((void ( *)(half *, half *, half *, const unsigned))mat_add), _Z7mat_addP6__halfS0_S0_j, (-1)); __cudaRegisterEntry(__T49, ((void ( *)(half *, half *, half *, const unsigned))mat_sub), _Z7mat_subP6__halfS0_S0_j, (-1)); __cudaRegisterEntry(__T49, ((void ( *)(half *, half *, float *))dot_wmma16x16), _Z13dot_wmma16x16P6__halfS0_Pf, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
