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
extern void __device_stub__Z7mat_subPfS_S_j(float *, float *, float *, const unsigned);
extern void __device_stub__Z7mat_addPfS_S_j(float *, float *, float *, const unsigned);
extern void __device_stub__Z9fill_zeroP6__halfS0_Pf(half *, half *, float *);
extern void __device_stub__Z14compose_matrixP6__halfS0_P7double2S2_(half *, half *, cuDoubleComplex *, cuDoubleComplex *);
extern void __device_stub__Z11isolate_vecPfS_S_S_S_(float *, float *, float *, float *, float *);
extern void __device_stub__Z11add_sub_vecPfS_S_S_S_S_(float *, float *, float *, float *, float *, float *);
extern void __device_stub__Z7combinePfS_P7double2(float *, float *, cuDoubleComplex *);
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
void __device_stub__Z7mat_subPfS_S_j( float *__par0,  float *__par1,  float *__par2,  const unsigned __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, const unsigned))mat_sub))); }
# 23 "main.cu"
void mat_sub( float *__cuda_0,float *__cuda_1,float *__cuda_2,const unsigned __cuda_3)
# 24 "main.cu"
{__device_stub__Z7mat_subPfS_S_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3);




}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z7mat_addPfS_S_j( float *__par0,  float *__par1,  float *__par2,  const unsigned __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, const unsigned))mat_add))); }
# 31 "main.cu"
void mat_add( float *__cuda_0,float *__cuda_1,float *__cuda_2,const unsigned __cuda_3)
# 32 "main.cu"
{__device_stub__Z7mat_addPfS_S_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3);




}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z9fill_zeroP6__halfS0_Pf( half *__par0,  half *__par1,  float *__par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(half *, half *, float *))fill_zero))); }
# 39 "main.cu"
void fill_zero( half *__cuda_0,half *__cuda_1,float *__cuda_2)
# 40 "main.cu"
{__device_stub__Z9fill_zeroP6__halfS0_Pf( __cuda_0,__cuda_1,__cuda_2);
# 49 "main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z14compose_matrixP6__halfS0_P7double2S2_( half *__par0,  half *__par1,  cuDoubleComplex *__par2,  cuDoubleComplex *__par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(half *, half *, cuDoubleComplex *, cuDoubleComplex *))compose_matrix))); }
# 95 "main.cu"
void compose_matrix( half *__cuda_0,half *__cuda_1,cuDoubleComplex *__cuda_2,cuDoubleComplex *__cuda_3)
# 96 "main.cu"
{__device_stub__Z14compose_matrixP6__halfS0_P7double2S2_( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 107 "main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z11isolate_vecPfS_S_S_S_( float *__par0,  float *__par1,  float *__par2,  float *__par3,  float *__par4) {  __cudaLaunchPrologue(5); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, float *, float *))isolate_vec))); }
# 116 "main.cu"
void isolate_vec( float *__cuda_0,float *__cuda_1,float *__cuda_2,float *__cuda_3,float *__cuda_4)
# 117 "main.cu"
{__device_stub__Z11isolate_vecPfS_S_S_S_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 123 "main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z11add_sub_vecPfS_S_S_S_S_( float *__par0,  float *__par1,  float *__par2,  float *__par3,  float *__par4,  float *__par5) {  __cudaLaunchPrologue(6); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 40UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, float *, float *, float *))add_sub_vec))); }
# 125 "main.cu"
void add_sub_vec( float *__cuda_0,float *__cuda_1,float *__cuda_2,float *__cuda_3,float *__cuda_4,float *__cuda_5)
# 126 "main.cu"
{__device_stub__Z11add_sub_vecPfS_S_S_S_S_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 132 "main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z7combinePfS_P7double2( float *__par0,  float *__par1,  cuDoubleComplex *__par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, cuDoubleComplex *))combine))); }
# 134 "main.cu"
void combine( float *__cuda_0,float *__cuda_1,cuDoubleComplex *__cuda_2)
# 135 "main.cu"
{__device_stub__Z7combinePfS_P7double2( __cuda_0,__cuda_1,__cuda_2);




}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z11fill_matrixP6__halfj( half *__par0,  const unsigned __par1) {  __cudaLaunchPrologue(2); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaLaunch(((char *)((void ( *)(half *, const unsigned))fill_matrix))); }
# 187 "main.cu"
void fill_matrix( half *__cuda_0,const unsigned __cuda_1)
# 188 "main.cu"
{__device_stub__Z11fill_matrixP6__halfj( __cuda_0,__cuda_1);




}
# 1 "main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T52) {  __nv_dummy_param_ref(__T52); __nv_save_fatbinhandle_for_managed_rt(__T52); __cudaRegisterEntry(__T52, ((void ( *)(half *, const unsigned))fill_matrix), _Z11fill_matrixP6__halfj, (-1)); __cudaRegisterEntry(__T52, ((void ( *)(float *, float *, cuDoubleComplex *))combine), _Z7combinePfS_P7double2, (-1)); __cudaRegisterEntry(__T52, ((void ( *)(float *, float *, float *, float *, float *, float *))add_sub_vec), _Z11add_sub_vecPfS_S_S_S_S_, (-1)); __cudaRegisterEntry(__T52, ((void ( *)(float *, float *, float *, float *, float *))isolate_vec), _Z11isolate_vecPfS_S_S_S_, (-1)); __cudaRegisterEntry(__T52, ((void ( *)(half *, half *, cuDoubleComplex *, cuDoubleComplex *))compose_matrix), _Z14compose_matrixP6__halfS0_P7double2S2_, (-1)); __cudaRegisterEntry(__T52, ((void ( *)(half *, half *, float *))fill_zero), _Z9fill_zeroP6__halfS0_Pf, (-1)); __cudaRegisterEntry(__T52, ((void ( *)(float *, float *, float *, const unsigned))mat_add), _Z7mat_addPfS_S_j, (-1)); __cudaRegisterEntry(__T52, ((void ( *)(float *, float *, float *, const unsigned))mat_sub), _Z7mat_subPfS_S_j, (-1)); __cudaRegisterEntry(__T52, ((void ( *)(half *, half *, float *))dot_wmma16x16), _Z13dot_wmma16x16P6__halfS0_Pf, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
