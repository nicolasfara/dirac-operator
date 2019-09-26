#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
extern void __device_stub__Z11dot_wmma4x4P6__halfS0_S0_(half *, half *, half *);
extern void __device_stub__Z7mat_subP6__halfS0_S0_j(half *, half *, half *, const unsigned);
extern void __device_stub__Z11fill_matrixP6__halfj(half *, const unsigned);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z11dot_wmma4x4P6__halfS0_S0_(half *__par0, half *__par1, half *__par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(half *, half *, half *))dot_wmma4x4)));}
# 9 "main.cu"
void dot_wmma4x4( half *__cuda_0,half *__cuda_1,half *__cuda_2)
# 10 "main.cu"
{__device_stub__Z11dot_wmma4x4P6__halfS0_S0_( __cuda_0,__cuda_1,__cuda_2);
# 19 "main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z7mat_subP6__halfS0_S0_j( half *__par0,  half *__par1,  half *__par2,  const unsigned __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(half *, half *, half *, const unsigned))mat_sub))); }
# 21 "main.cu"
void mat_sub( half *__cuda_0,half *__cuda_1,half *__cuda_2,const unsigned __cuda_3)
# 22 "main.cu"
{__device_stub__Z7mat_subP6__halfS0_S0_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3);




}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z11fill_matrixP6__halfj( half *__par0,  const unsigned __par1) {  __cudaLaunchPrologue(2); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaLaunch(((char *)((void ( *)(half *, const unsigned))fill_matrix))); }
# 29 "main.cu"
void fill_matrix( half *__cuda_0,const unsigned __cuda_1)
# 30 "main.cu"
{__device_stub__Z11fill_matrixP6__halfj( __cuda_0,__cuda_1);




}
# 1 "main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T3) {  __nv_dummy_param_ref(__T3); __nv_save_fatbinhandle_for_managed_rt(__T3); __cudaRegisterEntry(__T3, ((void ( *)(half *, const unsigned))fill_matrix), _Z11fill_matrixP6__halfj, (-1)); __cudaRegisterEntry(__T3, ((void ( *)(half *, half *, half *, const unsigned))mat_sub), _Z7mat_subP6__halfS0_S0_j, (-1)); __cudaRegisterEntry(__T3, ((void ( *)(half *, half *, half *))dot_wmma4x4), _Z11dot_wmma4x4P6__halfS0_S0_, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
