#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
extern void __device_stub__Z10fillMatrixP6float2j(cuFloatComplex *const, const unsigned);
extern void __device_stub__Z11checkMatrixPK6float2S1_jPb(const cuFloatComplex *const, const cuFloatComplex *const, const unsigned, bool *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z10fillMatrixP6float2j(cuFloatComplex *const __par0, const unsigned __par1){__cudaLaunchPrologue(2);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaLaunch(((char *)((void ( *)(cuFloatComplex *const, const unsigned))fillMatrix)));}
# 24 "main.cu"
void fillMatrix( cuFloatComplex *const __cuda_0,const unsigned __cuda_1)
# 25 "main.cu"
{__device_stub__Z10fillMatrixP6float2j( __cuda_0,__cuda_1);
# 31 "main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z11checkMatrixPK6float2S1_jPb( const cuFloatComplex *const __par0,  const cuFloatComplex *const __par1,  const unsigned __par2,  bool *__par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(const cuFloatComplex *const, const cuFloatComplex *const, const unsigned, bool *))checkMatrix))); }
# 33 "main.cu"
void checkMatrix( const cuFloatComplex *const __cuda_0,const cuFloatComplex *const __cuda_1,const unsigned __cuda_2,bool *__cuda_3)
# 34 "main.cu"
{__device_stub__Z11checkMatrixPK6float2S1_jPb( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 49 "main.cu"
}
# 1 "main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T9) {  __nv_dummy_param_ref(__T9); __nv_save_fatbinhandle_for_managed_rt(__T9); __cudaRegisterEntry(__T9, ((void ( *)(const cuFloatComplex *const, const cuFloatComplex *const, const unsigned, bool *))checkMatrix), _Z11checkMatrixPK6float2S1_jPb, (-1)); __cudaRegisterEntry(__T9, ((void ( *)(cuFloatComplex *const, const unsigned))fillMatrix), _Z10fillMatrixP6float2j, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
