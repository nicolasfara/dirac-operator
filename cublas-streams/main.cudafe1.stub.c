#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
extern void __device_stub__Z18kernel_fill_matrixP6__halfm(half *, size_t);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z18kernel_fill_matrixP6__halfm(half *__par0, size_t __par1){__cudaLaunchPrologue(2);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaLaunch(((char *)((void ( *)(half *, size_t))kernel_fill_matrix)));}
# 35 "cuda_utility.h"
void kernel_fill_matrix( half *__cuda_0,size_t __cuda_1)
# 36 "cuda_utility.h"
{__device_stub__Z18kernel_fill_matrixP6__halfm( __cuda_0,__cuda_1);



}
# 1 "main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T2) {  __nv_dummy_param_ref(__T2); __nv_save_fatbinhandle_for_managed_rt(__T2); __cudaRegisterEntry(__T2, ((void ( *)(half *, size_t))kernel_fill_matrix), _Z18kernel_fill_matrixP6__halfm, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
