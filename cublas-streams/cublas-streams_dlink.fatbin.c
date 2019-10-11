#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x00000000000002b8,0x0000005001010002,0x0000000000000268\n"
".quad 0x0000000000000000,0x0000003200010007,0x0000000800000040,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x2075632e6e69616d,0x0000000000000000\n"
".quad 0x33010102464c457f,0x0000000000000007,0x0000006500be0002,0x0000000000000000\n"
".quad 0x00000000000001c0,0x00000000000000c0,0x0038004000320532,0x0001000400400003\n"
".quad 0x7472747368732e00,0x747274732e006261,0x746d79732e006261,0x746d79732e006261\n"
".quad 0x78646e68735f6261,0x666e692e766e2e00,0x747368732e00006f,0x74732e0062617472\n"
".quad 0x79732e0062617472,0x79732e006261746d,0x6e68735f6261746d,0x692e766e2e007864\n"
".quad 0x00000000006f666e,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000300000001,0x0000000000000000,0x0000000000000000,0x0000000000000040\n"
".quad 0x0000000000000032,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x000000030000000b,0x0000000000000000,0x0000000000000000,0x0000000000000072\n"
".quad 0x0000000000000032,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x0000000200000013,0x0000000000000000,0x0000000000000000,0x00000000000000a8\n"
".quad 0x0000000000000018,0x0000000000000002,0x0000000000000008,0x0000000000000018\n"
".quad 0x0000000500000006,0x00000000000001c0,0x0000000000000000,0x0000000000000000\n"
".quad 0x00000000000000a8,0x00000000000000a8,0x0000000000000008,0x0000000500000001\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000008,0x0000000600000001,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000008\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[89];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 2, fatbinData, (void**)__cudaPrelinkedFatbins };
#ifdef __cplusplus
}
#endif