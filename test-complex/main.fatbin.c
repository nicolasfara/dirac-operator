#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x0000000000000eb0,0x0000004801010002,0x0000000000000e68\n"
".quad 0x0000000000000000,0x0000003d00010007,0x0000000700000040,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x0075632e6e69616d,0x33010102464c457f\n"
".quad 0x0000000000000007,0x0000006500be0002,0x0000000000000000,0x0000000000000dc0\n"
".quad 0x0000000000000a80,0x00380040003d053d,0x0001000d00400003,0x7472747368732e00\n"
".quad 0x747274732e006261,0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261\n"
".quad 0x666e692e766e2e00,0x2e747865742e006f,0x6c6c696631315a5f,0x5078697274616d5f\n"
".quad 0x6a666c61685f5f36,0x666e692e766e2e00,0x696631315a5f2e6f,0x697274616d5f6c6c\n"
".quad 0x6c61685f5f365078,0x732e766e2e006a66,0x5a5f2e6465726168,0x6d5f6c6c69663131\n"
".quad 0x5f36507869727461,0x2e006a666c61685f,0x74736e6f632e766e,0x315a5f2e30746e61\n"
".quad 0x616d5f6c6c696631,0x5f5f365078697274,0x642e006a666c6168,0x6e696c5f67756265\n"
".quad 0x642e6c65722e0065,0x6e696c5f67756265,0x65645f766e2e0065,0x656e696c5f677562\n"
".quad 0x722e00737361735f,0x65645f766e2e6c65,0x656e696c5f677562,0x6e2e00737361735f\n"
".quad 0x5f67756265645f76,0x007478745f787470,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x696631315a5f006f,0x697274616d5f6c6c,0x6c61685f5f365078,0x747865742e006a66\n"
".quad 0x6c696631315a5f2e,0x78697274616d5f6c,0x666c61685f5f3650,0x6e692e766e2e006a\n"
".quad 0x6631315a5f2e6f66,0x7274616d5f6c6c69,0x61685f5f36507869,0x2e766e2e006a666c\n"
".quad 0x5f2e646572616873,0x5f6c6c696631315a,0x365078697274616d,0x006a666c61685f5f\n"
".quad 0x736e6f632e766e2e,0x5a5f2e30746e6174,0x6d5f6c6c69663131,0x5f36507869727461\n"
".quad 0x5f006a666c61685f,0x642e006d61726170,0x6e696c5f67756265,0x642e6c65722e0065\n"
".quad 0x6e696c5f67756265,0x65645f766e2e0065,0x656e696c5f677562,0x722e00737361735f\n"
".quad 0x65645f766e2e6c65,0x656e696c5f677562,0x6e2e00737361735f,0x5f67756265645f76\n"
".quad 0x007478745f787470,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x000c00030000004b,0x0000000000000000,0x0000000000000000,0x000b0003000000b0\n"
".quad 0x0000000000000000,0x0000000000000000,0x00040003000000de,0x0000000000000000\n"
".quad 0x0000000000000000,0x00050003000000fa,0x0000000000000000,0x0000000000000000\n"
".quad 0x0006000300000126,0x0000000000000000,0x0000000000000000,0x000c101200000032\n"
".quad 0x0000000000000000,0x0000000000000100,0x00ce00020000010a,0x000a0efb01010000\n"
".quad 0x0100000001010101,0x696e2f656d6f682f,0x6f442f73616c6f63,0x2f73746e656d7563\n"
".quad 0x2f6370682f696e75,0x68736e7265746e69,0x63617269642f7069,0x6f74617265706f2d\n"
".quad 0x632d747365742f72,0x2f0078656c706d6f,0x616475632f74706f,0x2f2e2e2f6e69622f\n"
".quad 0x2f73746567726174,0x6c2d34365f363878,0x636e692f78756e69,0x616d00006564756c\n"
".quad 0x81010075632e6e69,0x75630bc905eca4e1,0x63697665645f6164,0x6d69746e75725f65\n"
".quad 0x00682e6970615f65,0x74a605eaeccede02,0x3170665f61647563,0xde02007070682e36\n"
".quad 0x0004bae305eaecce,0x0000000000020900,0x0113030104000000,0x0301380201031002\n"
".quad 0xd603030401200201,0xaa03010401200202,0x380202030108027d,0x004b010100380201\n"
".quad 0x0000001000020000,0x0101000a0efb0101,0x0000010000000101,0x0000000000020900\n"
".quad 0x0133030004000000,0x03f0810110026a03,0x020c03f001200202,0x8701080279030110\n"
".quad 0x01300204038084ea,0x0000000101003802,0x65762e0000000000,0x2e36206e6f697372\n"
".quad 0x65677261742e0034,0x0031365f6d732074,0x737365726464612e,0x343620657a69735f\n"
".quad 0x7369762e00000000,0x6e652e20656c6269,0x31315a5f20797274,0x74616d5f6c6c6966\n"
".quad 0x685f5f3650786972,0x702e00286a666c61,0x36752e206d617261,0x696631315a5f2034\n"
".quad 0x697274616d5f6c6c,0x6c61685f5f365078,0x6d617261705f6a66,0x7261702e002c305f\n"
".quad 0x203233752e206d61,0x6c6c696631315a5f,0x5078697274616d5f,0x6a666c61685f5f36\n"
".quad 0x315f6d617261705f,0x65722e007b002900,0x20646572702e2067,0x003b3e323c702509\n"
".quad 0x31622e206765722e,0x323c737225092036,0x206765722e003b3e,0x662509203233662e\n"
".quad 0x65722e003b3e323c,0x09203233622e2067,0x2e003b3e363c7225,0x3436622e20676572\n"
".quad 0x3e353c6472250920,0x702e646c0000003b,0x3436752e6d617261,0x202c316472250920\n"
".quad 0x6c696631315a5f5b,0x78697274616d5f6c,0x666c61685f5f3650,0x5f6d617261705f6a\n"
".quad 0x702e646c003b5d30,0x3233752e6d617261,0x5b202c3272250920,0x6c6c696631315a5f\n"
".quad 0x5078697274616d5f,0x6a666c61685f5f36,0x315f6d617261705f,0x2e766f6d00003b5d\n"
".quad 0x3372250920323375,0x782e64697425202c,0x33752e766f6d003b,0x202c347225092032\n"
".quad 0x3b782e6469746e25,0x3233752e766f6d00,0x25202c3572250920,0x3b782e6469617463\n"
".quad 0x2e6f6c2e64616d00,0x3172250920323373,0x25202c347225202c,0x3b337225202c3572\n"
".quad 0x672e707465730000,0x7025093233752e65,0x202c317225202c31,0x702540003b327225\n"
".quad 0x4209206172622031,0x0000003b325f3042,0x2e6f742e61747663,0x752e6c61626f6c67\n"
".quad 0x3264722509203436,0x003b31647225202c,0x2e6e722e74766300,0x093233752e323366\n"
".quad 0x317225202c316625,0x6320207b0000003b,0x31662e6e722e7476,0x7225203233662e36\n"
".quad 0x3b316625202c3173,0x6c756d000000007d,0x33752e656469772e,0x2c33647225092032\n"
".quad 0x3b32202c31722520,0x3436732e64646100,0x202c346472250920,0x7225202c32647225\n"
".quad 0x672e7473003b3364,0x31752e6c61626f6c,0x346472255b092036,0x3b31737225202c5d\n"
".quad 0x3a325f3042420000,0x7d003b7465720000,0x00082f0400000000,0x0000000500000006\n"
".quad 0x0000000600082304,0x0008120400000000,0x0000000000000006,0x0000000600081104\n"
".quad 0x0000300100000000,0x00080a0400002a01,0x000c014000000002,0x000c1704000c1903\n"
".quad 0x0008000100000000,0x000c17040011f000,0x0000000000000000,0x00ff1b030021f000\n"
".quad 0x0000001800041d04,0x0000005800081c04,0x00000000000000c8,0x00000000000000db\n"
".quad 0x0000000600000002,0x000000000000001d,0x0000000600000002,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x001c7c00e22007f6\n"
".quad 0x4c98078000870001,0xf0c8000002170000,0xf0c8000002570002,0x001fd840fec20ff1\n"
".quad 0x4e00000000270200,0x4f107f8000270203,0x5b30001800370200,0x001ff400fd4007ed\n"
".quad 0x4b6c038005270007,0x50b0000000070f00,0xe30000000000000f,0x001f8400e22207f0\n"
".quad 0x3828000001f70003,0x5cb8000000070a04,0x4c18808005070002,0x003fc400ffa00f15\n"
".quad 0x5ca8000000470904,0x4c10080005170303,0xeeda200000070204,0x001f9400fde007ef\n"
".quad 0x50b0000000070f00,0x50b0000000070f00,0x50b0000000070f00,0x001f8000ffe007ff\n"
".quad 0xe30000000007000f,0xe2400fffff87000f,0x50b0000000070f00,0x001f8000fc0007e0\n"
".quad 0x50b0000000070f00,0x50b0000000070f00,0x50b0000000070f00,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000300000001\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000040,0x0000000000000118\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x000000030000000b\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000158,0x0000000000000138\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x0000000200000013\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000290,0x00000000000000a8\n"
".quad 0x0000000500000002,0x0000000000000008,0x0000000000000018,0x00000001000000be\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000338,0x000000000000010e\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x00000001000000da\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000446,0x000000000000004f\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x0000000100000106\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000495,0x00000000000002e4\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x7000000000000029\n"
".quad 0x0000000000000000,0x0000000000000000,0x000000000000077c,0x0000000000000030\n"
".quad 0x0000000000000003,0x0000000000000004,0x0000000000000000,0x7000000000000051\n"
".quad 0x0000000000000000,0x0000000000000000,0x00000000000007ac,0x0000000000000050\n"
".quad 0x0000000c00000003,0x0000000000000004,0x0000000000000000,0x00000009000000ca\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000800,0x0000000000000010\n"
".quad 0x0000000400000003,0x0000000000000008,0x0000000000000010,0x00000009000000ee\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000810,0x0000000000000010\n"
".quad 0x0000000500000003,0x0000000000000008,0x0000000000000010,0x0000000100000097\n"
".quad 0x0000000000000002,0x0000000000000000,0x0000000000000820,0x000000000000014c\n"
".quad 0x0000000c00000000,0x0000000000000004,0x0000000000000000,0x0000000100000032\n"
".quad 0x0000000000000006,0x0000000000000000,0x0000000000000980,0x0000000000000100\n"
".quad 0x0500000600000003,0x0000000000000020,0x0000000000000000,0x0000000500000006\n"
".quad 0x0000000000000dc0,0x0000000000000000,0x0000000000000000,0x00000000000000a8\n"
".quad 0x00000000000000a8,0x0000000000000008,0x0000000500000001,0x0000000000000820\n"
".quad 0x0000000000000000,0x0000000000000000,0x000000000000024c,0x000000000000024c\n"
".quad 0x0000000000000008,0x0000000600000001,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000008\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[472];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 1, fatbinData, 0 };
#ifdef __cplusplus
}
#endif