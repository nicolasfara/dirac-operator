CC=nvcc

# K80
CC_CFLAGS_K80=-O3 -lcublas --ptxas-options=-v -lineinfo -gencode arch=compute_37,code=sm_37 -keep

# V100
CC_CFLAGS_V100=-O3 -lcublas --ptxas-options=-v -lineinfo -gencode arch=compute_70,code=sm_70 -keep

# GTX 1050
CC_CFLAGS_1050=-O3 -lcublas --ptxas-options=-v -lineinfo -gencode arch=compute_61,code=sm_61 -keep

# GTX 1050
CC_CFLAGS_M1200=-O3 -lcublas --ptxas-options=-v -lineinfo -gencode arch=compute_50,code=sm_50 -keep

###############################################################################

all: volta

###############################################################################


# NVIDIA CUDA COMPILER

k80: common-cuda.h
	nvcc $(LATTICE) $(CC_CFLAGS_K80) main.cu -o test-complex

volta: common-cuda.h
	nvcc $(LATTICE) $(CC_CFLAGS_V100) main.cu -o test-complex

gtx: common-cuda.h
	nvcc $(LATTICE) $(CC_CFLAGS_1050) main.cu -o test-complex

quadro: common-cuda.h
	nvcc $(LATTICE) $(CC_CFLAGS_M1200) main.cu -o test-complex

# CLEAN
clean:
	rm -f *.o *dlink* *.ii *.cubin *.gpu *.ptx *cudafe* *.fatbin* *.module_id test-complex
