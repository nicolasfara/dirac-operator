CC=nvcc

# K80
#CC_CFLAGS=-O3 --ptxas-options=-v -lineinfo -gencode arch=compute_37,code=sm_37 -keep

# V100
CC_CFLAGS=-O3 -lcublas --ptxas-options=-v -lineinfo -gencode arch=compute_70,code=sm_70 -keep

# GTX 1050
#CC_CFLAGS=-O3 -lcublas --ptxas-options=-v -lineinfo -gencode arch=compute_61,code=sm_61 -keep

###############################################################################

all: cuda

###############################################################################


# NVIDIA CUDA COMPILER

cuda: common-cuda.h
	nvcc $(LATTICE) $(CC_CFLAGS) main.cu -o test-complex-cublas

# CLEAN

clean: 
	rm -rf *.o 

