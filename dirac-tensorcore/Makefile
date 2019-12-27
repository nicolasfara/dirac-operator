
# Define Lattice size and cache blocking size:
#LATTICE=-Dnx=16 -Dny=16 -Dnz=16 -Dnt=16 -DDIM_BLOCK_X=8 -DDIM_BLOCK_Y=8 -DDIM_BLOCK_Z=8 -DDIM_BLOCK_T=4
#LATTICE=-Dnx=32 -Dny=32 -Dnz=32 -Dnt=32 -DDIM_BLOCK_X=8 -DDIM_BLOCK_Y=8 -DDIM_BLOCK_Z=8 -DDIM_BLOCK_T=4
LATTICE=-Dnx=16 -Dny=16 -Dnz=16 -Dnt=16 -DDIM_BLOCK_X=8 -DDIM_BLOCK_Y=8 -DDIM_BLOCK_Z=8 -DDIM_BLOCK_T=4

###############################################################################

CC=nvcc

# K80
#CC_CFLAGS=-O3 --ptxas-options=-v -lineinfo -gencode arch=compute_37,code=sm_37 -keep

# V100
CC_CFLAGS=-O3 --ptxas-options=-v -lineinfo -gencode arch=compute_70,code=sm_70 -keep

###############################################################################

all: cuda

###############################################################################


# NVIDIA CUDA COMPILER

cuda: common-cuda.h
	nvcc $(LATTICE) $(CC_CFLAGS) main.cu -o test-cuda-gpu

# CLEAN

clean: 
	rm -rf *.o 

