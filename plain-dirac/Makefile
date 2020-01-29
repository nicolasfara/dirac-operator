
# Define Lattice size and cache blocking size:
LATTICE=-Dnx=16 -Dny=16 -Dnz=16 -Dnt=16 -DDIM_BLOCK_X=8 -DDIM_BLOCK_Y=8 -DDIM_BLOCK_Z=8 -DDIM_BLOCK_T=4
#LATTICE=-Dnx=32 -Dny=32 -Dnz=32 -Dnt=32 -DDIM_BLOCK_X=8 -DDIM_BLOCK_Y=8 -DDIM_BLOCK_Z=8 -DDIM_BLOCK_T=4

###############################################################################

CC=nvcc

# K80
#CC_CFLAGS=-O3 --ptxas-options=-v -lineinfo -gencode arch=compute_37,code=sm_37 -keep

# V100
#CC_CFLAGS=-O3 --ptxas-options=-v -lineinfo -gencode arch=compute_70,code=sm_70 #-keep

# 1050
CC_CFLAGS=-O3 --ptxas-options=-v -lineinfo -gencode arch=compute_61,code=sm_61 #-keep

###############################################################################

all: cuda cuda-8 plain

###############################################################################


# NVIDIA CUDA COMPILER

cuda: common-cuda.h
	nvcc $(LATTICE) $(CC_CFLAGS) main.cu -o dirac-operator

cuda-8: common-cuda.h
	nvcc $(LATTICE) $(CC_CFLAGS) main-8.cu -o dirac-operator-8

openacc: common-acc.h
	gcc $(LATTICE) -fopenacc -foffload=nvptx-none -foffload="-O3" -O3 -o dirac-operator-acc

plain: common-acc.h
	gcc $(LATTICE) main-plain.c -o dirac-operator-plain
	gcc $(LATTICE) main-plain-8.c -fstack-protector -g -o dirac-operator-plain-8

# CLEAN

clean: 
	rm -rf *.o *dlink* *.ii *.cubin *.gpu *.ptx *cudafe* *.fatbin* *.module_id dirac-operator* EndFermion.fer

