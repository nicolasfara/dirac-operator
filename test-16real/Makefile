CC=nvcc

# V100
CC_CFLAGS_V100=-O3 --ptxas-options=-v -lineinfo -gencode arch=compute_70,code=sm_70 -lcublas #-keep
INLCUDE=-I./common/inc

###############################################################################

all: volta

###############################################################################

# NVIDIA CUDA COMPILER

volta:
	nvcc $(CC_CFLAGS_V100) $(INLCUDE) main.cu -o real-wmma

# CLEAN
clean:
	rm -f *.o *dlink* *.ii *.cubin *.gpu *.ptx *cudafe* *.fatbin* *.module_id real-wmma
