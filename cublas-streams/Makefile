#
# Makefile
# Nicolas Farabegoli, 2019-10-08 09:01
#

CC_CFLAGS=-O3 --ptxas-options=-v -lineinfo -gencode arch=compute_70,code=sm_70 -keep -lcublas
CC_CFLAGS-GTX=-O3 --ptxas-options=-v -lineinfo -gencode arch=compute_61,code=sm_61 -keep -lcublas
CC_CFLAGS-QUADRO=-O3 --ptxas-options=-v -lineinfo -gencode arch=compute_50,code=sm_50 -keep -lcublas
INCLUDE=-I./common/inc

all: volta

volta:
	nvcc $(CC_CFLAGS) $(INCLUDE) main.cu -o cublas-streams

gtx:
	nvcc $(CC_CFLAGS-GTX) $(INCLUDE) main.cu -o cublas-streams

quadro:
	nvcc $(CC_CFLAGS-QUADRO) $(INCLUDE) main.cu -o cublas-streams

$(EXE) : $(OBJS) $(CU_OBJS)
	$(CC) $(CFLAGS) $(NVCCFLAGS) -o $@ $?

clean:
	rm -f *.o *dlink* *.ii *.cubin *.gpu *.ptx *cudafe* *.fatbin* *.module_id cublas-streams

# vim:ft=make
#
