all: bin/phqdumpview bin/qdafilecleanup lib/squdistaccel.o lib/squdistaccel-fns.o

install: all
	cp bin/phqdumpview /usr/local/bin

CPPC = g++-4.7
CPPCFlags= $(miscflags) $(optimizations) $(warnings) $(CPPincludes)
CPPincludes = -I $(cqtxheader)
cqtxheader = /media/misc/progwrit/cpp/cqtx
optimizations= -march=native # -DCUDA_ACCELERATION -O2 -ffast-math
warnings= -Wall
miscflags= --std=c++11 -g

CUDAC = nvcc
CUDACFlags = $(CUDAincludes) $(CUDAarch) -O2
CUDAarch = -arch=sm_21
CUDAlib = $(CUDApath)/lib64

CUDAincludes = -I$(CUDApath)/include -I$(CULApath)/include
CUDApath = /usr/local/cuda
CULApath = /usr/local/cula
libraryflags = -L/usr/local/cuda/lib64 -lcublas -lcusparse -L/usr/local/cula/lib64 -lcula_core -lcula_lapack

CC = gcc $(profileflag)


bin/phqdumpview: apps/phqdumpview.cpp *.cpp
	$(CPPC) -o $@ $< $(CPPCFlags)

bin/qdafilecleanup: apps/qdafilecleanup.cpp *.cpp
	$(CPPC) -o $@ $< $(CPPCFlags)

lib/squdistaccel.o : squdistaccel.cu squdistaccel.hcu
	$(CUDAC) -c $< $(CUDACFlags)
	@mv squdistaccel.o $@

lib/squdistaccel-fns.o : squdistaccel-fns.cu squdistaccel-fns.hcu squdistaccel.hcu
	$(CUDAC) -c $< $(CUDACFlags)
	@mv squdistaccel-fns.o $@