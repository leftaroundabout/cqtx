all: bin/sqd_accel

CXXC = g++-4.6
CXXCFlags= $(miscflags) $(optimizations) $(warnings) $(CXXincludes)
CXXincludes = $(CQTXincludes) $(CUDAincludes)
optimizations= -march=native # -DCUDA_ACCELERATION -O2 -ffast-math
warnings= -Wall
miscflags= --std=c++0x -g

CQTXpath = ..
CQTXincludes = -I$(CQTXpath)
CQTXlibfolder = $(CQTXpath)/lib
CQTXlibs = $(CQTXlibfolder)/squdistaccel.o $(CQTXlibfolder)/squdistaccel-fns.o
CQTXsourcefiles = $(CQTXfolder)/*.cpp

gcppDEPS= $(CQTXsourcefiles) # Makefile

CUDAincludes = -I$(CUDApath)/include -I$(CULApath)/include
CUDApath = /usr/local/cuda
CUDAlibs = -L/usr/local/cuda/lib64 -lcudart -lcublas

CC = gcc $(profileflag)



bin/sqd_accel : sqd_accel.cpp $(CQTXpath)/*.?pp $(CQTXpath)/*cu
	$(CXXC) -c $< $(CXXCFlags)
	$(CC) -o $@ sqd_accel.o $(CQTXlibs) $(CUDAlibs)
	@rm sqd_accel.o