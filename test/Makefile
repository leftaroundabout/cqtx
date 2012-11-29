all: bin/sqd_accel bin/qdafilecleanup bin/algo_phqmfn_dblgauss_0

CXXC = g++-4.6
CXXCFlags= $(miscflags) $(optimizations) $(warnings) $(CXXincludes)
CXXincludes = $(CQTXincludes) $(CUDAincludes)
optimizations= -march=native -DCUDA_ACCELERATE_ALL # -O2 -ffast-math
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

bin/algo_phqmfn_dblgauss_0 : algo_phqmfn_dblgauss_0.cpp $(CQTXpath)/*.?pp $(CQTXpath)/*cu
	$(CXXC) -c $< $(CXXCFlags)
	$(CC) -o $@ algo_phqmfn_dblgauss_0.o $(CQTXlibs) $(CUDAlibs)
	@rm algo_phqmfn_dblgauss_0.o

bin/qdafilecleanup : qdafilecleanup.cpp $(CQTXpath)/*.?pp
	$(CXXC) -c $< $(CXXCFlags)
	$(CC) -o $@ qdafilecleanup.o $(CQTXlibs) $(CUDAlibs)
	@rm qdafilecleanup.o