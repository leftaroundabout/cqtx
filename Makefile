CPPC = g++-4.7
CPPCFlags= $(miscflags) $(optimizations) $(warnings) $(CPPincludes)
CPPincludes = -I $(cqtxheader)
cqtxheader = /media/misc/progwrit/cpp/cqtx
optimizations= -march=native # -DCUDA_ACCELERATION -O2 -ffast-math
warnings= -Wall
miscflags= --std=c++11 -g

HSCodeGenSrc = HsMacro/CodeGen/CXX
HSCodeGenObj = HsMacro/dist/build/CodeGen/CXX

CUDAC = nvcc
CUDACFlags = $(CUDAincludes) $(CUDAarch) -O2
CUDAarch = -arch=sm_21
CUDAlib = $(CUDApath)/lib64

CUDAincludes = -I$(CUDApath)/include -I$(CULApath)/include
CUDApath = /usr/local/cuda
CULApath = /usr/local/cula
libraryflags = -L/usr/local/cuda/lib64 -lcublas -lcusparse -L/usr/local/cula/lib64 -lcula_core -lcula_lapack

CC = gcc $(profileflag)

HSR = runhaskell

fullCABAL = cabal install && cabal haddock --hyperlink-source


all: hs-gend/fitfncmacros-0.h   \
     $(HSCodeGenObj)/Cqtx/PhqFn.o \
     bin/phqdumpview bin/qdafilecleanup \
     lib/squdistaccel.o \
     lib/squdistaccel-fns.o

install: all
	cp bin/phqdumpview /usr/local/bin

	
hs-gend/fitfncmacros-*.h: fitfncmacros.h
# 	there's a bug in ghc's `unlit` at the moment, it chokes at c preprocessor directives. Quick workaround:
	sed 's/\(^ *\)#/\1%/g' hs-gend/fitfncmacros.lhs > /tmp/fitfncmacros.lhs
# 	$(HSR) hs-gend/fitfncmacros.lhs
	$(HSR) /tmp/fitfncmacros.lhs
	
$(HSCodeGenObj)/Cqtx/PhqFn.o : $(HSCodeGenSrc)/Cqtx/PhqFn.hs $(HSCodeGenSrc)/Cqtx/PhqFn.hs
	cd HsMacro; $(fullCABAL)

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