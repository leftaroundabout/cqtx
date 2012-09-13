all : bin/phmscltest0

CPPC = g++-4.6
gccFLAGS= $(gccMISCFLAGS) $(gccLIBS) $(OPTIMIZATIONS) $(WARNINGS)
OPTIMIZATIONS= -march=native -O2 -ffast-math
WARNINGS= -Wall
gccMISCFLAGS= --std=c++0x -g
gccLIBS = -L/usr/local/cuda/lib64 -lcuda -lcudart

CCC = nvcc
CUDPPINCPATH = /usr/local/cudpp_src_2.0/include/
CUDFLAGS = -I $(CUDPPINCPATH) -arch sm_13
cudpplib = /usr/local/cudpp/lib/libcudpp.so /usr/local/cudpp/lib/libcudpp_hash.so

objects = fnpointtest.o cwrapper.o

cwrapper.o : cwrapper.cu   # cudakernels.cu
	$(CCC) -c cwrapper.cu $(CUDFLAGS)

fnpointtest.o : fnpointtest.cpp
	$(CPPC) -c fnpointtest.cpp $(gccFLAGS)

bin/phmscltest0 : fnpointtest.o cwrapper.o
	$(CPPC) -o bin/phmscltest0 $(objects) $(gccFLAGS)

