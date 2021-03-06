#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>

#include "interface.h"

extern "C" {


phmsqfnaccell_fpt *phmsqfnaccell_device_allocate(int n) {
  phmsqfnaccell_fpt *dptr;
  cudaError_t alloc_result = cudaMalloc( (void **) &dptr, n * sizeof(phmsqfnaccell_fpt) );
  if (alloc_result != cudaSuccess) {
    fprintf(stderr, "Unable to allocate memory on CUDA device.");
    exit(-1);
  }
  return dptr;
}

void phmsqfnaccell_device_injectdata( int ncalcs, int calcblocksize
                                    , int injoffset, int injcolumns
                                    , phmsqfnaccell_fpt *source[]
                                    , phmsqfnaccell_fpt *destination         ) {
  int memsize = ncalcs * calcblocksize * sizeof(phmsqfnaccell_fpt);
  phmsqfnaccell_fpt *hptr = (phmsqfnaccell_fpt *) malloc(memsize);
  for (int i=0; i<ncalcs; ++i) {
    for (int j=0; j<injcolumns; ++j) {
      hptr[i*calcblocksize + injoffset + j] = source[j][i];
    }
  }

  cudaError_t inject_result = cudaMemcpy( destination, hptr, memsize, cudaMemcpyHostToDevice );
  if (inject_result != cudaSuccess) {
    fprintf(stderr, "Unable to inject data to CUDA device.");
    exit(-1);
  }
}

void phmsqfnaccell_device_retrievedata( int ncalcs, int calcblocksize
                                      , int injoffset, int injcolumns
                                      , phmsqfnaccell_fpt *destination[]
                                      , phmsqfnaccell_fpt *source         ) {
  int memsize = ncalcs * calcblocksize * sizeof(phmsqfnaccell_fpt);
  phmsqfnaccell_fpt *hptr = (phmsqfnaccell_fpt *) malloc(memsize);

  cudaError_t inject_result = cudaMemcpy( hptr, source, memsize, cudaMemcpyDeviceToHost );
  if (inject_result != cudaSuccess) {
    fprintf(stderr, "Unable to retrieve data from CUDA device.");
    exit(-1);
  }
  for (int i=0; i<ncalcs; ++i) {
    for (int j=0; j<injcolumns; ++j) {
      destination[j][i] = hptr[i*calcblocksize + injoffset + j];
    }
  }
}

void phmsqfnaccell_device_freeall(device_concrrcalc_domains domns) {
  if (domns.stt_args!=domns.returns)
    cudaFree(domns.stt_args);
  cudaFree(domns.returns);
}

}

//      //      //      //      //      //      //      //      //      //



//      //      //      //      //      //      //      //      //      //

#define CUDA_PHMSQFNACCELL_BLOCKSIZE 512


__global__ void
calc_polynomial1_on_cuda ( int ncalcs
                         , int paramblocksize, int paramsoffset, phmsqfnaccell_fpt *params
                         , int resultblocksize, int resultoffset, phmsqfnaccell_fpt *results
                         , phmsqfnaccell_fpt coeff0, phmsqfnaccell_fpt coeff1               )  {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx<ncalcs)
    results[idx*resultblocksize + resultoffset]
      = coeff1 * params[idx*paramblocksize + paramsoffset]
      + coeff0;
}

extern "C"
void phmsqfnaccell_polynomial_calc( int degree, device_concrrcalc_domains domns ) {

  int block_size = CUDA_PHMSQFNACCELL_BLOCKSIZE;
  int n_blocks = domns.ncalcs/block_size + (domns.ncalcs%block_size == 0 ? 0:1);

  switch (degree) {
   case 1:
    calc_polynomial1_on_cuda <<< n_blocks, block_size >>>
                      ( domns.ncalcs
                      , domns.stt_argsblocksize, domns.stt_argsoffset, domns.stt_args
                      , domns.retsblocksize, domns.retsoffset, domns.returns
                      , domns.dyn_args[0],domns.dyn_args[1]
                      );
    break;
   default:
    fprintf(stderr, "Tried to calculate %inth-order polynomial, which is not yet supported on CUDA."
           ,                            degree);
  }
  
}



__global__ void
squaredistances_on_cuda ( int ncalcs
                        , int paramblocksize, int paramsoffset, phmsqfnaccell_fpt *params
                        , int resultblocksize, int resultoffset, phmsqfnaccell_fpt *results )  {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx<ncalcs){
    results[idx*resultblocksize + resultoffset]
      = params[idx*paramblocksize + paramsoffset] - params[idx*paramblocksize + paramsoffset + 1];
    results[idx*resultblocksize + resultoffset] *= results[idx*resultblocksize + resultoffset];
  }
}

extern "C"
void phmsqfnaccell_squaredists_calc( int zero, device_concrrcalc_domains domns ) {
  assert(zero==0);
 
  int block_size = CUDA_PHMSQFNACCELL_BLOCKSIZE;
  int n_blocks = domns.ncalcs/block_size + (domns.ncalcs%block_size == 0 ? 0:1);

  squaredistances_on_cuda <<< n_blocks, block_size >>>
                    ( domns.ncalcs
                    , domns.stt_argsblocksize, domns.stt_argsoffset, domns.stt_args
                    , domns.retsblocksize, domns.retsoffset, domns.returns
                    );
}





#if 0
void fix_dynaparams_to_cuda(device_concrrcalc_domains domns) {
  if (domns.dyn_argsblocksize==0) return;
  int coeffs_memsz = domns.dyn_argsblocksize * sizeof(phmsqfnaccell_fpt);
  if (!*domns.dyn_args_mem) {
    cudaError_t alloc_result = cudaMalloc( (void **) domns.dyn_args_mem, coeffs_memsz );
    if (alloc_result != cudaSuccess) {
      fprintf(stderr, "Unable to allocate memory for dynamic parameters on CUDA device.");
      exit(-1);
    }
  }
  cudaError_t inject_result = cudaMemcpy( *domns.dyn_args_mem, domns.dyn_args, coeffs_memsz, cudaMemcpyHostToDevice );
  if (inject_result != cudaSuccess) {
    fprintf(stderr, "Unable to inject dynamic data to CUDA device. (Error %i)", inject_result);
    exit(-1);
  }
}
#endif
