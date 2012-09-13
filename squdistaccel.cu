  // Copyright 2011-2012 Justus Sagemüller.

  // This file is part of the Cqtx library.
   //This library is free software: you can redistribute it and/or modify
  // it under the terms of the GNU General Public License as published by
 //  the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
   //This library is distributed in the hope that it will be useful,
  // but WITHOUT ANY WARRANTY; without even the implied warranty of
 //  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
  // You should have received a copy of the GNU General Public License
 //  along with this library.  If not, see <http://www.gnu.org/licenses/>.


#include "squdistaccel.hcu"
#include<stdlib.h>
#include<stdio.h>


int cudaFnToMeasureseqSquaredistCalcHandle_bad
              ( const cudaFnToMeasureseqSquaredistCalcHandle* handle ) {
  return handle->fixed_params.cudastat   != cudaSuccess
      || handle->fixed_params.cublasstat != CUBLAS_STATUS_SUCCESS;
}



void print_cublas_status_message(const cublasStatus_t& stat) {
  switch (stat) {
   case CUBLAS_STATUS_SUCCESS          : printf("CUBLAS_STATUS_SUCCESS"          ); break;
   case CUBLAS_STATUS_NOT_INITIALIZED  : printf("CUBLAS_STATUS_NOT_INITIALIZED"  ); break;
   case CUBLAS_STATUS_ALLOC_FAILED     : printf("CUBLAS_STATUS_ALLOC_FAILED"     ); break;
   case CUBLAS_STATUS_INVALID_VALUE    : printf("CUBLAS_STATUS_INVALID_VALUE"    ); break;
   case CUBLAS_STATUS_ARCH_MISMATCH    : printf("CUBLAS_STATUS_ARCH_MISMATCH"    ); break;
   case CUBLAS_STATUS_MAPPING_ERROR    : printf("CUBLAS_STATUS_MAPPING_ERROR"    ); break;
   case CUBLAS_STATUS_EXECUTION_FAILED : printf("CUBLAS_STATUS_EXECUTION_FAILED" ); break;
   case CUBLAS_STATUS_INTERNAL_ERROR   : printf("CUBLAS_STATUS_INTERNAL_ERROR"   ); break;
   default                             : printf("CUBLAS_STATUS_UNKNOWN"          );
  }
}



void cudaFnToMeasureseqSquaredistCalcHandle_printerrmsg
              (const cudaFnToMeasureseqSquaredistCalcHandle* handle){
  if(handle->fixed_params.cudastat != cudaSuccess)
    printf("CUDA error: %s\n", cudaGetErrorString(handle->fixed_params.cudastat));
  if(handle->fixed_params.cublasstat != CUBLAS_STATUS_SUCCESS)
    print_cublas_status_message(handle->fixed_params.cublasstat);
}


/*
void watch_h(const cudastoredMeasureseqRepHandle* fixed) {
  double* peakprev = (double*) malloc(fixed->n_measures * sizeof(double));
  double* peakprev1 = (double*) malloc(fixed->n_measures * sizeof(double));

  printf("Inspect, %d values...\n", fixed->n_measures);
  cublasGetVector(fixed->n_measures, sizeof(double), fixed->measurevalseqs[0], 1, peakprev, 1);
  cublasGetVector(fixed->n_measures, sizeof(double), fixed->measurevalseqs[1], 1, peakprev1, 1);
  for(int k=0; k<fixed->n_measures; ++k) printf("%f\t @ %f\n", peakprev[k], peakprev1[k]);

  free(peakprev); free(peakprev1);
}
*/


cudaFnToMeasureseqSquaredistCalcHandle new_cudaFnToMeasureseqSquaredistCalcHandle
              ( const double** fixed_params
              , const double** fixed_prmsuncrts
              , const double* returncomps
              , const double* returnuncrts
              , unsigned n_measures
              , unsigned n_fixedparams, unsigned n_fixedprmsuncrts
              , unsigned n_fitparams
              , cudaNonlinSqdistEvalFunction evalfunction
              , const cublasHandle_t* cublashandle ) {
/*  printf("Creating squaredist-accelerator over %d measures with %d fixed parameters, and %d variables...\n", n_measures, n_fixedparams, n_fitparams);
  printf("Namely:\n");
  for(int k=0; k<n_measures; ++k) printf("%f \t@ %f\n", returncomps[k], fixed_params[0][k]);*/

  cudaFnToMeasureseqSquaredistCalcHandle handle;

  handle.fixed_params.cublashandle = cublashandle;

  handle.fixed_params.n_measures = n_measures;
  handle.n_fitparams = n_fitparams;
  handle.sqdist_function = evalfunction;
  
  handle.fixed_params.cublasstat = CUBLAS_STATUS_SUCCESS;

  int n_staticstore = 1 + n_fixedparams + !!returnuncrts + n_fixedprmsuncrts;

  double** fixedval_arrays
    = (double**) malloc(n_staticstore * sizeof(double*));

  if(!fixedval_arrays) handle.fixed_params.cudastat = cudaErrorMemoryAllocation;

//printf("Allocate function-ret-val-buffer...\n");
  if(fixedval_arrays)
    handle.fixed_params.cudastat
            = cudaMalloc( (void**)&handle.fixed_params.fnresbuffer
                        , n_measures * sizeof(double)              );

  handle.fixed_params.n_valspermeasure = 0;
  for( unsigned i=0
     ; i<n_staticstore && handle.fixed_params.cudastat==cudaSuccess && handle.fixed_params.cublasstat == CUBLAS_STATUS_SUCCESS
     ; ++i                                                          ) {
//printf("Allocate measure sequence...\n");
    handle.fixed_params.cudastat
       = cudaMalloc( (void**)&fixedval_arrays[i]
                   , n_measures * sizeof(double) );
    if(handle.fixed_params.cudastat==cudaSuccess) {

      const double* srcdataloc
         =  i == 0 ?
                     returncomps                                           :(
            returnuncrts && i==1+n_fixedparams ?
                     returnuncrts                                          :(
            i < n_fixedparams + 1 ? 
                     fixed_params[i-1]                                     :(
            i >= 1 + n_fixedparams + !!returnuncrts ?                      
                     fixed_prmsuncrts[i - (1+n_fixedparams+!!returnuncrts)]   :NULL)));

//printf("Fill measure sequence...\n");

      handle.fixed_params.cublasstat
               = cublasSetVector( n_measures, sizeof(double)
                                , srcdataloc
                                , 1
                                , fixedval_arrays[i]
                                , 1                          );
      handle.fixed_params.n_valspermeasure = i+1;
    }
  }

  handle.fixed_params.measurevalseqs = fixedval_arrays;

  return handle;  
}



cudaFnToMeasureseqSquaredistCalcHandle copy_cudaFnToMeasureseqSquaredistCalcHandle
              ( cudaFnToMeasureseqSquaredistCalcHandle* cpy )                {
  cudaFnToMeasureseqSquaredistCalcHandle handle = *cpy;

//  printf("Copying squaredist-accelerator over %d measures with %d fixed parameters (including return comp.), and %d variables...\n", handle.fixed_params.n_measures, handle.fixed_params.n_valspermeasure, handle.n_fitparams);

  handle.fixed_params.cublasstat = CUBLAS_STATUS_SUCCESS;

  double** fixedval_arrays
    = (double**) malloc(handle.fixed_params.n_valspermeasure * sizeof(double*));

  if(!fixedval_arrays) handle.fixed_params.cudastat = cudaErrorMemoryAllocation;
   else handle.fixed_params.cudastat
           = cudaMalloc( (void**)&handle.fixed_params.fnresbuffer
                       , handle.fixed_params.n_measures * sizeof(double) );

  handle.fixed_params.n_valspermeasure = 0;
  for( unsigned i=0
     ; i<cpy->fixed_params.n_valspermeasure && handle.fixed_params.cudastat==cudaSuccess && handle.fixed_params.cublasstat == CUBLAS_STATUS_SUCCESS
     ; ++i                                                          ) {
    handle.fixed_params.cudastat
       = cudaMalloc( (void**)&fixedval_arrays[i]
                   , handle.fixed_params.n_measures * sizeof(double) );
    if(handle.fixed_params.cudastat==cudaSuccess) {
      handle.fixed_params.cublasstat
               = cublasDcopy( *handle.fixed_params.cublashandle
                            , handle.fixed_params.n_measures
                            , cpy->fixed_params.measurevalseqs[i]
                            , 1
                            , fixedval_arrays[i]
                            , 1                                   );
      handle.fixed_params.n_valspermeasure = i+1;
    }
  }

  handle.fixed_params.measurevalseqs = fixedval_arrays;

  return handle;  
}



void delete_cudaFnToMeasureseqSquaredistCalcHandle
              ( cudaFnToMeasureseqSquaredistCalcHandle* handle ) {
//  printf("Deleting squaredist-accelerator over %d measures with %d fixed parameters, and %d variables...\n", handle->fixed_params.n_measures, handle->fixed_params.n_valspermeasure, handle->n_fitparams);

  for( unsigned i=0; i<handle->fixed_params.n_valspermeasure; ++i )
    cudaFree((void*) handle->fixed_params.measurevalseqs[i] );
  free((void*)handle->fixed_params.measurevalseqs);
  cudaFree((void*) handle->fixed_params.fnresbuffer);
}

cudaFnToMeasureseqSquaredistCalcHandle null_cudaFnToMeasureseqSquaredistCalcHandle() {
  cudaFnToMeasureseqSquaredistCalcHandle handle;

  handle.fixed_params.cublashandle = NULL;
  handle.fixed_params.n_measures = 0;
  handle.fixed_params.n_valspermeasure = 0;

  handle.n_fitparams = 0;
  handle.sqdist_function = NULL;

  handle.fixed_params.measurevalseqs = NULL;
  
  return handle;
}

#if 0
void nullify_cudaFnToMeasureseqSquaredistCalcHandle
              ( cudaFnToMeasureseqSquaredistCalcHandle* handle ) {
  for( unsigned i=0; i<handle->fixed_params.n_valspermeasure; ++i )
    handle->fixed_params.measurevalseqs[i] = NULL;
  handle->fixed_params.measurevalseqs = NULL;
  handle->fixed_params.fnresbuffer = NULL;
}
#endif



double evaluate_cudaFnToMeasureseqSquaredistCalc
              ( cudaFnToMeasureseqSquaredistCalcHandle* handle
              , const double* fitparams                        ) {
  if(!cudaFnToMeasureseqSquaredistCalcHandle_bad(handle))
    return handle->sqdist_function(&handle->fixed_params, fitparams);

  return -1;
}



