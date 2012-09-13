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


#include "squdistaccel-fns.hcu"
#include "cublas_v2.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>


__global__ void ca_d_gaussian_var_x0_invtwosigmasq_A
                ( const double* x
                , double x0
                , double inv_twosigmasq
                , double A
                , const double* rcmp
                , double* resc
                , int N                    ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    double deltax = x[i] - x0;
    resc[i] = rcmp[i] - A*exp(-deltax*deltax*inv_twosigmasq);
  }
}


double cudaaccelsqd_gaussian_VAR_x0_sigma_A
                ( cudastoredMeasureseqRepHandle* fixed
                , const double* fitparams              ) {
  static int block_size = 256;
  int n_blocks = fixed->n_measures/block_size
              + (fixed->n_measures%block_size==0? 0 : 1);
  ca_d_gaussian_var_x0_invtwosigmasq_A<<<n_blocks, block_size>>>
                         ( fixed->measurevalseqs[1]
                         , fitparams[0]
                         , 1./(2.*fitparams[1]*fitparams[1])
                         , fitparams[2]
                         , fixed->measurevalseqs[0]
                         , fixed->fnresbuffer
                         , fixed->n_measures               );
  double result;

  fixed->cublasstat
     = cublasDnrm2( *fixed->cublashandle
                  , fixed->n_measures
                  , fixed->fnresbuffer
                  , 1
                  , &result              );
  return result*result;  //nrm2 returns √(∑ x²).
}


__global__ void ca_d_gaussian_VAR_x0_invtwosigmasq_A_UNCRT_RET
                ( const double* x
                , double x0
                , double inv_twosigmasq
                , double A
                , const double* rcmp
                , const double* rcmpuncrt
                , double* resc
                , int N                    ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    double deltax = x[i] - x0;
    resc[i] = (rcmp[i] - A*exp(-deltax*deltax*inv_twosigmasq))
                                   / rcmpuncrt[i];
  }
}

double cudaaccelsqd_gaussian_VAR_x0_sigma_A_UNCRT_RET
                ( cudastoredMeasureseqRepHandle* fixed
                , const double* fitparams              ) {
  static int block_size = 256;
  int n_blocks = fixed->n_measures/block_size
              + (fixed->n_measures%block_size==0? 0 : 1);
  ca_d_gaussian_VAR_x0_invtwosigmasq_A_UNCRT_RET<<<n_blocks, block_size>>>
                         ( fixed->measurevalseqs[1]
                         , fitparams[0]
                         , 1./(2.*fitparams[1]*fitparams[1])
                         , fitparams[2]
                         , fixed->measurevalseqs[0]
                         , fixed->measurevalseqs[2]
                         , fixed->fnresbuffer
                         , fixed->n_measures               );
  double result;

  fixed->cublasstat
     = cublasDnrm2( *fixed->cublashandle
                  , fixed->n_measures
                  , fixed->fnresbuffer
                  , 1
                  , &result              );
  return result*result;  //nrm2 returns √(∑ x²).
}


__global__ void ca_d_gaussian_VAR_x0_invtwosigmasq_A_UNCRT_x
                ( const double* x
                , const double* xuncrt
                , double x0
                , double inv_twosigmasq
                , double A
                , const double* rcmp
                , double* resc
                , int N                    ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    double deltax = x[i] - x0
         , exp_argcoeff = -deltax*inv_twosigmasq
         , expf_res = A*exp(deltax*exp_argcoeff)
         , expf_uncrt = xuncrt[i] * expf_res * 2 * exp_argcoeff;
    resc[i] = (rcmp[i] - expf_res) / expf_uncrt;
  }
}

double cudaaccelsqd_gaussian_VAR_x0_sigma_A_UNCRT_x
                ( cudastoredMeasureseqRepHandle* fixed
                , const double* fitparams              ) {
  static int block_size = 256;
  int n_blocks = fixed->n_measures/block_size
              + (fixed->n_measures%block_size==0? 0 : 1);
  ca_d_gaussian_VAR_x0_invtwosigmasq_A_UNCRT_x<<<n_blocks, block_size>>>
                         ( fixed->measurevalseqs[1]
                         , fixed->measurevalseqs[2]
                         , fitparams[0]
                         , 1./(2.*fitparams[1]*fitparams[1])
                         , fitparams[2]
                         , fixed->measurevalseqs[0]
                         , fixed->fnresbuffer
                         , fixed->n_measures               );
  double result;

  fixed->cublasstat
     = cublasDnrm2( *fixed->cublashandle
                  , fixed->n_measures
                  , fixed->fnresbuffer
                  , 1
                  , &result              );
  return result*result;  //nrm2 returns √(∑ x²).
}


__global__ void ca_sqd_gaussian_VAR_x0_invtwosigmasq_A_UNCRT_RET_x
                ( const double* x
                , const double* xuncrt
                , double x0
                , double inv_twosigmasq
                , double A
                , const double* rcmp
                , const double* rcmpuncrt
                , double* resc
                , int N                    ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    double deltax = x[i] - x0
         , exp_argcoeff = -deltax*inv_twosigmasq
         , expf_res = A*exp(deltax*exp_argcoeff)
         , expf_uncrt = xuncrt[i] * expf_res * 2 * exp_argcoeff
         , deltaret = rcmp[i] - expf_res;
    resc[i] = deltaret*deltaret
                / (expf_uncrt*expf_uncrt + rcmpuncrt[i]*rcmpuncrt[i]);
  }
}

double cudaaccelsqd_gaussian_VAR_x0_sigma_A_UNCRT_RET_x
                ( cudastoredMeasureseqRepHandle* fixed
                , const double* fitparams              ) {
  static int block_size = 256;
  int n_blocks = fixed->n_measures/block_size
              + (fixed->n_measures%block_size==0? 0 : 1);
  ca_sqd_gaussian_VAR_x0_invtwosigmasq_A_UNCRT_RET_x<<<n_blocks, block_size>>>
                         ( fixed->measurevalseqs[1]
                         , fixed->measurevalseqs[3]
                         , fitparams[0]
                         , 1./(2.*fitparams[1]*fitparams[1])
                         , fitparams[2]
                         , fixed->measurevalseqs[0]
                         , fixed->measurevalseqs[2]
                         , fixed->fnresbuffer
                         , fixed->n_measures                 );
  double result;

  fixed->cublasstat
     = cublasDasum( *fixed->cublashandle
                  , fixed->n_measures
                  , fixed->fnresbuffer
                  , 1
                  , &result              );
  return result;
}




template<unsigned NPeaks>
struct multigaussian_VAR_x0_invtwosigmasq_A_PARAMS {
  double x0[NPeaks], inv_twosigmasq[NPeaks], A[NPeaks];
};

template<>
struct multigaussian_VAR_x0_invtwosigmasq_A_PARAMS<0> {
  union { double* x0; double* inv_twosigmasq; double* A; };
};

template<unsigned NPeaks>
__global__ void ca_d_multigaussian_VARS_x0_invtwosigmasq_A
                ( const double* x
                , multigaussian_VAR_x0_invtwosigmasq_A_PARAMS<NPeaks> ps
                , const double* rcmp
                , double* resc
                , int N                    ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  //static_assert(NPeaks>0);
  if(i<N) {
    double fnres = 0;
    for(unsigned j=0; j<NPeaks; ++j) {
      double deltax = x[i] - ps.x0[j];
      fnres += ps.A[j] * exp(-deltax*deltax * ps.inv_twosigmasq[j]);
    }
    resc[i] = rcmp[i] - fnres;
  }
}


template<unsigned NPeaks>
double cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A
                ( cudastoredMeasureseqRepHandle* fixed
                , const double* fitparams              ) {
/*  double* peakprev = (double*) malloc(fixed->n_measures * sizeof(double));
  double* peakprev1 = (double*) malloc(fixed->n_measures * sizeof(double));

  printf("Create arg tgt preview, %d values...\n", fixed->n_measures);
  cublasGetVector(fixed->n_measures, sizeof(double), fixed->measurevalseqs[0], 1, peakprev, 1);
  cublasGetVector(fixed->n_measures, sizeof(double), fixed->measurevalseqs[1], 1, peakprev1, 1);
  for(int k=0; k<fixed->n_measures; ++k) printf("%f\t @ %f\n", peakprev[k], peakprev1[k]);
*/
  static int block_size = 256;

  multigaussian_VAR_x0_invtwosigmasq_A_PARAMS<NPeaks> peaksargs;

  for(unsigned j=0; j<NPeaks; ++j) {
    peaksargs.x0[j] = fitparams[0 + 3*j];
    peaksargs.inv_twosigmasq[j] = 1./(2.*fitparams[1 + 3*j]*fitparams[1 + 3*j]);
    peaksargs.A[j] = fitparams[2 + 3*j];
  }

  int n_blocks = fixed->n_measures/block_size
              + (fixed->n_measures%block_size==0? 0 : 1);
  ca_d_multigaussian_VARS_x0_invtwosigmasq_A<NPeaks>
            <<<n_blocks, block_size>>>
                         ( fixed->measurevalseqs[1]
                         , peaksargs
                         , fixed->measurevalseqs[0]
                         , fixed->fnresbuffer
                         , fixed->n_measures        );
  double result;
/*
  printf("Create fn result preview, %d values...\n", fixed->n_measures);
  cublasGetVector(fixed->n_measures, sizeof(double), fixed->fnresbuffer, 1, peakprev, 1);
  for(int k=0; k<fixed->n_measures; ++k) printf("%f\n", peakprev[k]);
*/
  fixed->cublasstat
     = cublasDnrm2( *fixed->cublashandle
                  , fixed->n_measures
                  , fixed->fnresbuffer
                  , 1
                  , &result              );

//  free(peakprev);  free(peakprev1);

  return result*result;  //nrm2 returns √(∑ x²).
}

template<>
double cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<0>
                ( cudastoredMeasureseqRepHandle* fixed
                , const double* fitparams              ) {
  double result;
  fixed->cublasstat
     = cublasDnrm2( *fixed->cublashandle
                  , fixed->n_measures
                  , fixed->measurevalseqs[0]
                  , 1
                  , &result              );
  return result*result;  //nrm2 returns √(∑ x²).
}


const cudaNonlinSqdistEvalFunction
   cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_vtable[]
        = { cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<0>
//          , cudaaccel_sqdf_gaussian_VAR_x0_sigma_A
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<1>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<2>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<3>  /*
          , ...                                        ... */ , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<4>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<5>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<6>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<7>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<8>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<9>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<10>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<11>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<12>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<13>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<14>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A<15>  };
const unsigned n_instantiated_cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A
                                                       = 16;

cudaNonlinSqdistEvalFunction cudaaccelsqd_multigaussian_VARS_x0_sigma_A
              ( unsigned npeaks ) {
  if(npeaks < n_instantiated_cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A)
    return cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_vtable[npeaks];
   else
    return NULL;
}




template<unsigned NPeaks>
__global__ void ca_d_multigaussian_VARS_x0_invtwosigmasq_A_UNCRT_RET
                ( const double* x
                , multigaussian_VAR_x0_invtwosigmasq_A_PARAMS<NPeaks> ps
                , const double* rcmp
                , const double* rcmpuncrt
                , double* resc
                , int N                    ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  //static_assert(NPeaks>0);
  if(i<N) {
    double fnres = 0, xi=x[i];
    for(unsigned j=0; j<NPeaks; ++j) {
      double deltax = xi - ps.x0[j];
      fnres += ps.A[j] * exp(-deltax*deltax * ps.inv_twosigmasq[j]);
    }
    resc[i] = (rcmp[i] - fnres)/rcmpuncrt[i];
  }
}


template<unsigned NPeaks>
double cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET
                ( cudastoredMeasureseqRepHandle* fixed
                , const double* fitparams              ) {

  static int block_size = 256;

  multigaussian_VAR_x0_invtwosigmasq_A_PARAMS<NPeaks> peaksargs;

  for(unsigned j=0; j<NPeaks; ++j) {
    peaksargs.x0[j] = fitparams[0 + 3*j];
    peaksargs.inv_twosigmasq[j] = 1./(2.*fitparams[1 + 3*j]*fitparams[1 + 3*j]);
    peaksargs.A[j] = fitparams[2 + 3*j];
  }

  int n_blocks = fixed->n_measures/block_size
              + (fixed->n_measures%block_size==0? 0 : 1);
  ca_d_multigaussian_VARS_x0_invtwosigmasq_A_UNCRT_RET<NPeaks>
            <<<n_blocks, block_size>>>
                         ( fixed->measurevalseqs[1]
                         , peaksargs
                         , fixed->measurevalseqs[0]
                         , fixed->measurevalseqs[2]
                         , fixed->fnresbuffer
                         , fixed->n_measures        );
  double result;

  fixed->cublasstat
     = cublasDnrm2( *fixed->cublashandle
                  , fixed->n_measures
                  , fixed->fnresbuffer
                  , 1
                  , &result              );

  return result*result;  //nrm2 returns √(∑ x²).
}


const cudaNonlinSqdistEvalFunction
   cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_vtable[]
        = { cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<0>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<1>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<2>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<3>  /*
          , ...                                                  ... */ , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<4>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<5>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<6>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<7>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<8>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<9>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<10>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<11>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<12>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<13>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<14>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET<15>  };
const unsigned n_instantiated_cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET
                                                                 = 16;

cudaNonlinSqdistEvalFunction cudaaccelsqd_multigaussian_VARS_x0_sigma_A_UNCRT_RET
              ( unsigned npeaks ) {
  if(npeaks < n_instantiated_cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET)
    return cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_vtable[npeaks];
   else
    return NULL;
}




template<unsigned NPeaks>
__global__ void ca_sqd_multigaussian_VARS_x0_invtwosigmasq_A_UNCRT_x
                ( const double* x
                , const double* xuncrt
                , multigaussian_VAR_x0_invtwosigmasq_A_PARAMS<NPeaks> ps
                , const double* rcmp
                , double* resc
                , int N                                                  ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  //static_assert(NPeaks>0);
  if(i<N) {
    double fnres = 0, fnderivative = 0, xi=x[i];

    for(unsigned j=0; j<NPeaks; ++j) {
      double deltax = xi - ps.x0[j]
           , exp_argcoeff = -deltax*ps.inv_twosigmasq[j]
           , expf_res = ps.A[j]*exp(deltax*exp_argcoeff);

      fnres += expf_res;
      fnderivative += expf_res * 2 * exp_argcoeff;
    }
    double deltaret = rcmp[i] - fnres
         , resuncrt = xuncrt[i] * fnderivative;
    resc[i] = deltaret * deltaret
                 / ( resuncrt*resuncrt );
  }
}


template<unsigned NPeaks>
double cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x
                ( cudastoredMeasureseqRepHandle* fixed
                , const double* fitparams              ) {
  static int block_size = 256;

  multigaussian_VAR_x0_invtwosigmasq_A_PARAMS<NPeaks> peaksargs;

  for(unsigned j=0; j<NPeaks; ++j) {
    peaksargs.x0[j] = fitparams[0 + 3*j];
    peaksargs.inv_twosigmasq[j] = 1./(2.*fitparams[1 + 3*j]*fitparams[1 + 3*j]);
    peaksargs.A[j] = fitparams[2 + 3*j];
  }

  int n_blocks = fixed->n_measures/block_size
              + (fixed->n_measures%block_size==0? 0 : 1);
  ca_sqd_multigaussian_VARS_x0_invtwosigmasq_A_UNCRT_x<NPeaks>
            <<<n_blocks, block_size>>>
                         ( fixed->measurevalseqs[1]
                         , fixed->measurevalseqs[2]
                         , peaksargs
                         , fixed->measurevalseqs[0]
                         , fixed->fnresbuffer
                         , fixed->n_measures        );
  double result;

  fixed->cublasstat
     = cublasDasum( *fixed->cublashandle
                  , fixed->n_measures
                  , fixed->fnresbuffer
                  , 1
                  , &result              );

  return result;
}



const cudaNonlinSqdistEvalFunction
   cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x_vtable[]
        = { cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<0>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<1>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<2>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<3>  /*
          , ...                                                ... */ , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<4>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<5>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<6>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<7>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<8>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<9>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<10>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<11>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<12>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<13>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<14>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x<15>  };
const unsigned n_instantiated_cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x
                                                               = 16;

cudaNonlinSqdistEvalFunction cudaaccelsqd_multigaussian_VARS_x0_sigma_A_UNCRT_x
              ( unsigned npeaks ) {
  if(npeaks < n_instantiated_cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x)
    return cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_x_vtable[npeaks];
   else
    return NULL;
}



template<unsigned NPeaks>
__global__ void ca_sqd_multigaussian_VARS_x0_invtwosigmasq_A_UNCRT_RET_x
                ( const double* x
                , const double* xuncrt
                , multigaussian_VAR_x0_invtwosigmasq_A_PARAMS<NPeaks> ps
                , const double* rcmp
                , const double* rcmpuncrt
                , double* resc
                , int N                    ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  //static_assert(NPeaks>0);
  if(i<N) {
    double fnres = 0, fnderivative = 0, xi=x[i];

    for(unsigned j=0; j<NPeaks; ++j) {
      double deltax = xi - ps.x0[j]
           , exp_argcoeff = -deltax*ps.inv_twosigmasq[j]
           , expf_res = ps.A[j]*exp(deltax*exp_argcoeff);

      fnres += expf_res;
      fnderivative += expf_res * 2 * exp_argcoeff;
         // ∂ₓ A⋅exp(-x²/(2σ²)) = Aexp(-x²/(2σ²))⋅2⋅(-x/(2σ²))
    }

    double deltaret = rcmp[i] - fnres
         , resuncrt = xuncrt[i] * fnderivative;
    resc[i] = deltaret * deltaret
                 / ( resuncrt*resuncrt + rcmpuncrt[i]*rcmpuncrt[i] );
  }
}


template<unsigned NPeaks>
double cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x
                ( cudastoredMeasureseqRepHandle* fixed
                , const double* fitparams              ) {

  static int block_size = 256;

  multigaussian_VAR_x0_invtwosigmasq_A_PARAMS<NPeaks> peaksargs;

  for(unsigned j=0; j<NPeaks; ++j) {
    peaksargs.x0[j] = fitparams[0 + 3*j];
    peaksargs.inv_twosigmasq[j] = 1./(2.*fitparams[1 + 3*j]*fitparams[1 + 3*j]);
    peaksargs.A[j] = fitparams[2 + 3*j];
  }

  int n_blocks = fixed->n_measures/block_size
              + (fixed->n_measures%block_size==0? 0 : 1);
  ca_sqd_multigaussian_VARS_x0_invtwosigmasq_A_UNCRT_RET_x<NPeaks>
            <<<n_blocks, block_size>>>
                         ( fixed->measurevalseqs[1]
                         , fixed->measurevalseqs[3]
                         , peaksargs
                         , fixed->measurevalseqs[0]
                         , fixed->measurevalseqs[2]
                         , fixed->fnresbuffer
                         , fixed->n_measures        );
  double result;

  fixed->cublasstat
     = cublasDasum( *fixed->cublashandle
                  , fixed->n_measures
                  , fixed->fnresbuffer
                  , 1
                  , &result              );

  return result;
}


const cudaNonlinSqdistEvalFunction
   cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x_vtable[]
        = { cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<0>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<1>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<2>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<3>  /*
          , ...                                                    ... */ , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<4>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<5>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<6>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<7>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<8>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<9>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<10>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<11>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<12>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<13>, cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<14>
          , cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x<15>  };
const unsigned n_instantiated_cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x
                                                                   = 16;

cudaNonlinSqdistEvalFunction cudaaccelsqd_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x
              ( unsigned npeaks ) {
  if(npeaks < n_instantiated_cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x)
    return cudaaccel_sqdf_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x_vtable[npeaks];
   else
    return NULL;
}
