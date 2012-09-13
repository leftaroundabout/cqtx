#ifndef ACCELERATED_SQUAREDISTANCE_EVALUATORS
#define ACCELERATED_SQUAREDISTANCE_EVALUATORS

#include "../lambdalike/maybe.hpp"
#include<vector>
#include<cassert>
#include<memory>


#ifdef ACCELERATE_FUNCTIONFITS
#define ACCELERATE_SQUAREDISTANCE_EVALUATORS
#endif

#if defined(CUDA_ACCELERATION) && defined(ACCELERATE_SQUAREDISTANCE_EVALUATORS)\
       || defined(CUDA_ACCELERATE_ALL)
#define USE_CUDA_TO_ACCELERATE_NONLIN_SQUAREDISTANCE
#endif


#ifdef USE_CUDA_TO_ACCELERATE_NONLIN_SQUAREDISTANCE
#include "cublas_v2.h"
#include "squdistaccel.hcu"
#include "squdistaccel-fns.hcu"
namespace squaredistanceAccel {
class nonlinSqD_handle;
class accelHandle {
  cublasHandle_t cublashandle;
  cublasStatus_t cublasstatus;
 public:
  accelHandle() {
    cublasstatus = cublasCreate(&cublashandle);
    assert(cublasstatus==CUBLAS_STATUS_SUCCESS);
  }
  accelHandle(const accelHandle& cp) =delete;
  accelHandle(accelHandle&& mov) noexcept
    : cublashandle(mov.cublashandle)
    , cublasstatus(mov.cublasstatus)
  {}
  accelHandle&operator=(accelHandle cp) {
    cublasDestroy(cublashandle);
    cublashandle = cp.cublashandle;
    cublasstatus = cp.cublasstatus;
    return *this;
  }
  ~accelHandle() {
    cublasDestroy(cublashandle);
  }

  friend class nonlinSqD_handle;

};
}//leccAecnatsiderauqs ecapseman
#  define CUDAerrcheckcdH            \
 if(!cudahandle){                    \
   printout_error(cudahandle);       \
   assert(!"CUDA operation");        \
 }
#  define CUDAerrcheckcdA(handle)    \
 if(!(handle)){                      \
   printout_error(handle);           \
   assert(!"CUDA operation");        \
 }
#  define CUDAerrcheckcdO(obj)       \
 if(!(obj).cudahandle){              \
   printout_error((obj).cudahandle); \
   assert(!"CUDA operation");        \
 }

#else//if !(defined(USE_CUDA_TO_ACCELERATE_NONLIN_SQUAREDISTANCE)
namespace squaredistanceAccel {
struct accelHandle{
  accelHandle(const accelHandle& cp) =delete;
  accelHandle&operator=(const accelHandle& cp) =delete;
  accelHandle() {};
};
}
#endif




namespace squaredistanceAccel_fns{

class gaussian_VAR_x0_sigma_A{};// "A⋅exp(-(x-x₀)²/(2σ²))"

class multigaussian_VARS_x0_sigma_A{};// "∑ⱼ Aⱼ⋅exp(-(x-x₀ⱼ)²/(2σⱼ²))"

}



namespace squaredistanceAccel {

typedef std::vector<double> llArray;
typedef std::vector<std::vector<double>> llMeasureseq;

using namespace lambdalike;

#ifdef USE_CUDA_TO_ACCELERATE_NONLIN_SQUAREDISTANCE

auto operator!(const cudaFnToMeasureseqSquaredistCalcHandle& h) -> bool {
  return cudaFnToMeasureseqSquaredistCalcHandle_bad(&h)!=0;             }
void printout_error(const cudaFnToMeasureseqSquaredistCalcHandle& h) {
  cudaFnToMeasureseqSquaredistCalcHandle_printerrmsg(&h);            }

class nonlinSqD_handle{
  mutable cudaFnToMeasureseqSquaredistCalcHandle cudahandle;
  std::shared_ptr<accelHandle> ah;
  bool migratedaway;
  
  nonlinSqD_handle( const llMeasureseq& fixed_params
                  , const std::map<int,llArray>& fixedprms_uncrts
                  , const llArray& tgtrets
                  , const maybe<llArray>& tgtret_unctrs
                  , unsigned n_fitparams
                  , cudaNonlinSqdistEvalFunction f       )
    : ah(std::make_shared<accelHandle>())
    , migratedaway(false)                  {

    unsigned n_measures = tgtrets.size(); 

    std::vector<const double*>fixedparamptrs;
    for(auto& p: fixed_params) {
      assert(p.size() == n_measures);
      fixedparamptrs.push_back(p.data());
    }
    std::vector<const double*>fixedparamuncrtptrs;
    for(auto& p: fixedprms_uncrts) {
      assert(p.second.size() == n_measures);
      fixedparamuncrtptrs.push_back(p.second.data());
    }

    cudahandle = new_cudaFnToMeasureseqSquaredistCalcHandle
                                           ( fixedparamptrs.data()
                                           , fixedparamuncrtptrs.data()
                                           , tgtrets.data()
                                           , tgtret_unctrs.is_just()
                                                 ? (*tgtret_unctrs).data()
                                                 : nullptr
                                           , n_measures
                                           , fixed_params.size()
                                           , fixedprms_uncrts.size()
                                           , n_fitparams
                                           , f
                                           , &ah->cublashandle             ); CUDAerrcheckcdH
//    std::cout << "Created nonlinSqD_handle" << std::endl;
  }

 public:

  auto operator()(const llArray& varparams)const -> double {
    double result = evaluate_cudaFnToMeasureseqSquaredistCalc(&cudahandle, varparams.data()); CUDAerrcheckcdH
    return result;
  }

  nonlinSqD_handle(nonlinSqD_handle&& mov) noexcept
    : cudahandle(mov.cudahandle)
    , ah(std::move(mov.ah))
    , migratedaway(false)   {
    mov.migratedaway = true;
//    std::cout << "Moved nonlinSqD_handle" << std::endl;
    //mov.cudahandle = null_cudaFnToMeasureseqSquaredistCalcHandle();
  }
  nonlinSqD_handle(const nonlinSqD_handle& cpy)
    : cudahandle(copy_cudaFnToMeasureseqSquaredistCalcHandle(&cpy.cudahandle))
    , ah(cpy.ah)
    , migratedaway(false)
  { CUDAerrcheckcdH }
//      std::cout << "Copied nonlinSqD_handle" << std::endl; }
      
  nonlinSqD_handle& operator=(const nonlinSqD_handle& cpy) =delete;

  ~nonlinSqD_handle() {
    if(!migratedaway) {
      delete_cudaFnToMeasureseqSquaredistCalcHandle(&cudahandle);
/*      std::cout << "delete nonlinSqD_handle" << std::endl;
     }else{
      std::cout << "delete (empty) nonlinSqD_handle" << std::endl; */
    }
  }
  
  template<typename FnSpecify>
  friend class nonlinSqD_staticDispatch;
};



template<typename FnSpecify>
struct nonlinSqD_staticDispatch {
  auto operator()( const llMeasureseq& fixed_params
                 , const std::map<int,llArray>& fixedprms_uncrts
                 , const llArray& tgtrets
                 , const maybe<llArray>& tgtret_unctrs
                 , unsigned n_fitparams                                         )
                     -> maybe<nonlinSqD_handle> {
    return nothing;
  }
  nonlinSqD_staticDispatch() {}
//  nonlinSqD_staticDispatch(const accelHandle& _) {}
};


template<>// "A⋅exp(-(x-x₀)²/(2σ²))"
struct nonlinSqD_staticDispatch<squaredistanceAccel_fns::gaussian_VAR_x0_sigma_A> {
  auto operator()( const llMeasureseq& fixed_params
                 , const std::map<int,llArray>& fixedprms_uncrts
                 , const llArray& tgtrets
                 , const maybe<llArray>& tgtret_unctrs
                 , unsigned n_fitparams             )
                     -> maybe<nonlinSqD_handle> {
    return just(nonlinSqD_handle
                 ( fixed_params
                 , fixedprms_uncrts
                 , tgtrets
                 , tgtret_unctrs
                 , n_fitparams
                 , !tgtret_unctrs.is_just()
                     ? (fixedprms_uncrts.size()==0
                         ? cudaaccelsqd_gaussian_VAR_x0_sigma_A 
                         : cudaaccelsqd_gaussian_VAR_x0_sigma_A_UNCRT_x )
                     : (fixedprms_uncrts.size()==0
                         ? cudaaccelsqd_gaussian_VAR_x0_sigma_A_UNCRT_RET
                         : cudaaccelsqd_gaussian_VAR_x0_sigma_A_UNCRT_RET_x ) ) );
  }
  nonlinSqD_staticDispatch() {}
};
template<>// "∑ⱼ Aⱼ⋅exp(-(x-x₀ⱼ)²/(2σⱼ²))"
struct nonlinSqD_staticDispatch<
          squaredistanceAccel_fns::multigaussian_VARS_x0_sigma_A > {
  auto operator()( const llMeasureseq& fixed_params
                 , const std::map<int,llArray>& fixedprms_uncrts
                 , const llArray& tgtrets
                 , const maybe<llArray>& tgtret_unctrs
                 , unsigned n_fitparams             )
                     -> maybe<nonlinSqD_handle> {

    assert(n_fitparams%3 == 0);
    unsigned npeaks = n_fitparams/3;

    auto sqdfnpt
       = !tgtret_unctrs.is_just()
           ? (fixedprms_uncrts.size()==0
               ? cudaaccelsqd_multigaussian_VARS_x0_sigma_A(npeaks)
               : cudaaccelsqd_multigaussian_VARS_x0_sigma_A_UNCRT_x(npeaks) )
           : (fixedprms_uncrts.size()==0
               ? cudaaccelsqd_multigaussian_VARS_x0_sigma_A_UNCRT_RET(npeaks)
               : cudaaccelsqd_multigaussian_VARS_x0_sigma_A_UNCRT_RET_x(npeaks) );
    if(sqdfnpt)
      return just( nonlinSqD_handle( fixed_params
                                   , fixedprms_uncrts
                                   , tgtrets
                                   , tgtret_unctrs
                                   , n_fitparams
                                   , sqdfnpt       ) );
     else return nothing;
  }
  nonlinSqD_staticDispatch() {}
};




template<typename FnSpecify> auto
nonlin_sqd_handle( const llMeasureseq& fixed_params
                 , const std::map<int,llArray>& fixedprms_uncrts
                 , const llArray& tgtrets
                 , const maybe<llArray>& tgtret_unctrs
                 , unsigned n_fitparams                                        )
         -> maybe<nonlinSqD_handle> {
  nonlinSqD_staticDispatch<FnSpecify> dispatched;
  return dispatched(fixed_params, fixedprms_uncrts
                   , tgtrets    , tgtret_unctrs
                   , n_fitparams);
}



#else//if (!defined(USE_CUDA_TO_ACCELERATE_NONLIN_SQUAREDISTANCE))


using namespace lambdalike;

struct nonlinSqD_handle{
  auto operator()(const llArray& varparams)const -> double {
    return 0;                                                          }
};

template<typename FnSpecify> auto
nonlin_sqd_handle( const llMeasureseq& fixed_params
                 , const std::map<int,llArray>& fixedprms_uncrts
                 , const llArray& tgtrets
                 , const maybe<llArray>& tgtret_unctrs
                 , unsigned n_fitparams                                        )
         -> maybe<nonlinSqD_handle> {
  return nothing;
}


#endif//defined(USE_CUDA_TO_ACCELERATE_NONLIN_SQUAREDISTANCE)


}//leccAecnatsiderauqs ecapseman


#endif