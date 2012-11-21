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


//Standard fittable functions, for use in cqtx with the minimization algorithms (e.g. via fitdist_fntomeasures).


#include "fitfncmacros.h"
                         

namespace_cqtxnamespace_OPEN



                                                              COPYABLE_PDERIVED_CLASS(/*
class*/fittable_gaussianfn,/*: public*/fittable_phmsqfn) {
  _4_FITVARIABLES(x, x0, sigma, A)
//  FITVARIABLE(0, x)  FITVARIABLE(1, x0)  FITVARIABLE(2, sigma)  FITVARIABLE(3, A)
 public:
  fittable_gaussianfn() { allocate_fitvarsbuf();
            cptof_x("x");          cptof_x0("x_0");         cptof_sigma("\\sigma");        cptof_A("A");  }      

  physquantity operator() (const measure &thisparams) const{
    ps = &thisparams;
    if (sigma() != 0)
      return A() * exp( ( ( x() - x0() ) / sigma() ).squared() * (-1/2.) );
     else return (x() == x0())? A() : 0;
  }

  measure example_parameterset(const measure &constraints, const physquantity &desiredret) const{
    measure example; ps = &constraints;

    if (ps->has(*cptof_A())){
      if (! desiredret.compatible(A())) {
        cerr << "Amplitude " << *cptof_A() << " incompatible with desired return value"
             << "\n(" << desiredret << ", while " << *cptof_A() << " = " << A() << ")\n";
      }
     }else{
      example.push_back( desiredret .plusminus (desiredret)  .label(*cptof_A()) );
    }

    if (ps->has(*cptof_x())) {
      if (ps->has(*cptof_x0())) {
        if (! x().compatible(x0())){
          cerr << "Offset " << *cptof_x0() << " incompatible with parameter " << *cptof_x()
               << "\n(" << *cptof_x0() << " = " << x0() << ", " << *cptof_x() << " = " << x() << ")\n";
          abort();
        }
       }else{
        example.push_back( x() .plusminus (x())  .label(*cptof_x0()) );
      }
      if (ps->has(*cptof_sigma())) {
        if (! x().compatible(sigma())){
          cerr << "Width " << *cptof_sigma() << " incompatible with parameter " << *cptof_x()
               << "\n(" << *cptof_sigma() << " = " << sigma() << ", " << *cptof_x() << " = " << x() << ")\n";
          abort();
        }
       }else{
        if(x().error() > 0) {
          example.push_back( x().error() .plusminus (x().error()) .label(*cptof_sigma()) );
         }else{
          example.push_back( x() .plusminus (x()) .label(*cptof_sigma()) );
        }
      }
     }else{
      if (ps->has(*cptof_x0())) {
        example.push_back( x0() .plusminus (x0())  .label(*cptof_x()) );
        if (ps->has(*cptof_sigma())) {
          if (! x().compatible(sigma())){
            cerr << "Width " << *cptof_sigma() << " incompatible with offset " << *cptof_x0()
                 << "\n(" << *cptof_sigma() << " = " << sigma() << ", " << *cptof_x0() << " = " << x0() << ")\n";
            abort();
          }
         }else{
          if(x().error() > 0) {
            example.push_back( x0().error() .plusminus (x0().error()) .label(*cptof_sigma()) );
           }else{
            example.push_back( x0() .plusminus (x0()) .label(*cptof_sigma()) );
          }
        }
       }else{
        if (ps->has(*cptof_sigma())) {
          example.push_back( sigma() .plusminus (sigma())  .label(*cptof_x()) );
          example.push_back( sigma() .plusminus (sigma())  .label(*cptof_x0()) );
         }else{
          cerr << "Insufficient constraints to determine the dimension of parameter " << *cptof_x()
               << ", offset " << *cptof_x0() << " or width " << *cptof_sigma() << " of gaussian function\n";
          abort();
        }
      }
    }
    
    
    return example;
  }
 
};



    //A spectrum is a fittable_phmsqfn that has one "evaluation" variable x
   // and three "parameter" variables per peak xⱼ, σⱼ, Aⱼ. fittable_multigaussianfn is
  //  the "prototype" spectrum; every derived spectrum should behave comparably
 //   in the sense that the peaks are almost compactly-supported on the interval
//    [x₀-2σ₀, x₀+2σ₀] and have an area ∫ℝ dx s(x, x₀, σ₀, A₀) ∝ A₀.



                                                                   COPYABLE_PDERIVED_CLASS(/*
class*/fittable_multigaussianfn,/*: public*/fittable_phmsqfn) {
  unsigned npeaks;
  _4_FITVARIABLES(x, x0, sigma, A)
  NMULTIFITVARIABLES(3)
 public:
  fittable_multigaussianfn(int nnpeaks=1): npeaks(nnpeaks) {
    allocate_fitvarsbuf(npeaks);
    cptof_x("x");
    for (unsigned i = 0; i<npeaks; ++i) {
      cptof_x0(i,    "x"       + LaTeX_subscript(i));
      cptof_sigma(i, "\\sigma" + LaTeX_subscript(i));
      cptof_A(i,     "A"       + LaTeX_subscript(i));
    }
    
  }

  physquantity operator() (const measure &thisparams) const{
    ps = &thisparams;
    physquantity result=0;
    for (unsigned j = 0; j<npeaks; ++j) {
      if (sigma(j) != 0)
        result += A(j) * exp( ( ( x() - x0(j) ) / sigma(j) ).squared() * (-1/2.) );
       else {
        result += (x() == x0(j))? A(j) : 0;
      }
    }
    return result;
  }

  auto
  squaredist_accel(const measureseq& fixedparams, const msq_dereferencer& retcomp_drf)const
            -> p_maybe<phmsq_function> {
             
    assert(fixedparams.size() > 0);
    ps = &fixedparams.front();

    std::vector<const phUnit*> agdfunits(3*npeaks);
    const phUnit* retunit;
    
    std::vector<captfinder> varfinds(3*npeaks);

  // we go for multigaussian_vars_x0_sigma_A.
  // x0, x and sigma have the same physical dimension.
    if(!ps->has(*cptof_x())) return nothing;
    agdfunits[0] = x().tryfindUnit().preferredUnit();
    assert(agdfunits[0] != NULL);
    
    agdfunits[1] = agdfunits[0];

    if(auto fst_r = retcomp_drf.tryfind(*ps)) {
      agdfunits[2] = fst_r->tryfindUnit().preferredUnit();
      assert(agdfunits[2] != NULL);
      retunit = agdfunits[2];
     }else return nothing;

    for(unsigned j=3; j<3*npeaks; ++j) agdfunits[j] = agdfunits[j%3];

    for(unsigned i=0; i<npeaks; ++i) {
      varfinds[0+3*i] = cptof_x0(i);
      varfinds[1+3*i] = cptof_sigma(i);
      varfinds[2+3*i] = cptof_A(i);
    }


    bool allhave_error = true;

    for(auto& axs : fixedparams) {
      if(auto x = cptof_x().tryfind(axs)) {
        if(auto rt = retcomp_drf.tryfind(axs)) {
          if(x->error()==0 && rt->error()==0) allhave_error=false;
        }else return nothing;
      }else return nothing;
    }
    

    std::vector<double> rtcomps, rtcomps_uncrt;
    std::vector<std::vector<double>> fixed_dat(1);
    std::map<int,std::vector<double>> fixed_dat_uncrt;
    if(allhave_error) fixed_dat_uncrt[0] = {};

    for(auto& axs : fixedparams) {
      auto x = cptof_x()(axs);
      auto rt = retcomp_drf(axs);
      if(allhave_error || ( x.error()==0 && rt.error()==0 )) {
        fixed_dat[0].push_back(x[*agdfunits[0]]);
        rtcomps.push_back(rt[*agdfunits[2]]);
      }
      if(allhave_error) {
        fixed_dat_uncrt.begin()->second
                     .push_back(x.error()[*agdfunits[0]]);
        rtcomps_uncrt.push_back(rt.error()[*agdfunits[2]]);
      }
    }


    for(auto& h: squaredistanceAccel::nonlin_sqd_handle
                   <squaredistanceAccel_fns::multigaussian_VARS_x0_sigma_A>
                                                ( fixed_dat
                                                , fixed_dat_uncrt
                                                , rtcomps
                                                , allhave_error
                                                    ? just(rtcomps_uncrt)
                                                    : nothing
                                                , 3*npeaks)                 )
      return just( fittable_phmsqfn::
                      build_sqdistaccel( std::move(h)
                                       , varfinds
                                       , std::make_pair( agdfunits
                                                       , allhave_error
                                                           ? &real1
                                                           : retunit   ) ) );

    return nothing;
  }

  measure example_parameterset(const measure &constraints, const physquantity &desiredret) const{
    measure example; ps = &constraints;

    for (unsigned i = 0; i<npeaks; ++i) {
      if (ps->has(*cptof_A(i))){
        if (! desiredret.compatible(A(i))) {
          cerr << "Amplitude " << *cptof_A(i) << " incompatible with desired return value"
               << "\n(" << desiredret << ", while " << *cptof_A(i) << " = " << A(i) << ")\n";
        }
       }else{
        example.push_back( desiredret .plusminus (desiredret)  .label(*cptof_A(i)) );
      }
 
      if (ps->has(*cptof_x())) {
        if (ps->has(*cptof_x0(i))) {
          if (! x().compatible(x0(i))){
            cerr << "Offset " << *cptof_x0(i) << " incompatible with parameter " << *cptof_x()
                 << "\n(" << *cptof_x0(i) << " = " << x0(i) << ", " << *cptof_x() << " = " << x() << ")\n";
            goto insufficient;
          }
         }else{
          example.push_back( x() .plusminus (x())  .label(*cptof_x0(i)) );
        }
        if (ps->has(*cptof_sigma(i))) {
          if (! x().compatible(sigma(i))){
            cerr << "Width " << *cptof_sigma(i) << " incompatible with parameter " << *cptof_x()
               << "\n(" << *cptof_sigma(i) << " = " << sigma(i) << ", " << *cptof_x() << " = " << x() << ")\n";
            goto insufficient;
          }
         }else{
          if(x().error() > 0) {
            example.push_back( x().error() .plusminus (x().error()) .label(*cptof_sigma(i)) );
           }else{
            example.push_back( x() .plusminus (x()) .label(*cptof_sigma(i)) );
          }
        }
       }else{
        if (ps->has(*cptof_x0(i))) {
          example.push_back( x0(i) .plusminus (x0(i))  .label(*cptof_x()) );
          if (ps->has(*cptof_sigma(i))) {
            if (! x().compatible(sigma(i))){
              cerr << "Width " << *cptof_sigma(i) << " incompatible with offset " << *cptof_x0(i)
                   << "\n(" << *cptof_sigma(i) << " = " << sigma(i) << ", " << *cptof_x0(i) << " = " << x0(i) << ")\n";
              goto insufficient;
            }
           }else{
            if(x0().error() > 0) {
              example.push_back( x0().error() .plusminus (x0().error()) .label(*cptof_sigma(i)) );
             }else{
              example.push_back( x0(i) .plusminus (x0(i)) .label(*cptof_sigma(i)) );
            }
          }
         }else{
          if (ps->has(*cptof_sigma(i))) {
            example.push_back( sigma(i) .plusminus (sigma(i))  .label(*cptof_x0(i)) );
           }else{
            cerr << "Insufficient constraints to determine the dimension of offset "
                 << *cptof_x0(i) << " or width " << *cptof_sigma(i) << " of spectrum function\n";
            goto insufficient;
          }
        }
      }
    }
    
    if (!ps->has(*cptof_x())) {
      for (unsigned i = 0; i<npeaks; ++i) {
        if (ps->has(*cptof_x0(i))) {
          example.push_back( x0(i) .plusminus (x0(i))  .label(*cptof_x()) );
          break;
         }else if(ps->has(*cptof_sigma(i))){
          example.push_back( sigma(i) .plusminus (sigma(i))  .label(*cptof_x()) );
          break;
         }else if(i==npeaks-1){
          cerr << "Insufficient constraints to determine the dimension of parameter " << *cptof_x()
               << ", offset " << *cptof_x0(i) << " or width " << *cptof_sigma(i) << " of gaussian function\n";
          goto insufficient;
        }
      }
    }
    return example;

    insufficient:
    cout << "Constraints were:\n" << *ps;
    abort();
  }
  
};


                                                                              COPYABLE_PDERIVED_CLASS(/*
class*/combinedPeaks_fittable_spectrum,/*: public*/fittable_phmsqfn) {
  std::unique_ptr<fittable_phmsqfn> base_spectr; // the spectrum that's actually used for evaluating the combination 
  fittable_multigaussianfn exampleparams_dummy;  // this one is only used to generate example parameters
  unsigned npeaks, basepeaks_per_cpeak;
  
  std::string intern_x0caption, intern_sigmacaption, intern_Acaption;
  
  _4_FITVARIABLES(x, x0, sigma, A)
  NMULTIFITVARIABLES(3)    //x₀, σ and A together describe one peak.

 public:
  typedef std::function<std::vector<measure>(const measure&)> PeaksCombiner;
 private:
  PeaksCombiner peakscombiner;
  
                                                                     COPYABLE_DERIVED_STRUCT(/*
  struct*/squaredistAccel,/*: public*/phmsq_function) {
    _3_FITVARIABLES(x0, sigma, A)
    NMULTIFITVARIABLES(3)
    std::shared_ptr<phmsq_function> basespectr_acceld;
    unsigned npeaks, basepeaks_per_cpeak;
    std::string intern_x0caption, intern_sigmacaption, intern_Acaption;
    PeaksCombiner peakscombiner;
    
    auto operator() (const measure& thisparams)const -> physquantity {
      ps = &thisparams;
      measure pass_params;
      for(unsigned i=0; i<npeaks; ++i){
        measure thispeak;
        thispeak.let(intern_x0caption   ) = x0(i);
        thispeak.let(intern_sigmacaption) = sigma(i);
        thispeak.let(intern_Acaption    ) = A(i);
        unsigned j = i * basepeaks_per_cpeak;
        for(auto& thissubpeak: peakscombiner(thispeak)) {
          pass_params.let(   "x"    + LaTeX_subscript(j)) = thissubpeak[intern_x0caption   ];
          pass_params.let("\\sigma" + LaTeX_subscript(j)) = thissubpeak[intern_sigmacaption];
          pass_params.let(   "A"    + LaTeX_subscript(j)) = thissubpeak[intern_Acaption    ];
          ++j;
        }
      }
      return (*basespectr_acceld)(pass_params);
    }
    
    squaredistAccel( std::unique_ptr<phmsq_function> basespectr_acceld
                   , unsigned basepeaks_per_cpeak
                   , PeaksCombiner peakscombiner
                   , std::string intern_x0caption
                   , std::string intern_sigmacaption
                   , std::string intern_Acaption
                   , const std::vector<std::array<std::string, 3>>& param_captions )
      : basespectr_acceld(std::move(basespectr_acceld))
      , npeaks(param_captions.size()), basepeaks_per_cpeak(basepeaks_per_cpeak)
      , peakscombiner(std::move(peakscombiner))                                   {
      allocate_fitvarsbuf(npeaks);
      for (unsigned i = 0; i<npeaks; ++i) {
        cptof_x0(   i, param_captions[i][0] );
        cptof_sigma(i, param_captions[i][1] ); 
        cptof_A(    i, param_captions[i][2] );
      }
    }
    
  };
  
 public:
  template<class BaseSpectrum>
  combinedPeaks_fittable_spectrum( BaseSpectrum base_spectr
                                 , unsigned basepeaks_per_cpeak
                                 , PeaksCombiner peakscombiner
                                 , std::string intern_x0caption
                                 , std::string intern_sigmacaption
                                 , std::string intern_Acaption
                                 , unsigned npeaks=1                )
    : base_spectr(base_spectr.moved())
    , exampleparams_dummy(npeaks)
    , npeaks(npeaks), basepeaks_per_cpeak(basepeaks_per_cpeak)
    , intern_x0caption(intern_x0caption), intern_sigmacaption(intern_sigmacaption), intern_Acaption(intern_Acaption)
    , peakscombiner(std::move(peakscombiner))                  {
    allocate_fitvarsbuf(npeaks);
    cptof_x("x");
    for (unsigned i = 0; i<npeaks; ++i) {
      cptof_x0(   i,    "x"    + LaTeX_subscript(i));
      cptof_sigma(i, "\\sigma" + LaTeX_subscript(i));
      cptof_A(    i,    "A"    + LaTeX_subscript(i));
    }
  }
  
  combinedPeaks_fittable_spectrum(
      const combinedPeaks_fittable_spectrum& cpy)
    : base_spectr(cpy.base_spectr->clone())
    , exampleparams_dummy(cpy.npeaks)
    , npeaks(cpy.npeaks)
    , basepeaks_per_cpeak(cpy.basepeaks_per_cpeak)
    , intern_x0caption(cpy.intern_x0caption), intern_sigmacaption(cpy.intern_sigmacaption), intern_Acaption(cpy.intern_Acaption)
    , peakscombiner(cpy.peakscombiner) {
    allocate_fitvarsbuf(npeaks);
    cptof_x("x");
    for (unsigned i = 0; i<npeaks; ++i) {
      cptof_x0(   i,    "x"    + LaTeX_subscript(i));
      cptof_sigma(i, "\\sigma" + LaTeX_subscript(i));
      cptof_A(    i,    "A"    + LaTeX_subscript(i));
    }
  }
  
  auto operator() (const measure& thisparams)const -> physquantity {
    ps = &thisparams;
    measure pass_params;
    pass_params.let("x") = x();
    for(unsigned i=0; i<npeaks; ++i){
      measure thispeak;
      thispeak.let(intern_x0caption   ) = x0(i);
      thispeak.let(intern_sigmacaption) = sigma(i);
      thispeak.let(intern_Acaption    ) = A(i);
      unsigned j = i * basepeaks_per_cpeak;
      for(auto& thissubpeak: peakscombiner(thispeak)) {
        pass_params.let(   "x"    + LaTeX_subscript(j)) = thissubpeak[intern_x0caption   ];
        pass_params.let("\\sigma" + LaTeX_subscript(j)) = thissubpeak[intern_sigmacaption];
        pass_params.let(   "A"    + LaTeX_subscript(j)) = thissubpeak[intern_Acaption    ];
        ++j;
      }
    }
    return (*base_spectr)(pass_params);
  }
  
  auto
  squaredist_accel(const measureseq& fixedparams, const msq_dereferencer& retcomp_drf)const
            -> p_maybe<phmsq_function> {
    measureseq pass_fixedps;
    for(auto& axs: fixedparams) {
      ps = &axs;
      pass_fixedps.push_back(measure());
      if(axs.has(*cptof_x())) {
        pass_fixedps.back().let("x") = x();
      }else return nothing;
      if(auto rt = retcomp_drf.tryfind(axs)) {
        pass_fixedps.back().let("I") = *rt;
      }else return nothing;
    }
    auto delegated = base_spectr->squaredist_accel(pass_fixedps, captfinder("I"));
    if(delegated.is_nothing()) return nothing;
    
    std::vector<std::array<std::string, 3>> accelcaptions;
    for(unsigned i=0; i<npeaks; ++i) {
      accelcaptions.push_back(std::array<std::string, 3>
               {{ *cptof_x0(i), *cptof_sigma(i), *cptof_A(i) }});
    }
    
    return just (
       squaredistAccel( std::unique_ptr<phmsq_function>((*delegated).moved())
                      , basepeaks_per_cpeak
                      , peakscombiner
                      , intern_x0caption, intern_sigmacaption, intern_Acaption
                      , accelcaptions                                         )
           );
  }
  
  measure example_parameterset(const measure &constraints, const physquantity &desiredret) const{
    measure pass_params; ps = &constraints;
    if(ps->has(*cptof_x()))
      pass_params.let("x") = x();
    
    for(unsigned i=0; i<npeaks; ++i) {
      if(ps->has(*cptof_x(i)))
        pass_params.let(   "x"    + LaTeX_subscript(i)) = x(i);
      if(ps->has(*cptof_sigma(i)))
        pass_params.let("\\sigma" + LaTeX_subscript(i)) = sigma(i);
      if(ps->has(*cptof_A(i)))
        pass_params.let(   "A"    + LaTeX_subscript(i)) = A(i);
    }
    
    measure example = exampleparams_dummy.example_parameterset(pass_params, desiredret)
          , pass_example;
    ps = &example;

    if(ps->has("x"))
      pass_example.let(*cptof_x()) = example["x"];
    
    for(unsigned i=0; i<npeaks; ++i) {
      if(ps->has(   "x"    + LaTeX_subscript(i)))
        pass_example.let(*cptof_x(i))     = example[   "x"    + LaTeX_subscript(i)];
      if(ps->has("\\sigma" + LaTeX_subscript(i))  )
        pass_example.let(*cptof_sigma(i)) = example["\\sigma" + LaTeX_subscript(i)];
      if(ps->has(   "A"    + LaTeX_subscript(i)))
        pass_example.let(*cptof_A(i))     = example[   "A"    + LaTeX_subscript(i)];
    }
    return pass_example;
  }
};



                                                               COPYABLE_PDERIVED_CLASS(/*
class*/fittable_exponentialfn,/*: public*/fittable_phmsqfn) {
  _3_FITVARIABLES(x, lambda, A)
 public:
  fittable_exponentialfn() { allocate_fitvarsbuf();
              cptof_x("x");             cptof_lambda("\\lambda");           cptof_A("A");  }      

  physquantity operator() (const measure &thisparams) const{
    ps = &thisparams;
    return A() * exp( lambda() * x() );
  }

  measure example_parameterset(const measure &constraints, const physquantity &desiredret) const{
    measure example; ps = &constraints;
    if (ps->has(*cptof_x())) {
      if (ps->has(*cptof_lambda())) {
        if (! x().conjugates(lambda())){
          cerr << "Exponential slope " << *cptof_lambda() << " incompatible with parameter " << *cptof_x()
               << "\n(" << *cptof_lambda() << " = " << lambda() << ", " << *cptof_x() << " = " << x() << ")\n";
          abort();
        }
       }else{
        example.push_back( x().inv() .plusminus (x().inv()) .label(*cptof_lambda()) );
      }
     }else{
      if (ps->has(*cptof_lambda())) {
        example.push_back( lambda().inv() .plusminus (lambda().inv())  .label(*cptof_x()) );
       }else{
        cerr << "Insufficient constraints to determine the dimension of parameter " << *cptof_x()
             << " or slope " << *cptof_lambda() << " of exponential function\n";
        abort();
      }
    }
    
    if (ps->has(*cptof_A())){
      if (! desiredret.compatible(A())) {
        cerr << "Amplitude " << *cptof_A() << " incompatible with desired return value"
             << "\n(" << desiredret << ", while " << *cptof_A() << " = " << A() << ")\n";
      }
     }else{
      example.push_back( desiredret .plusminus (desiredret)  .label(*cptof_A()) );
    }
    
    return example;
  }
 
};






namespace_cqtxnamespace_CLOSE