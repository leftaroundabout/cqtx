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


#define SET_FITVARIABLE_REFERENCEFOR(oref,callr) \
  pvarref callr() const {return argdrfs[oref](*ps);} \
  pvarref callr(unsigned i) const {return argdrfs[oref + multifitvariables_spacing * i](*ps);}
#define SET_FITVARIABLENAME_FOR(oref,callr) \
  qvarname cptof_##callr() const {return argdrfs[oref];} \
  void cptof_##callr(const std::string ncapt) {argdrfs[oref] = ncapt;} \
  qvarname cptof_##callr(unsigned i) const {return argdrfs[oref + multifitvariables_spacing * i];} \
  void cptof_##callr(unsigned i, const std::string ncapt) {argdrfs[oref + multifitvariables_spacing * i] = ncapt;}
#define INTRODUCE_FITVARIABLE(oref,callr) SET_FITVARIABLE_REFERENCEFOR(oref,callr) SET_FITVARIABLENAME_FOR(oref,callr)
#define CREATE_FITVARIABLESALLOCATOR(n) \
  void allocate_fitvarsbuf() { argdrfs.resize(n); } \
  void allocate_fitvarsbuf(unsigned m) { argdrfs.resize(n - multifitvariables_spacing + multifitvariables_spacing*m); }
#define _1_FITVARIABLE(varn0) CREATE_FITVARIABLESALLOCATOR(1) \
  INTRODUCE_FITVARIABLE(0,varn0)
#define _2_FITVARIABLES(varn0, varn1) CREATE_FITVARIABLESALLOCATOR(2) \
  INTRODUCE_FITVARIABLE(0,varn0) INTRODUCE_FITVARIABLE(1,varn1)
#define _3_FITVARIABLES(varn0, varn1, varn2) CREATE_FITVARIABLESALLOCATOR(3) \
  INTRODUCE_FITVARIABLE(0,varn0) INTRODUCE_FITVARIABLE(1,varn1) INTRODUCE_FITVARIABLE(2,varn2)
#define _4_FITVARIABLES(varn0, varn1, varn2, varn3) CREATE_FITVARIABLESALLOCATOR(4) \
  INTRODUCE_FITVARIABLE(0,varn0) INTRODUCE_FITVARIABLE(1,varn1) \
  INTRODUCE_FITVARIABLE(2,varn2) INTRODUCE_FITVARIABLE(3,varn3)
const unsigned multifitvariables_spacing = 0;
#define NMULTIFITVARIABLES(n) static const unsigned multifitvariables_spacing = n;

#define NORMAL_FITTINGFUNCTION(caption, declarations, allocations, definition) \
COPYABLE_PDERIVED_CLASS(caption##function, fittable_phmsqfn) {                 \
  declarations                                                                 \
 public:                                                                       \
  caption##function() { allocate_fitvarsbuf();                                 \
    allocations                                                                \
  }                                                                            \
  physquantity operator() (const measure &thisparams) const{                   \
    ps = &thisparams;                                                          \
    definition                                                                 \
  }                                                                            \
}caption;

#define FITTINGFUNCTION__3_VARIABLES(                         \
      fncapt,vc1,vn1,vc2,vn2,vc3,vn3,fndefinition)            \
NORMAL_FITTINGFUNCTION( fncapt                                \
  , _3_FITVARIABLES(vc1, vc2, vc3)                            \
  , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3);     \
  , fndefinition                                              \
)
#define FITTINGFUNCTION__4_VARIABLES(                         \
      fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,fndefinition)    \
NORMAL_FITTINGFUNCTION( fncapt                                \
  , _4_FITVARIABLES(vc1, vc2, vc3, vc4)                       \
  , cptof_##vc1(vn1); cptof_##vc2(vn2);                       \
    cptof_##vc3(vn3); cptof_##vc4(vn4);                       \
  , fndefinition                                              \
)



// old form
#if 0
#define FITTINGFUNCTION__4_VARIABLES(fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,fndefinition) \
COPYABLE_PDERIVED_CLASS(fncapt, fittable_phmsqfn) {                                       \
  _4_FITVARIABLES(vc1, vc2, vc3, vc4)                                                     \
 public:                                                                                  \
  fncapt() { allocate_fitvarsbuf();                                                       \
    cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3); cptof_##vc4(vn4);               \
  }                                                                                       \
  physquantity operator() (const measure &thisparams) const{                              \
    ps = &thisparams;                                                                     \
    fndefinition                                                                          \
  }                                                                                       \
};
#endif





   // attempt to define a general remapping, i.e. transforming a function
  //  f(x,y,z) to something like g(w,q) := f(w, w-q, q). It turns out to be defficult
 //   to do this as a macro, probably not feasible.
#if 0
#define NORMAL_MULTIREMAPPED_FITTINGFUNCTION(caption, origfunction, declarations, allocations, definition) \
COPYABLE_PDERIVED_CLASS(caption##function, fittable_phmsqfn) {                 \
  declarations                                                                 \
  origfunction original_fn;                                                    \
  
  COPYABLE_DERIVED_STRUCT(squaredist_acceled, phmsq_function) {
    declarations
    origfunction original_fn;
    physquantity operator() (const measure& thisparams) const{
      measure mdfparams = thisparams;
      ps = &mdfparams;
      definition
      return original_fn(mdfparams);
    }
  };
 public:                                                                       \
  auto                                                                         \
  squaredist_accel( const measureseq& fixedparams                              \
                  , const msq_dereferencer& retcomp_drf) const                 \
           -> p_maybe<phmsq_function> {                                        \
    for(auto& acc: original_fn.squaredist_accel(fixedparams, retcomp_drf)) {
      return squaredist_acceled
    }
    measureseq procd_params(fixedparams.size());                               \
    for(int i = 0; i < fixedparams.size(); ++i) {
      ps = &fixedparams[i]; rs = &procd_params[i];
      try {
        definition
       }catch(...) {
        return nothing;
      }
    }
  }
  caption##function() { allocate_fitvarsbuf();                                 \
    allocations                                                                \
  }                                                                            \
  physquantity operator() (const measure &thisparams) const{                   \
    ps = &thisparams;                                                          \
    definition                                                                 \
  }                                                                            \
}caption;
#endif