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


namespace_cqtxnamespace_OPEN


#ifndef FITFNMACROS_FOR_PRODUCING_SIMPLY_PHYSQFUNCTIONS
  #define FITFNMACROS_FOR_PRODUCING_SIMPLY_PHYSQFUNCTIONS

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
  COPYABLE_DERIVED_STRUCT(caption##function, fittable_phmsqfn) {                 \
   private:                                                                      \
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

  #define FITTINGFUNCTION__1_VARIABLE(                    \
        fncapt,vc,vn,fndefinition)                        \
  NORMAL_FITTINGFUNCTION( fncapt                          \
    , _1_FITVARIABLE(vc)                                  \
    , cptof_##vc(vn);                                     \
    , fndefinition                                        \
  )
  #define FITTINGFUNCTION__2_VARIABLES(                     \
        fncapt,vc1,vn1,vc2,vn2,fndefinition)                \
  NORMAL_FITTINGFUNCTION( fncapt                            \
    , _2_FITVARIABLES(vc1, vc2)                             \
    , cptof_##vc1(vn1); cptof_##vc2(vn2);                   \
    , fndefinition                                          \
  )
  #define FITTINGFUNCTION__3_VARIABLES(                       \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,fndefinition)          \
  NORMAL_FITTINGFUNCTION( fncapt                              \
    , _3_FITVARIABLES(vc1, vc2, vc3)                          \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3);   \
    , fndefinition                                            \
  )
  #define FITTINGFUNCTION__4_VARIABLES(                         \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,fndefinition)    \
  NORMAL_FITTINGFUNCTION( fncapt                                \
    , _4_FITVARIABLES(vc1, vc2, vc3, vc4)                       \
    , cptof_##vc1(vn1); cptof_##vc2(vn2);                       \
      cptof_##vc3(vn3); cptof_##vc4(vn4);                       \
    , fndefinition                                              \
  )


#endif



namespace_cqtxnamespace_CLOSE