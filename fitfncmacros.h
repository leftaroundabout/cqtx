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
  #define _5_FITVARIABLES(varn0, varn1, varn2, varn3, varn4) CREATE_FITVARIABLESALLOCATOR(5) \
    INTRODUCE_FITVARIABLE(0,varn0) INTRODUCE_FITVARIABLE(1,varn1) \
    INTRODUCE_FITVARIABLE(2,varn2) INTRODUCE_FITVARIABLE(3,varn3) \
    INTRODUCE_FITVARIABLE(4,varn4)
  #define _6_FITVARIABLES(varn0, varn1, varn2, varn3, varn4, varn5) CREATE_FITVARIABLESALLOCATOR(6) \
    INTRODUCE_FITVARIABLE(0,varn0) INTRODUCE_FITVARIABLE(1,varn1) \
    INTRODUCE_FITVARIABLE(2,varn2) INTRODUCE_FITVARIABLE(3,varn3) \
    INTRODUCE_FITVARIABLE(4,varn4) INTRODUCE_FITVARIABLE(5,varn5)
  #define _7_FITVARIABLES(varn0, varn1, varn2, varn3, varn4, varn5, varn6) CREATE_FITVARIABLESALLOCATOR(7) \
    INTRODUCE_FITVARIABLE(0,varn0) INTRODUCE_FITVARIABLE(1,varn1) \
    INTRODUCE_FITVARIABLE(2,varn2) INTRODUCE_FITVARIABLE(3,varn3) \
    INTRODUCE_FITVARIABLE(4,varn4) INTRODUCE_FITVARIABLE(5,varn5) \
    INTRODUCE_FITVARIABLE(6,varn6)
  #define _8_FITVARIABLES(varn0, varn1, varn2, varn3, varn4, varn5, varn6, varn7) CREATE_FITVARIABLESALLOCATOR(8) \
    INTRODUCE_FITVARIABLE(0,varn0) INTRODUCE_FITVARIABLE(1,varn1) \
    INTRODUCE_FITVARIABLE(2,varn2) INTRODUCE_FITVARIABLE(3,varn3) \
    INTRODUCE_FITVARIABLE(4,varn4) INTRODUCE_FITVARIABLE(5,varn5) \
    INTRODUCE_FITVARIABLE(6,varn6) INTRODUCE_FITVARIABLE(7,varn7)
  #define _9_FITVARIABLES(varn0, varn1, varn2, varn3, varn4, varn5, varn6, varn7, varn8) CREATE_FITVARIABLESALLOCATOR(9) \
    INTRODUCE_FITVARIABLE(0,varn0) INTRODUCE_FITVARIABLE(1,varn1) \
    INTRODUCE_FITVARIABLE(2,varn2) INTRODUCE_FITVARIABLE(3,varn3) \
    INTRODUCE_FITVARIABLE(4,varn4) INTRODUCE_FITVARIABLE(5,varn5) \
    INTRODUCE_FITVARIABLE(6,varn6) INTRODUCE_FITVARIABLE(7,varn7) \
    INTRODUCE_FITVARIABLE(8,varn8)
  #define _10_FITVARIABLES(varn0, varn1, varn2, varn3, varn4, varn5, varn6, varn7, varn8, varn9) CREATE_FITVARIABLESALLOCATOR(10) \
    INTRODUCE_FITVARIABLE(0,varn0) INTRODUCE_FITVARIABLE(1,varn1) \
    INTRODUCE_FITVARIABLE(2,varn2) INTRODUCE_FITVARIABLE(3,varn3) \
    INTRODUCE_FITVARIABLE(4,varn4) INTRODUCE_FITVARIABLE(5,varn5) \
    INTRODUCE_FITVARIABLE(6,varn6) INTRODUCE_FITVARIABLE(7,varn7) \
    INTRODUCE_FITVARIABLE(8,varn8) INTRODUCE_FITVARIABLE(9,varn9)
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
  #define FITTINGFUNCTION__5_VARIABLES(                                 \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,fndefinition)    \
  NORMAL_FITTINGFUNCTION( fncapt                                        \
    , _5_FITVARIABLES(vc1, vc2, vc3, vc4, vc5)                          \
    , cptof_##vc1(vn1); cptof_##vc2(vn2);                               \
      cptof_##vc3(vn3); cptof_##vc4(vn4); cptof_##vc5(vn5);             \
    , fndefinition                                                      \
  )
  #define FITTINGFUNCTION__6_VARIABLES(                                         \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,fndefinition)    \
  NORMAL_FITTINGFUNCTION( fncapt                                                \
    , _6_FITVARIABLES(vc1, vc2, vc3, vc4, vc5, vc6)                             \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3);                     \
      cptof_##vc4(vn4); cptof_##vc5(vn5); cptof_##vc6(vn6);                     \
    , fndefinition                                                              \
  )
  #define FITTINGFUNCTION__7_VARIABLES(                                                 \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,vc7,vn7,fndefinition)    \
  NORMAL_FITTINGFUNCTION( fncapt                                                        \
    , _7_FITVARIABLES(vc1, vc2, vc3, vc4, vc5, vc6, vc7)                                \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3);                             \
      cptof_##vc4(vn4); cptof_##vc5(vn5); cptof_##vc6(vn6); cptof_##vc7(vn7);           \
    , fndefinition                                                                      \
  )
  #define FITTINGFUNCTION__8_VARIABLES(                                                         \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,vc7,vn7,vc8,vn8,fndefinition)    \
  NORMAL_FITTINGFUNCTION( fncapt                                                                \
    , _7_FITVARIABLES(vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8)                                   \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3); cptof_##vc4(vn4);                   \
      cptof_##vc5(vn5); cptof_##vc6(vn6); cptof_##vc7(vn7); cptof_##vc8(vn8);                   \
    , fndefinition                                                                              \
  )
  #define FITTINGFUNCTION__9_VARIABLES(                                                                 \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,vc7,vn7,vc8,vn8,vc9,vn9,fndefinition)    \
  NORMAL_FITTINGFUNCTION( fncapt                                                                        \
    , _7_FITVARIABLES(vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9)                                      \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3); cptof_##vc4(vn4);                           \
      cptof_##vc5(vn5); cptof_##vc6(vn6); cptof_##vc7(vn7); cptof_##vc8(vn8); cptof_##vc9(vn9);         \
    , fndefinition                                                                                      \
  )


#endif



namespace_cqtxnamespace_CLOSE