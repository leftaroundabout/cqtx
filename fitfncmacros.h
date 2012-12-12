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

/*
\begin{code}
import System.IO
import Data.List
hMacroPrn h mac = mapM_ (\l -> hPutStrLn h 
                          $ l ++ replicate(maxlen - length l)' ' ++ "\\"
                     ) macrolines >> hPutStrLn h ""
 where macrolines = lines mac
       maxlen = maximum $ map length macrolines
main :: IO()
main =
\end{code} */

namespace_cqtxnamespace_OPEN


#ifndef FITFNMACROS_FOR_PRODUCING_SIMPLY_PHYSQFUNCTIONS
  #define FITFNMACROS_FOR_PRODUCING_SIMPLY_PHYSQFUNCTIONS

  #define SET_FITVARIABLE_REFERENCEFOR(oref,callr)                                               \
    pvarref callr() const {return argdrfs[oref](*ps);}                                           \
    pvarref callr(unsigned i) const {return argdrfs[oref + multifitvariables_spacing * i](*ps);} \
    std::vector<physquantity> all_##callr##s()const {                                            \
      std::vector<physquantity> result(n_allocd_multivar_groups);                                \
      for(unsigned i = 0; i<n_allocd_multivar_groups; ++i)                                       \
        result[i] = callr(i);                                                                    \
      return result;                                                                             \
    }
  #define SET_FITVARIABLENAME_FOR(oref,callr) \
    qvarname cptof_##callr() const {return argdrfs[oref];} \
    void cptof_##callr(const std::string ncapt) {argdrfs[oref] = ncapt;} \
    qvarname cptof_##callr(unsigned i) const {return argdrfs[oref + multifitvariables_spacing * i];} \
    void cptof_##callr(unsigned i, const std::string ncapt) {argdrfs[oref + multifitvariables_spacing * i] = ncapt;}
  #define INTRODUCE_FITVARIABLE(oref,callr) SET_FITVARIABLE_REFERENCEFOR(oref,callr) SET_FITVARIABLENAME_FOR(oref,callr)
  #define CREATE_FITVARIABLESALLOCATOR(n)                                          \
    void allocate_fitvarsbuf() { argdrfs.resize(n); }                              \
    unsigned n_allocd_multivar_groups;                                             \
    void allocate_fitvarsbuf(unsigned m) {                                         \
      n_allocd_multivar_groups = m;                                                \
      argdrfs.resize(n - multifitvariables_spacing + multifitvariables_spacing*m); \
    }
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
  
  #define NORMAL_FITTINGFUNCTION_NMULTIS(caption, nmultivars, constrarg, multiinstt, declarations, allocations, definition) \
  COPYABLE_DERIVED_STRUCT(caption##function, fittable_phmsqfn) {                 \
   private:                                                                      \
    declarations                                                                 \
    NMULTIFITVARIABLES(nmultivars)                                                \
   public:                                                                       \
    caption##function(constrarg) { allocate_fitvarsbuf(multiinstt);                                 \
      allocations                                                                \
    }                                                                            \
    physquantity operator() (const measure &thisparams) const{                   \
      ps = &thisparams;                                                          \
      definition                                                                 \
    }                                                                            \
  }caption;

  #include "hs-gend/fitfncmacros-0.h"  /*
\begin{code}
  withFile "hs-gend/fitfncmacros-0.h" WriteMode (\h -> do
   mapM_ (hMacroPrn h) [
      "  #define FITTINGFUNCTION__"++show n++"_VARIABLE"++(if n>1 then "S" else "")++"(  \n"++
      "        fncapt,"++varsAssocList++",fndefinition)"                              ++"\n"++
      "  NORMAL_FITTINGFUNCTION( fncapt"                                              ++"\n"++
      "    , _"++show n++"_FITVARIABLES("++varsCnList++")"                            ++"\n"++
      "    , "++initAssocList                                                         ++"\n"++
      "    , fndefinition"                                                            ++"\n"++
      "  )"
    | n<-[1..9]
    , let varsCns = ["vc"++show i | i<-[1..n]]
          varsNns = ["vn"++show i | i<-[1..n]]
          varsAssocList = lshow $ zipWith (\c n->c++","++n) varsCns varsNns
          varsCnList    = lshow varsCns
          initAssocList = concat $ zipWith (\c n->"cptof_##"++c++"("++n++"); ") varsCns varsNns
          lshow = concat . intersperse ","
    ]
   mapM_ (hMacroPrn h) [
      "  #define FITTINGFUNCTION_"++show n++"VARIABLE"++(if n>1 then "S" else "")
                                                                        ++"_NMULTIS( \n\
      \        caption,nmultivars,"++varsAssocList++",definition)                    \n\
      \  COPYABLE_DERIVED_STRUCT(caption, fittable_phmsqfn) {                        \n\
      \   private:                                                                   \n\
      \    _"++show n++"_FITVARIABLES("++varsCnList++")                              \n\
      \    NMULTIFITVARIABLES(nmultivars)                                            \n\
      \   public:                                                                    \n\
      \    caption(unsigned nmultiinsts) { allocate_fitvarsbuf(nmultiinsts);         \n"
             ++initAssocList                                                       ++ "\
      \    }                                                                         \n\
      \    physquantity operator() (const measure &thisparams) const{                \n\
      \      ps = &thisparams;                                                       \n\
      \      definition                                                              \n\
      \    }                                                                         \n\
      \  };"
    | n<-[1..9]
    , let varsCns = ["vc"++show i | i<-[1..n]]
          varsNns = ["vn"++show i | i<-[1..n]]
          varsAssocList = lshow $ zipWith (\c n->c++","++n) varsCns varsNns
          varsCnList    = lshow varsCns
          initAssocList = unlines
           [ "      if(nmultivars>"++show(n-i)++") {                        \n\
             \        for(unsigned "++j++"=0;"++j++"<nmultiinsts; ++"++j++") \n\
             \          cptof_##vc"++show i++"("++j++", vn"++show i
                             ++" + LaTeX_subscript("++j++"));                \n\
             \       }else{                                                  \n\
             \        cptof_##vc"++show i++"(vn"++show i++");                \n\
             \      }"
           | i<-[1..n] ]
          lshow = concat . intersperse ","
          j = "junique_index"
    ])
\end{code} */
  



#endif



namespace_cqtxnamespace_CLOSE