  #define FITTINGFUNCTION__1_VARIABLE(  \
        fncapt,vc1,vn1,fndefinition)    \
  NORMAL_FITTINGFUNCTION( fncapt        \
    , _1_FITVARIABLES(vc1)              \
    , cptof_##vc1(vn1);                 \
    , fndefinition                      \
  )                                     \

  #define FITTINGFUNCTION__2_VARIABLES(     \
        fncapt,vc1,vn1,vc2,vn2,fndefinition)\
  NORMAL_FITTINGFUNCTION( fncapt            \
    , _2_FITVARIABLES(vc1,vc2)              \
    , cptof_##vc1(vn1); cptof_##vc2(vn2);   \
    , fndefinition                          \
  )                                         \

  #define FITTINGFUNCTION__3_VARIABLES(                     \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,fndefinition)        \
  NORMAL_FITTINGFUNCTION( fncapt                            \
    , _3_FITVARIABLES(vc1,vc2,vc3)                          \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3); \
    , fndefinition                                          \
  )                                                         \

  #define FITTINGFUNCTION__4_VARIABLES(                                       \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,fndefinition)                  \
  NORMAL_FITTINGFUNCTION( fncapt                                              \
    , _4_FITVARIABLES(vc1,vc2,vc3,vc4)                                        \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3); cptof_##vc4(vn4); \
    , fndefinition                                                            \
  )                                                                           \

  #define FITTINGFUNCTION__5_VARIABLES(                                                         \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,fndefinition)                            \
  NORMAL_FITTINGFUNCTION( fncapt                                                                \
    , _5_FITVARIABLES(vc1,vc2,vc3,vc4,vc5)                                                      \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3); cptof_##vc4(vn4); cptof_##vc5(vn5); \
    , fndefinition                                                                              \
  )                                                                                             \

  #define FITTINGFUNCTION__6_VARIABLES(                                                                           \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,fndefinition)                                      \
  NORMAL_FITTINGFUNCTION( fncapt                                                                                  \
    , _6_FITVARIABLES(vc1,vc2,vc3,vc4,vc5,vc6)                                                                    \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3); cptof_##vc4(vn4); cptof_##vc5(vn5); cptof_##vc6(vn6); \
    , fndefinition                                                                                                \
  )                                                                                                               \

  #define FITTINGFUNCTION__7_VARIABLES(                                                                                             \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,vc7,vn7,fndefinition)                                                \
  NORMAL_FITTINGFUNCTION( fncapt                                                                                                    \
    , _7_FITVARIABLES(vc1,vc2,vc3,vc4,vc5,vc6,vc7)                                                                                  \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3); cptof_##vc4(vn4); cptof_##vc5(vn5); cptof_##vc6(vn6); cptof_##vc7(vn7); \
    , fndefinition                                                                                                                  \
  )                                                                                                                                 \

  #define FITTINGFUNCTION__8_VARIABLES(                                                                                                               \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,vc7,vn7,vc8,vn8,fndefinition)                                                          \
  NORMAL_FITTINGFUNCTION( fncapt                                                                                                                      \
    , _8_FITVARIABLES(vc1,vc2,vc3,vc4,vc5,vc6,vc7,vc8)                                                                                                \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3); cptof_##vc4(vn4); cptof_##vc5(vn5); cptof_##vc6(vn6); cptof_##vc7(vn7); cptof_##vc8(vn8); \
    , fndefinition                                                                                                                                    \
  )                                                                                                                                                   \

  #define FITTINGFUNCTION__9_VARIABLES(                                                                                                                                 \
        fncapt,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,vc7,vn7,vc8,vn8,vc9,vn9,fndefinition)                                                                    \
  NORMAL_FITTINGFUNCTION( fncapt                                                                                                                                        \
    , _9_FITVARIABLES(vc1,vc2,vc3,vc4,vc5,vc6,vc7,vc8,vc9)                                                                                                              \
    , cptof_##vc1(vn1); cptof_##vc2(vn2); cptof_##vc3(vn3); cptof_##vc4(vn4); cptof_##vc5(vn5); cptof_##vc6(vn6); cptof_##vc7(vn7); cptof_##vc8(vn8); cptof_##vc9(vn9); \
    , fndefinition                                                                                                                                                      \
  )                                                                                                                                                                     \

  #define FITTINGFUNCTION_1VARIABLE_NMULTIS(                                               \
        caption,nmultivars,vc1,vn1,definition)                                             \
  COPYABLE_DERIVED_STRUCT(caption, fittable_phmsqfn) {                                     \
   private:                                                                                \
    _1_FITVARIABLES(vc1)                                                                   \
    NMULTIFITVARIABLES(nmultivars)                                                         \
   public:                                                                                 \
    caption(unsigned nmultiinsts) { allocate_fitvarsbuf(nmultiinsts);                      \
      if(nmultivars<0) {                                                                   \
        cptof_##vc1(vn1);                                                                  \
       }else{                                                                              \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)           \
          cptof_##vc1(junique_index, vn1 + LaTeX_subscript(junique_index));                \
      }                                                                                    \
    }                                                                                      \
    physquantity operator() (const measure &thisparams) const{                             \
      ps = &thisparams;                                                                    \
      definition                                                                           \
    }                                                                                      \
  };                                                                                       \

  #define FITTINGFUNCTION_2VARIABLES_NMULTIS(                                              \
        caption,nmultivars,vc1,vn1,vc2,vn2,definition)                                     \
  COPYABLE_DERIVED_STRUCT(caption, fittable_phmsqfn) {                                     \
   private:                                                                                \
    _2_FITVARIABLES(vc1,vc2)                                                               \
    NMULTIFITVARIABLES(nmultivars)                                                         \
   public:                                                                                 \
    caption(unsigned nmultiinsts) { allocate_fitvarsbuf(nmultiinsts);                      \
      if(nmultivars<1) {                                                                   \
        cptof_##vc1(vn1);                                                                  \
       }else{                                                                              \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)           \
          cptof_##vc1(junique_index, vn1 + LaTeX_subscript(junique_index));                \
      }                                                                                    \
      if(nmultivars<0) {                                                                   \
        cptof_##vc2(vn2);                                                                  \
       }else{                                                                              \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)           \
          cptof_##vc2(junique_index, vn2 + LaTeX_subscript(junique_index));                \
      }                                                                                    \
    }                                                                                      \
    physquantity operator() (const measure &thisparams) const{                             \
      ps = &thisparams;                                                                    \
      definition                                                                           \
    }                                                                                      \
  };                                                                                       \

  #define FITTINGFUNCTION_3VARIABLES_NMULTIS(                                              \
        caption,nmultivars,vc1,vn1,vc2,vn2,vc3,vn3,definition)                             \
  COPYABLE_DERIVED_STRUCT(caption, fittable_phmsqfn) {                                     \
   private:                                                                                \
    _3_FITVARIABLES(vc1,vc2,vc3)                                                           \
    NMULTIFITVARIABLES(nmultivars)                                                         \
   public:                                                                                 \
    caption(unsigned nmultiinsts) { allocate_fitvarsbuf(nmultiinsts);                      \
      if(nmultivars<2) {                                                                   \
        cptof_##vc1(vn1);                                                                  \
       }else{                                                                              \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)           \
          cptof_##vc1(junique_index, vn1 + LaTeX_subscript(junique_index));                \
      }                                                                                    \
      if(nmultivars<1) {                                                                   \
        cptof_##vc2(vn2);                                                                  \
       }else{                                                                              \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)           \
          cptof_##vc2(junique_index, vn2 + LaTeX_subscript(junique_index));                \
      }                                                                                    \
      if(nmultivars<0) {                                                                   \
        cptof_##vc3(vn3);                                                                  \
       }else{                                                                              \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)           \
          cptof_##vc3(junique_index, vn3 + LaTeX_subscript(junique_index));                \
      }                                                                                    \
    }                                                                                      \
    physquantity operator() (const measure &thisparams) const{                             \
      ps = &thisparams;                                                                    \
      definition                                                                           \
    }                                                                                      \
  };                                                                                       \

  #define FITTINGFUNCTION_4VARIABLES_NMULTIS(                                              \
        caption,nmultivars,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,definition)                     \
  COPYABLE_DERIVED_STRUCT(caption, fittable_phmsqfn) {                                     \
   private:                                                                                \
    _4_FITVARIABLES(vc1,vc2,vc3,vc4)                                                       \
    NMULTIFITVARIABLES(nmultivars)                                                         \
   public:                                                                                 \
    caption(unsigned nmultiinsts) { allocate_fitvarsbuf(nmultiinsts);                      \
      if(nmultivars<3) {                                                                   \
        cptof_##vc1(vn1);                                                                  \
       }else{                                                                              \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)           \
          cptof_##vc1(junique_index, vn1 + LaTeX_subscript(junique_index));                \
      }                                                                                    \
      if(nmultivars<2) {                                                                   \
        cptof_##vc2(vn2);                                                                  \
       }else{                                                                              \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)           \
          cptof_##vc2(junique_index, vn2 + LaTeX_subscript(junique_index));                \
      }                                                                                    \
      if(nmultivars<1) {                                                                   \
        cptof_##vc3(vn3);                                                                  \
       }else{                                                                              \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)           \
          cptof_##vc3(junique_index, vn3 + LaTeX_subscript(junique_index));                \
      }                                                                                    \
      if(nmultivars<0) {                                                                   \
        cptof_##vc4(vn4);                                                                  \
       }else{                                                                              \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)           \
          cptof_##vc4(junique_index, vn4 + LaTeX_subscript(junique_index));                \
      }                                                                                    \
    }                                                                                      \
    physquantity operator() (const measure &thisparams) const{                             \
      ps = &thisparams;                                                                    \
      definition                                                                           \
    }                                                                                      \
  };                                                                                       \

  #define FITTINGFUNCTION_5VARIABLES_NMULTIS(                                                     \
        caption,nmultivars,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,definition)                    \
  COPYABLE_DERIVED_STRUCT(caption, fittable_phmsqfn) {                                            \
   private:                                                                                       \
    _5_FITVARIABLES(vc1,vc2,vc3,vc4,vc5)                                                          \
    NMULTIFITVARIABLES(nmultivars)                                                                \
   public:                                                                                        \
    caption(unsigned nmultiinsts) { allocate_fitvarsbuf(nmultiinsts);                             \
      if(nmultivars<4) {                                                                          \
        cptof_##vc1(vn1);                                                                         \
       }else{                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                  \
          cptof_##vc1(junique_index, vn1 + LaTeX_subscript(junique_index));                       \
      }                                                                                           \
      if(nmultivars<3) {                                                                          \
        cptof_##vc2(vn2);                                                                         \
       }else{                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                  \
          cptof_##vc2(junique_index, vn2 + LaTeX_subscript(junique_index));                       \
      }                                                                                           \
      if(nmultivars<2) {                                                                          \
        cptof_##vc3(vn3);                                                                         \
       }else{                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                  \
          cptof_##vc3(junique_index, vn3 + LaTeX_subscript(junique_index));                       \
      }                                                                                           \
      if(nmultivars<1) {                                                                          \
        cptof_##vc4(vn4);                                                                         \
       }else{                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                  \
          cptof_##vc4(junique_index, vn4 + LaTeX_subscript(junique_index));                       \
      }                                                                                           \
      if(nmultivars<0) {                                                                          \
        cptof_##vc5(vn5);                                                                         \
       }else{                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                  \
          cptof_##vc5(junique_index, vn5 + LaTeX_subscript(junique_index));                       \
      }                                                                                           \
    }                                                                                             \
    physquantity operator() (const measure &thisparams) const{                                    \
      ps = &thisparams;                                                                           \
      definition                                                                                  \
    }                                                                                             \
  };                                                                                              \

  #define FITTINGFUNCTION_6VARIABLES_NMULTIS(                                                             \
        caption,nmultivars,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,definition)                    \
  COPYABLE_DERIVED_STRUCT(caption, fittable_phmsqfn) {                                                    \
   private:                                                                                               \
    _6_FITVARIABLES(vc1,vc2,vc3,vc4,vc5,vc6)                                                              \
    NMULTIFITVARIABLES(nmultivars)                                                                        \
   public:                                                                                                \
    caption(unsigned nmultiinsts) { allocate_fitvarsbuf(nmultiinsts);                                     \
      if(nmultivars<5) {                                                                                  \
        cptof_##vc1(vn1);                                                                                 \
       }else{                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                          \
          cptof_##vc1(junique_index, vn1 + LaTeX_subscript(junique_index));                               \
      }                                                                                                   \
      if(nmultivars<4) {                                                                                  \
        cptof_##vc2(vn2);                                                                                 \
       }else{                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                          \
          cptof_##vc2(junique_index, vn2 + LaTeX_subscript(junique_index));                               \
      }                                                                                                   \
      if(nmultivars<3) {                                                                                  \
        cptof_##vc3(vn3);                                                                                 \
       }else{                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                          \
          cptof_##vc3(junique_index, vn3 + LaTeX_subscript(junique_index));                               \
      }                                                                                                   \
      if(nmultivars<2) {                                                                                  \
        cptof_##vc4(vn4);                                                                                 \
       }else{                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                          \
          cptof_##vc4(junique_index, vn4 + LaTeX_subscript(junique_index));                               \
      }                                                                                                   \
      if(nmultivars<1) {                                                                                  \
        cptof_##vc5(vn5);                                                                                 \
       }else{                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                          \
          cptof_##vc5(junique_index, vn5 + LaTeX_subscript(junique_index));                               \
      }                                                                                                   \
      if(nmultivars<0) {                                                                                  \
        cptof_##vc6(vn6);                                                                                 \
       }else{                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                          \
          cptof_##vc6(junique_index, vn6 + LaTeX_subscript(junique_index));                               \
      }                                                                                                   \
    }                                                                                                     \
    physquantity operator() (const measure &thisparams) const{                                            \
      ps = &thisparams;                                                                                   \
      definition                                                                                          \
    }                                                                                                     \
  };                                                                                                      \

  #define FITTINGFUNCTION_7VARIABLES_NMULTIS(                                                                     \
        caption,nmultivars,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,vc7,vn7,definition)                    \
  COPYABLE_DERIVED_STRUCT(caption, fittable_phmsqfn) {                                                            \
   private:                                                                                                       \
    _7_FITVARIABLES(vc1,vc2,vc3,vc4,vc5,vc6,vc7)                                                                  \
    NMULTIFITVARIABLES(nmultivars)                                                                                \
   public:                                                                                                        \
    caption(unsigned nmultiinsts) { allocate_fitvarsbuf(nmultiinsts);                                             \
      if(nmultivars<6) {                                                                                          \
        cptof_##vc1(vn1);                                                                                         \
       }else{                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                  \
          cptof_##vc1(junique_index, vn1 + LaTeX_subscript(junique_index));                                       \
      }                                                                                                           \
      if(nmultivars<5) {                                                                                          \
        cptof_##vc2(vn2);                                                                                         \
       }else{                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                  \
          cptof_##vc2(junique_index, vn2 + LaTeX_subscript(junique_index));                                       \
      }                                                                                                           \
      if(nmultivars<4) {                                                                                          \
        cptof_##vc3(vn3);                                                                                         \
       }else{                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                  \
          cptof_##vc3(junique_index, vn3 + LaTeX_subscript(junique_index));                                       \
      }                                                                                                           \
      if(nmultivars<3) {                                                                                          \
        cptof_##vc4(vn4);                                                                                         \
       }else{                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                  \
          cptof_##vc4(junique_index, vn4 + LaTeX_subscript(junique_index));                                       \
      }                                                                                                           \
      if(nmultivars<2) {                                                                                          \
        cptof_##vc5(vn5);                                                                                         \
       }else{                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                  \
          cptof_##vc5(junique_index, vn5 + LaTeX_subscript(junique_index));                                       \
      }                                                                                                           \
      if(nmultivars<1) {                                                                                          \
        cptof_##vc6(vn6);                                                                                         \
       }else{                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                  \
          cptof_##vc6(junique_index, vn6 + LaTeX_subscript(junique_index));                                       \
      }                                                                                                           \
      if(nmultivars<0) {                                                                                          \
        cptof_##vc7(vn7);                                                                                         \
       }else{                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                  \
          cptof_##vc7(junique_index, vn7 + LaTeX_subscript(junique_index));                                       \
      }                                                                                                           \
    }                                                                                                             \
    physquantity operator() (const measure &thisparams) const{                                                    \
      ps = &thisparams;                                                                                           \
      definition                                                                                                  \
    }                                                                                                             \
  };                                                                                                              \

  #define FITTINGFUNCTION_8VARIABLES_NMULTIS(                                                                             \
        caption,nmultivars,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,vc7,vn7,vc8,vn8,definition)                    \
  COPYABLE_DERIVED_STRUCT(caption, fittable_phmsqfn) {                                                                    \
   private:                                                                                                               \
    _8_FITVARIABLES(vc1,vc2,vc3,vc4,vc5,vc6,vc7,vc8)                                                                      \
    NMULTIFITVARIABLES(nmultivars)                                                                                        \
   public:                                                                                                                \
    caption(unsigned nmultiinsts) { allocate_fitvarsbuf(nmultiinsts);                                                     \
      if(nmultivars<7) {                                                                                                  \
        cptof_##vc1(vn1);                                                                                                 \
       }else{                                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                          \
          cptof_##vc1(junique_index, vn1 + LaTeX_subscript(junique_index));                                               \
      }                                                                                                                   \
      if(nmultivars<6) {                                                                                                  \
        cptof_##vc2(vn2);                                                                                                 \
       }else{                                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                          \
          cptof_##vc2(junique_index, vn2 + LaTeX_subscript(junique_index));                                               \
      }                                                                                                                   \
      if(nmultivars<5) {                                                                                                  \
        cptof_##vc3(vn3);                                                                                                 \
       }else{                                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                          \
          cptof_##vc3(junique_index, vn3 + LaTeX_subscript(junique_index));                                               \
      }                                                                                                                   \
      if(nmultivars<4) {                                                                                                  \
        cptof_##vc4(vn4);                                                                                                 \
       }else{                                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                          \
          cptof_##vc4(junique_index, vn4 + LaTeX_subscript(junique_index));                                               \
      }                                                                                                                   \
      if(nmultivars<3) {                                                                                                  \
        cptof_##vc5(vn5);                                                                                                 \
       }else{                                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                          \
          cptof_##vc5(junique_index, vn5 + LaTeX_subscript(junique_index));                                               \
      }                                                                                                                   \
      if(nmultivars<2) {                                                                                                  \
        cptof_##vc6(vn6);                                                                                                 \
       }else{                                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                          \
          cptof_##vc6(junique_index, vn6 + LaTeX_subscript(junique_index));                                               \
      }                                                                                                                   \
      if(nmultivars<1) {                                                                                                  \
        cptof_##vc7(vn7);                                                                                                 \
       }else{                                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                          \
          cptof_##vc7(junique_index, vn7 + LaTeX_subscript(junique_index));                                               \
      }                                                                                                                   \
      if(nmultivars<0) {                                                                                                  \
        cptof_##vc8(vn8);                                                                                                 \
       }else{                                                                                                             \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                          \
          cptof_##vc8(junique_index, vn8 + LaTeX_subscript(junique_index));                                               \
      }                                                                                                                   \
    }                                                                                                                     \
    physquantity operator() (const measure &thisparams) const{                                                            \
      ps = &thisparams;                                                                                                   \
      definition                                                                                                          \
    }                                                                                                                     \
  };                                                                                                                      \

  #define FITTINGFUNCTION_9VARIABLES_NMULTIS(                                                                                     \
        caption,nmultivars,vc1,vn1,vc2,vn2,vc3,vn3,vc4,vn4,vc5,vn5,vc6,vn6,vc7,vn7,vc8,vn8,vc9,vn9,definition)                    \
  COPYABLE_DERIVED_STRUCT(caption, fittable_phmsqfn) {                                                                            \
   private:                                                                                                                       \
    _9_FITVARIABLES(vc1,vc2,vc3,vc4,vc5,vc6,vc7,vc8,vc9)                                                                          \
    NMULTIFITVARIABLES(nmultivars)                                                                                                \
   public:                                                                                                                        \
    caption(unsigned nmultiinsts) { allocate_fitvarsbuf(nmultiinsts);                                                             \
      if(nmultivars<8) {                                                                                                          \
        cptof_##vc1(vn1);                                                                                                         \
       }else{                                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                                  \
          cptof_##vc1(junique_index, vn1 + LaTeX_subscript(junique_index));                                                       \
      }                                                                                                                           \
      if(nmultivars<7) {                                                                                                          \
        cptof_##vc2(vn2);                                                                                                         \
       }else{                                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                                  \
          cptof_##vc2(junique_index, vn2 + LaTeX_subscript(junique_index));                                                       \
      }                                                                                                                           \
      if(nmultivars<6) {                                                                                                          \
        cptof_##vc3(vn3);                                                                                                         \
       }else{                                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                                  \
          cptof_##vc3(junique_index, vn3 + LaTeX_subscript(junique_index));                                                       \
      }                                                                                                                           \
      if(nmultivars<5) {                                                                                                          \
        cptof_##vc4(vn4);                                                                                                         \
       }else{                                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                                  \
          cptof_##vc4(junique_index, vn4 + LaTeX_subscript(junique_index));                                                       \
      }                                                                                                                           \
      if(nmultivars<4) {                                                                                                          \
        cptof_##vc5(vn5);                                                                                                         \
       }else{                                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                                  \
          cptof_##vc5(junique_index, vn5 + LaTeX_subscript(junique_index));                                                       \
      }                                                                                                                           \
      if(nmultivars<3) {                                                                                                          \
        cptof_##vc6(vn6);                                                                                                         \
       }else{                                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                                  \
          cptof_##vc6(junique_index, vn6 + LaTeX_subscript(junique_index));                                                       \
      }                                                                                                                           \
      if(nmultivars<2) {                                                                                                          \
        cptof_##vc7(vn7);                                                                                                         \
       }else{                                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                                  \
          cptof_##vc7(junique_index, vn7 + LaTeX_subscript(junique_index));                                                       \
      }                                                                                                                           \
      if(nmultivars<1) {                                                                                                          \
        cptof_##vc8(vn8);                                                                                                         \
       }else{                                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                                  \
          cptof_##vc8(junique_index, vn8 + LaTeX_subscript(junique_index));                                                       \
      }                                                                                                                           \
      if(nmultivars<0) {                                                                                                          \
        cptof_##vc9(vn9);                                                                                                         \
       }else{                                                                                                                     \
        for(unsigned junique_index=0;junique_index<nmultiinsts; ++junique_index)                                                  \
          cptof_##vc9(junique_index, vn9 + LaTeX_subscript(junique_index));                                                       \
      }                                                                                                                           \
    }                                                                                                                             \
    physquantity operator() (const measure &thisparams) const{                                                                    \
      ps = &thisparams;                                                                                                           \
      definition                                                                                                                  \
    }                                                                                                                             \
  };                                                                                                                              \

