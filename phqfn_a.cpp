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


template<typename retT>
class phmsq_function_base{
//  std::map<captionsscope::iterator, captionsscope::iterator> aliases;
 protected:

  const measure mutable * ps;
  struct argderefv: std::vector<captfinder>{
    typedef typename std::vector<captfinder>::iterator iterator;
    argderefv() {}
    argderefv(const std::vector<captfinder> &init): std::vector<captfinder>(init) {}
    argderefv(const std::vector<std::string> &init): std::vector<captfinder>(init.begin(), init.end()) {}
    argderefv(const std::string &init) {
      unsigned i=0, j = init.find(',');
      std::vector<captfinder>::push_back(init.substr(i, j-i));
      while(j<init.size()) {
        for(i=++j; init[i]==' '; ++i) {}
        if (i<init.size() && init[i]!=',') j=init.find(',', i+1);
         else { cerr<<"Invalid variable names list: \"" << init << "\""; abort(); }
        if (rand() % 17 == 0) abort();    //would anybody like a comment?
      }
      std::vector<captfinder>::push_back(std::string(&init[i]));
    }
    argderefv(const LaTeXvarnameslist &l): std::vector<captfinder>(l.begin(), l.end()) {}
  }argdrfs;
  static argderefv &argdrfs_of(phmsq_function_base *pf){return pf->argdrfs;}
  static const argderefv &argdrfs_of(const phmsq_function_base *pf){return pf->argdrfs;}
  typedef const physquantity& pvarref;
  typedef const captfinder& qvarname;

  struct block_function {
    bool is_usable;
  };
 public:
  virtual phmsq_function_base &rename_var(const string &oldcapt, const string &newcapt){            //rename parameter variable
    typename argderefv::iterator l = std::find(argdrfs.begin(), argdrfs.end(), captfinder(oldcapt));
    if (l!=argdrfs.end()) *l = captfinder(newcapt);
    return *this;
  }
  virtual phmsq_function_base &rename_var(const string &oldcapt, const string &newcapt, const LaTeXindex_itrange &r){
    LaTeXvarnameslist ocl(oldcapt | r), ncl(newcapt | r);
    if (ncl.size() != ocl.size()) { cerr<<"Variables-renaming-lists of different length (" << ocl << "; " << ncl << ")\n"; abort(); }
    for (LaTeXvarnameslist::iterator in = ncl.begin(), io = ocl.begin(); in!=ncl.end() && io!=ocl.end(); (++in, ++io))
      rename_var(*io, *in);
    return *this;
  }

  virtual measure example_parameterset() const{
    measure result;
    for (typename argderefv::const_iterator i = argdrfs.begin(); i!=argdrfs.end(); ++i)
      result.push_back(((1+lusminus(1))*real1, p_label(**i)));
    return result;
  } 
//  virtual measure example_parameterset(const physquantity &desiredret) =0; 
//  virtual phmsq_function() =0;
  virtual retT operator() (const measure &thisparams) const =0;
  virtual bool can_use_with_one_parameter() const { return argdrfs.size()==1; }
  virtual retT operator() (const physquantity &singleparam) const {  // avoid overloading this function, the 
    if (argdrfs.size()==1) {                                        //  measure-accepting form should stay standard.
      return operator()(measure(singleparam.wlabel(*argdrfs.front())));
     }else{
      std::cerr << "Called a phqfn that doesn't have one canonical argument with one parameter;\n";
      std::cerr << "use can_use_with_one_parameter() in doubt to check whether this IS possible\n";
      abort();
    }
  }
  virtual phmsq_function_base* clone() const=0;      //for copy-by-baseobjpointer-able derived objects
  virtual ~phmsq_function_base() {}

 // template <class NonevaledFn>
//  friend struct ptlevaled_fittable_phmsq_function;
};

typedef phmsq_function_base<physquantity> phmsq_function;


QTeXdiagmaster &QTeXdiagmaster::plot_phmsq_function(const phmsq_function &f, phq_interval rng, const measure &consts, int res, const QTeXgrcolor &cl, std::string nfname) {

  make_safe_qcvfilename(nfname);
  QTeXgrofstream crvfile(nfname);
  if (!crvfile) {cout<<"Bad filename \"" << nfname << "\""; abort();}

  if (rng.width()==0) return *this;     // Nothing to plot on a vanishing interval!

  if (res<0) res = default_resolution_for_phfunction_plots;
  physquantity plotstep = rng.width()/res;
  if (rng.l().is_identical_zero()) rng.l() = rng.width()*0;
  
  phq_interval yrng( f(consts.combined_with(rng.l())), f(consts.combined_with(rng.r())) );

  measure fnarg = consts;
  fnarg.push_back(rng.randompoint());
//  cout << rng << endl << yrng << endl;
  for (physquantity &x=fnarg.back(); yrng.width()==0; x=rng.randompoint()) {
//    cout << "(" << x << "," << f(fnarg) << ")\n"; for(int i=30792340; i>0 ;--i) {i=-i;i=-i;}
                                      //search the interval for a function value that
    yrng.widen_to_include(f(fnarg)); // does not equal the one at the borders,
  }                                 //  so as to have an estimate over the area the
  widen_window(rng, yrng);         //   graph will need for display

  struct schedstack{physquantity x,y; bool lineto; schedstack *last;}schedule;


  for (int outline_errmargins=0; outline_errmargins<2; ++outline_errmargins) {
    (fnarg=consts).push_back(rng.l());
    
    physquantity &x = fnarg.back();
    physquantity fnval = f(fnarg);

    if (!outline_errmargins)
      schedule = schedstack{x, fnval, false, NULL};
    if (outline_errmargins)
      plotstep *= 3/sqrt(1+consts.size());

    for (x += plotstep; x<rng.r(); x+=plotstep) {
      if (outline_errmargins) {
        unsigned i = rand()%(consts.size()+1);
        if (i==consts.size()) {
          fnval = f(fnarg);
          fnval += fnval.error() * ((rand()%2)*2-1);
         }else{
          fnarg[i] = consts[i] + consts[i].error()*((rand()%2)*2-1);
          fnval = f(fnarg);
          fnarg[i] = consts[i];
        }
        schedule = schedstack{x, fnval, false, new schedstack(schedule)};
       }else{
        fnval = f(fnarg);
        if (fnval.couldbe_anything()){
          schedule.lineto=false;
         }else{
          yrng.widen_to_include(fnval);
          schedule = schedstack{x, fnval, true, new schedstack(schedule)};
        }
      }
      yrng.widen_to_include(fnval);
    }
    schedule.lineto=false;
  }

  crvfile.cwrite('p');
  crvfile.dpwrite(schedule.x[*origxUnit], schedule.y[*origyUnit]);
  crvfile.cwrite(' ');
  for (auto l=schedule.last; l!=NULL; schedule=*l, delete l, l=schedule.last) {
    if (l->lineto) {
      crvfile.cwrite('l');
     }else{
      crvfile.cwrite('p');
    }
    crvfile.dpwrite(l->x[*origxUnit], l->y[*origyUnit]);
    crvfile.cwrite(' ');
  }
  widen_window(rng, yrng);
   
  link_to_graph_file(nfname, cl);
    
  return *this;
}


//COPYABLE_PDERIVED_CLASS(wrapped_compactsupport_phqfn, phmsq_function) {
class wrapped_compactsupport_phqfn: public phmsq_function {
  const compactsupportfn<physquantity,physquantity> *wrappedfn;
  wrapped_compactsupport_phqfn(const compactsupportfn<physquantity,physquantity> &cstrt)
    : wrappedfn(&cstrt)    { // not a safe pointer for wrappedfn, that's why this is a private constructor
    phmsq_function::argdrfs = argderefv("x");
  }
 public:
  phmsq_function_base<physquantity> *clone() const {
    cerr << "Tried to clone a wrapped_compactsupport_phqfn\n";
    cerr.flush();   // as the next instruction is likely to crash the program
    return nullptr;
  }
  measure example_parameterset() const{
    return measure(wrappedfn->support().location().wlabel(*argdrfs[0]));
  }
  physquantity operator() (const measure &thisparams) const {
    return (*wrappedfn)(phmsq_function::argdrfs[0](thisparams));
  }
  friend QTeXdiagmaster &QTeXdiagmaster::plot_phmsq_function(const compactsupportfn<physquantity, physquantity> &, const QTeXgrcolor &);
};
QTeXdiagmaster &QTeXdiagmaster::plot_phmsq_function(const compactsupportfn<physquantity, physquantity> &f, const QTeXgrcolor &cl){
  wrapped_compactsupport_phqfn wfn(f);
  plot_phmsq_function( wfn
                     , label(f.support(), wfn.example_parameterset()[0].cptstr())
                     , cl
                     );
 // insertCurve(rasterize(f, 2048), cl);
  return *this;
}
#if 0
COPYABLE_PDERIVED_CLASS(added_phqfns, phmsq_function) {
  std::vector<const compactsupportfn<physquantity,physquantity> *> wrappedfns;
 public:
  added_phqfns(const compactsupportfn<physquantity,physquantity> &cstrt)
    : wrappedfn(&cstrt)    {         // TODO: safe pointer for wrappedfn
    phmsq_function::argdrfs = argderefv("x");
  }
  measure example_parameterset() const{
    return measure(wrappedfn->support().location().wlabel(*argdrfs[0]));
  }
  physquantity operator() (const measure &thisparams) const {
    return (*wrappedfn)(phmsq_function::argdrfs[0](thisparams));
  }
};
#endif

namespace_cqtxnamespace_CLOSE
#include "squdistaccel.hpp"
namespace_cqtxnamespace_OPEN

//using namespace squaredistanceAccel;

class fittable_phmsqfn : public phmsq_function{

 public:
  class squaredistAccelerator;
 protected:
  typedef squaredistanceAccel::nonlinSqD_handle accHandle;
  typedef std::pair<std::vector<const phUnit*>, const phUnit*> vpUnits;
  static auto
  build_sqdistaccel(accHandle ah, phmsq_function::argderefv vadfs, vpUnits vus)
                           -> squaredistAccelerator;
 public:

  virtual auto
  squaredist_accel(const measureseq& fixedparams, const msq_dereferencer& retlk)const
                    -> maybe<squaredistAccelerator> {
    return nothing;
  }

  fittable_phmsqfn &rename_var(const string &oldcapt, const string &newcapt){       //rename parameter variable
    phmsq_function::rename_var(oldcapt, newcapt); return *this;
  }
  fittable_phmsqfn &rename_var(const string &oldcapt, const string &newcapt, const LaTeXindex_itrange &r){       //rename parameter variable
    phmsq_function::rename_var(oldcapt, newcapt, r); return *this;
  }
  virtual measure example_parameterset( const measure &constrainingparams
                                      , const physquantity &desiredret) const{
    measure cprms = constrainingparams;
    for (measure::iterator i=cprms.begin(); i!=cprms.end(); ++i)
      i->seterror(*i);     // make values sufficiently uncertain

    measure expprms;
    int min_variables_combinations=1, optional_variables_combinations=0;
    bool consider_negative_exponents=false;
    while(1) {
      expprms.clear();
      for (argderefv::const_iterator i = phmsq_function::argdrfs.begin()
          ; i!=phmsq_function::argdrfs.end()
          ; ++i) {
        if (cprms.has(**i)) {
          expprms.push_back(cprms[**i]);
         }else{
          expprms.push_back(1*real1);
          if (min_variables_combinations==1&&optional_variables_combinations==0&&!consider_negative_exponents)
            expprms.back() = cprms[rand()%cprms.size()];
           else{
            for (int j=min_variables_combinations
                            + rand()%(optional_variables_combinations+1)
                        ; j>0; --j) {
              int trialexampleextr = rand()%cprms.size();
              if ( consider_negative_exponents && rand()%2)
                expprms.back() /= cprms[trialexampleextr];
               else
                expprms.back() *= cprms[trialexampleextr];
            }
          }
          expprms.back().label(**i);
        }
      }
      try {
        std::cout << "Trying to eval phmsq_function with parameters\n"
                  << expprms << "... ";
        desiredret - operator()(expprms);
        std::cout << "... Succeded!\n";

//        cout << "\nQuantities that might need to be removed again:\n" << constrainingparams << endl;
        for (measure::iterator i = expprms.begin(); i!=expprms.end();) {
          if (cprms.has(i->caption)){
//            cout << "Removing " << i->cptstr() << endl;
            i=expprms.erase(i);             // remove those quantities that came
//            cout << "which leaves:\n" << expprms << endl;
          }           else                            //  from the constraining parameterset
//            cout << "Has no " << i->cptstr() << endl;
            ++i;
        }
//        cout << "Result:\n" << expprms << endl;
        return expprms;
      }catch(...){
        std::cout << "... Failed!\n";
        if (0==rand()%(15*phmsq_function::argdrfs.size()))
          ++optional_variables_combinations;
        if (0==rand()%(39*phmsq_function::argdrfs.size()))
          consider_negative_exponents=true;
        if (0==rand()%(158*phmsq_function::argdrfs.size()))
          min_variables_combinations = 0;
      }
    }
  }
  virtual measure example_parameterset(const physquantity &desiredret) const { return example_parameterset(measure(), desiredret); } 
  virtual fittable_phmsqfn* clone() const=0;      //for copy-by-baseobjpointer-able derived objects
};


COPYABLE_PDERIVED_CLASS(fittable_phmsqfn::squaredistAccelerator, fittable_phmsqfn) {
  accHandle h;
  vpUnits varunits;
  using fittable_phmsqfn::argdrfs;

  squaredistAccelerator(accHandle ah, phmsq_function::argderefv vadfs, vpUnits vus)
    : h(std::move(ah))
    , varunits(vus)   {
    argdrfs = std::move(vadfs);
//    std::cout << "Constructed sqdistaccel" << std::endl;
  }
 public:
  auto operator()(const measure& fitparams)const -> physquantity {
    std::vector<double> natparams(argdrfs.size());
    for(unsigned i=0; i<argdrfs.size(); ++i)
      natparams.at(i) = argdrfs[i](fitparams)[*varunits.first.at(i)];
    return h(natparams) * *varunits.second;
  }
 friend class fittable_phmsqfn;
};
auto fittable_phmsqfn
::build_sqdistaccel(accHandle ah, phmsq_function::argderefv vadfs, vpUnits vus)
                           -> squaredistAccelerator                            {
    return squaredistAccelerator(std::move(ah), vadfs, vus);                   }

// TEMPLATIZED_COPYABLE_DERIVED_STRUCT( base_phmsqfn
//                                    , phmsqfitfn_paramRemap
//                                    , fittable_phmsqfn      ) {
//   
// }

   //evaluate the average square distance between some measure values and a
  // fit function trying to describe those values though some fit parameters.
 //  The result is relative to the margin allowed by the measures' uncertainties,
//   i.e. if the deviations are of the order predicted by those, 1 is returned.
COPYABLE_PDERIVED_CLASS(fitdist_fntomeasures, phmsq_function) {
  measureseq ftgt;
  measure constargs;
  std::unique_ptr<fittable_phmsqfn> f;
  maybe<fittable_phmsqfn::squaredistAccelerator> accelerated;
  msqDereferencer returndrf;
  bool take_fnreturns_as_exact;

  physquantity straightforwardsqdistance(const measure &fpm) const{
    physquantity result = 0;
    measure ffpm = fpm; ffpm.remove_errors();
    for(auto& x : ftgt){
      measure wkd = ffpm;
      wkd.append(x);
      physquantity idealret = returndrf(x);

      if (take_fnreturns_as_exact) {
        assert(1==0);
        physquantity thiscmperrsq = idealret.error();
        idealret.seterror(0);
        idealret.SubstractAsExactVal((*f)(wkd));
        result += (idealret /= thiscmperrsq).squared();
       }else{
        idealret -= (*f)(wkd);
        result += (idealret.werror(0) / idealret.error()).squared();
      }

    }
    return result;                                     
  }



  void enforcenormalizableerrors() {

    bool allhaveerror=true;
    measureseq::iterator i = ftgt.begin();
    
    physquantity minimumabsval = abs(returndrf(*i));
    while (minimumabsval==0) {
      minimumabsval = abs(returndrf(*(i++)));
      if (i==ftgt.end()) {
        cerr << ftgt << "(measureseq size: " << ftgt.size()
             << ")\nWhy would you want to fit some function to a measure sequence that is 0 everywhere?";
        abort();
      }
    }
    
    for (i = ftgt.begin(); i!=ftgt.end(); ++i){
      if (returndrf(*i).error()==0) allhaveerror=false;
      if (returndrf(*i)!=0) minimumabsval.push_downto(abs(returndrf(*i)));
    }

    if (!allhaveerror) {
      minimumabsval/=2;
      i = ftgt.begin();
      while (i!=ftgt.end()) {
        if (returndrf(*i).error()==0) returndrf(*(i++)).error(minimumabsval);
         else i = ftgt.erase(i);
      }
    }

  }

 public:

  fitdist_fntomeasures( const measureseq& nftgt
                      , const fittable_phmsqfn& nf
                      , msqDereferencer nreturndrf
                      , const measure &constraints=measure() )
    : ftgt(nftgt)
    , constargs(constraints)
    , f(nf.clone())
    , accelerated(f->squaredist_accel(ftgt, nreturndrf))
    , returndrf(std::move(nreturndrf))
    , take_fnreturns_as_exact(false) {
    enforcenormalizableerrors();
  }
  
  fitdist_fntomeasures(const fitdist_fntomeasures& cpy)
    : ftgt(cpy.ftgt)
    , constargs(cpy.constargs)
    , f(cpy.f->clone())
    , accelerated(cpy.accelerated)
    , returndrf(cpy.returndrf)
    , take_fnreturns_as_exact(cpy.take_fnreturns_as_exact)
  {}

  fitdist_fntomeasures(fitdist_fntomeasures&& mov)
    : ftgt(std::move(mov.ftgt))
    , constargs(std::move(mov.constargs))
    , f(std::move(mov.f))
    , accelerated(std::move(mov.accelerated))
    , returndrf(std::move(mov.returndrf))
    , take_fnreturns_as_exact(mov.take_fnreturns_as_exact)
  {}

  measure example_parameterset() const{
    measure cconstraints = constargs;
    cconstraints.append( ftgt.randomrepresentatives() );
    return f->example_parameterset(cconstraints, ftgt.randomrepresentative(returndrf));
  } 


  physquantity operator() (const measure &thisparams) const {
    for(auto& acc: accelerated) return acc(thisparams) / ftgt.size();
    return straightforwardsqdistance(thisparams) / ftgt.size();
  }

};




namespace_cqtxnamespace_CLOSE

#include "fitfncmacros.h"    //helper macros, to make it easier to write the functions in a manner
                            // both readable and automation-friendly
#include "stdfitfn.cpp"  //some standard fitting functions, like gaussian peaks etc.

namespace_cqtxnamespace_OPEN


                                         TEMPLATIZED_COPYABLE_DERIVED_STRUCT(/*
template < class*/NonevaledFn/*>  // NonevaledFn should be a phmsq_function*/,/*
struct*/ptlevaled_fittable_phmsq_function, fittable_phmsqfn) {

  using copyable_derived<fittable_phmsqfn,ptlevaled_fittable_phmsq_function>::fittable_phmsqfn::argdrfs;
  using copyable_derived<fittable_phmsqfn,ptlevaled_fittable_phmsq_function>::fittable_phmsqfn::argdrfs_of;

  fittable_phmsqfn *ofn;
  mutable measure parameters;
  unsigned nstatic, ndynamic;

  void register_eval_lack() {
    for (auto param : argdrfs_of(ofn))
      if (!parameters.has(*param)) {
        argdrfs.push_back(param);
        parameters.push_back((0*real1).wlabel(*param));
        ++ndynamic;
      }
  }
#if 0
  void multi_partial_eval(const measureseq &stparams) {
    multi_partial_eval(stparams, [](const std::string &s){return s;});
    for (auto param : ofn->argdrfs) {
      std::string c = *param;
      maybe<int> subscr = try_splitoff_subscript<int>(c);
      if (subscr.is_just() && stparams[*subscr].has(c)) {
        parameters.push_back(stparams[*subscr][c]);
        ++nstatic;
       }else{
        argdrfs.push_back(c);
        parameters.push_back(0);
        ++ndynamic;
      }
    }
  }
#endif
  template<class param_assoc_function>
  void multi_partial_eval(const measureseq &stparams, const param_assoc_function &asc) {
    measure dynaparms;
    for (auto param : argdrfs_of(ofn)) {
      std::string c = *param;
      maybe<unsigned> subscr = try_splitoff_subscript<unsigned>(c);
      if (subscr.is_just() && *subscr < stparams.size()
             && stparams[*subscr].has(asc(c))) {
        parameters.push_back(stparams[*subscr][asc(c)].wlabel(*param));
        ++nstatic;
       }else{
        argdrfs.push_back(asc(c));
        ++ndynamic;
        dynaparms.push_back((0*real1).wlabel(c));
      }
    }
    for (auto q : dynaparms)
      parameters.push_back(q);
  }
 public:
  ptlevaled_fittable_phmsq_function(const measure &stparams) // simple partial evaluation, for ordinary
    : ofn(new NonevaledFn())                                //  phmsq_functions with default constructor
    , parameters(stparams)
    , nstatic(parameters.size())
    , ndynamic(0)                {
    register_eval_lack();
  }
  ptlevaled_fittable_phmsq_function(const NonevaledFn &f, const measure &stprs)
    : ofn(f.clone())
    , parameters(stprs)
    , nstatic(parameters.size())
    , ndynamic(0)                {
    register_eval_lack();
  }
  ptlevaled_fittable_phmsq_function(const measureseq &stparams)   // multi-partial evaluation, for array-like
    : ofn(new NonevaledFn(stparams.size()))                      //  phmsq_functions with constructor taking
    , parameters(0)
    , nstatic(0)
    , ndynamic(0)                                                     {
    multi_partial_eval(stparams, [](const std::string &s){return s;});
  }
  template<class ParamAssocFunction>
  ptlevaled_fittable_phmsq_function(const measureseq &stparams, const ParamAssocFunction &asc)
    : ofn(new NonevaledFn(stparams.size()))
    , parameters(0)
    , nstatic(0)
    , ndynamic(0)                       {
    multi_partial_eval(stparams, asc);
  }
  template<class ParamAssocFunction>
  ptlevaled_fittable_phmsq_function(const NonevaledFn &f, const measureseq &stparams, const ParamAssocFunction &asc)
    : ofn(f.clone())
    , parameters(0)
    , nstatic(0)
    , ndynamic(0)                       {
    multi_partial_eval(stparams, asc);
  }
  ptlevaled_fittable_phmsq_function(const NonevaledFn &f, const measureseq &stparams)
    : ptlevaled_fittable_phmsq_function(f, stparams, [](const std::string &s){return s;}) {}

  physquantity operator() (const measure &thisparams) const{
    for (unsigned i=0; i<ndynamic; ++i) {
      parameters[nstatic+i] = argdrfs[i](thisparams);
      //cout << "dynamic @\"" << *argdrfs[i] << "\": " << parameters[nstatic+i].cptstr() << " = " << parameters[nstatic+i] << endl;
    }
    //cout << "evaluate function with parameters\n" << parameters;
    return (*ofn)(parameters);
  }

  ~ptlevaled_fittable_phmsq_function() {delete ofn;}
};

template<typename PtlParams>
auto partially_evaluated(const fittable_phmsqfn &f, PtlParams c)
        -> ptlevaled_fittable_phmsq_function<fittable_phmsqfn> {
  return ptlevaled_fittable_phmsq_function<fittable_phmsqfn>(f, c);
}

template<typename RedrData>
class redirection_map {
  std::map<RedrData, RedrData> rmap;
 public:
  redirection_map(std::initializer_list<std::initializer_list<RedrData>> init) {
    for (auto i : init) {
      assert(i.size()==2);
      auto j = i.begin();
      RedrData key = *j;
      ++j;
      rmap[key] = *j;
    }
  }
  const RedrData &operator()(const RedrData &key) const{
    auto loc = rmap.find(key);
    if (loc!=rmap.end()) return loc->second;                     
     else return key;
  }
  template<typename ORedrData>
  const ORedrData &operator()(const ORedrData &key) const {
    auto loc = rmap.find(RedrData(key));
    if (loc!=rmap.end()) return ORedrData(loc->second);
     else return ORedrData(key);
  }
};
template<typename RedrData>
auto redirect(std::initializer_list<std::initializer_list<RedrData>> init)
       -> redirection_map<RedrData> {
  return redirection_map<RedrData>(init);
}
auto redirect_name(std::initializer_list<std::initializer_list<std::string>> init)
       -> redirection_map<std::string>         {
  return redirection_map<std::string>(init);
}




typedef phmsq_function_base<bool> boolean_phmsq_function;
class nonholonomic_optimizationconstraint: public boolean_phmsq_function {
/*
 protected:
  struct cstrtInfoPrimitive {
    std::vector<captfinder> actees;
    std::vector<physquantity> params;
    enum {
      less_than;                                    virtual measure suggestion() const{
      more_than;                                   std::vector<cstrtInfoPrimitive> prms = primitives();
      sum_less_than;                               for (auto i = prms.begin
      sum_more_than;                               }
      difference_less_than;
      difference_more_than;
    };
  };
  virtual std::vector<cstrtInfoPrimitive> primitives() const=0;
//  static measure suggestion_crt(const std::vector<cstrtInfoPrimitive> &cstrt_primitives) {  }
*/
 public:
  virtual void make_conformant(measure &trial) const=0;
  virtual nonholonomic_optimizationconstraint* clone() const=0;
};

COPYABLE_DERIVED_STRUCT(trivial_nonholonomic_optimizationconstraint, nonholonomic_optimizationconstraint) {
/*std::vector<nonholonomic_optimizationconstraint::cstrtInfoPrimitive> primitives() const{
    return std::vector<nonholonomic_optimizationconstraint::cstrtInfoPrimitive>(0);
  }*/
  bool operator() (const measure &thisparams) const{ return true; }
  void make_conformant(measure &trial) const{}
}trivial_constraint;

COPYABLE_DERIVED_STRUCT(enforce_positiveness, nonholonomic_optimizationconstraint) {
  enforce_positiveness(argderefv init) { argdrfs = init; }
  enforce_positiveness(const std::string &init) { argdrfs = argderefv(init); }
  bool operator() (const measure &thisparams) const{
    for (argderefv::const_iterator i = argdrfs.begin(); i!= argdrfs.end(); ++i){
      if ((*i)(thisparams).negative()) return false;
    }
    return true;
  }
  void make_conformant(measure &trial) const{
    for (auto drv : argdrfs){
      physquantity &q = drv(trial);
      while (q.negative())
        q = ((q*0).werror(q)).uncertain_collapsed();
    }
  }
};

COPYABLE_DERIVED_STRUCT(enforce_each_smaller_than, nonholonomic_optimizationconstraint) {
 private:
  measure suprema;
 public:
  enforce_each_smaller_than(const measure &compr): suprema(compr) {
    for (measure::iterator i = suprema.begin(); i!=suprema.end(); ++i)
      argdrfs.push_back(i->caption);
  }
  enforce_each_smaller_than(argderefv init, const physquantity &p) {
    argdrfs = init;
    for (argderefv::const_iterator i = argdrfs.begin(); i!= argdrfs.end(); ++i){
      suprema.push_back(p.wlabel(**i));
    }
  }
  enforce_each_smaller_than(const std::string &init, const physquantity &p) { argdrfs = argderefv(init);
    for (argderefv::const_iterator i = argdrfs.begin(); i!= argdrfs.end(); ++i){
      suprema.push_back(p.wlabel(**i));
    }
  }
  bool operator() (const measure &thisparams) const{
    for (unsigned i = 0; i<argdrfs.size(); ++i){
      if ((argdrfs[i])(thisparams) >= suprema[i]) {
   //     cout << *argdrfs[i] << " = " << (argdrfs[i])(thisparams) << " ≥ " << suprema[i] << endl;
        return false;
      }
    }
    return true;
  }
  void make_conformant(measure &trial) const{
    for (unsigned i = 0; i<argdrfs.size(); ++i){
      physquantity &q = argdrfs[i](trial), s = suprema[i];
      while (q>=s)
        q = (s.werror(q-s)).uncertain_collapsed();
    }
  }
};

COPYABLE_DERIVED_STRUCT(enforce_each_bigger_than, nonholonomic_optimizationconstraint) {
 private:
  measure infima;
 public:
  enforce_each_bigger_than(const measure &compr): infima(compr) {
    for (measure::iterator i = infima.begin(); i!=infima.end(); ++i)
      argdrfs.push_back(i->caption);
  }
  enforce_each_bigger_than(argderefv init, const physquantity &p) {
    argdrfs = init;
    for (argderefv::const_iterator i = argdrfs.begin(); i!= argdrfs.end(); ++i){
      infima.push_back(p.wlabel(**i));
    }
  }
  enforce_each_bigger_than(const std::string &init, const physquantity &p) { argdrfs = argderefv(init);
    for (argderefv::const_iterator i = argdrfs.begin(); i!= argdrfs.end(); ++i){
      infima.push_back(p.wlabel(**i));
    }
  }
  bool operator() (const measure &thisparams) const{
    for (unsigned i = 0; i<argdrfs.size(); ++i){
      if ((argdrfs[i])(thisparams) <= infima[i]) {
//       cout << *argdrfs[i] << " = " << (argdrfs[i])(thisparams) << " ≤ " << infima[i] << endl;
        return false;
      }
    }
    return true;
  }
  void make_conformant(measure &trial) const{
    for (unsigned i = 0; i<argdrfs.size(); ++i){
      physquantity &q = argdrfs[i](trial), s = infima[i];
      while (q<=s)
        q = (s.werror(q-s)).uncertain_collapsed();
    }
  }
};

COPYABLE_DERIVED_STRUCT(enforce_each_sum_bigger_than, nonholonomic_optimizationconstraint) {
 private:
  measure infima;
  unsigned npairs;
 public:
  enforce_each_sum_bigger_than(const measure &smds1, const measure &smds2) {
    assert(smds1.size() == smds2.size());
    npairs = smds1.size();
    infima = smds1;
    for (measure::iterator i = infima.begin(); i!=infima.end(); ++i)
      argdrfs.push_back(i->caption);
    for (unsigned i = 0; i<npairs; ++i) {
      infima[i]+=smds2[i];
      argdrfs.push_back(smds2[i].caption);
    }
  }
  enforce_each_sum_bigger_than(argderefv smds1, argderefv smds2, const physquantity &p) {
    if (smds1.size()==1){
      npairs = smds2.size();
      for (unsigned i = 0; i!= npairs; ++i)
        argdrfs.push_back(smds1[0]);
      for (unsigned i = 0; i!= npairs; ++i){
        argdrfs.push_back(smds2[i]);
        infima.push_back(p.wlabel(*smds1[0] + " + " + *smds2[i]));
      }
     }else if(smds2.size()==1){
      npairs = smds1.size();
      argdrfs = smds1;
      for (unsigned i = 0; i!= npairs; ++i){
        argdrfs.push_back(smds2[0]);
        infima.push_back(p.wlabel(*smds1[i] + " + " + *smds2[0]));
      }
     }else{
      assert(smds1.size() == smds2.size());
      npairs = smds1.size();
      argdrfs = smds1;
      for (unsigned i = 0; i!= npairs; ++i){
        argdrfs.push_back(smds2[i]);
        infima.push_back(p.wlabel(*smds1[i] + " + " + *smds2[i]));
      }
    }
  }
  bool operator() (const measure &thisparams) const{
    for (unsigned i = 0; i<npairs; ++i){
      if ((argdrfs[i])(thisparams) + (argdrfs[i+npairs])(thisparams) <= infima[i]) {
        return false;
      }
    }
    return true;
  }
  void make_conformant(measure &trial) const{
    for (unsigned i = 0; i<npairs; ++i){
      physquantity &q1 = argdrfs[i](trial), &q2 = argdrfs[i+npairs](trial)
                 , s = infima[i];
      while (q1 + q2 <= s) {
        physquantity delta = (s - (q1+q2))/2;
        q1 = ((q1+delta).werror(delta)).uncertain_collapsed();
        q2 = ((q2+delta).werror(delta)).uncertain_collapsed();
      }
    }
  }
};
COPYABLE_DERIVED_STRUCT(enforce_each_difference_smaller_than, nonholonomic_optimizationconstraint) {
 private:
  measure suprema;
  unsigned npairs;
 public:
  enforce_each_difference_smaller_than(const measure &smds1, const measure &smds2) {
    assert(smds1.size() == smds2.size());
    npairs = smds1.size();
    suprema = smds1;
    for (measure::iterator i = suprema.begin(); i!=suprema.end(); ++i)
      argdrfs.push_back(i->caption);
    for (unsigned i = 0; i<npairs; ++i) {
      suprema[i]-=smds2[i];
      argdrfs.push_back(smds2[i].caption);
    }
  }
  enforce_each_difference_smaller_than(argderefv smds1, argderefv smds2, const physquantity &p) {
    if (smds1.size()==1){
      npairs = smds2.size();
      for (unsigned i = 0; i!= npairs; ++i)
        argdrfs.push_back(smds1[0]);
      for (unsigned i = 0; i!= npairs; ++i){
        argdrfs.push_back(smds2[i]);
        suprema.push_back(p.wlabel(*smds1[0] + " - " + *smds2[i]));
      }
     }else if(smds2.size()==1){
      npairs = smds1.size();
      argdrfs = smds1;
      for (unsigned i = 0; i!= npairs; ++i){
        argdrfs.push_back(smds2[0]);
        suprema.push_back(p.wlabel(*smds1[i] + " - " + *smds2[0]));
      }
     }else{
      assert(smds1.size() == smds2.size());
      npairs = smds1.size();
      argdrfs = smds1;
      for (unsigned i = 0; i!= npairs; ++i){
        argdrfs.push_back(smds2[i]);
        suprema.push_back(p.wlabel(*smds1[i] + " - " + *smds2[i]));
      }
    }
  }
  bool operator() (const measure &thisparams) const{
    for (unsigned i = 0; i<npairs; ++i){
      if ((argdrfs[i])(thisparams) - (argdrfs[i+npairs])(thisparams) >= suprema[i]) {
        return false;
      }
    }
    return true;
  }
  void make_conformant(measure &trial) const{
    for (unsigned i = 0; i<npairs; ++i){
      physquantity &q1 = argdrfs[i](trial), &q2 = argdrfs[i+npairs](trial)
                 , s = suprema[i];
      while (q1 - q2 >= s) {
        physquantity delta = (q1-q2 - s)/2;
        q1 = ((q1-delta).werror(delta)).uncertain_collapsed();
        q2 = ((q2+delta).werror(delta)).uncertain_collapsed();
      }
    }
  }
};




COPYABLE_PDERIVED_CLASS(combined_nonholonomic_optimizationconstraint, nonholonomic_optimizationconstraint) {
  typedef std::vector<const nonholonomic_optimizationconstraint*> delegationcnt;
  delegationcnt cstrs;
 public:
  combined_nonholonomic_optimizationconstraint(const nonholonomic_optimizationconstraint &c1, const nonholonomic_optimizationconstraint &c2) {
    cstrs.push_back(c1.clone());
    cstrs.push_back(c2.clone());
  }
  combined_nonholonomic_optimizationconstraint(const combined_nonholonomic_optimizationconstraint &cpfr): cstrs(cpfr.cstrs){
    for (delegationcnt::iterator i = cstrs.begin(); i!=cstrs.end(); ++i)
      *i = (*i)->clone();
  }
  bool operator() (const measure &thisparams) const{
    for (delegationcnt::const_iterator i = cstrs.begin(); i!=cstrs.end(); ++i){
      if (!(**i)(thisparams)) return false;
    }
    return true;
  }
  void make_conformant(measure &trial) const{
    while (!operator()(trial)) {
      int i = rand()%cstrs.size();
      (*cstrs[i]).make_conformant(trial);
    }
  }
  ~combined_nonholonomic_optimizationconstraint(){
    for (delegationcnt::iterator i = cstrs.begin(); i!=cstrs.end(); ++i)
      delete *i;
  }
};


combined_nonholonomic_optimizationconstraint operator &&(const nonholonomic_optimizationconstraint &l, const nonholonomic_optimizationconstraint &r){
  return combined_nonholonomic_optimizationconstraint(l, r);
}



class evolution_minimizer {
  const phmsq_function &f;
  const nonholonomic_optimizationconstraint &constrained;
  measure minmzstate;

  struct cons {
    measure t;
    physquantity e;
    cons(const measure &trial, const physquantity &anderror):
      t(trial), e(anderror) {}
  };
  struct schedule: std::queue<cons>{
    void clear() { c.clear(); }
    const measure &random_location() const{
      return c[rand() * float(c.size()) / RAND_MAX].t;
    }
  }sched;
  unsigned aim_schedlength;    //≥92;
  unsigned static_niterations;  //≈2^(≈12);
  double meanerrstosdv; //=std::sqrt(aim_schedlength);

  physquantity avgd, acgmin, distatest, meanexpdistdiffdiff;

  typedef std_normaldistributed_randomgen ranGenT;
  ranGenT &r;
  std::ostream *msgstream;
  
  
  bool is_finished() {                // Not used at the moment
if(0)cout << "minimum: " << acgmin
         << "\naverage: " << avgd
         << "\nmeanexpdistdiffdiff: " << meanexpdistdiffdiff << "\n\n";
    return meanexpdistdiffdiff < 1 && avgd.error() /* meanerrstosdv*/ < acgmin / 4;
//        && distatest < avgd - avgd.error();
  }

  double multiplicity_stretching_factor;
  unsigned multiplicity_for(const physquantity &td) {             //Multiplicity aiming at a Boltzmann distribution
if(0)cout<< "Calc multiplicity.\nAverage dist: " << avgd
         << "\nDist here now: " << td
         << "\nmultiplicity_stretching_factor: " << multiplicity_stretching_factor
         << "\n(td.minusexactly(avgd)/avgd.error()).dbl() * multiplicity_stretching_factor: "
         << (td.minusexactly(avgd)/avgd.error()).dbl() * multiplicity_stretching_factor
         << "\nsched.size(): "
         << sched.size()
         << "\naim_schedlength: "
         << aim_schedlength
         << "\n(double(sched.size()) - aim_schedlength) / aim_schedlength: "
         << (double(sched.size()) - aim_schedlength) / aim_schedlength
         << "\n(td.minusexactly(avgd)/avgd.error()).dbl() + (double(sched.size()) - aim_schedlength) / aim_schedlength: "
         << (td.minusexactly(avgd)/avgd.error()).dbl() + double(sched.size() - aim_schedlength) / aim_schedlength
         << "\nBoltzmann: "
         << exp(- (td.minusexactly(avgd)/avgd.error()).dbl() * multiplicity_stretching_factor
                                           - (double(sched.size()) - aim_schedlength) / aim_schedlength )
         << "\ndithered: ";
    unsigned result= std::min (unsigned(dither_round( exp( ( - (td.minusexactly(avgd)/avgd.error()).dbl()
                                           - (double(sched.size()) - aim_schedlength) / aim_schedlength )
                                            * multiplicity_stretching_factor ) ) ),
                aim_schedlength/4)     ;
    if (result==1) multiplicity_stretching_factor *= 1 + 2./aim_schedlength;
     else multiplicity_stretching_factor *= 1 - 1./aim_schedlength;
   // cout << result << "\n\n";
    return result;
  }
  
  measure staterandomizer() {
    measure result = -minmzstate;
    if (float(rand())/RAND_MAX > 1) {
      for (measure::iterator i = result.begin(); i!=result.end(); ++i)
        *i = i->error() /* meanerrstosdv */* (r() / 2.);
     }else{
      result.operator=(sched.random_location())
            .remove_errors()
            .by_i_substract(sched.random_location())
            .multiply_all_by(r()/2);
    }
    return result;
  }
  
 public:
 
  void prepare_minimize() {
    sched.clear();
    measureseq qs, exqs;

    distatest = f(minmzstate);
    for (unsigned i=0; i<aim_schedlength; ++i) {
   //   cout << "minmzstate: \n" << minmzstate << endl;
      measure m;
      unsigned misses=0, rescales=0;
      do{
        if (misses++ > (i*i+1)*aim_schedlength*(1+rescales*rescales*rescales)) {
          minmzstate.scale_allerrors(sqrt(sqrt(sqrt(sqrt(2.)))));
          ++rescales;
          if(msgstream)(*msgstream) << "Cannot find fitting start; scale range -{\n"
                                    << minmzstate
                                    << " - - - - - - - - - - - - - - - - - - - -}\n\n";
        }
        m = minmzstate.uncertain_collapsed();
   //     cout << m;
    //    for (int i=0; i<89257698; ++i){}
        if(!constrained(m)) {
         //misses=misses;}//cout << "[FAILED]\n\n";
          constrained.make_conformant(m);
         }else{
    //      cout << "[SUCCEEDED]\n\n";
          misses = 0;
        }
      }while (!constrained(m));
  //    cout << "m: \n" << m << endl;
      physquantity fh = f(m);
      if (i) acgmin.push_downto(fh);
       else acgmin=fh;
       
      qs.push_back(measure(fh));

      sched.push(cons(m, fh));
    }
    avgd = meanofElem(qs, measureindexpointer(0));
    avgd.error(avgd.error() * meanerrstosdv);
    meanexpdistdiffdiff = 0;
    for (measureseq::iterator i = qs.begin(); i!= qs.end(); ++i) {
 if(0)cout << "distatest: " << distatest
           << "\nfh: " << (*i)[0]
           << "\nDelta: " << distatest-(*i)[0]
           << "\navgd.error(): " << avgd.error()
           << "\nexpstuff: " << exp((distatest-(*i)[0])/avgd.error())<<"\n\n";  //(avgd.error()*aim_schedlength))
      meanexpdistdiffdiff += exp((distatest - (*i)[0])/avgd.error()); //(avgd.error()*aim_schedlength)
    }
    meanexpdistdiffdiff /= aim_schedlength;
  }

  void minimize(unsigned minziterations) {
    if(msgstream)(*msgstream) << "minmzstate (start): \n" << minmzstate << endl
                              << "begin optimization\n";
    unsigned c=0, d=0, t=0;
    while (t < minziterations ) {
      physquantity fh = sched.front().e;
 if(0)cout << "distatest: " << distatest
           << "\nfh: " << fh
           << "\nDelta: " << distatest-fh
           << "\navgd.error(): " << avgd.error()
           << "\nexpstuff: " << exp((distatest-fh)/avgd.error())<<"\n\n";  //(avgd.error()*aim_schedlength))
      meanexpdistdiffdiff.statistically_approach( exp((distatest-fh)/avgd.error()), 1./aim_schedlength );  //(avgd.error()*aim_schedlength)

      for (int i = multiplicity_for(fh); i>0; --i) {
   //     cout << "sched.front(): \n" << sched.front().t << endl;
   //     cout << "staterandomizer(): \n" << staterandomizer() << endl;
        measure m;
        do {
          m = sched.front().t.by_i_plus(staterandomizer());
        }while (!constrained(m));
        minmzstate.by_i_statistically_approach(m, 1./aim_schedlength);

        fh = f(m);
        if (fh < acgmin) {
          d=0;
          acgmin = fh;
         }else ++d;
        avgd.statistically_approach(fh, 1./aim_schedlength);
        
        sched.push(cons(m, fh));
      }
      sched.pop();
      if (sched.size() > aim_schedlength/4.) {
   if(0)cout << "sched size: " << sched.size() << endl;
       }else{
        cout << "Out of schedule, weird.\n";
        break;
      }
            
      if ( ((++c)%=int(meanerrstosdv)) == 0) {
        distatest = f(minmzstate);
      }
      ++t;
      if(msgstream && t%aim_schedlength == 0 && r()>1.5) {
        (*msgstream) << " --- I t e r a t i o n s :   " << t << " / " << minziterations << endl;
        (*msgstream)  << "minmzstate: \n" << minmzstate << endl;
        (*msgstream)  << "minimum: " << acgmin
                      << "\naverage: " << avgd << endl << endl;
         //  << "\nmeanexpdistdiffdiff: " << meanexpdistdiffdiff << "\n\n"
           ;
      }
    }

  }
  
  class solutioncertaintycriteria {
    bool needsdoublevalue;
   public:
    struct doublevalue {};
    solutioncertaintycriteria(): needsdoublevalue(false){}
    solutioncertaintycriteria(const doublevalue &d): needsdoublevalue(true){}
    const bool &needs_doublevalue() const { return needsdoublevalue; }
  }solutioncertaintycriterium;
  
  bool certaintyexpanduncertainty(){
    if (solutioncertaintycriterium.needs_doublevalue()){ //with this criterium, the value of the
      physquantity bestachievmt = f(minmzstate);        // function to minimize at any of the
                                                       //  uncertainty delimiters needs to be at
                                                      //   least 2^(1/n) the minimum value
                                                     //    (n=minmzstate.size()) to be acknowledged
                                                    //     as a statistically significant
                                                   //      solution to the minimization problem.
      for (unsigned i = 0; i<minmzstate.size(); ++i) {
        measure m = minmzstate;
        bool toosmall;
        do {
          toosmall = false;
          int runintoconstraint=0;
          for (int s=-1; s!=0; s=(s<0)? 1 : 0) {
            m[i] = minmzstate[i] + s*minmzstate[i].error();
            if (!constrained(m)) {                   //that means there is no error-significant surrounding
              if (runintoconstraint++) return false;// to the prepared solution inside the allowed range 
             }else if (f(m) < bestachievmt*std::pow(2,1./minmzstate.size())) {
              toosmall = true; break;
            }
          }
          minmzstate[i].scale_error(std::pow(2,.25/minmzstate.size()));
        } while(toosmall);
      }
    }
    return true;
  }

  
  measure minimized() {
    return minmzstate;
  }
  measure result() {
    return minmzstate;
  }
  physquantity minimum() {
    return f(minmzstate);
  }
  
  evolution_minimizer(const phmsq_function &nf,
                      const nonholonomic_optimizationconstraint &nc=trivial_constraint,
                      const solutioncertaintycriteria &solcertaintycriterium=solutioncertaintycriteria(),
                      const measure &startstate = measure(),
                      ranGenT &rg = defaultsimplenormaldistributedrandomgenerator)
    : f(nf)
    , constrained(nc)
    , minmzstate(startstate)
    , r(rg)
#ifdef EVOLUTIONFIT_STATUSMESSAGESDEFAULTSTREAM
    , msgstream(&EVOLUTIONFIT_STATUSMESSAGESDEFAULTSTREAM)
#else
    , msgstream(nullptr)
#endif
    , multiplicity_stretching_factor(1)
    , solutioncertaintycriterium(solcertaintycriterium) {
    minmzstate.complete_carefully(f.example_parameterset());
    aim_schedlength = std::max(6 * minmzstate.size() * minmzstate.size(), size_t(92u));
// older version:   aim_schedlength = std::max(4 * minmzstate.size() * minmzstate.size(), size_t(92u));
    static_niterations = aim_schedlength * 256; //128;    //empirically
    meanerrstosdv = std::sqrt(aim_schedlength);
    
    int totalruns=0;
    do {
      prepare_minimize();
      minimize(static_niterations);
      for (int refiningruns=0; refiningruns < 4; ++refiningruns) {
        if (certaintyexpanduncertainty()) goto minimization_successful;
        minimize(static_niterations/2);
      }
 //     minmzstate = startstate;   //try the whole process once again
      minmzstate.complete_carefully(f.example_parameterset());
      if (totalruns++ > 2) {
        cerr << "Aborting search for statistically significant solution.";
        goto minimization_successful;
        throw *this;  //after successlessly having tried again, abort
      }
    //  cout << "\nMinimization did not succeed. Try again with different set of starting parameters.\n";
    } while (0);
    minimization_successful:{}
    if(msgstream)(*msgstream) << "Result of minimization:\n" << minmzstate << endl;
  }
  /*montecarlofit(const measureseq &fsrc, const msq_dereferencer &retdrf,
                               const fittable_phmsqfn &fn, //const measure &startparams,
                               measure constraints=measure(),
                               ranGenT &rg = defaultsimplenormaldistributedrandomgenerator):
    ftgt(fsrc),
    f(fn),
    returndrf(retdrf),
    r(rg),
    distancer(fsrc, fn, retdrf) {//defaultsimplenormaldistributedrandomgenerator)                               {
    measure cconstraints = constraints;
    cconstraints.append( fsrc.randomrepresentatives() );
    cout << "\ncconstraints:\n" << cconstraints;
    constraints.append( f.example_parameterset(cconstraints, fsrc.randomrepresentative(retdrf)) );
    cout << "\nfitting start:\n" << constraints << endl;
    cout << "\nfitting result:\n" << (fitres = fittomsqseq(constraints)) << endl;
  }*/
};


struct fit_of_phqfn_to_msq_object {
  measure fitresult;
  const measure &result() const {return fitresult;}
};

fit_of_phqfn_to_msq_object fit_phq_to_measureseq(
          const fittable_phmsqfn& nf
        , const measureseq& nftgt
        , msqDereferencer nreturndrf
        , const nonholonomic_optimizationconstraint& nc=trivial_constraint
        , evolution_minimizer::solutioncertaintycriteria
                 sccrit=evolution_minimizer::solutioncertaintycriteria::doublevalue()  ) {
  fit_of_phqfn_to_msq_object ret;
  fitdist_fntomeasures d(nftgt, nf, std::move(nreturndrf));
  evolution_minimizer mnz(d, nc, sccrit);
  ret.fitresult = mnz.result();
  return ret;
}

typedef int spectfit_linkingt;

const spectfit_linkingt link_peaks_width  = 0x1    // set this flag to have all peaks have the same width.
                      , link_peaks_height = 0x2    // this for all same height.
                      , link_peaks_pos    = 0x4    // you probably do not want all same positions, it's there just for the sake of completeness.
                      , global_link_peaks_width  = 0x8    // global versions of the flags;
                      , global_link_peaks_height = 0x10  //  instead of enforcing the fit of
                      , global_link_peaks_pos    = 0x20;//   individual peak groups to feature
                                                       //    the value as a constant, this will
                                                      //     disallow gradual changes in the values.
                                                     //      Not supported yet.

std::string spectrpeakwidth_name="\\sigma";          

template<class MultiPeakFn = fittable_multigaussianfn>
measure ffit_spectrum( const measureseq &fnplot  // w/noise level track please!
                     , const captfinder &xfind
                     , const captfinder &Afind
                     , const captfinder &noisefind
                     , unsigned npeaks = 1
                     , spectfit_linkingt linkings = 0x0
                     ) {
  MultiPeakFn fgfn(npeaks);

  LaTeXindex_itrange indexrng = LaTeXindex("i").from(0).unto(npeaks);

  fgfn.rename_var("x, x_i, A_i", *xfind + ", " + *xfind+"_i, " + *Afind+"_i", indexrng);

  LaTeXvarnameslist widthdrfs, heightdrfs, positiondrfs;
  if (linkings & link_peaks_width) {
    widthdrfs = spectrpeakwidth_name;
    for (unsigned i=0; i<npeaks; ++i)
      fgfn.rename_var(spectrpeakwidth_name+LaTeX_subscript(i), spectrpeakwidth_name);
   }else widthdrfs = (spectrpeakwidth_name+"_i")|indexrng;
  if (linkings & link_peaks_height) {
    heightdrfs = *Afind+"_0";
    for (unsigned i=1; i<npeaks; ++i) fgfn.rename_var(*Afind+LaTeX_subscript(i), *Afind+"_0");
   }else heightdrfs = (*Afind+"_i")|indexrng;
  if (linkings & link_peaks_pos) {
    positiondrfs = *xfind+"_0";
    for (unsigned i=1; i<npeaks; ++i) fgfn.rename_var(*xfind+LaTeX_subscript(i), *xfind+"_0");
   }else positiondrfs = (*xfind+"_i")|indexrng;
  
  physquantity biggestx = maxval(fnplot, xfind),
               smallestx = minval(fnplot, xfind),
               rasterx = (biggestx-smallestx)/fnplot.size(),
               noiselvl = RMSofElem(fnplot, noisefind);

  fitdist_fntomeasures d(fnplot, fgfn, Afind);

  evolution_minimizer fitthisback( d
                           // prohibit left runout
                                 , enforce_each_difference_smaller_than( positiondrfs,
                                                         /*     minus */ widthdrfs,
                                                         /* less than */ biggestx )
                           // prohibit right runout
                                && enforce_each_sum_bigger_than( positiondrfs,
                                                 /*      plus */ widthdrfs,
                                                 /* more than */ smallestx )
                           // prohibit squeeze between samples
                                && enforce_each_bigger_than( widthdrfs, 2*rasterx )
                           // prohibit vanish in noise
                                && enforce_each_bigger_than( heightdrfs, noiselvl*2 )
                           // prohibit flatten over range
                                && enforce_each_smaller_than( widthdrfs, biggestx-smallestx )
                                 , evolution_minimizer::solutioncertaintycriteria::doublevalue()
                                 );
  
//  #endif
  return fitthisback.result();
}


  // determine noise level as non-corellation between following events
captfinder add_noisetrack( measureseq &signal       // mutable-signal version
                         , const captfinder &Afind
                         ) {
  captfinder noisefind("\\nu_{"+*Afind+"}");
  physquantity lowpass6 = abs(Afind(signal[1]) - (Afind(signal[0])+Afind(signal[2]))/2);
  lowpass6.label(*noisefind);
  signal[0].push_back(lowpass6);
  double lp_coeff = 1/sqrt(sqrt(signal.size()));
  for (unsigned i = 1; i<signal.size()-1; ++i) {
    physquantity noisehere = abs(Afind(signal[i]) - (Afind(signal[i-1])+Afind(signal[i+1]))/2);
    lowpass6 += lp_coeff * noisehere;
    lowpass6 /= (1+lp_coeff);
    signal[i].push_back(lowpass6);
  }
  signal.back().push_back(lowpass6);
  for (unsigned i = signal.size()-1; i-->0;) {
    physquantity noisehere = noisefind(signal[i]);
    lowpass6 += lp_coeff * noisehere;
    lowpass6 /= (1+lp_coeff);
    noisefind(signal[i]) = lowpass6;
  }
  return noisefind;
}
measureseq noisetrack( const measureseq &signal   // pure / non-mutable-signal version
                     , const captfinder &Afind
                     ) {
  assert(1==0);  // TODO: either remove this function or get it to work
  measureseq diffrs;
  
  std::string diffslabel = Afind(signal[0]).cptstr() + "<diff>",
              vtimelabel = "<virtual_time_variable>";

  for (unsigned i = 1; i<signal.size(); ++i) {
    diffrs.push_back(
              measure (
                        ( Afind(signal[i]) - Afind(signal[i-1]) ).wlabel(diffslabel),
                        ( i*real1 ).wlabel(vtimelabel)
                      )
                    );
  }
  return diffrs;
}
void noise_as_uncertainty( measureseq &signal
                         , const captfinder &Afind
                         ) {
  measureseq noisy = signal;
  captfinder noisefind = add_noisetrack(noisy, Afind);
  for (unsigned i=0; i<signal.size(); ++i){
    Afind(signal[i]).error(noisefind(noisy[i]));
  }

}




#if 0
bool spectrumfitoutcome_compatible( const ptlfit_outcome &l
                                  , const ptlfit_outcome &r
                                  , const captfinder &xfind
                                  , const physquantity ) {
  
}
#endif

template<class MultiPeakFn = fittable_multigaussianfn>
class spectrumfitter {
  captfinder xfind, Afind, noisefind;
  spectfit_linkingt linkings;
  measureseq start_fnplot;
#if 0
  struct ptlfit_outcome {
    measureseq *peaksptr;
    bool failed() {return peaksptr==NULL;}
    measureseq &peaks(){
      if(failed()) {
        cerr << "Tried to get the peaks from an unsuccessful ptlfit_outcome!";
        abort();
      }
      return *peaksptr;
    }

    ptlfit_outcome(): peaksptr(NULL) {}
    ptlfit_outcome(const measureseq &pks): peaksptr(new measureseq(pks)) {
//      cout<<"Copied measureseq to ptlfit_outcome:\n"/*<<peaks()*/<<"(to address "<<peaksptr<<")\n";
    }
    ptlfit_outcome(ptlfit_outcome &&cp): peaksptr(cp.peaksptr) {
      cp.peaksptr=nullptr;
//      if(peaksptr)cout<<"Move-Copied ptlfit_outcome with measureseq:\n"/*<<peaks()*/;else cout<<"Move-copied failure ptlfit_outcome\n";
    }
    ptlfit_outcome &operator=(const ptlfit_outcome &cp) {
      delete peaksptr;
      peaksptr = cp.peaksptr;
//      if(peaksptr)cout<<"Assigned ptlfit_outcome with measureseq:\n"/*<<peaks()*/;else cout<<"Assigned failure ptlfit_outcome\n";
      return *this;
    }
    ptlfit_outcome &operator=(ptlfit_outcome &&cp) {
      delete peaksptr;
      peaksptr = cp.peaksptr;
      cp.peaksptr=nullptr;
//      if(peaksptr)cout<<"Move-assigned ptlfit_outcome with measureseq:\n"/*<<peaks()*/;else cout<<"Move-assigned failure ptlfit_outcome\n";
      return *this;
    }
    ptlfit_outcome(const ptlfit_outcome &cp):peaksptr(cp.peaksptr) {
      if(peaksptr) peaksptr = new measureseq(*peaksptr);
//      if(peaksptr)cout<<"Copied ptlfit_outcome with measureseq:\n"/*<<peaks()*/<<"(old address: "<<cp.peaksptr<<", new address: "<<peaksptr<<")\n";else cout<<"Copied failure ptlfit_outcome\n";
    }
    ~ptlfit_outcome(){
//      if(peaksptr)cout<<"Deleting ptlfit_outcome with measureseq:\n"/*<<peaks()*/<<"(address: "<<peaksptr<<")\n";else cout<<"Deleting ptlfit_outcome with no peaks.\n";
      delete peaksptr;
    }
  };
  ptlfit_outcome ptlfit_failure() {return ptlfit_outcome();}
#else
  typedef maybe<measureseq> ptlfit_outcome;
#endif
  static const unsigned minimum_spectrumfit_mpoints = 96
                      , min_spectrumfit_mpoints_per_peak = 64;
 
  struct delegate_id {
    int i;
    delegate_id &operator++(){i+=2;return*this;}
    bool operator!=(const delegate_id &other) const{return i!=other.i;}
    delegate_id other(){return delegate_id{-i};}
    bool ok(){return i<2;}
  };
  static delegate_id delegation_l(){return delegate_id{-1};}
  static delegate_id delegation_r(){return delegate_id{1};}
  static delegate_id delegate_end(){return delegate_id{3};}
  template<typename DlgDat>
  struct delegations {
    DlgDat l, r;
    DlgDat &operator[](delegate_id i){return i.i<0? l : r;}
    struct iterator{
      delegations<DlgDat> *domain; delegate_id i;
      DlgDat &operator*(){return (*domain)[i];}
      iterator &operator++(){++i; return *this;}
      bool operator!=(const iterator &other) const{return i!=other.i;}
    };
    iterator begin(){return iterator{this,delegation_l()};}
    iterator end(){return iterator{this,delegate_end()};}
  };
  struct identify_peaksrepmatch_canditate_rankcmp{
    const measure *basemsr; captfinder xfind;
    identify_peaksrepmatch_canditate_rankcmp &operator=(const identify_peaksrepmatch_canditate_rankcmp &cpf){
      basemsr=cpf.basemsr; xfind=cpf.xfind; return *this;
    }
    physquantity badness(const measure &x)  {
      return abs(xfind(x)-xfind(*basemsr));  }
    bool operator()(const measure *l, const measure *r) {
      return badness(*l)<badness(*r);                    }
  };
  
  static std::string indent(int nestdepth){return string(nestdepth*2,' ');}

  bool use_multithreads(){return false;}
 public:

  
                               // fnplot must be sorted so that xfind(front)≤xfind(...)≤xfind(back) and with noise level track
  ptlfit_outcome fit_spectrum_delegations(const measureseq &fnplot, int nestdepth) {
    
    const double intersection_size = 1./3;
    if (fnplot.size() * (1-intersection_size) < minimum_spectrumfit_mpoints) return nothing;
    
    delegations<measureseq::const_iterator> intersection_itbords{
        fnplot.begin()+fnplot.size()*(.5-intersection_size/2)
      , fnplot.begin()+fnplot.size()*(.5+intersection_size/2)
    };
    unsigned intersect_cardinality = intersection_itbords.r - intersection_itbords.l;
    auto splitpoint = intersection_itbords.l + intersect_cardinality/2;
    for ( double gapfind = .1
        ; xfind(*splitpoint)>noisefind(*splitpoint)
         && abs(gapfind)<1
        ; gapfind *= -((double) intersect_cardinality+2)/intersect_cardinality )
      splitpoint = intersection_itbords.l + (int)(
         intersect_cardinality * (1 + gapfind)/2 );

    if (xfind(*splitpoint)>noisefind(*splitpoint)) return nothing;
    
    delegations<measureseq> sections {
        measureseq(fnplot.begin(), splitpoint+intersect_cardinality/4)
      , measureseq(splitpoint-intersect_cardinality/4,   fnplot.end())
    };
    delegations<physquantity> intvborders { xfind(fnplot.front())
                                         , xfind(fnplot.back())     };
    delegations<physquantity> spborders { xfind(sections.r.front())
                                        , xfind(sections.l.back())  };
    physquantity intersect_width = spborders.r - spborders.l;
#ifdef SPECTRUMFITTER_PROCESSMESSAGES
    cout << indent(nestdepth) << "delegations with borders ";
    for (auto b : spborders) cout << b << ", ";
    cout << endl;
#endif
    delegations<ptlfit_outcome> delegatedwn{nothing,nothing};

    if (use_multithreads()) {
      cerr<<"Multithreading for spectrumfit not implemented yet.\n";abort();
     }else{
      for (auto       d:fit_spectrum_process(sections.l, nestdepth+1)) {
        delegatedwn.l = just(d);
        delegatedwn.r = fit_spectrum_process(sections.r, nestdepth+1);
      }
    }
    for(auto s : delegatedwn)
      if(s == nothing) return nothing;
#ifdef SPECTRUMFITTER_PROCESSMESSAGES
    cout << indent(nestdepth) << "both delegations succeded, try to match them...\n";
#endif

    delegations<phq_interval> prefIntvs{
        phq_interval(intvborders.l-intersect_width, spborders.l+intersect_width/2)
      , phq_interval(spborders.r-intersect_width/2, intvborders.r+intersect_width)
    };

    ptlfit_outcome result(measureseq(0));
    
    for (auto side=delegation_l(); side.ok(); ++side) {
      for (auto p : *delegatedwn[side]) {
        if (prefIntvs[side].strictly_includes(xfind(p)))
          (*result).push_back(p);
      }
    }
#if 0    
    typedef const measure* Cmsrp;
    typedef identify_peaksrepmatch_canditate_rankcmp idccmp;
    delegations< std::vector< std::set< Cmsrp, idccmp > > > intersection_pks;
    for (auto side=delegation_l(); side.ok(); ++side) {
      for (auto i : *delegatedwn[side])
        if (xfind(i)>spborders.l && xfind(i)<spborders.r)
          intersection_pks[side].push_back(std::set<Cmsrp,idccmp>(
            idccmp{&i, xfind}
          ));
    }
  /*
    for (auto i in delegatedwnr.peaks())
      if (xfind(*i)>liborder && xfind(*)<riborder) intersect_pks_r.push_back(make_pair(*i, nullptr));
  */
    
    for (auto side=delegation_l(); side.ok(); ++side) {
      for (auto i : intersection_pks[side]) {
        for (auto j : *delegatedwn[side.other()]) {
          i.insert(&j);
        }
      }
    }
    std::map<Cmsrp,Cmsrp> agreed;
    for (auto i : intersection_pks.l) {
      bool agrees=true;
      for (auto j : intersection_pks.r) {
        if ( (*i.begin() == j.key_comp().basemsr)
           ^ (*j.begin() == i.key_comp().basemsr) ) {
          agrees=false;
          break;
        }
      }
      if (agrees)
        agreed[*i.begin()] = i.key_comp().basemsr;
    }
    for (auto i : intersection_pks.r) {
      if (agreed.find(i.key_comp().basemsr) == agreed.end())
        agreed[*i.begin()] = i.key_comp().basemsr;
    }
    std::set<Cmsrp> all_intersect_finds;
    
    for (auto i : agreed) {
      all_intersect_finds.insert(i.first); all_intersect_finds.insert(i.second); }
    
    ptlfit_outcome result(measureseq(0));
    
    for(auto s : delegatedwn){
      for(auto i : *s){
        if (all_intersect_finds.find(&i)==all_intersect_finds.end())
          (*result).push_back(i);
      }
    }
    
    for (auto i : agreed) {
      if (xfind(*i.first).error() < xfind(*i.second).error())
        (*result).push_back(*i.first);
       else
        (*result).push_back(*i.second);
    }
#endif
    if (spectrumfit_is_ok(fnplot, *result)) {
#ifdef SPECTRUMFITTER_PROCESSMESSAGES
    cout << indent(nestdepth) << "matching of delegations successful.\n";
#endif
      return result;
     }else{
#ifdef SPECTRUMFITTER_PROCESSMESSAGES
    cout << indent(nestdepth) << "matching of delegations resulted in unsuitable fit function.\n";
#endif
      return nothing;
    }
  }

  bool spectrumfit_is_ok(const measureseq &fnplot, const measureseq &fittedpeaks) {
    captfinder wfind(spectrpeakwidth_name);
    unsigned npeaks = fittedpeaks.size();
    fittable_multigaussianfn fgfn(npeaks);

    LaTeXindex_itrange indexrng = LaTeXindex("i").from(0).unto(npeaks);
    fgfn.rename_var("x, x_i, A_i", *xfind + ", " + *xfind+"_i, " + *Afind+"_i", indexrng);

    measure testprm;
    for (unsigned i=0; i<npeaks; ++i){
      testprm.push_back(xfind(fittedpeaks[i]).wlabel(*xfind+LaTeX_subscript(i)));
      testprm.push_back(Afind(fittedpeaks[i]).wlabel(*Afind+LaTeX_subscript(i)));
      testprm.push_back(wfind(fittedpeaks[i]).wlabel(*wfind+LaTeX_subscript(i)));
    }
    testprm.push_back(0);
    physquantity fitdists=0, noises=0;
    for (auto m : fnplot) {
      if (Afind(m).positive()) {
        testprm.back() = xfind(m);
        fitdists += (fgfn(testprm) - Afind(m)).squared();
      }
      noises   += noisefind(m).squared();
    }

    return fitdists < noises;
    
  }

  ptlfit_outcome fit_spectrum_process(const measureseq &fnplot, int nestdepth) {
#ifdef SPECTRUMFITTER_PROCESSMESSAGES
    cout << indent(nestdepth) << "trying to delegate fit of range "
                 << fnplot.range_of(xfind) << "...\n";
#endif
    for (int i=0; i<1; ++i)                                         // try delegation twice since
      for (auto r : fit_spectrum_delegations(fnplot, nestdepth+1)) //  fails may be flukes, if either
        return just(r);                                           //   succeeds return its result.
#ifdef SPECTRUMFITTER_PROCESSMESSAGES
    cout << indent(nestdepth) << "since fit delegation failed, fit whole range "
                 << fnplot.range_of(xfind) << "...\n";
#endif
    ptlfit_outcome result(measureseq(0));
    for ( unsigned npeaks = 0
        ; npeaks*min_spectrumfit_mpoints_per_peak+minimum_spectrumfit_mpoints<=fnplot.size()
        ; ++npeaks) {
#ifdef SPECTRUMFITTER_PROCESSMESSAGES
      cout << indent(nestdepth)<<"            "<<npeaks<<"peaks...\n";
#endif
      if (npeaks>0)
        *result = ffit_spectrum<MultiPeakFn>
                        ( fnplot, xfind, Afind, noisefind, npeaks, linkings )
                             .pack_subscripted();
      if(spectrumfit_is_ok(fnplot, *result)) {
#ifdef SPECTRUMFITTER_PROCESSMESSAGES
        cout << indent(nestdepth) << "success, result is: (...)\n";// << *result;
#endif
        return result;
      }
    }
#ifdef SPECTRUMFITTER_PROCESSMESSAGES
    cout << indent(nestdepth) << "process-fit in range failed\n";
#endif
    return nothing;
  }
 public:
  spectrumfitter( const measureseq &fnplot
                , const captfinder &xfind
                , const captfinder &Afind
                , spectfit_linkingt linkings = 0x0 )
    : xfind(xfind)
    , Afind(Afind)
    , linkings(linkings)
    , start_fnplot(fnplot) {
    noisefind = add_noisetrack(start_fnplot, Afind);
    std::sort( start_fnplot.begin(), start_fnplot.end()
             , [xfind](const measure &m1, const measure &m2)
                 { return xfind(m1) < xfind(m2); }
             );
  }
  measureseq result() {
    for(auto res : fit_spectrum_process(start_fnplot,0)) return res;
#ifdef SPECTRUMFITTER_PROCESSMESSAGES
 //   cout << "FINAL result:\n(@ address " << res.peaksptr << ")\n" << res.peaks();
#endif
    cerr << "Spectrum fit failed.\n";
    abort();
    return result();
  }
};

template<class MultiPeakFn = fittable_multigaussianfn>
measureseq fit_spectrum( const measureseq fnplot
                       , const captfinder &xfind
                       , const captfinder &Afind
                       , spectfit_linkingt linkings = 0x0
                       ) {
  spectrumfitter<MultiPeakFn> fitter(fnplot,xfind,Afind,linkings);
  return fitter.result();
}



#if 0
class montecarlofit {      //use deprecated at the moment
  measureseq ftgt;
  const fittable_phmsqfn &f;
  measure fitres;
  const msq_dereferencer &returndrf;
  
  fitdist_fntomeasures distancer;

  typedef std_normaldistributed_randomgen ranGenT;
  ranGenT &r;



  measure errorsmoothproximitygradientandval (const measure &fpm) {
                                  //Gradient and value of the scalar field decribing the square
                                 // distance of a fitting function from some measure data,
                                //  weightedly averaged over some range of fitting parameters
    measure result(fpm.size()+1, 0);

    for (measureseq::const_iterator i=ftgt.begin(); i!=ftgt.end(); ++i){
      measure wkd = fpm;
      for (measure::iterator j = wkd.begin(); j!=wkd.end(); ++j)
        (*j += r() * j->error()).error(0);
      wkd.append(*i);
      physquantity thisret = returndrf(*i),
                   thiscmperr = thisret.error(); //thiscmperrsq=thiscmperrsq.abs(); //thiscmperrsq.squareme();
  //    result += ( (thisret - f(wkd)) / thiscmperr ).squareme().dbl();
      physquantity sqproxhere = thiscmperr / (( thisret.seterror(0).minusexactly(f(wkd)) ).abs() + thiscmperr);
      for (unsigned j = 0; j<fpm.size(); ++j){
        if (fpm[j].error()>0) {
          physquantity drvdisturbmem=wkd[j],
                       d = fpm[j].error() * (r()*(1./50));
          while (d==0) d = fpm[j].error() * (r()*(1./50));
          wkd[j] += d;
//cout << ( ( ( thisret.minusexactly(f(wkd)) ).squared() ) - sqdisthere ) / ( d *= thiscmperrsq ) << endl;
///cout << result[j] << endl;
          physquantity sqpt = thiscmperr / (( thisret.minusexactly(f(wkd)) ).abs() + thiscmperr);
          result[j] += (sqpt - sqproxhere) / d;
          wkd[j] = drvdisturbmem;
        }
      }
      result[fpm.size()] += sqproxhere;
    }
    return result;                                     
  }


  measure errorsmoothsqrtdistancegradientandval (const measure &fpm) {
                                  //Gradient and value of the scalar field decribing the square
                                 // distance of a fitting function from some measure data,
                                //  weightedly averaged over some range of fitting parameters
    measure result(fpm.size()+1, 0);

    for (measureseq::const_iterator i=ftgt.begin(); i!=ftgt.end(); ++i){
      measure wkd = fpm;
      for (measure::iterator j = wkd.begin(); j!=wkd.end(); ++j)
        (*j += r() * j->error() / 3).error(0);
      wkd.append(*i);
      physquantity thisret = returndrf(*i),
                   thiscmperrsq = thisret.error(); thiscmperrsq=thiscmperrsq;//.sqrt();
  //    result += ( (thisret - f(wkd)) / thiscmperr ).squareme().dbl();
      physquantity sqdisthere = ( thisret.seterror(0).minusexactly(f(wkd)) ).abs();//.sqrt();
      for (unsigned j = 0; j<fpm.size(); ++j){
        if (fpm[j].error()>0) {
          physquantity drvdisturbmem=wkd[j],
                       d = fpm[j].error() * (r()*(1./50));
          while (d==0) d = fpm[j].error() * (r()*(1./50));
          wkd[j] += d;
//cout << ( ( ( thisret.minusexactly(f(wkd)) ).squared() ) - sqdisthere ) / ( d *= thiscmperrsq ) << endl;
///cout << result[j] << endl;
          result[j] += ( ( ( thisret.minusexactly(f(wkd)) ).abs() ) - sqdisthere )
                     / ( d *= thiscmperrsq );
          wkd[j] = drvdisturbmem;
        }
      }
      result[fpm.size()] += (sqdisthere /= thiscmperrsq);
    }
    return result;                                     
  }


  static const double gradclcspread = 3./5;

  measure errorsmoothsqdistancegradientandvalandnodiv (const measure &fpm, bool no_div=false) {
                                  //Gradient and value of the scalar field decribing the square
                                 // distance of a fitting function from some measure data,
                                //  weightedly averaged over some range of fitting parameters
    std::vector<double> resultbins(fpm.size()*2+1, 2), resbinnormlzs(fpm.size()*2+1, 2);
    measureseq resbinctrs(fpm.size()*2+1, measure(fpm.size()));
    
    for (unsigned j=0; j<resbinctrs.size(); ++j) {
      for (unsigned k=0; k<fpm.size(); ++k) {
        resbinctrs[j][k] = fpm[k].werror(0);
//    cout << " fbg " << resbinctrs[j][k] << ", ";
        if (k == j)
          resbinctrs[j][k] += fpm[k].error() * (1-gradclcspread);
         else if (k+fpm.size() == j)
          resbinctrs[j][k] -= fpm[k].error() * (1-gradclcspread);
  //  cout << " fin " << resbinctrs[j][k] << ", ";
      }
//      cout << endl;
    }

    std::vector<double> ranperturbs(fpm.size());
    double ttldistsq;


    while (resbinnormlzs.back() < 64.) {
      measureseq::const_iterator i=ftgt.begin() + size_t(float(rand())*ftgt.size() / RAND_MAX);
      measure wkd = fpm;
      if (no_div) {
        do {
          ttldistsq = 0;
          for (unsigned j = 0; j<fpm.size(); ++j) {
            ranperturbs[j] = r();
            ttldistsq += ranperturbs[j]*ranperturbs[j];
          }
        } while (ttldistsq > gradclcspread*gradclcspread);
        for (unsigned j = 0; j<fpm.size(); ++j)
          (wkd[j] += ranperturbs[j] * wkd[j].error()).error(0);
       }else{
        for (measure::iterator k = wkd.begin(); k!=wkd.end(); ++k)
          (*k += r() * k->error()).error(0);
      }
      wkd.append(*i);
      physquantity thisret = returndrf(*i),
                   thiscmperr = thisret.error();  // thiscmperrsq=thiscmperrsq.squareme();
 /*   dbgbgn;
      cout << "thisret: " << thisret << endl;
      cout << "f(wkd): " << f(wkd) << endl;
      cout << "thiscmperr: " << thiscmperr << endl; */
      double sqdisthere = ( thisret.seterror(0).minusexactly(f(wkd)) / thiscmperr ).squared().dbl();
//    dbgend;

      for (unsigned j = no_div? resbinctrs.size()-1 : 0; j<resbinctrs.size(); ++j) {
        double thisreldistsq = 0;
        if (no_div) {
          thisreldistsq = ttldistsq;
         }else{
          for (unsigned k=0; k<fpm.size(); ++k) {
            thisreldistsq += ( (wkd[k] - resbinctrs[j][k]) / fpm[k].error() ).squared().dbl();
          }
          thisreldistsq /= gradclcspread*gradclcspread;
        }
   //       cout << "(" << (wkd[k] - resbinctrs[j][k]) << " / " << fpm[k].error() << ")²";
     //     cout << " = " << ( (wkd[k] - resbinctrs[j][k]) / fpm[k].error() ).squared() << endl;
    //    cout << thisreldistsq << " >-=> ";
        double thisweight = Friedrichs_mollifier_onsquares(thisreldistsq);
        resbinnormlzs[j] += thisweight;
      //  cout << thisweight << endl;
        resultbins[j] += sqdisthere*thisweight;
      }

    }

    measure result;
    
    for (unsigned k=0; k<fpm.size(); ++k) {
 //     cout << resultbins[k] << "/" << resbinnormlzs[k] << " - " << resultbins[k+fpm.size()] << "/" << resbinnormlzs[k+fpm.size()];
   //   cout << " = " << resultbins[k]/resbinnormlzs[k] - resultbins[k+fpm.size()]/resbinnormlzs[k+fpm.size()] << endl;
      result.push_back(
              ( resultbins[k]/resbinnormlzs[k] - resultbins[k+fpm.size()]/resbinnormlzs[k+fpm.size()] )
             /( resbinctrs[k][k] - resbinctrs[k+fpm.size()][k] )
      );
    }
    result.push_back( resultbins[fpm.size()*2] * real1 / resbinnormlzs[fpm.size()*2] );

    return result;                                     

/*
      for (int j = 0; j<fpm.size(); ++j){
        if (fpm[j].error()>0) {
          physquantity drvdisturbmem=wkd[j],
                       d = fpm[j].error() * (r()*(1./50));
          while (d==0) d = fpm[j].error() * (r()*(1./50));
          wkd[j] += d;
//cout << ( ( ( thisret.minusexactly(f(wkd)) ).squared() ) - sqdisthere ) / ( d *= thiscmperrsq ) << endl;
///cout << result[j] << endl;
          result[j] += ( ( ( thisret.minusexactly(f(wkd)) ).squared() ) - sqdisthere )
                     / ( d *= thiscmperrsq );
          wkd[j] = drvdisturbmem;
        }
      }
      result[fpm.size()] += (sqdisthere /= thiscmperrsq);
    }
*/
  }


  measure errorsmoothsqdistancegradientandval (const measure &fpm) {
                                  //Gradient and value of the scalar field decribing the square
                                 // distance of a fitting function from some measure data,
                                //  weightedly averaged over some range of fitting parameters
    measure result(fpm.size()+1, 0);

    for (measureseq::const_iterator i=ftgt.begin(); i!=ftgt.end(); ++i){
      measure wkd = fpm;
      for (measure::iterator j = wkd.begin(); j!=wkd.end(); ++j)
        (*j += r() * j->error() / 36).error(0);
      wkd.append(*i);
      physquantity thisret = returndrf(*i),
                   thiscmperrsq = thisret.error(); thiscmperrsq=thiscmperrsq.squareme();
  //    result += ( (thisret - f(wkd)) / thiscmperr ).squareme().dbl();
      physquantity sqdisthere = ( thisret.seterror(0).minusexactly(f(wkd)) ).squared();
      for (unsigned j = 0; j<fpm.size(); ++j){
        if (fpm[j].error()>0) {
          physquantity drvdisturbmem=wkd[j],
                       d = fpm[j].error() * (r()*(1./50));
          while (d==0) d = fpm[j].error() * (r()*(1./50));
          wkd[j] += d;
//cout << ( ( ( thisret.minusexactly(f(wkd)) ).squared() ) - sqdisthere ) / ( d *= thiscmperrsq ) << endl;
///cout << result[j] << endl;
          result[j] += ( ( ( thisret.minusexactly(f(wkd)) ).squared() ) - sqdisthere )
                     / ( d *= thiscmperrsq );
          wkd[j] = drvdisturbmem;
        }
      }
      result[fpm.size()] += (sqdisthere /= thiscmperrsq);
    }
    return result;                                     
  }



  physquantity errorsmoothsqdistance (const measure &fpm) {
                                  //Value of the smooth square distance scalar field.
                                 //    Yet to be optimized!
    return errorsmoothsqdistancegradientandvalandnodiv(fpm, true)[fpm.size()];                                     
  }



  physquantity straightforwardsqdistance (const measure &fpm) {
                                  //The square
                                 // distance of the fitting function from some measure data,
                                //  weightedly averaged over some range of fitting parameters
    physquantity result = 0;
    for (measureseq::const_iterator i=ftgt.begin(); i!=ftgt.end(); ++i){
      measure wkd = fpm;
      wkd.remove_errors().append(*i);
      physquantity thisret = returndrf(*i),
                   thiscmperrsq = thisret.error();// thiscmperrsq.squareme();
  //    result += ( (thisret - f(wkd)) / thiscmperr ).squareme().dbl();

      physquantity sqdisthere = thisret.seterror(0);
      sqdisthere.SubstractAsExactVal(f(wkd));

      result += (sqdisthere /= thiscmperrsq).squared();
    }
    return result;                                     
  }




  measure errorsmoothsuggestforminimizingsqdistance(const measure &fpm) {
  
    for (double plausibleproxdist = 1./4; plausibleproxdist>0; plausibleproxdist/=4){
    
      cout << endl << "State now:\n" << fpm << endl;
  
      measure direction = errorsmoothsqdistancegradientandvalandnodiv (fpm);

      physquantity sqdisthere = errorsmoothsqdistance(fpm);  direction.pop_back();
      double normlz = 0;
      for (unsigned i=0; i < fpm.size(); ++i){
//dbgbgn;
//      cout << direction[i] << " * -" << fpm[i].error() << " = " << direction[i] * -fpm[i].error() << endl;
        normlz += ( direction[i] *= fpm[i].error() ).squared().dbl();
  //dbgend;
        direction[i] *= -fpm[i].error();    //calculate optimizing direction from gradient
      }
      normlz = sqrt(normlz);

    direction.multiply_all_by(1/normlz);
    cout << "Proposed change:\n" << direction << endl;
    direction.multiply_all_by(normlz);
    fpm.by_i_plus_multiple_of(direction, 1/normlz);

      std::vector<physquantity> sqdisttries( 1,
                 errorsmoothsqdistance(fpm.by_i_plus_multiple_of(direction, 1/normlz) ) );
      double epsilon=1;
    
                                
      while (epsilon > plausibleproxdist) {
        epsilon /= sqrt(2);
        physquantity lasttry = sqdisttries.back();
        sqdisttries.push_back( errorsmoothsqdistance(fpm.by_i_plus_multiple_of(direction, epsilon/normlz) ) );
        if ( sqdisttries.back() < sqdisthere ) {
   //       cout << sqdisttries.back() << " < " << sqdisthere << endl;
          if ( sqdisttries.back() < lasttry )
//{        cout << "also " << sqdisttries.back() << " < " << lasttry << endl;
            return fpm.by_i_plus_multiple_of(direction, epsilon/normlz);
//}
  //   cout << "but not " << sqdisttries.back() << " < " << lasttry << endl;
          return fpm.by_i_plus_multiple_of(direction, sqrt(2) * epsilon/normlz);
         }else{
   //       cout << sqdisttries.back() << " >= " << sqdisthere << endl;

        }
      }
    
    }
    
    cerr << "Weird error in monte carlo fit"; abort();

  }


  void enforcenormalizableerrors() {

    bool allhaveerror=true;
    measureseq::iterator i = ftgt.begin();
    
    physquantity minimumabsval = abs(returndrf(*i));
    while (minimumabsval==0) {
      minimumabsval = abs(returndrf(*(i++)));
      if (i==ftgt.end()) {
        cerr << ftgt << "(measureseq size: " << ftgt.size()
             << ")\nWhy would you want to fit some function to a measure sequence that is 0 everywhere?";
        abort();
      }
    }
    
    for (i = ftgt.begin(); i!=ftgt.end(); ++i){
      if (returndrf(*i).error()==0) allhaveerror=false;
      if (returndrf(*i)!=0) minimumabsval.push_downto(abs(returndrf(*i)));
    }

    if (!allhaveerror) {
      minimumabsval/=2;
      i = ftgt.begin();
      while (i!=ftgt.end()) {
        if (returndrf(*i).error()==0) returndrf(*(i++)).error(minimumabsval);
         else i = ftgt.erase(i);
      }
    }

  }
  
  float makesureseesminimum(measure &fpm, int id, float safetyoverwall = 0) {
    physquantity minimumsqd = straightforwardsqdistance(fpm);

    measure uerrbrd=fpm, lerrbrd=fpm;
    uerrbrd[id] = fpm[id].u_errorbord();
    lerrbrd[id] = fpm[id].l_errorbord();

    physquantity usqd = straightforwardsqdistance(uerrbrd), lsqd = straightforwardsqdistance(uerrbrd);
    
    while ( usqd < minimumsqd && lsqd < minimumsqd ) {
      fpm[id] = fpm[id] + fpm[id].error() * r();
      minimumsqd = straightforwardsqdistance(fpm);
      uerrbrd[id] = fpm[id].u_errorbord();
      lerrbrd[id] = fpm[id].l_errorbord();
      usqd = straightforwardsqdistance(uerrbrd); lsqd = straightforwardsqdistance(lerrbrd);
      cout << fpm << endl;
    }

    float inflated = 1, inflator = 1;
    physquantity initerr = fpm[id].error();

    while (true) {
      
      if ( usqd < minimumsqd * (safetyoverwall+1) || lsqd < minimumsqd * (safetyoverwall+1)) {
        if (inflated < 16/inflator) {
          fpm[id].scale_error(pow(2,inflator));
          inflated *= pow(2,inflator);
          cout << "infltt: " << inflated << endl;
          cout << "infltr: " << inflator << endl;
          cout << fpm[id] << endl;
         }else{
          inflated = 1;
          inflator = 1 / ((1/inflator) + 3);
          if (inflator < .015) {
            cerr << "inflator < .015 (for fit variable " << fpm[id].cptstr() << ")\n";
            return -1;
          }
          fpm[id].error(initerr);
        }
//        safetyoverwall /= sqrt(2);
   //     for ( int brake=0; brake<328; brake+=(1 + (0*straightforwardsqdistance(uerrbrd)).dbl()) ) {}
     //   cout << fpm << endl;
        uerrbrd[id] = fpm[id].u_errorbord();
        lerrbrd[id] = fpm[id].l_errorbord();
        usqd = straightforwardsqdistance(uerrbrd); lsqd = straightforwardsqdistance(lerrbrd);

       }else{
        return inflated;
      }


    }
  }
  
  
 public:


  measure result() { return fitres; }
  
  double fit_badness() {
    return errorsmoothsqdistance(result()).dbl() / ftgt.size();
  }




  static const int fitoptimizationsteps = 48;

  measure fittomsqseq(measure fitstate) {
  
    enforcenormalizableerrors();

    cout << "Badness: " << errorsmoothsqdistance(fitstate).dbl() / ftgt.size() << endl;

    if (fitstate.size()==0) return fitstate;

    measure moved(fitstate.size(), 0), lastmoves;

    for (int i = 0; i<fitoptimizationsteps; ++i){
    
      for (measure::iterator j=fitstate.begin(); j!=fitstate.end();){   //remove exactly-fixed
        if (j->error() == 0) {                                         // parameters from fitting
          for (measureseq::iterator k=ftgt.begin(); k!=ftgt.end(); ++k){
            k->push_back(*j);
          }
          moved.erase(j);
          j = fitstate.erase(j);
         }else{
          ++j;
        }
      }
    
      measure improved = errorsmoothsuggestforminimizingsqdistance(fitstate);
      lastmoves=moved;
      moved = improved.by_i_minus(fitstate);
      double relmoved=0;
      for (unsigned j = 0; j<fitstate.size(); ++j) {
        relmoved += ( moved[j]/fitstate[j].error() ).squared().dbl();
      }
      relmoved = sqrt(relmoved);
      cout << "Moved relative to error margin: " << relmoved << endl;
      
      for (unsigned j = 0; j<fitstate.size(); ++j) {
        if (moved[j]==0) fitstate[j] = improved[j].werror(0);
         else{
          physquantity newuncertainty = fitstate[j].error() * (relmoved*relmoved + .25)  //.98; //
                + (moved[j]+lastmoves[j]).squared()/(2*fitstate[j].error());
          fitstate[j] = improved[j].werror(newuncertainty);
        }
      }
      
      if ( float(rand()) * i * fitstate.size() / fitoptimizationsteps / RAND_MAX > .5 ) {
        if (makesureseesminimum(fitstate, float(rand()) * fitstate.size() / RAND_MAX) < 0) return fitstate;
      }
      
      cout << "Badness: " << errorsmoothsqdistance(fitstate).dbl() / ftgt.size() << endl;
    }
    return fitstate;
  
    /*struct mranmv {    //random generator for monte carlo deviations
      physquantity operator() (physquantity sd){
        return (sd *= (rand() - RAND_MAX/2)) /= (RAND_MAX/2);
      }
    } m_ranfn;*/

 /*   typedef std::vector<captfinder> derefarr;
    derefarr derefs;
    for (measure::const_iterator i=result.begin(); i!=result.end(); ++i)
      derefs.push_back(i->caption);  */
      
    
    
    
  
  /*  std::string preretvartempname="retvar", rnm=prretvartempname;
    for {int ui=0; result.has(retvartempname); ++ui}{
      rnm=(std::stringstream(preretvartempname)<<ui).str();
    }
    captfinder rdrf(rnm);*/
  
  #if 0
    for (measureseq::iterator i=fsrc.begin; i!=fsrc.end(); ++i)
      i->append(result).append(constraints);
  
  
    for (int i=0; i<montecarlofititerations; ++i){
      
      for (derefarr::iterator j=derefs.begin(); j!=derefs.end(); ++j){
        measureseq tresult=result;
        for (derefarr::iterator l=derefs.begin(); l!=derefs.end(); ++l)
          (*l)(tresult) = (*l)(result).error(0);
        measureseq eval(0);
        for (int k=0; k<montecarlofittrialpoints && k<fsrc.size(); ++k){
          measure *w = &fsrc.randommeasure();
          (*j)(tresult) = (*j)(result).error(0) + m_ranfn((*j)(result).error());
          for (derefarr::iterator l=derefs.begin(); l!=derefs.end(); ++l){
            (*l)(*w) = (*l)(tresult);
          }
          eval.push_back(measure(
            (*j)(tresult),
            (retdrf(*w) - f(*w)).squared()
          ));
        } 
      }
    }
  #endif

    return fitstate;
  }


  /*montecarlofit(const measureseq &fsrc, const msq_dereferencer &retdrf,
                               const fittable_phmsqfn &fn, //const measure &startparams,
                               measure constraints=measure(),
                               ranGenT &rg = defaultsimplenormaldistributedrandomgenerator):
    ftgt(fsrc),
    f(fn),
    returndrf(retdrf),
    r(rg),
    distancer(fsrc, fn, retdrf) {//defaultsimplenormaldistributedrandomgenerator)                               {
    measure cconstraints = constraints;
    cconstraints.append( fsrc.randomrepresentatives() );
    cout << "\ncconstraints:\n" << cconstraints;
    constraints.append( f.example_parameterset(cconstraints, fsrc.randomrepresentative(retdrf)) );
    cout << "\nfitting start:\n" << constraints << endl;
    cout << "\nfitting result:\n" << (fitres = fittomsqseq(constraints)) << endl;
  }*/

 /* montecarlofit(const measureseq &fsrc, const msq_dereferencer &retdrf,
                               const fittable_phmsqfn &fn, //const measure &startparams,
                               measure constraints=measure(),
                               ranGenT &rg = defaultsimplenormaldistributedrandomgenerator):
    ftgt(fsrc),
    f(fn),
    returndrf(retdrf),
    r(rg),
    distancer(fsrc, fn, retdrf)                                {
    evolution_minimizer m(distancer);
    cout << "\nfitting result:\n" << (fitres = m.minimized()) << endl;
  } */


#if 0
//const int montecarlofititerations=4;
//const int montecarlofittrialpoints=256;
const int max_available_space_for_fittingnodes = 9765625;  //=5^10, the minimum number of grid nodes
                                                     // necessary for fitting 10 parameters
double min_fitgridsmoothcellradius_fordim(int d) {
  if (d>10) return 1;
  return sqrt(12. - d);
}

  struct fitspreadgrid {
    int ttlngridnodes, ngridnodesperdim, ngridnodesspread;
    double smoothrngnodesspread;
    measure range, smoothrange;    //Attention: this entire class depends on a consistent order
    int rngdimns;                 // of the values in measures representing fitting considerations.
    const measureseq *fnsrc;
    const fittable_phmsqfn *f;
    const msq_dereferencer *returndrf;
    struct gridnode {
      bool isknown; physquantity msqdist;
      gridnode(): isknown(false) {}
    };
    typedef std::vector<gridnode> grid;
    grid g;

    fitspreadgrid(const measure &rng, const measureseq &nfnsrc,
                  const fittable_phmsqfn &nfn, const msq_dereferencer &nretdrf){
      cerr << "Warning: the fitspreadgrid class has never been testet. You're on your own!";
      fnsrc = &nfnsrc;
      range = rng;
      rngdimns = range.size();
      ttlngridnodes = fnsrc->size(); //ceil(sqrt(double(fnsrc->size())));
      ngridnodesperdim = ceil(pow(ttlngridnodes, 1./rng.size()));
      ngridnodesspread = ngridnodesperdim/2.;
      smoothrngnodesspread = 2 * sqrt(float(ngridnodesspread));
      if (smoothrngnodesspread<min_fitgridsmoothcellradius_fordim(rngdimns))
        smoothrngnodesspread = min_fitgridsmoothcellradius_fordim(rngdimns);
      ngridnodesspread = smoothrngnodesspread + smoothrngnodesspread*(smoothrngnodesspread/4) + .5;
      ngridnodesperdim = ngridnodesspread * 2 + 1;
      double tttlngridnds = pow(ngridnodesperdim, rng.size());
      if (tttlngridnds>max_available_space_for_fittingnodes)     {
        cerr << "Too many parameters for fitting function";  abort(); }
      ttlngridnodes = tttlngridnds;
      g.resize(ttlngridnodes);
      
      smoothrange = errors_in(range);
      for (measure::iterator i = smoothrange.begin(); i!=smoothrange.end(); ++i)
        *i *= smoothrngnodesspread/ngridnodesspread;
      
      f = &nfn;
      returndrf = &nretdrf;
    }

    
    typedef std::vector<int> coordingrid;
    typedef long rawcoordingrid;
    
    rawcoordingrid grid_idof(const coordingrid &c) const{
      rawcoordingrid result=c[0], o=1;
      for (int e=1; e<rngdimns; ++e){
        o*=ngridnodesperdim;
        result+=o*c[e];
      }
      return result;
    }
    measure rngposat(const coordingrid &c) const{
      measure result=range;
      for(int d=0; d<rngdimns; ++d){
        result[d]+=(result[d].error()*(c[d]-ngridnodesspread))/ngridnodesspread;
        result[d].error(0);
      }
      return result;
    }
    coordingrid cigposof(const measure &m) const{
      coordingrid result(rngdimns, ngridnodesspread);
      for(int d=0; d<rngdimns; ++d)
        result[d] += ((m[d] - range[d])/range[d].error()).dbl() * ngridnodesspread;
      return result;
    }
    const physquantity &calcsqdfor(const coordingrid &c, rawcoordingrid i){  //i should always be
                                                                            // =grid_idof(c)
      g[i].isknown=true;
      measure argshere = fnsrc->randommeasure();
      argshere.append(rngposat(c));
      return (g[i].msqdist = ((*returndrf)(argshere) - (*f)(argshere)).squared());
    }
    const physquantity &sqd_at(const coordingrid &c, rawcoordingrid i){  //i should always be
                                                                        // =grid_idof(c)
      if (g[i].isknown) return g[i].msqdist;
      return calcsqdfor(c,i);
    }
    
    struct fullellipsoid {
      fitspreadgrid *embggrid;
      measure center, radii;
  //    long intrad;
      class iterator {
        friend class ::fitspreadgrid::fullellipsoid;
        const fullellipsoid* sph;
        rawcoordingrid nowpos, seqend;
        coordingrid dac;
       public:
        bool operator==(const iterator &other) const{ return nowpos==other.nowpos; }
        bool operator!=(const iterator &other) const{ return nowpos!=other.nowpos; }

       private:

        double sqnormedradiusduetodimsfrom(int e) const{
          double result=0;
          for (int i=e; i < sph->embggrid->rngdimns; ++i){
            physquantity dine = sph->embggrid->range[i];
            dine.seterror(0);
            ((dine += (sph->embggrid->range[i].error()*(dac[i] - sph->embggrid->ngridnodesspread))/sph->embggrid->ngridnodesspread
                 ) -= sph->center[i]
                 ) /= sph->radii[i];
            result += dine.squared().dbl();
          }
          return result;
        }
        double reset_dim(int e){
          if (e > sph->embggrid->rngdimns - 2) return -1;  //counts as false: last coord should not be reset
          ++dac[e+1];
          double irmd = 1 - sqnormedradiusduetodimsfrom(e+1);
          if (irmd < 0){
            if (reset_dim(e+1)<0) return -1;
            irmd = 1 - sqnormedradiusduetodimsfrom(e+1);
          }
          irmd = sqrt(irmd);
          dac[e] = ((sph->center[e] - sph->radii[e] * irmd ).minusexactly(sph->embggrid->range[e])
                              /sph->embggrid->range[e].error()
                    ).dbl() * sph->embggrid->ngridnodesspread + 1;
          return irmd;
        }
        void find_seqend(double rsrm){
          int dacmem=dac[0];
          dac[0] = ((sph->center[0] + sph->radii[0] * rsrm ).minusexactly(sph->embggrid->range[0])
                            /sph->embggrid->range[0].error()
                    ).dbl() * sph->embggrid->ngridnodesspread;
          seqend = sph->embggrid->grid_idof(dac);
          dac[0] = dacmem;
        }
 
       public:
 
        iterator &operator++(){
          if (nowpos < 0) { cerr << "Incrementing iterator to a solid sphere over end()"; abort(); }
          ++dac[0];
          if (nowpos++ < seqend) return *this;
          double rsrm = reset_dim(0);
          if (rsrm>=0){
            nowpos = sph->embggrid->grid_idof(dac);
            find_seqend(rsrm);
           }else{
            nowpos = -1;
          }

          return *this;
        }
        ::fitspreadgrid::gridnode &operator*(){
          return sph->embggrid->g[nowpos];
        }
        ::fitspreadgrid::gridnode *operator->(){
          return &sph->embggrid->g[nowpos];
        }
        const coordingrid &coordm() {
          return dac;               }
        rawcoordingrid coordr() {
          return nowpos;        }
        
        void preparethispos(){
          nowpos = sph->embggrid->grid_idof(dac);
          double irmd = 1 - sqnormedradiusduetodimsfrom(1);
          if (irmd < 0){
            nowpos=-1;  return;
          }
          find_seqend(irmd);
        }
        
        iterator(const fullellipsoid* csph, const coordingrid &ndac) : sph(csph), dac(ndac){
          preparethispos();
        }
        
        struct begintreat { begintreat(){} };
        iterator(const fullellipsoid* csph, const begintreat &t) :
          sph(csph), dac(csph->embggrid->cigposof(csph->center)) {
          int ld = csph->embggrid->rngdimns - 1;
          dac[ld] = ((sph->center[ld] - sph->radii[ld]).minusexactly(sph->embggrid->range[ld])
                            /sph->embggrid->range[ld].error()
                    ).dbl() * sph->embggrid->ngridnodesspread;
          preparethispos();
        }
        struct endtreat { endtreat(){} };
        iterator(const fullellipsoid* csph, const endtreat &t) : sph(csph), nowpos(-1) {}
      };
      
      iterator begin() const { return iterator(this, iterator::begintreat()); }
      const iterator static_end;
      const iterator &end() const { return static_end; }
      
      fullellipsoid(fitspreadgrid *nembggrid, const measure &ncenter, const measure &nradii):
        embggrid(nembggrid), center(ncenter), radii(nradii),
        static_end(this, iterator::endtreat()) {
      }
      
      static double normsqdisttocenter(const iterator &i) {
        return i.sqnormedradiusduetodimsfrom(0);
      }
      
    };
    
    physquantity smoothsqd_at(const measure &a){
//      coordingrid c = cigposof(a);
      fullellipsoid e(this, a, smoothrange);
      physquantity result=0; double normalizer=0, thisnorm;
      for (fullellipsoid::iterator i = e.begin(); i!=e.end(); ++i){
        thisnorm = Friedrichs_mollifier_onsquares(e.normsqdisttocenter(i));
        normalizer += thisnorm;
        result += thisnorm * sqd_at(i.coordm(), i.coordr());
      }
      return result/=normalizer;
    }
    
    int dim_out_smoothcalc_range(const measure &a) {
      for (int i=0; i<a.size(); ++i) {
        if (a[i]-smoothrange[i] <= range[i].l_errorbord()
         || a[i]+smoothrange[i] >= range[i].u_errorbord())
          return i;
      }
      return -1;
    }
    bool can_smoothcalc_at(const measure &a) {
      return dim_out_smoothcalc_range(a) == -1;
    }



  };
#endif  

};
#endif

namespace_cqtxnamespace_CLOSE
