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


//debugging stuff
#define DBGBGN (std::cout << "⸮").flush()
#define DBGEND (std::cout << "?").flush()
#define DBGERRINFO (dynamic_cast< std::ostringstream & >(std::ostringstream() << std::dec << "ERROR MESSAGE, from " << __FILE__ << ", line " << __LINE__ << " —\n").str())
#define ERRMSGSTREAM cerr << DBGERRINFO
#define CND_DBGEVAL(e) if (globaldebugflag) (cout << e << endl).flush()


namespace_cqtxnamespace_OPEN

bool globaldebugflag=false;

 //platform-independent-sized integer types; this should work on gcc
// and conform with the BOOST int32_t, uint32_t, int64_t, uint64_t types.
#if __SIZEOF_POINTER__ == 4    // x86
typedef int int32_t;
typedef unsigned uint32_t;
//typedef long long int64_t;       // already given in sys/types.h
typedef unsigned long long uint64_t;
#elif __SIZEOF_POINTER__ == 8    // UNIX x86-64
typedef int int32_t;
typedef unsigned uint32_t;
typedef long int64_t;
typedef unsigned long uint64_t;
#endif


const double pi = 3.141592653589793;

const double MuchMoreThanOne = 128;     // for the much-greater-than comparison of physical quantities

bool thiscritical=false;


std::mt19937 standard_random_engine;
auto defaultsimplenormaldistributedrandomgenerator
  = std::bind(std::normal_distribution<>(), standard_random_engine);
typedef decltype(defaultsimplenormaldistributedrandomgenerator)
        std_normaldistributed_randomgen;

namespace_cqtxnamespace_CLOSE


#include "phq_base.h"   // the basic definitions of physical quantities

#include "stdtemplatespclz.hpp" // specializations of some standard classes (e.g.
                               //  complex, random distributions) for physquantity

#include "intn_tex.h"


namespace_cqtxnamespace_OPEN

namespace stdPhysUnitsandConsts {
  unitsscope stdUnits("stdPhysUnits");
}
//unitsscope *defaultUnitsc = &stdPhysUnits;
std::vector<unitsscope *> defaultUnitscps(1, &stdPhysUnitsandConsts::stdUnits);

phUnit cgsUnitof(const physquantity &src);

const phUnit &newUnit(const physquantity &baseElem, string newName,
           const std::vector<unitsscope *> &scopes = defaultUnitscps){
  physquantity source = baseElem;
  if (source.tunit!=NULL) source.fixcgs();
  const phUnit *putmethere = new phUnit(baseElem.Dimension(), source.valincgs, newName);
  bool ok=true, isnew=true;
  for (std::vector<unitsscope*>::const_iterator s=scopes.begin(); s!=scopes.end(); ++s){
    std::pair<unitsscope::iterator, bool> pinscope = (*s)->insert(putmethere);
    ok=pinscope.second;
    if (!ok) {
      physquantity conflicting=1***(pinscope.first);
      if (source.Dimension()==conflicting.Dimension() && source==conflicting){
        cerr << "Unit\"" << newName << "\" is already in scope. Do nothing here." << endl ;
        if (isnew){
          for (std::vector<unitsscope*>::const_iterator srmp=scopes.begin(); srmp!=s; ++srmp){
            cerr << "Warning: redundant junk pointers over multiple unit scopes are not handled properly yet."
                 << endl;
          }
          delete putmethere;
          putmethere = *pinscope.first;
          isnew = false;
        }
        ok=true;
       }else{
        phUnit cgsuniteqv=cgsUnitof(conflicting);
        cerr << "A unit with symbol \""
             << newName
             << "\" already exists ( = "
             << conflicting[cgsuniteqv]
             << " " << cgsuniteqv.uName
             << ")";
        if ((*s)->caption!="")
          cerr << " in scope \"" << (*s)->caption << "\"";
        cerr << ".\nOverloading of unit names inside a scope is not supported.";
        abort();
      }
    }
  }
  return *putmethere;
}
const phUnit &newUnit(const physquantity &baseElem, string newName, unitsscope *scope){
  if (scope!=NULL)
    return newUnit(baseElem, newName, std::vector<unitsscope *>(1, scope));
  return newUnit(baseElem, newName, std::vector<unitsscope *>());
}
const phUnit tmpUnit(const physquantity &baseElem, string newName){
  physquantity source = baseElem;
  if (source.tunit!=NULL) source.fixcgs();
  return phUnit(baseElem.Dimension(), source.valincgs, newName);
}

inline const phUnit *findUnit(const string &whatname, const unitsscope *scope = defaultUnitsc){
  unitsscope::iterator where = scope->find(whatname);
  if (where==scope->end()) return NULL;
  return *where;
}

inline const phUnit *suitableUnitfor(const physquantity &whatq, const unitsscope *scope){
  if(scope==NULL) return NULL;
  std::vector<const phUnit *> tmpUs(scope->suitableUnitsfor(whatq));
  if (tmpUs.size()==0) return NULL;
  return tmpUs[0];
}

phUnit cgsUnitof(const physquantity &src){
  stringstream nm, tm;
  if (src.preferredUnit()!=NULL){
    src.fixcgs();
  }
  if(src.myDim.c()!=0) {nm << "cm";
    if(src.myDim.c()!=1) { tm<<src.myDim.c();
      if(tm.str().size()>1)
        nm << "^{" << tm.str() << "}";
       else if(src.myDim.c()!=0)
        nm << "^" << tm.str();
    if(src.myDim.g()!=0 || src.myDim.s()!=0) nm << "{\\cdot}";
  }}
  tm.str("");
  if(src.myDim.g()!=0) {nm << "g";
    if(src.myDim.g()!=1) { tm<<src.myDim.g();
      if(tm.str().size()>1)
        nm << "^{" << tm.str() << "}";
       else if(src.myDim.g()!=0)
        nm << "^" << tm.str();
    if(src.myDim.s()!=0) nm << "{\\cdot}";
  }}
  tm.str("");
  if(src.myDim.s()!=0) {nm << "s";
    if(src.myDim.s()!=1) { tm<<src.myDim.s();
      if(tm.str().size()>1)
        nm << "^{" << tm.str() << "}";
       else if(src.myDim.s()!=0)
        nm << "^" << tm.str();
  }}
  tm.clear();
  return phUnit(src.myDim.c(), src.myDim.g(), src.myDim.s(), 1, nm.str());
}



namespace stdPhysUnitsandConsts {
  unitsscope exoticunits("exotic");

  //physikalische Einheiten:
  const phUnit
                seconds = newUnit(  1 * seconds_base,                       "s"  )
   ,       milliseconds = newUnit(  1e-3 * seconds,                        "ms"  )
   ,       microseconds = newUnit(  1e-6 * seconds,                    "\\mu s"  )
   ,        nanoseconds = newUnit(  1e-9 * seconds,                        "ns"  )
   ,            minutes = newUnit(  60 * seconds,                         "min"  )
   ,              hours = newUnit(  60 * minutes,                           "h"  )
   ,               days = newUnit(  24 * hours,                             "d"  )
   ,              years = newUnit(  365.25 * days,                          "a"  )
   ,             meters = newUnit(  1 * meters_base,                        "m"  )
   ,          kilograms = newUnit(  1 * kilograms_base,                    "kg"  )
   ,            amperes = newUnit(  1 * amperes_base,                       "A"  )
   ,             hertzs = newUnit(  seconds.to(-1),                        "Hz"  )
   ,    inverse_seconds = newUnit(  seconds.to(-1),                    "s^{-1}"  )
   ,        centimeters = newUnit(  meters / 100,                          "cm"  )
   ,        millimeters = newUnit(  meters / 1000,                         "mm"  )
   ,        micrometers = newUnit(  meters * 1e-6,                     "\\mu m"  )
   ,         nanometers = newUnit(  meters * 1e-9,                         "nm"  )
   ,         picometers = newUnit(  meters * 1e-12,                        "pm"  )
   ,        megaparsecs = newUnit(  3.085677581e+25 * millimeters,        "Mpc"  , &exoticunits)
   ,          angstroms = newUnit(  1e-10 * meters,                "\\Angstrom"  )
   ,       squaremeters = newUnit(  meters.to(2),                         "m^2"  )
   ,  squarecentimeters = newUnit(  (meters/100).to(2),                  "cm^2"  )
   ,              acres = newUnit(  meters.to(2),                           "a"  , &exoticunits)
   ,        cubicmeters = newUnit(  meters.to(3),                         "m^3"  )
   ,   cubiccentimeters = newUnit(  (meters/100).to(3),                  "cm^3"  )
   ,     invcentimeters = newUnit(  (meters/100).inv(),               "cm^{-1}"  )
   ,             liters = newUnit(  (meters/10).to(3),                      "l"  )
   ,        milliliters = newUnit(  (meters/100).to(3),                    "ml"  )
   ,              grams = newUnit(  kilograms/1000,                         "g"  )
   ,         milligrams = newUnit(  kilograms/1000000,                     "mg"  )
   ,             tonnes = newUnit(  1000*kilograms,                         "t"  )
   ,            newtons = newUnit(  kilograms * meters / seconds.to(2),     "N"  )
   ,             joules = newUnit(  newtons*meters,                         "J"  )
   ,      electronvolts = newUnit(  1.60217653e-19 * joules,               "eV"  )
   ,  kiloelectronvolts = newUnit(  1000 * electronvolts,                 "keV"  )
   ,  megaelectronvolts = newUnit(  1e+6 * electronvolts,                 "MeV"  )
   ,  gigaelectronvolts = newUnit(  1e+9 * electronvolts,                 "GeV"  )
   , millielectronvolts = newUnit(  1.60217653e-22 * joules,              "meV"  )
   ,       kelvins_base = newUnit(  1*joules,                               "K"  , NULL)     //Joule und Kelvin sind in cgs identisch
   ,            kelvins = newUnit(  1*kelvins_base,                         "K"  )
   ,              watts = newUnit(  newtons*meters/seconds,                 "W"  )
   ,       milesperhour = newUnit(  .447*meters/seconds,                  "mph"  )
   ,            pascals = newUnit(  newtons/meters.to(2),                  "Pa"  )
   ,       hectopascals = newUnit(  100*pascals,                          "hPa"  )
   ,        megapascals = newUnit(  1e+6 * pascals,                       "MPa"  )
   ,               bars = newUnit(  1e+5 * pascals,                       "bar"  )
   ,          millibars = newUnit(  1e-3 * bars,                         "mbar"  )
   ,           kilobars = newUnit(  1e+8 * pascals,                      "kbar"  )
   ,           megabars = newUnit(  1e+11 * pascals,                     "Mbar"  )
   ,        atmospheres = newUnit(  101325 * pascals,                     "atm"  )
   ,           coulombs = newUnit(  amperes*seconds,                        "C"  )
   ,       milliamperes = newUnit(  1e-3 * amperes,                        "mA"  )
   ,       microamperes = newUnit(  1e-6 * amperes,                   "{\\mu}A"  )
   ,              volts = newUnit(  joules/coulombs,                        "V"  )
   ,         millivolts = newUnit(  1e-3 * volts,                          "mV"  )
   ,         microvolts = newUnit(  1e-6 * volts,                     "{\\mu}V"  )
   ,               ohms = newUnit(  volts/amperes,                    "\\Omega"  )
   ,          ohmmeters = newUnit(  ohms*meters,              "\\Omega\\cdot m"  )
   ,               mhos = newUnit(  amperes/volts,                      "\\mho"  )
   ,             teslas = newUnit(  volts*seconds/meters.to(2),             "T"  )
   ,               mols = newUnit(  6.02214179e+23*real1,                 "mol"  )
   ,          millimols = newUnit(  mols/1000,                           "mmol"  )
   ,       molsperliter = newUnit(  mols/liters,               "\\frac{mol}{l}"  )
   ,   molspercubemeter = newUnit(  mols/cubicmeters,        "\\frac{mol}{m^2}"  )
   ,           percents = newUnit(  real1/100,                            "\\%"  );

  //...und Konstanten:
  const physquantity
              kboltzmann = ((1.3806504   +lusminus(.0000024))   * 1e-23 * joules/kelvins)           .wlabel("\\kbar")
   ,      electroncharge = ((1.602176487 +lusminus(.000000040)) * 1e-19 * coulombs)                 .wlabel("e")
   ,        electronmass = ((9.10938215 +lusminus(.00000045)) * 1e-31 * kilograms)                  .wlabel("m_e")
   ,         neutronmass = ((1.67492729 +lusminus(.00000028)) * 1e-27 * kilograms)                  .wlabel("m_n")
   ,          lightspeed = (299792458 * meters/seconds)                                             .wlabel("c")
   ,          hbarplanck = ((1.054571628 +lusminus(.000000053)) * 1e-34 * joules*seconds)           .wlabel("\\hbar")
   , vacuum_permeability = ((4*pi)                              * 1e-7  * teslas*meters/amperes)    .wlabel("\\mu_0")
   ,        bohrmagneton = (electroncharge * hbarplanck / (2 * electronmass) )                      .wlabel("\\mu_B")
   ,     gravityconstant = ((6.67384 +lusminus(.0008)) * 1e-11 * newtons*(meters/kilograms).to(2) ) .wlabel("G")

   , earth_gravityaccell_cologne = ((9.78913+lusminus(.00005)) * meters/seconds.to(2))              .wlabel("g\\rmsc{cologne}")
   , standardgravity = ((9.80665)           * meters/(seconds.to(2)))                               .wlabel("g");


  unitsscope planck_units("Planck units");
  namespace Planck {
    const phUnit
           length = newUnit( sqrt(hbarplanck*gravityconstant/lightspeed.to(3)), "l_P",    &planck_units )
     ,       mass = newUnit( sqrt(hbarplanck*lightspeed/gravityconstant)      , "m_P",    &planck_units )
     ,       time = newUnit( sqrt(hbarplanck*gravityconstant/lightspeed.to(5)), "t_P",    &planck_units )
     ,  frequency = newUnit( 1/time,                                            "\\nu_P", &planck_units )
     , wavenumber = newUnit( 1/length                                         , "k_P",    &planck_units )
     ,     energy = newUnit( sqrt(hbarplanck*lightspeed.to(5)/gravityconstant), "E_P",    &planck_units )
     ,     action = newUnit( hbarplanck,                                        "\\hbar", &planck_units );
  }

  unitsscope all_known_units("all known units", {stdUnits, planck_units});
}

unitsscope *default_defaultUnitsc(){return &stdPhysUnitsandConsts::stdUnits;}
using namespace stdPhysUnitsandConsts;

/*struct namestable_phqaccess {
//  namestable_phqaccess(const physquantity &naccess) : physquantity(naccess) {}

  namestable_phqaccess &hello() { std::cout << "Hi"; return *this; }
  
  physquantity &operator=(const physquantity &cpyfrom){
    hello();
//    cout << "Assigning phq ";
    captionsscope::iterator oldcapt = caption;
//    if (oldcapt!=captNULL){
  //    cout << *oldcapt << " ( = " << *this << ") the new value;
    
    //}
    physquantity::operator=(cpyfrom);
    caption = oldcapt;
    return *static_cast<physquantity *>(this);
  }
  
};*/

class measure;
class measureseq;



      // measure is a container for physquantities, designed for looking these up
     //  by name rather than index. Even though such a lookup is usually best done
    //   with hash maps or balanced trees, we use a plain unordered array here;
   //    this is complexity-slower for big lookups but has less overhead in both
  //     space and time, so it's ideal when there's only a few quantities in a
 //      measure, e.g. one length, one current, one force. That's the intended way
//       of using this container.
  //     If used for larger sets of quantities and speed-relevant algorithms, direct
 //      O(n) by-name lookup becomes inefficient: in this case use captfinder objects
//       for the lookups, which support index memoization.
class measure : public std::vector<physquantity> {
 public:
  measure(){}
  measure(size_t num) : std::vector<physquantity>(num) {}
  measure(size_t num, physquantity allelems) : std::vector<physquantity>(num, allelems) {}
  measure(physquantity singleelem) : std::vector<physquantity>(1, singleelem) {}
  measure(physquantity elem0, physquantity elem1){push_back(elem0); push_back(elem1);}
  measure(const std::pair<physquantity,physquantity> &elems){push_back(elems.first); push_back(elems.second);}
  measure(physquantity elem0, physquantity elem1, physquantity elem2){push_back(elem0); push_back(elem1); push_back(elem2);}

  measure& operator=(measure m) {
    std::vector<physquantity>::clear();
    std::vector<physquantity>::operator=(std::move(m));
    return *this;
  }
  
  bool has(const captT &cptsc) const{
    for (std::vector<physquantity>::const_iterator i = begin(); i!= end(); ++i){
      if (i->caption==cptsc) return true;
    }
    return false;
  }
  bool has(const string &cptsc) const{
    for (std::vector<physquantity>::const_iterator i = begin(); i!= end(); ++i){
      if (i->cptstr()==cptsc) return true;
    }
    return false;
  }

  physquantity &called(const string &cptsc){
    for (std::vector<physquantity>::iterator i = begin(); i!= end(); ++i){
      if (i->cptstr()==cptsc) return *i;
    }
    cout << "Requested a phq with caption \"" << cptsc << "\". There exists none in this measure, only:\n";
    for (std::vector<physquantity>::iterator i = begin(); i!= end(); ++i){
      cout << i->cptstr() << ", ";
    }
    cout << "\nAbort."<<std::endl; abort();
    push_back( physquantity()/*(0*real1)*/ ); back().label(cptsc);
    return back();
  }
  const physquantity &called(const string &cptsc) const{
    for (std::vector<physquantity>::const_iterator i = begin(); i!= end(); ++i){
      if (i->cptstr()==cptsc) return *i;
    }
    cerr << "Requested a phq with caption \"" << cptsc << "\". There exists none in this measure, only:\n";
    for (std::vector<physquantity>::const_iterator i = begin(); i!= end(); ++i){
      cerr << i->cptstr() << ", ";
    }
    cerr << "\nUnable to insert new, as this measure is const.\n";
    abort();
  }
  
  measure &rename(const std::string &cptsc, const std::string &nname) {
    for (auto i = begin(); i!= end(); ++i)
      if (i->cptstr()==cptsc) i->label(nname);
    return *this;
  }
  measure renamed(const std::string &cptsc, const std::string &nname) const{
//    measure cpy(*this);
  //  cout << cpy;
    return measure(*this).rename(cptsc,nname);
  }
/*  maybe<const physquantity &> lookup(const string &cptsc) const{
    for (auto i = begin(); i!= end(); ++i)
      if (i->cptstr()==cptsc) return just(*i);
    return nothing;
  }
  maybe<physquantity &> lookup(const string &cptsc) {
    for (auto i = begin(); i!= end(); ++i)
      if (i->cptstr()==cptsc) return just(*i);
    return nothing;
  }
*/
  physquantity &operator[](int cpti){ return std::vector<physquantity>::operator[](cpti); }
  const physquantity &operator[](int cpti) const{ return std::vector<physquantity>::operator[](cpti); }
  physquantity/*::namestable_access*/ &operator[](const string &cptsc){ return called(cptsc); }
  const physquantity &operator[](const string &cptsc) const{ return called(cptsc); }

  measure &likeUnit(const phUnit &thatone){
    for (std::vector<physquantity>::iterator i = begin(); i!= end(); ++i)
      i->tryconvto(thatone);
    return *this;
  }
  
  
/*  class l{
    
   public:
    physquantity operator()(measure x){
      
    }
  };*/

/*
  template<class createfnobjT>
  measure &let(const createfnobjT &createfnobj){
//    physquantity nu = createfnobj(*this);
    push_back(createfnobj(*this));
    return *this;
  }
*/

  measure &letnow(const string &nnm, const physquantity &nphq){
    if (!has(nnm)){
      push_back(nphq);  back().label(nnm);
     }else
      operator[](nnm) = nphq;
    return *this;
  }
  measure &let(const string &nnm, const physquantity &nphq){
    push_back(nphq);  back().label(nnm);
    return *this;
  }

  class letdefobj {
    measure *deftgt;
    std::vector<std::string> dsrdnames;
    letdefobj(measure *t, const std::string& dn)
      : deftgt(t)
      , dsrdnames(1, dn)
    {}
    auto operator&(letdefobj also)const -> letdefobj{
      assert(also.deftgt == deftgt);
      for(auto& nname: dsrdnames)
        also.dsrdnames.push_back(nname);
      return also;
    }
   public:
    physquantity &operator=(const physquantity &ldf) {
      for(auto&tg : dsrdnames)
        deftgt -> let(tg, ldf);
      return deftgt->called(dsrdnames.front());
    }
    physquantity &operator=(double ldf) {
      for(auto&tg : dsrdnames)
        deftgt -> let(tg, ldf*real1);
      return deftgt->called(dsrdnames.front());
    }
    
    struct unitfree {
      letdefobj* defobj;
      const phUnit* unit;
    };
    struct const_unitfree {
      const letdefobj* defobj;
      const phUnit* unit;
    };
//     double& operator[](const phUnit& u) const{
//       return unitfree{this, &u};            }

    friend class measure;
    friend std::istream& operator>>(std::istream&, const unitfree&);
  };
  letdefobj let(const std::string& nnm) {
    return letdefobj(this, nnm);        }
  template<typename... Further>
  letdefobj let(const std::string& nnm, Further... other) {
    return letdefobj(this, nnm) & let(other...);          }

  letdefobj newvar(const std::string& nnm) {  
    return letdefobj(this, nnm);           }
//  private:
//   letdefobj newvars(const std::string& nnm) {
//     return letdefobj(this, nnm);            }
//  public:
//   template<typename... Further>
//   letdefobj newvars(const std::string& nnm, Further... other) {
//     return letdefobj(this, nnm) & newvars(other...);          }

  measure &erase(const string &nnm){
    iterator i = begin();
    while(i->cptstr() != nnm){
      ++i; if (i==end()) return *this;
    }
    for(iterator j=i++; i!=end(); ++i,++j)
      j->overwritewith(*i);
    std::vector<physquantity>::resize(std::vector<physquantity>::size()-1);
    return *this;
  }
  iterator erase(iterator i){
    if (i==end()) return i;
    unsigned ii = i-begin();
    for(iterator j=i++; i!=end(); ++i,++j)
      j->overwritewith(*i);
    std::vector<physquantity>::resize(std::vector<physquantity>::size()-1);
    return begin() + ii;
  }

  measure &append(const measure &mi){
    for(const_iterator i=mi.begin(); i!=mi.end(); ++i)
      push_back(*i);
    return *this;
  }
  measure &append(const physquantity &mi){
    push_back(mi);
    return *this;
  }
  measure &complete_carefully(const measure &mi){
    for(const_iterator i=mi.begin(); i!=mi.end(); ++i) {
      if (!has(i->caption))
        push_back(*i);
    }
    return *this;
  }

  template<class A>
  measure combined_with(const A &mi) const{
    return measure(*this).append(mi);
  }
  
  measureseq pack_subscripted() const;


  measure &multiply_all_by(double lambda) {
    for(unsigned i=0; i<size(); ++i)
      operator[](i) *= lambda;
    return *this;
  }

  measure &by_i_add_multiple_of(const measure &other, double lambda) {
    for(unsigned i=0; i<size(); ++i)
      operator[](i) += other[i] * lambda;
    return *this;
  }
  measure by_i_plus_multiple_of(const measure &other, double lambda) const{
    measure result(*this);
    result.by_i_add_multiple_of(other, lambda);
    return result;
  }
  measure &by_i_add(const measure &other) {
//     for(unsigned i=0; i<size(); ++i)
    for(unsigned i=std::min(size(), other.size()); i-->0;)
      operator[](i) += other[i];
    return *this;
  }
  measure by_i_plus(const measure &other) const{
    measure result(*this);
    result.by_i_add(other);
    return result;
  }
  measure &by_i_substract(const measure &other) {
    for(unsigned i=std::min(size(), other.size()); i-->0;)
      operator[](i) -= other[i];
    return *this;
  }
  measure operator-() const{
    measure result = *this;
    for (iterator i = result.begin(); i!=result.end(); ++i)
      *i *= -1;
    return result;
  }
  measure by_i_minus(const measure &other) const{
    measure result(*this);
    result.by_i_substract(other);
    return result;
  }
  measure &by_i_leterrors(const measure &other) {
    for(unsigned i=std::min(size(), other.size()); i-->0;)
      operator[](i).error(other[i]);
    return *this;
  }
  measure by_i_witherrors(const measure &other) const{
    measure result(*this);
    result.by_i_leterrors(other);
    return result;
  }
  
  measure &by_i_statistically_approach(const measure &other, double significance) {
//     for(unsigned i=0; i<size(); ++i)
    for(unsigned i=std::min(size(), other.size()); i-->0;)
      operator[](i).statistically_approach(other[i], significance);
    return *this;
  }
  
  measure &remove_errors() {
    for (iterator i = begin(); i!=end(); ++i)
      i->error(0);
    return *this;
  }
  measure &scale_allerrors(double lambda) {
    for (iterator i = begin(); i!=end(); ++i)
      i->error(i->error() * lambda);
    return *this;
  }
  measure &uncertain_collapse(std_normaldistributed_randomgen &r = defaultsimplenormaldistributedrandomgenerator) {
    for (iterator i = begin(); i!=end(); ++i)
      i->uncertain_collapse(r);
    return *this;
  }
  measure uncertain_collapsed(std_normaldistributed_randomgen &r = defaultsimplenormaldistributedrandomgenerator)
         const{  return measure(*this).uncertain_collapse(r);  }
  
  

};


std::ostream &operator<<(std::ostream &tgt, const measure &src){
  for (auto& q: src) {
    if (q.caption != cptNULL) tgt << miniTeX(*q.caption) << ": ";
     else tgt << "(unnamed) ";
    tgt << q << std::endl;
  }
  return tgt;
}

std::istream& operator>>( std::istream &src
                        , const measure::letdefobj::unitfree &tgter ) {
  double numget;
  src >> numget;
  *tgter.defobj = numget * (*tgter.unit);
  return src;
}

measure errors_in(const measure &source) {
  measure result = source;
  for (measure::iterator i=result.begin(); i!=result.end(); ++i){
    if (i->caption != cptNULL) i->label("\\Err"+*i->caption);
    (*i) = i->error();
  }
  return result;
}



template<typename fieldT>
class interval;


class measureseq : std::vector<measure>{
 public:
  measureseq(){}
  explicit measureseq(size_t numbmsrs) : std::vector<measure>(numbmsrs) {}
  explicit measureseq(size_t numbmsrs, const measure &ainit) : std::vector<measure>(numbmsrs, ainit) {}
  measureseq(const std::vector<measure> &src) : std::vector<measure>(src.size()){
    for (unsigned int i=0; i<src.size(); ++i)
      at(i) = src[i];
  }
//  measureseq(std::vector<measure> &&src) : std::vector<measure>(std::move(src)){}
  explicit measureseq(const const_iterator &linit, const const_iterator &rinit)
     : std::vector<measure>(linit, rinit) {}
  measureseq (std::initializer_list<std::initializer_list<physquantity>> init){
    for (auto m : init) {
      measure mf; for (auto q : m) mf.push_back(q);
      std::vector<measure>::push_back(mf);
    }
  }
  

  using std::vector<measure>::size;
  using std::vector<measure>::clear;

  auto front() -> measure& { return std::vector<measure>::front(); }
  auto front()const -> const measure& { return std::vector<measure>::front(); }
  auto back() -> measure& { return std::vector<measure>::back(); }
  auto back()const -> const measure& { return std::vector<measure>::back(); }

  void push_back(const measure& m) { std::vector<measure>::push_back(m); }
  void push_back(measure&& m) { std::vector<measure>::push_back(std::move(m)); }

  auto operator[](int i) -> measure& { return std::vector<measure>::operator[](i); }
  auto operator[](int i)const -> const measure& { return std::vector<measure>::operator[](i); }
  

  measureseq &likeUnit(const phUnit &thatone){
    for (auto i = begin(); i!= end(); ++i)
      i->likeUnit(thatone);
    return *this;
  }
  template<class createfnobjT>
  measureseq &foreach_let(const createfnobjT &createfnobj){
    for (iterator i = begin(); i!=end(); ++i)
      i->let(createfnobj);
    return *this;
  }
  measureseq &foreach_append(const physquantity &x){
    for (iterator i = begin(); i!=end(); ++i)
      i->append(x);
    return *this;
  }

 private:  
  struct access : ::std::iterator< std::random_access_iterator_tag
                                 , measure                         > {
    typedef std::vector<measure>::iterator iterator;
    iterator i;

    access(iterator i) : i(i){}
    access() : i(){}

    auto operator==(access other)const -> bool {return i==other.i;}
    auto operator!=(access other)const -> bool {return i!=other.i;}
    auto operator<(access other)const -> bool {return i<other.i;}
    auto operator>(access other)const -> bool {return i>other.i;}
    auto operator>=(access other)const -> bool {return i<=other.i;}
    auto operator<=(access other)const -> bool {return i>=other.i;}

    auto operator*()const -> measure& { return *i; }
    auto operator->()const -> measure* { return &(*i); }

    access& operator++() {++i; return *this;}
    access operator++(int _) {return i++;}
    access& operator--() {--i; return *this;}
    access operator--(int _) {return i--;}

    access& operator+=(std::ptrdiff_t offs) {i+=offs; return *this;}
    access& operator-=(std::ptrdiff_t offs) {i-=offs; return *this;}

    auto operator+(std::ptrdiff_t offs)const -> access {
      return access(i + offs);                         }
    auto operator-(std::ptrdiff_t offs)const -> access {
      return access(i - offs);                         }

    auto operator-(access other)const -> std::ptrdiff_t {
      return i - other.i;                               }
    
    auto operator[](const string &longfor) -> physquantity& {
      return (*this)->called(longfor);                      }
  };
  struct const_access : ::std::iterator< std::input_iterator_tag
                                       , measure                 > {
    typedef std::vector<measure>::const_iterator iterator;
    iterator i;

    const_access(iterator i) : i(i){}
    const_access(access macc) : i(macc.i){}
    const_access() : i(){}

    auto operator==(const_access other)const -> bool {return i==other.i;}
    auto operator!=(const_access other)const -> bool {return i!=other.i;}
    auto operator<(const_access other)const -> bool {return i<other.i;}
    auto operator>(const_access other)const -> bool {return i>other.i;}
    auto operator<=(const_access other)const -> bool {return i<=other.i;}
    auto operator>=(const_access other)const -> bool {return i>=other.i;}

    auto operator*()const -> const measure& { return *i; }
    auto operator->()const -> const measure* { return &(*i); }

    const_access& operator++() {++i; return *this;}
    const_access operator++(int _) {return i++;}
    const_access& operator--() {--i; return *this;}
    const_access operator--(int _) {return i--;}

    const_access& operator+=(std::ptrdiff_t offs) {i+=offs; return *this;}
    const_access& operator-=(std::ptrdiff_t offs) {i-=offs; return *this;}

    auto operator+(std::ptrdiff_t offs)const -> const_access {
      return const_access(i + offs);                         }
    auto operator-(std::ptrdiff_t offs)const -> const_access {
      return const_access(i - offs);                         }

    auto operator-(const_access other)const -> std::ptrdiff_t {
      return i - other.i;                                     }
    
    auto operator[](const string &longfor) -> const physquantity& {
      return (*this)->called(longfor);                            }
  };

//   typedef std::vector<measure>::iterator iterator;
//   typedef std::vector<measure>::const_iterator const_iterator;
 public:
  typedef access iterator;
  typedef const_access const_iterator;

  measureseq(const_access bgn, const_access end)
    : std::vector<measure>(bgn.i, end.i) {}

  access begin() { return std::vector<measure>::begin(); }
  const_access begin() const { return std::vector<measure>::begin(); }
  const_access cbegin() const { return std::vector<measure>::begin(); }
  access end() { return std::vector<measure>::end(); }
  const_access end() const { return std::vector<measure>::end(); }
  const_access cend() const { return std::vector<measure>::end(); }

  access erase(access a) {return std::vector<measure>::erase(a.i);}
  
  measureseq giveme(const std::vector<string> &desires){
    measureseq result;
    for(access i = begin(); i!=end(); ++i){
      result.push_back(measure());
      for(std::vector<string>::const_iterator j=desires.begin(); j!=desires.end(); ++j)
        result.back().push_back(i[*j]);
    }
    return result;
  }
  measureseq giveme(const string &d0, const string &d1, const string &d2){
    std::vector<string> d(3);  d[0]=d0; d[1]=d1; d[2]=d2;  return giveme(d);  }
  measureseq giveme(const string &d0, const string &d1, const string &d2, const string &d3){
    std::vector<string> d(4);    d[0]=d0; d[1]=d1; d[2]=d2; d[3]=d3;    return giveme(d);       }

//  measure &randommeasure(){ return operator[]((rand()*size())/RAND_MAX); }
  const measure &randommeasure() const{ return operator[]((float(rand())*size())/RAND_MAX); }

 private: static const unsigned representativefindcounts=64;public:
  template<class derefT>
  physquantity randomrepresentative(const derefT &drf) const{
    physquantity roughmean=0;
    unsigned i;
    for(i=0; ( i<representativefindcounts && i<size() ) || ( roughmean==0 && i<size()*4 ); ++i){
      roughmean+=drf(randommeasure());
    }
    roughmean/=std::min(size_t(representativefindcounts), size());
  //  cout << "Rough mean: " << roughmean << endl;
    int j, bj=(float(rand())*size())/RAND_MAX;
    physquantity lsqd=(drf(operator[](bj)) - roughmean).squared(), mndistsq=0;
    
    for(i=0; (i<representativefindcounts && i<size()) || (i<2*size() && drf(operator[](bj))==0); ++i){
      j = (float(rand())*size())/RAND_MAX;
      physquantity tsquared = (drf(operator[](j)) - roughmean).squared();
      mndistsq += tsquared;
      if ( ( drf(operator[](j))!=0 && tsquared < lsqd ) || drf(operator[](bj))!=0) {
        bj=j; lsqd = tsquared;
      }
    }
    mndistsq /= i;
    return roughmean/*drf(operator[](bj))*/.werror(sqrt(mndistsq));
  }

  measure randomrepresentatives() const;  
  
  measureseq &append(const measureseq &ma) {  
    for(const_iterator i=ma.begin(); i!=ma.end(); ++i)
      push_back(*i);
    return *this;
  }
  measureseq &append(const measure &mi){
    push_back(mi);
    return *this;
  }
  template<class A>
  measureseq combined_with(const A &mi) const{
    return measureseq(*this).append(mi);
  }



  template<class Lookup>
  interval<physquantity> range_of(const Lookup &lkup) const;
  
};


std::ostream &operator<<(std::ostream &tgt, const measureseq &src){
  for (unsigned i=0; i!=src.size(); ++i){
    tgt << " --- --- --- Measure #" << i << " --- --- --- \n";
    tgt << src[i];
  }
  return tgt;
}


measureseq measure::pack_subscripted() const {
  std::multimap<int, physquantity> indexmap;
  int biggestindex=0, smallestindex=0;
  for (const_iterator i = begin(); i!=end(); ++i) {
    string s = i->cptstr();
    for (int j : try_splitoff_subscript<int>(s)) {
      indexmap.insert(std::make_pair(j, i->wlabel(s)));
      if (j < smallestindex) smallestindex=j;
      if (j > biggestindex) biggestindex=j;
    }
  }
  measureseq result(biggestindex-smallestindex + 1);
  for (auto insp : indexmap)
    result[insp.first-smallestindex] .push_back(insp.second);
  return result;
}


measureseq datathinned(const measureseq &alldat, unsigned dtskips=16) {
  measureseq result;
  for (unsigned i=0; i<alldat.size(); ++i+=dtskips)
    result.push_back(alldat[i]);
  return result;
}





template<typename fieldT>
class centered_about_object {
  fieldT c;
 public:
  centered_about_object(const fieldT &c): c(c) {}
  friend class interval<fieldT>;
};
template<typename fieldT>
centered_about_object<fieldT> centered_about(const fieldT &c) {return centered_about_object<fieldT>(c);}
template<typename fieldT>
class interval_size_object {
  fieldT s;
 public:
  interval_size_object(const fieldT &si): s(si) { if (s>=0); else s=-s; }
  friend class interval<fieldT>;
};
template<typename fieldT>
interval_size_object<fieldT> interval_size(const fieldT &s) {return interval_size_object<fieldT>(s);}


template<typename fieldT>
class interval {
  std::pair<fieldT, fieldT> b;
  bool is_empty;
 public:
  interval(): b(0,0), is_empty(true) {}
  interval(const fieldT &linit, const fieldT &rinit): is_empty(false) {
    if (rinit>linit) { b.first=linit;
                       b.second=rinit; }
     else            { b.first=rinit;
                       b.second=linit; }
  }
  interval(const fieldT &linit, const fieldT &rinit, const std::string &cpt): is_empty(false) {
    if (rinit>linit) { b.first=linit;
                       b.second=rinit; }
     else            { b.first=rinit;
                       b.second=linit; }
    b.first.label(cpt);
    b.second.label(cpt);
  }
  interval(const centered_about_object<fieldT> &c, const interval_size_object<fieldT> &s): is_empty(false) {
    b.first = c.c - s.s;
    b.second = c.c + s.s;
  }
  interval(const centered_about_object<fieldT> &c, const interval_size_object<fieldT> &s, const std::string &cpt): is_empty(false) {
    (b.first = c.c - s.s).label(cpt);
    (b.second = c.c + s.s).label(cpt);
  }
  interval(const interval<fieldT> &binit)
    : b(binit.b), is_empty(binit.is_empty) {}
  interval(const std::pair<fieldT, fieldT> &binit): is_empty(false){
    if (binit.second>binit.first) { b.first=binit.first; b.second=binit.second; }
     else                        { b.first=binit.second; b.second=binit.first; }
  }
  
  interval &widen_to_include(const fieldT &incl) {
    if (is_empty) {
      b.first = b.second = incl;
      is_empty=false;
     }else if (incl<b.first) b.first = incl;
      else if (incl>b.second) b.second = incl;
    return *this;
  }
  interval &widen_to_include(const interval<fieldT> &incl) {
    if (is_empty) {
      *this=incl;
     }else{
      if (incl.b.first<b.first) b.first = incl.b.first;
      if (incl.b.second>b.second) b.second = incl.b.second;
    }
    return *this;
  }
    
  fieldT &l() { return b.first; }
  fieldT &r() { return b.second; }
  const fieldT &l() const{ return b.first; }
  const fieldT &r() const{ return b.second; }

  fieldT width() const { return b.second-b.first; }

  fieldT location() const { return b.first + width()/2; }
  fieldT mid() const { return b.first + width()/2; }

  fieldT randompoint() const { return l() + rand() * width() / RAND_MAX; }
  
  bool includes (const fieldT &elem) const {return (elem >= l() && elem <= r());}
  bool strictly_includes (const fieldT &elem) const {return (elem > l() && elem < r());}
};
typedef interval<physquantity> phq_interval;
template<>
physquantity interval<physquantity>::location() const {
  return (l().werror(0) + width().werror(0)/2) .werror(width()/2);
}
template<>
physquantity interval<physquantity>::randompoint() const { return l().werror(0) + rand() * width().werror(0) / RAND_MAX; }

template<>
bool interval<physquantity>::strictly_includes (const physquantity &elem) const{
  return (elem.l_errorbord() > l() && elem.u_errorbord() < r());
}


interval<physquantity> label(const interval<physquantity> &i, const std::string &nlabel){
  return interval<physquantity>(i.l().wlabel(nlabel), i.r().wlabel(nlabel));
}

template<typename fieldT>
std::ostream &operator<<(std::ostream &tgt, const interval<fieldT> &i){
  tgt << "[" << i.l() << "," << i.r() << "]";
  return tgt;
}


template<class Lookup>
interval<physquantity> measureseq::range_of(const Lookup &lkup) const{
  phq_interval result( centered_about(lkup(front()))
                     , interval_size(0*real1)        );
  for (auto q : *this)
    result.widen_to_include(lkup(q));
  return result;
}


measureseq equidist_prepared_measureseq( const string& pcapt
                                       , const phq_interval& rng
                                       , int resolution         ){
  measureseq prepare;
  physquantity step = rng.width()/resolution;
  for ( physquantity t=rng.l()
      ; prepare.size()<unsigned(resolution)
      ; t+=step                              )
    prepare.push_back(measure().let(pcapt) = t);
  return prepare;        
}
measureseq log_equidist_prepared_measureseq( const string& pcapt
                                           , const phq_interval& rng
                                           , const physquantity& logoffset
                                           , int resolution
                                           , bool use_antialiasing = true ){
  measureseq prepare;
  assert(rng.includes(logoffset));
  physquantity multiplier = ( (rng.r()+logoffset)
                               /(rng.l()+logoffset) ).to(1./resolution);
  for ( physquantity t=rng.l()+logoffset
      ; prepare.size()<unsigned(resolution)
      ; t*=multiplier                        ) {
    if(use_antialiasing){
      prepare.push_back(measure().let(pcapt)
         = phq_interval(t-logoffset, t*multiplier-logoffset).randompoint() );
     }else{
      prepare.push_back(measure().let(pcapt) = t - logoffset);
    }
  }
  return prepare;
}


struct msq_dereferencer {
  virtual physquantity &operator() (measure &v) const =0;
  virtual const physquantity &operator() (const measure &v) const =0;
  virtual auto tryfind(measure& v)const -> physquantity* =0;
  virtual auto tryfind(const measure& v)const -> const physquantity* =0;
  virtual bool operator==(const msq_dereferencer &other) const =0;
  virtual bool operator!=(const msq_dereferencer &other) const {return !operator==(other);}
  virtual ~msq_dereferencer() {}
};

struct msqDereferencer : msq_dereferencer {   //copyable version,
                                             // initializable from any
                                            //  implementing instance
  std::shared_ptr<msq_dereferencer> implementation;

  template<class Implementation>
  msqDereferencer(Implementation impl)
    : implementation(std::make_shared<Implementation>(std::move(impl)))
  {}

  auto operator() (measure &v) const -> physquantity& {
    return (*implementation)(v);              }
  auto operator() (const measure &v) const -> const physquantity& {
    return (*implementation)(v);                                  }
  auto tryfind(measure& v)const -> physquantity* {
    return implementation->tryfind(v);           }
  auto tryfind(const measure& v)const -> const physquantity* {
    return implementation->tryfind(v);                       }
  bool operator==(const msq_dereferencer &other) const {
    return (*implementation) == other;                 }
  bool operator!=(const msq_dereferencer &other) const {
    return (*implementation) != other;                 }
};


class captfinder : public msq_dereferencer{
  captT thatone;
  mutable unsigned int mmry;
  
  struct captionNotFoundException : std::exception {
    std::string message;
    captionNotFoundException(captT thatone)
      : message("Value with caption \""+*thatone+"\" not found in measure.") {}
    const char* what()const noexcept {
      return message.c_str();        }
    ~captionNotFoundException() noexcept {}
  };
//  bool hasdefault; physquantity defaultval;

  void locatein(const measure &fld) const{
    if (thatone==cptNULL){
      cerr << "Trying to dereference something according to a nonspecified variable name.";
      abort();
    }
    int thr=0;
    if (mmry >= fld.size()) mmry = 0;
    while (fld[mmry].caption != thatone){
      if (++mmry >= fld.size()) {
        mmry = 0;
        if (++thr > 1) {
     /*     if (hasdefault){
            mmry=-1; return;
           }else*/
            throw captionNotFoundException(thatone);
        }
      }
    }
  }

  bool can_locatein(const measure &fld) const{
    if (thatone==cptNULL) return false;
    int thr=0;
    if (mmry >= fld.size()) mmry = 0;
    while (fld[mmry].caption != thatone){
      if (++mmry >= fld.size()) {
        mmry = 0;
        if (++thr > 1) return false;
      }
    }
    return true;
  }

 public:
  captfinder &operator=(const string &s) {
    thatone = globalcaptionsscope.insert(s).first;
    return *this;
  }

  captfinder(): thatone(cptNULL), mmry(0) {} //, hasdefault(false) {}
  captfinder(const string &s): mmry(0) { //, hasdefault(false) {
    operator=(s);
  }

  captfinder(const captT &ni) : thatone(ni) {}

  auto tryfind(const measure& v)const -> const physquantity* {
    if(can_locatein(v)) return &v[mmry];
     else return nullptr;
  }
  auto tryfind(measure& v)const -> physquantity* {
    if(can_locatein(v)) return &v[mmry];
     else return nullptr;
  }

  physquantity/*::namestable_access*/ &operator() (measure &fld) const{
    locatein(fld); return fld[mmry];  }
  const physquantity & operator() (const measure &fld) const{
    locatein(fld); /*if (mmry==-1) return defaultval;*/ return fld[mmry];  }

  bool operator==(const msq_dereferencer &other) const{
    if (typeid(other) != typeid(*this)) return false;
    return thatone == dynamic_cast<const captfinder*>(&other)->thatone;
  }
  
  const string &operator *()const{return *thatone;}

};


measure measureseq::randomrepresentatives() const{
  measure result(randommeasure());
  for (measure::iterator i=result.begin(); i!=result.end(); ++i){
    *i = randomrepresentative(captfinder(i->caption));
  }
  return result;
}  


class p_label{
  string l;
 public:
  p_label(const string &nl){l=nl;}
  friend physquantity operator,(physquantity q, const p_label &nl);
};
physquantity operator,(physquantity q, const p_label &nl){
  return q.label(nl.l);
}
class g_label{
  string l;
 public:
  g_label(const string &nl){l=nl;}
  friend physquantity operator,(physquantity &q, const g_label &nl);
};
physquantity operator,(physquantity &q, const g_label &nl){
  return q.label(nl.l);
}



class measureindexpointer : public msq_dereferencer {
  unsigned int thatone;
 public:
  measureindexpointer(){thatone = 0;}
  measureindexpointer(const measureindexpointer &cpyfrom){thatone = cpyfrom.thatone;}
  measureindexpointer(const unsigned int &thisplease){thatone = thisplease;}
  
  physquantity & operator() (measure &fld) const{
    return fld[thatone];
  }
  const physquantity & operator() (const measure &fld) const{
    return fld[thatone];
  }

  auto tryfind(const measure& v)const -> const physquantity* {
    if(thatone<v.size()) return &v[thatone];
     else return nullptr;
  }
  auto tryfind(measure& v)const -> physquantity* {
    if(thatone<v.size()) return &v[thatone];
     else return nullptr;
  }

  bool operator== (const msq_dereferencer &other) const{
    if (typeid(other) != typeid(*this)) return false;
    return thatone == dynamic_cast<const measureindexpointer*>(&other)->thatone;
  }
};

template<typename selector>
class distancetothat {
  physquantity thatone; selector guideme;
 public:
  distancetothat(){}
  distancetothat(const physquantity &thisplease){thatone = thisplease;}
  distancetothat(physquantity thisplease, selector newguide){
    thatone = thisplease;  guideme = newguide;}
  physquantity operator() (measure &fld) const {
/*    cout << thatone << " - ";
    cout << guideme(fld) << " = ";
    cout << (guideme(fld) - thatone).abs() << endl;*/
    return (guideme(fld) - thatone).abs();
  }
};

template<int guideme>
class sdistancetobyindex {
  physquantity thatone;
 public:
  sdistancetobyindex(const physquantity &thisplease){thatone = thisplease;}
  physquantity operator() (measure &fld) {
//  cout << "a";
  //cout << fld[2];
    return (fld[guideme] - thatone).abs();
  }
};


/*    for(measure::iterator it = fld.begin(); it!=fld.end(); ++it){
      if (thatone.compatible(*it)) return (*it-thatone).abs();
    }
    return (thatone*0).to(0);
*/



namespace_cqtxnamespace_CLOSE





#include "cqtx_io.cpp"
#include "cqtx_alg.h"
