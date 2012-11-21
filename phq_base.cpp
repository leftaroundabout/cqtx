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


#include <exception>
#include <array>

namespace_cqtxnamespace_OPEN


typedef double phq_underlying_float;     // TODO: actually utilize this type in the physquantity definitions, rather than plain doubles.

class phDimension{
  typedef int32_t dimensionsfixedp;
  static const dimensionsfixedp intrepmultiplier = 39916800;    // ≡ 11!
  dimensionsfixedp cint,gint,sint;            //cgs-Dekomposition der Einheit
 public:
  phDimension() {}   //undefinierte Dimension
  phDimension(const double &ci, const double &gi, const double &si):
    cint(ci * intrepmultiplier),
    gint(gi * intrepmultiplier),
    sint(si * intrepmultiplier)
  {}
  float c() {return float(cint)/intrepmultiplier;}
  float g() {return float(gint)/intrepmultiplier;}
  float s() {return float(sint)/intrepmultiplier;}
  bool operator ==(const phDimension &cmpd) const{
    return (cint==cmpd.cint && gint==cmpd.gint && sint==cmpd.sint);  }
  bool operator !=(const phDimension &cmpd) const{
    return (cint!=cmpd.cint || gint!=cmpd.gint || sint!=cmpd.sint);  }
  bool operator ==(const int &cmpd) const{
    return (cint==0 && gint==0 && sint==0);  }
  bool conjugates(const phDimension &cmpd) const{
    return (cint==-cmpd.cint && gint==-cmpd.gint && sint==-cmpd.sint);  }
  bool operator !=(const int &cmpd) const{
    return (cint!=0 || gint!=0 || sint!=0);  }
  phDimension &operator *=(const phDimension &mtpv){
    cint+=mtpv.cint; gint+=mtpv.gint; sint+=mtpv.sint;
    return (*this);  }
  phDimension &operator /=(const phDimension &mtpv){
    cint-=mtpv.cint; gint-=mtpv.gint; sint-=mtpv.sint;
    return (*this);  }
  phDimension &operator ^=(const float &pwv){
    cint*=pwv; gint*=pwv; sint*=pwv;
    return (*this);  }
  bool lessincollationordering(const phDimension &other) const{
    if (cint<other.cint) return true; if (cint>other.cint) return false;
    if (gint<other.gint) return true; if (gint>other.gint) return false;
    if (sint<other.sint) return true; /*if (sint>=other.sint)*/ return false;
  }
  
  enum class cgsSelect : size_t {
    c=0, g, s                          };

 private:
  static auto gcd(dimensionsfixedp n, dimensionsfixedp m) -> dimensionsfixedp {
    return n%m==0 ? m : gcd(m,n%m);                                           }
 public:
  
  std::string fractionshow(cgsSelect dcd)const {
    dimensionsfixedp vi = *(&cint + (size_t)(dcd));
    std::stringstream result;
    if(vi<0) {
      result << '-';
      vi *= -1;
    }
    dimensionsfixedp gcdi = gcd(vi, intrepmultiplier);
    vi /= gcdi;
    if(gcdi<intrepmultiplier) {
      result << vi << '/' << intrepmultiplier/gcdi;
     }else{
      result << vi;
    }
    return result.str();
  }
  
  friend class phqdata_binstorage;
};
struct phUnit{
  phDimension Dimension;
  double cgsPrefactor;
  string uName;
  phUnit(const phDimension &nuDim, const double &pfi, const std::string &iName):
    Dimension(nuDim),
    cgsPrefactor(pfi),
    uName(iName)
  {}
  phUnit(const double &ci, const double &gi, const double &si, const double &pfi, const string &iName):
    Dimension(ci, gi, si),
    cgsPrefactor(pfi),
    uName(iName)
  {}
  bool operator==(const phUnit &other) const{
    return Dimension==other.Dimension && cgsPrefactor==other.cgsPrefactor;}
  bool operator!=(const phUnit &other) const{ return !(operator==(other)); }
  bool Usymbol_alphabetically_less(const phUnit &other) const{return uName<other.uName;}
  const class physquantity to(const double &expn) const;
  const string uCaption() const{return (uName.size()>0)? uName : "scalar";}
};
const phUnit   //Basiseinheiten:
        //        c     g      s
real1            (0,    0,     0,   1,          ""),
seconds_base     (0,    0,     1,   1,          "s"),
meters_base      (1,    0,     0,   100,        "m"),
kilograms_base   (0,    1,     0,   1000,       "kg"),
amperes_base     (3./2, 1./2, -2,   2997924580.,"A");




//template<unsigned NUnitsInvolved, unsigned NUnknownInvolved=0>
struct physDimensionException : std::exception {
  typedef std::vector<const phUnit*> InvdUnitsCnt;
  typedef std::vector<phDimension> InvdUnknwCnt;
  InvdUnitsCnt units_involved;
  InvdUnknwCnt unknowns_involved;

  mutable std::string message; //it's presumably bad practise to have heap-allocated
                              // objects in an exception. FIXME, or, don't.

  template<typename ...UnitsList>
  physDimensionException(UnitsList... l) : units_involved{{l...}} {}
  physDimensionException( std::initializer_list<const phUnit*> us
                        , std::initializer_list<phDimension> uks )
    : units_involved(us), unknowns_involved(uks) {}

  const char* what()const noexcept {
    try {
      std::stringstream readout;
      readout << "physical-dimension-exception ";
      if(unknowns_involved.size() > 0) {
        readout << "involving units:";
        for(auto& u: units_involved) readout << ' ' << u->uName;
      }
      if(unknowns_involved.size() > 0) {
        if(unknowns_involved.size() > 0)
          readout << "\n    and unknown units of dimension (   c   ,   g   ,   s     )\n";
         else
          readout << "\n      involving units of dimension (   c   ,   g   ,   s     )\n";
        for(auto& u: unknowns_involved) readout
           << "                                    " << std::left
           << std::setw(7) << u.fractionshow(phDimension::cgsSelect::c) << ','
           << std::setw(7) << u.fractionshow(phDimension::cgsSelect::g) << ','
           << std::setw(7) << u.fractionshow(phDimension::cgsSelect::s) << '\n';
       }else{
        readout << ".\n";
      }
      message = readout.str();
      return message.c_str();
     }catch(...){
      return "physical-dimension-exception (failed to fetch information about the involved units)";
    }
  }
  
  ~physDimensionException() noexcept {}
};



typedef phUnit t_unit;

struct compunitpointerbyalphabeticalsymbolorder{
  bool operator()(const phUnit *one, const phUnit *other) const{return one->Usymbol_alphabetically_less(*other) ;}};
struct compunitpointerbydimension{
  bool operator()(const phUnit *one, const phUnit *other) const{return one->Dimension.lessincollationordering(other->Dimension) ;}};
//  typedef std::multiset<const phUnit *, cophUnit(dsrd,1,"")mpunitpointerbyalphabeticalsymbolorder> alphabetical;

class physquantity;

class unitsscope{
  typedef std::set<const phUnit *, compunitpointerbyalphabeticalsymbolorder> alphabetical;
  alphabetical by_name;
  typedef std::multiset<const phUnit *, compunitpointerbydimension> dimensional;
  dimensional by_dimension;
 public:
  typedef alphabetical::iterator iterator;
  typedef std::pair<dimensional::iterator,dimensional::iterator> dimrange;
  iterator begin() const{ return by_name.begin(); }
  iterator end() const{ return by_name.end(); }
  
  string caption;

  std::pair<iterator, bool> insert(const phUnit *insm){
    std::pair<iterator,bool> res = by_name.insert(insm);
    if (res.second) by_dimension.insert(insm);
    return res;
  }

  unitsscope(){ insert(&real1); }
  unitsscope(string ncaption) : caption(ncaption) { insert(&real1); }
  template<typename UnitPointersContainer>
  unitsscope(string ncaption, const UnitPointersContainer &cnt)
    : caption(ncaption) {
    for(auto u: cnt) insert(u);
  }
  unitsscope(string ncaption, std::initializer_list<unitsscope> cps) {
    for (auto sc : cps) {
      for (auto u : sc)
        insert(u);
    }
  }
  
  iterator find(const string &dsrd) const{
/*  phUnit *t = new phUnit(0,0,0,1, dsrd); iterator res = by_name.find(t); delete t; */ //???
    phUnit t(0,0,0,1, dsrd);
    iterator res = by_name.find(&t);
    return res;
  }
  iterator find(const phDimension &dsrd) const{
    phUnit t(dsrd,1,"");
    dimensional::iterator b = by_dimension.find(&t);

    if (b == by_dimension.end()) return end();
    return by_name.find(*b);
  }
  dimrange equal_range(const phDimension &dsrd) const{
    phUnit t(dsrd,1,"");
    dimrange res = by_dimension.equal_range(&t);
//    delete t;
    return res;
  }
  const phUnit *fineqrange(const phDimension &dsrd){
    return *equal_range(dsrd).first;
  }
  iterator find(const phUnit *dsrd) const{
    dimrange b = by_dimension.equal_range(dsrd);
    for(dimensional::iterator a=b.first; a!=b.second; ++a)
      if ((*a)->cgsPrefactor == dsrd->cgsPrefactor) return by_name.find(*a);
    return end();
  }
  bool has(const phUnit *dsrd) const{return by_dimension.find(dsrd)!=by_dimension.end();}
  
  dimrange suitablefor(const physquantity &) const;
  
  std::vector<const phUnit *>gatherdimrange(const dimrange &rng) const{
  /*  dimensional::iterator i=rng.first;
    cout << (*rng.first)->uName;
    if (rng.first == rng.second)
      cout << ",";
    */
   return std::vector<const phUnit *>(rng.first, rng.second);
  }
  std::vector<const phUnit *> suitableUnitsfor(const physquantity &src) const{
    return gatherdimrange(suitablefor(src)); }
  
};
unitsscope *default_defaultUnitsc();
unitsscope *defaultUnitsc=default_defaultUnitsc();

inline const phUnit *suitableUnitfor(const physquantity &, const unitsscope * =defaultUnitsc);




//string unnamedphysicalquantity("unnamed physical quantity");
typedef std::set<string> captionsscope;
typedef captionsscope::iterator captT;
captionsscope globalcaptionsscope;
//const std::set<string, captcomparator>captionsNULLscope;
const captT cptNULL=globalcaptionsscope.end(); // = captionsNULLscope.begin();
//#define cptNULL NULL






class physquantity {                           //Klasse f\"ur physikalische Gr\"o{\ss}en
  mutable phDimension myDim;  //trotz mutable-Deklaration NICHT const zu modifizieren, nur im Rahmen
 protected:                  // einer Einheitenspezifizierung \"andern!
  mutable const phUnit *tunit;   //Einheitenumrechnung ist nicht-destruktiv, daher mutable
  mutable double valintu;       // Wert der Variablen in der Einheit tunit
  mutable double valincgs;     //  Wert der Variablen in cgs
  
  const phDimension &Dimension() const{
    if (tunit==NULL) return myDim; return tunit->Dimension;
  }
  
  struct ErrorDat{
//    physquantity *src;
    double valincgs;
    double valintu;
    ErrorDat(double vcn, double vtn): valincgs(vcn), valintu(vtn) {}
    ErrorDat() {}
  }mutable myError;
  
//  std::map<physquantity * , ErrorDat> ErrOrigin;
//  std::stack<physquantity *> ErrHeritage;

  bool disjointErrorAncestry(const physquantity &other){
    return true;
  }

  void chkUnitCompatible(const phUnit &other) const{
    if (tunit!=NULL){
      if (tunit->Dimension != other.Dimension){         //Dies bedeutet dass die Einheiten inkompatibel sind
        throw physDimensionException(tunit, &other);
      /*unitconversion utrouble(tunit, &other);
        throw (utrouble);*/
      }
     }else{
      if (myDim != other.Dimension)         //Dies bedeutet dass die Einheiten inkompatibel sind
        //cout << "Danger: weird units trouble";
        throw physDimensionException({&other}, {myDim});
//        throw other;
    }
  }
 public:
  bool isUnitCompatible(const physquantity &other) const{
    if (tunit!=NULL && other.tunit!=NULL){
      if (tunit->Dimension != other.tunit->Dimension){          // => incomatible units
        return false;
      }
     }else if(tunit!=NULL){
      if (tunit->Dimension != other.myDim)
        return false;
     }else if(other.tunit!=NULL){
      if (myDim != other.tunit->Dimension)
        return false;
     }else{
      if (myDim != other.myDim){
        return false;
      }
    }
    return true;
  }
 protected:
  void chkUnitCompatible(const physquantity &other) const{
    if (tunit!=NULL && other.tunit!=NULL){
      if (tunit->Dimension != other.tunit->Dimension){         // => incomatible units
        throw physDimensionException(tunit, other.tunit);
      }
     }else if(tunit!=NULL){
      if (tunit->Dimension != other.myDim)
        throw physDimensionException({tunit},{other.myDim});
     }else if(other.tunit!=NULL){
      if (myDim != other.tunit->Dimension)
        throw physDimensionException({other.tunit},{myDim});
     }else{
      if (myDim != other.myDim){
        throw physDimensionException({},{myDim,other.myDim});
//        throw std::make_pair(myDim, other.myDim);
        abort();
      }
    }
  }
  void fixcgs() const{valincgs = valintu * tunit->cgsPrefactor;
                myError.valincgs = myError.valintu * tunit->cgsPrefactor;
                myDim = tunit->Dimension;}                                       //Diese Funktionen niemals
  void fixtu() const{valintu = valincgs / tunit->cgsPrefactor;                  // aufrufen wenn tunit==NULL!
               myError.valintu = myError.valincgs / tunit->cgsPrefactor;}

  void freefromunitsys(){ if (tunit!=NULL){
    fixcgs();    tunit=NULL;
  }}
//  void assimilate(const physquantity &model){  }

  
 public:
  captT caption;
//  string *caption;

  physquantity(const physquantity &cpyfrom):                   //Standard-Kopie
  	myDim(cpyfrom.myDim),
    tunit(cpyfrom.tunit),
    caption(cpyfrom.caption)
#if 0
,   valintu(cpyfrom.valintu),
    valincgs(cpyfrom.valincgs),
    myError(cpyfrom.myError.valincgs, cpyfrom.myError.valintu)
  {
 #else
  {
    if (tunit!=NULL) {
      valintu = cpyfrom.valintu;
      myError.valintu = cpyfrom.myError.valintu;
     }else{
      valincgs = cpyfrom.valincgs;
      myError.valincgs = cpyfrom.myError.valincgs;
    }
#endif
  }
  explicit physquantity(const phDimension &nDim):       //Erstelle eine null-Instanz der Gr\"o{\ss}e
    myDim(nDim),
    tunit(NULL),
    valintu(0),
    valincgs(1),
    myError(0, 0),
    caption(cptNULL)
  {}
  physquantity(double nValue, const phUnit &nunit):       //Erstelle mit Wert nValue in der angegebenen Einheit
    myDim(nunit.Dimension),
    tunit(&nunit),
    valintu(nValue),
    caption(cptNULL)      {
    myError.valintu = 0;
  }
  physquantity(const phUnit &nunit):                   //Erstelle mit Wert 1 in der angegebenen Einheit
    myDim(nunit.Dimension),
    tunit(&nunit),
    valintu(1),
    caption(cptNULL)      {
    myError.valintu = 0;
  }
  physquantity():       //Erstelle eine nicht-definierte Instanz der Gr\"o{\ss}e
  //  myDim(0,0,0),
    tunit(NULL),
    caption(cptNULL)    {
 //   myError.valincgs = valincgs = 0;
  }
  physquantity(const int &intinit):
    myDim(0,0,0),
    valintu(intinit),
    valincgs(intinit),
	  caption(cptNULL)   {
    if (intinit==0) tunit = NULL; else tunit = &real1;
    myError.valintu = myError.valincgs = 0;
  }
  physquantity(const lambdalike::polymorphicNumLiteral& init):
    myDim(0,0,0),
    valintu(init),
    valincgs(init),
          caption(cptNULL)   {
    if (double(init)==0.) tunit = NULL; else tunit = &real1;
    myError.valintu = myError.valincgs = 0;
  }
//  physquantity(const physquantity &cpyfrom, const string &nameme){
  //  *this = cpyfrom;
    //caption = globalcaptionsscope.insert(nameme).first;
 // }
  

  physquantity &operator=(const int &intinit){
    myDim = phDimension(0,0,0);
    if (intinit==0) tunit = NULL; else tunit = &real1;
    valintu = valincgs = intinit;
    myError.valintu = myError.valincgs = 0;
    return *this;
  }
  physquantity &operator=(const physquantity &cpyfrom){
    //if (globaldebugflag) cout << "operator=" << endl;
    if (cpyfrom.caption!=cptNULL //||
        && caption==cptNULL)
      { caption = cpyfrom.caption; }
    tunit = cpyfrom.tunit;
    if (tunit!=NULL) {
   //   if (globaldebugflag) cout << "tunit!=NULL" << endl;
      valintu = cpyfrom.valintu;
      myError.valintu = cpyfrom.myError.valintu;
     }else{
 //     if (globaldebugflag) cout << "tunit==NULL, Dim==" << cpyfrom.myDim << endl;
      myDim = cpyfrom.myDim;
      valincgs = cpyfrom.valincgs;
      myError.valincgs = cpyfrom.myError.valincgs;
    }
    return *this;
  }
  void overwritewith(const physquantity &cpyfrom) {  //more ruthless version of operator=, also overwrites the caption always
    tunit = cpyfrom.tunit;
    myDim = cpyfrom.myDim;
    valincgs = cpyfrom.valincgs;
    valintu = cpyfrom.valintu;
    myError.valincgs = cpyfrom.myError.valincgs;
    myError.valintu = cpyfrom.myError.valintu;
    caption = cpyfrom.caption;
  }

  struct namestable_access{
    physquantity *accessp;
    namestable_access(physquantity &nphr) : accessp(&nphr) {}
    physquantity &operator=(const physquantity &cpyfrom){
      accessp->tunit = cpyfrom.tunit;
      if (accessp->tunit!=NULL) {
        accessp->valintu = cpyfrom.valintu;
        accessp->myError.valintu = cpyfrom.myError.valintu;
       }else{
        accessp->myDim = cpyfrom.myDim;
        accessp->valincgs = cpyfrom.valincgs;
        accessp->myError.valincgs = cpyfrom.myError.valincgs;
      }
      return *accessp;
    }
    physquantity &operator+=(const physquantity &);
    physquantity &operator-=(const physquantity &);
    physquantity operator+(const physquantity &);
    physquantity operator-(const physquantity &);
    bool operator<(const physquantity &);
  };
  physquantity(const namestable_access &cpyfrom){
    *this = *cpyfrom.accessp;
  }
  
//  physquantity &namestable_access::operator=(const physquantity &cpyfrom) {

  double &operator [](const phUnit &tgunit){     //Referenz auf die Variable in der Einheit tgunit
    if (&tgunit == NULL){ cout << "Must use unit to dereference to unit!"; abort();}
    if (tunit!=&tgunit){
      if ((tunit==NULL)? valincgs!=0 : valintu!=0)  chkUnitCompatible(tgunit);
      if (tunit!=NULL) fixcgs();
      tunit = &tgunit;
      fixtu();
    }
//    cout << myDim.c << myDim.g << myDim.s << endl;
    return(valintu);
  }
  const double &operator [](const phUnit &tgunit)const{     //Referenz auf die Variable in der Einheit tgunit
    if (&tgunit == NULL){ cout << "Must use unit to dereference to unit!"; abort();}
    if (tunit!=&tgunit){
      if ((tunit==NULL)? valincgs!=0 : valintu!=0)  chkUnitCompatible(tgunit);
      if (tunit!=NULL) fixcgs();
      tunit = &tgunit;
      fixtu();
    }
//    cout << myDim.c << myDim.g << myDim.s << endl;
    return(valintu);
  }
  double value_in_unit(const phUnit &tgunit) const{           //Die Variable in der Einheit tgunit
    if (&tgunit == NULL){ cout << "Must use unit to dereference to unit!"; abort();}
    if (tunit!=&tgunit){
      if (valincgs!=0 || tunit!=NULL) chkUnitCompatible(tgunit);
      if (tunit!=NULL) return valintu * tunit->cgsPrefactor / tgunit.cgsPrefactor;
      return valincgs / tgunit.cgsPrefactor;
    }
    return(valintu);
  }
  physquantity operator ()(const phUnit &tgunit) const{     //Kopie der Variable, Einheit tgunit
    if ( is_identical_zero() ) return physquantity (0, tgunit);
    physquantity result = *this;
    if (result.tunit!=&tgunit){
      result.chkUnitCompatible(tgunit);
      if (result.tunit!=NULL) result.fixcgs();
      result.tunit = &tgunit;
      result.fixtu();
    }
    return(result);
  }
  physquantity tryfindUnit(const unitsscope *sc=defaultUnitsc) const{
    if (tunit==NULL){
      const phUnit *ttU = suitableUnitfor(*this, sc);
      if (ttU!=NULL){
        operator[](*ttU);
      }
    }
    return *this;
  }

  bool is_identical_zero() const{
    if (tunit==NULL){ if(valincgs==0 && myError.valincgs==0) return true; }
     else { if(valintu==0 && myError.valintu==0) return true; }
    return false;
  }
  
  physquantity &tryconvto(const phUnit &tgunit){     //Nehme, sofern m\"oglich, diese Einheit an
    if (&tgunit!=NULL && tunit!=&tgunit){
      if (tunit!=NULL) {
        if (tunit->Dimension == tgunit.Dimension){
          fixcgs();
          tunit = &tgunit;
          fixtu();
        }
       }else{
        if (myDim == tgunit.Dimension){
          tunit = &tgunit;
          fixtu();
        }
      }
     }
    return(*this);
  }
//  friend double dbl(const physquantity &src);
  explicit operator double()const {return dbl();}
  double dbl() const{
    if(is_identical_zero()) return 0;
    if (Dimension() != 1){
      if (tunit==NULL) {
        if (valincgs!=0 || myError.valincgs!=0){
          cerr << *this << "Fehler: nicht dimensionslos";
          //abort();
          throw Dimension();
        }
       }else{
        if (valintu!=0 || myError.valintu!=0)
          throw physDimensionException(tunit);
      }
    }
    if (tunit==NULL) {
      return valincgs;
     }else{
      return (valintu * tunit->cgsPrefactor);
    }
  }
  const phUnit *preferredUnit() const{
    return tunit;
  }
  physquantity error() const{                    //Messfehler der Gr\"o{\ss}e
    physquantity result = *this;
    result.valintu = myError.valintu;
    result.valincgs = myError.valincgs;
    result.myError.valintu =
    result.myError.valincgs = 0;
    return(result);
  }
  bool couldbe_anything(){
    return (tunit==NULL
            ? myError.valincgs==HUGE_VAL
            : myError.valintu==HUGE_VAL
           );
  }
  
  void error(const physquantity &nerr){                    //Setze Messfehler
    if (tunit == NULL){
      if (valincgs!=0) chkUnitCompatible(nerr);
      if (nerr.tunit == NULL){
        myError.valincgs = fabs(nerr.valincgs);
       }else{
        tunit = nerr.tunit;
        fixtu();
        myError.valintu = fabs(nerr.valintu);
      }
     }else{
      if (nerr.tunit == NULL){
        myError.valintu = fabs(nerr.valincgs/tunit->cgsPrefactor);
       }else{
        if (tunit==nerr.tunit)
          myError.valintu = fabs(nerr.valintu);
         else
          myError.valintu = fabs(nerr.valintu*nerr.tunit->cgsPrefactor
                                          / tunit->cgsPrefactor);
      }
    }
  }
  void error(int nerr){                    //Setze Messfehler
    if (nerr != 0){
      cerr << "Nonzero integer error for physical quantity??\n";
      abort();
    }
    if (tunit == NULL)
      myError.valincgs = 0;
     else
      myError.valintu = 0;
  }
  physquantity &seterror(const physquantity &nerr) { error(nerr);  return(*this);  }
  physquantity &seterror(int nerr) { error(nerr); return(*this);  }
  physquantity plusminus(const physquantity &nerr) const { return(physquantity(*this).seterror(nerr));  }
  physquantity werror(const physquantity &nerr) const { return(physquantity(*this).seterror(nerr));  }
  physquantity werror(int nerr) const { return(physquantity(*this).seterror(nerr));  }
  physquantity plusminusrel(const float &nerr) const { return(physquantity(*this).seterror(physquantity(*this)*nerr));  }
  physquantity u_errorbord() const{  return(*this + error());  }
  physquantity l_errorbord() const{  return(*this - error());  }
  physquantity &scale_error(double lambda) {
 //   error(error() * lambda);
    myError.valincgs *= lambda;
    myError.valintu *= lambda;
    return *this;
  }
  
  physquantity &uncertain_collapse(std_normaldistributed_randomgen &r = defaultsimplenormaldistributedrandomgenerator) {
    if (tunit == NULL) {
      valincgs += myError.valincgs * r();
      myError.valincgs = 0;
     }else{
      valintu += myError.valintu * r();
      myError.valintu = 0;
    }
    return *this;
  }
  physquantity uncertain_collapsed(std_normaldistributed_randomgen &r = defaultsimplenormaldistributedrandomgenerator)
           const { return physquantity(*this).uncertain_collapse(r); }

  string cptstr() const{
    if (caption == cptNULL) return "unnamed physical quantity";
    return(*caption);
  }
  string LaTeXcaptstr() const{
    if (caption == cptNULL) return "unnamed physical quantity";
    return("$" + *caption + "$");
  }
  physquantity &label(const string &lbl) { caption=globalcaptionsscope.insert(lbl).first; return(*this); }
//  physquantity label(const string &lbl) const { return physquantity(*this).label(lbl); }
  physquantity wlabel(const string &lbl) const { return physquantity(*this).label(lbl); }
  
  struct stringoutpT{
    string captstr, valstr, errstr, unitstr, decptstr;
    signed int decpot;
    static const int num_of_unnecessary_leading_zeroes=0;

    stringoutpT(physquantity *me, const phUnit &outptU){

      if (me->caption != cptNULL) captstr = *me->caption;

      stringstream vk, wk, pk;

      if (me->is_identical_zero()) {

        wk << 0;
        valstr = errstr = wk.str();
        decpot = 0;

       }else{

        me->chkUnitCompatible(outptU);
        me->tryconvto(outptU);
        double val=me->valintu, err = me->myError.valintu;
        wk << std::fixed;
        vk << std::fixed;
        if (val<0) {valstr="-"; val = -val;}
        if (val>0) {
          decpot = (int) floor(log10(val));
          decpot *= (decpot>2 || decpot<-2);
          if (err>0){
            wk << std::setprecision(decpot - (int)floor(log10(err/2)));
            vk << std::setprecision(decpot - (int)floor(log10(err/2)));
            vk << err*pow(10, -decpot);
            errstr = vk.str();
            if (*(errstr.begin())=='0'){
              errstr.erase(errstr.begin());
              for(int i = 0; i < num_of_unnecessary_leading_zeroes; ++i){
                errstr = "{_{^0}}" + errstr;
              }
            }
          }
          wk << val*pow(10, -decpot);
          valstr += wk.str();
          if (*(valstr.begin())=='0') {
            valstr.erase(valstr.begin());
            for(int i = 0; i < num_of_unnecessary_leading_zeroes; ++i){
              valstr = "{_{^0}}" + valstr;
            }
           }else if (decpot==0){
            if(val == (double) (long) val){
              while (*(valstr.rbegin())=='0') valstr.resize(valstr.size()-1);
              if (*(valstr.rbegin())=='.') valstr.resize(valstr.size()-1);
            }
          }
         }else{
          valstr="0";
          if (err>0){
            decpot = (int) floor(log10(err));
            decpot *= (decpot>2 || decpot<-2);
            vk << std::setprecision(decpot - (int)floor(log10(err/2)));
            vk << err*pow(10, -decpot);
            errstr = vk.str();
            if (*(errstr.begin())=='0') {
              errstr.erase(errstr.begin());
              for(int i = 0; i < num_of_unnecessary_leading_zeroes; ++i){
                errstr = "{_{^0}}" + errstr;
              }
            }
           }else{
            decpot = 0;
          }
        }

      }

      unitstr = outptU.uName;
      pk << decpot;
      decptstr = pk.str();

    }

  };
  std::string mathLaTeXval(bool shwCapt, bool shwVal, bool shwErr, bool shwUnit, const phUnit &outptU) {
    stringoutpT cmpts(this, outptU);
    string result="";
    if (shwCapt) {
      if (shwVal) result += cmpts.captstr + " = ";
       else if (shwErr) result += "\\Err{" + cmpts.captstr + "} = ";
       else result += cmpts.captstr;
    }
    if (shwVal && shwErr&&cmpts.errstr.size()>0 && (shwUnit || cmpts.decpot!=0)) result += "(";
    if (shwVal){
      result += cmpts.valstr;
    }
    if (shwErr){
      if ((shwVal && !shwCapt && shwErr && cmpts.errstr.size()>0) || !shwVal)
        result += "\\pm";
      if (cmpts.errstr.size()>0)
        result += cmpts.errstr;
       else if (!shwVal) result += "0";   //  && shwCapt
    }
    if (shwVal && shwErr&&cmpts.errstr.size()>0 && (shwUnit || cmpts.decpot!=0)) result += ")";
    if (shwVal || (shwErr&&cmpts.errstr.size()>0)){
      if (cmpts.decpot!=0) result += "\\cdot10^";
      if (cmpts.decptstr.size()>1) result += "{";
      if (cmpts.decpot!=0) result += cmpts.decptstr;
      if (cmpts.decptstr.size()>1) result += "}";
    }
    if (shwUnit) {
      if (shwCapt && !shwVal && !shwErr) result += "[";
       else if (shwVal) result += "\\:";
      if (shwCapt || shwVal || (shwErr&&cmpts.errstr.size()>0))
        result += "\\mathrm{" + cmpts.unitstr + "}";
      if (shwCapt && !shwVal && !shwErr) result += "]";
    }

    return result; 
  }
  std::string mathLaTeXval(bool shwCapt, bool shwVal, bool shwErr, bool shwUnit, const phUnit &outptU) const {
    return physquantity(*this).mathLaTeXval(shwCapt, shwVal, shwErr, shwUnit, outptU);                       }
  std::string LaTeXval(bool shwCapt, bool shwVal, bool shwErr, bool shwUnit, const phUnit &outptU) const{
    std::string result = mathLaTeXval(shwCapt, shwVal, shwErr, shwUnit, outptU);
    if (result.size()==0) result = " ";
    return "$" + result + "$";
  }
    
    
    
  physquantity &operator +=(const physquantity &addv){
    if (is_identical_zero())
      return (*this=addv);
     else if (addv.is_identical_zero())
      return *this;
     else
      chkUnitCompatible(addv);

    if (addv.tunit==NULL) {
      if(tunit!=NULL){
        fixcgs(); tunit=NULL;
      }
      valincgs += addv.valincgs;
      if (!(myError.valincgs==0 && addv.myError.valincgs==0))
        myError.valincgs = hypot(myError.valincgs, addv.myError.valincgs);
     }else{
      if(tunit!=NULL){
        if(tunit!=addv.tunit){
          fixcgs(); tunit=addv.tunit; fixtu();}
       }else{
        tunit=addv.tunit; fixtu();
      }
      valintu += addv.valintu;
      if (!(myError.valintu==0 && addv.myError.valintu==0))
        myError.valintu = hypot(myError.valintu, addv.myError.valintu);
    }
    return (*this);
  }
  physquantity &operator +=(double addv){
    if (tunit != &real1)
      (*this)[real1];
    valintu += addv;
    return *this;
  }
  physquantity &AddAsExactVal(const physquantity &addv){
    if (is_identical_zero()){
      if (addv.tunit!=NULL){
        tunit = addv.tunit;
        myDim = addv.myDim;
        valintu = addv.valintu;
       }else{
        tunit = NULL;
        myDim = addv.myDim;
        valincgs = addv.valincgs;
      }
      return *this;
     }else if (addv.is_identical_zero()){
      return *this;
     }else
      chkUnitCompatible(addv);

    if (addv.tunit==NULL) {
      if(tunit!=NULL){
        fixcgs(); tunit=NULL;
      }
      valincgs += addv.valincgs;
     }else{
      if(tunit!=NULL){
        if(tunit!=addv.tunit){
          fixcgs(); tunit=addv.tunit; fixtu();}
       }else{
        tunit=addv.tunit; fixtu();
      }
      valintu += addv.valintu;
    }
    return (*this);
  }
  physquantity operator +(const physquantity &addv) const{
    return physquantity(*this)+=addv;
  }
  physquantity plusexactly(const physquantity &addv) const{
    return physquantity(*this).AddAsExactVal(addv);
  }
  physquantity &operator *=(const double &mtpv){
    valintu *= mtpv;
    valincgs *= mtpv;
    myError.valintu *= std::abs(mtpv);
    myError.valincgs *= std::abs(mtpv);
    return (*this);
  }
  physquantity operator *(const double &mtpv) const{
    return physquantity(*this) *= mtpv;
  }

  template<class RHSMultiplier>
  auto operator*(const lhsMultipliable<RHSMultiplier> &mtpv) const
      -> decltype(mtpv.polymorphic().lefthand_multiplied(physquantity())) {
    return mtpv.polymorphic().lefthand_multiplied(*this);
  }

  physquantity &operator /=(const double &mtpv){
    valintu /= mtpv;
    valincgs /= mtpv;
    myError.valintu /= std::abs(mtpv);
    myError.valincgs /= std::abs(mtpv);
    return (*this);
  }
  const physquantity operator /(const double &mtpv) const{
    return physquantity(*this) /= mtpv;
  }
  const physquantity operator -() const{
    physquantity result(*this);
    result.valintu *= -1;
    result.valincgs *= -1;
    return result;
  }
  physquantity &operator -=(physquantity addv){
    return (*this += -addv);
  }
  physquantity operator -(const physquantity &addv) const{
    return (*this + -addv);
  }
  physquantity &SubstractAsExactVal(const physquantity &addv){
    return AddAsExactVal(-addv);
  }
  physquantity minusexactly(const physquantity &addv) const{
    return physquantity(*this).AddAsExactVal(-addv);
  }
  
  physquantity &statistically_approach(const physquantity &tgt, double significance) {
    // assert ( 0 < significance < 1 )
    double us = 1-significance, d;
    
    chkUnitCompatible(tgt);

    if (tgt.tunit==NULL) {
      if(tunit!=NULL){
        fixcgs(); tunit=NULL;
      }
      d = tgt.valincgs - valincgs;
//      d *= significance;
      valincgs *= us;
      valincgs += tgt.valincgs * significance;
      myError.valincgs = std::sqrt(myError.valincgs*myError.valincgs*us + d*d * significance);
     }else{
      if(tunit!=NULL){
        if(tunit!=tgt.tunit) {
          fixcgs(); tunit=tgt.tunit; fixtu(); }
       }else{
        tunit=tgt.tunit; fixtu();
      }
      d = tgt.valintu - valintu;
  //    d *= significance;
      valintu *= us;
      valintu += tgt.valintu * significance;
      myError.valintu = std::sqrt(myError.valintu*myError.valintu*us*us + d*d * significance);
    }
    
    return *this;
  }
  
  physquantity& operator= (const double &mtpv){
    myDim = real1.Dimension;
    valincgs = mtpv;
    tunit = NULL;
    return (*this);
  }
  physquantity& operator*= (const physquantity &mtpv){

    if(is_identical_zero()) return *this;
    if(mtpv.is_identical_zero()) return (*this = 0);

    if (mtpv.tunit == &real1){
      if (tunit==NULL){
        if (!(myError.valincgs==0 && mtpv.myError.valintu==0))
          myError.valincgs = hypot(myError.valincgs * mtpv.valintu,
                                   mtpv.myError.valintu * valincgs);
        valincgs *= mtpv.valintu;
       }else{
          if (!(myError.valintu==0 && mtpv.myError.valintu==0))
            myError.valintu = hypot(myError.valintu * mtpv.valintu,
                                   mtpv.myError.valintu * valintu);
          valintu *= mtpv.valintu;
      }
      return *this;
    }

    if (tunit!=NULL){
      fixcgs();
      if (tunit != &real1 || mtpv.tunit != &real1){
        tunit=NULL;
      }
    }
    caption = cptNULL;
    if (!disjointErrorAncestry(mtpv)){
      cerr << "Error crash in operator *="; abort();
    }
    if (mtpv.tunit!=NULL){
      myDim*=mtpv.tunit->Dimension;
      double otherincgs = mtpv.valintu * mtpv.tunit->cgsPrefactor;
      if (myError.valincgs!=0 || mtpv.myError.valintu!=0)
        myError.valincgs = hypot(myError.valincgs * otherincgs,
                                 mtpv.myError.valintu*mtpv.tunit->cgsPrefactor * valincgs);
      valincgs *= otherincgs;
     }else{
      myDim*=mtpv.myDim;
      if (myError.valincgs!=0 || mtpv.myError.valincgs!=0)
        myError.valincgs = hypot(myError.valincgs * mtpv.valincgs,
                                 mtpv.myError.valincgs * valincgs);

      valincgs *= mtpv.valincgs;

    }
/*    if (mtpv.tunit!=NULL){
      (valincgs *= mtpv.valintu) *= mtpv.tunit->cgsPrefactor;
     }else{
      valincgs *= mtpv.valincgs;
    }*/

    if (tunit == &real1){
      tunit=mtpv.tunit;
      fixtu();
    }

    return *this;
  }
  const physquantity operator *(const physquantity &mtpv) const{
    return physquantity(*this) *= mtpv;
  }
  physquantity &operator /=(const physquantity &mtpv){
    if(is_identical_zero() && !mtpv.is_identical_zero()) return *this;
    if (mtpv.tunit == &real1){
      if (tunit==NULL){
        if (!(myError.valincgs==0 && mtpv.myError.valintu==0))
          myError.valincgs = hypot(myError.valincgs / mtpv.valintu,
                                   mtpv.myError.valintu * valincgs / (mtpv.valintu*mtpv.valintu));
        valincgs /= mtpv.valintu;
       }else{
          if (!(myError.valintu==0 && mtpv.myError.valintu==0))
            myError.valintu = hypot(myError.valintu / mtpv.valintu,
                                   mtpv.myError.valintu * valintu / (mtpv.valintu*mtpv.valintu));
          valintu /= mtpv.valintu;
      }
      return *this;
    }

    freefromunitsys();
    caption = cptNULL;
    if (!disjointErrorAncestry(mtpv)){
      cerr << "Error crash in operator /="; abort();
    }
    if (mtpv.tunit!=NULL){
      myDim/=mtpv.tunit->Dimension;
      if (mtpv.valintu != 0){
        double otherincgs = mtpv.valintu * mtpv.tunit->cgsPrefactor;
        valincgs /= otherincgs;
        if (!(myError.valincgs==0 && mtpv.myError.valintu==0))
          myError.valincgs = hypot(myError.valincgs / otherincgs,
                                   mtpv.myError.valintu * valincgs / mtpv.valintu);
       }else{
        myError.valincgs = HUGE_VAL;
      }
     }else{
      myDim/=mtpv.myDim;
      if (mtpv.valincgs != 0){
        valincgs /= mtpv.valincgs;
        if (!(myError.valincgs==0 && mtpv.myError.valincgs==0))
          myError.valincgs = hypot(myError.valincgs / mtpv.valincgs,
                                   mtpv.myError.valincgs * valincgs / mtpv.valincgs);
       }else{
        myError.valincgs = HUGE_VAL;
      }
    }
    return (*this);
  }
  const physquantity operator /(const physquantity &mtpv) const{
    return physquantity(*this) /= mtpv;
  }
  physquantity &operator ^=(const double &expn){
    freefromunitsys();
    if (myError.valincgs != 0){
      myError.valincgs *= fabs(expn * pow(valincgs, expn-1));
    }
    caption = cptNULL;
    valincgs = pow(valincgs, expn);
    myDim ^= expn;
    return (*this);
  }
  const physquantity to (const double &expn) const{
    return physquantity(*this) ^= expn;
  }
  const physquantity invme(){
    freefromunitsys();
    myDim ^= -1;
    if(valincgs==0 || myError.valincgs==HUGE_VAL) {
      myError.valincgs = HUGE_VAL;
      return *this;
    }
    if (myError.valincgs != 0){
      myError.valincgs *= 1/(valincgs*valincgs);
    }
    valincgs = 1/valincgs;
    caption = cptNULL;
    return (*this);
  }
  const physquantity inv () const{
    return physquantity(*this).invme();
  }
  const physquantity sqrtme(){       // non-gaussian uncertainty propagation
    freefromunitsys();              //  in the neighbourhood of 0.
    if (valincgs < myError.valincgs && -valincgs < myError.valincgs) {
      double rsqnet = (myError.valincgs-valincgs);
      myError.valincgs /= std::sqrt(4*valincgs+rsqnet*rsqnet/myError.valincgs);
      valincgs = valincgs>0? std::sqrt(valincgs) : 0;
     }else{
      valincgs = std::sqrt(valincgs);
      if (myError.valincgs != 0){
        myError.valincgs *= .5 / valincgs;
      }
    }
    caption = cptNULL;
    myDim ^= .5;
    return (*this);
  }
  const physquantity sqrt () const{
    return physquantity(*this).sqrtme();
  }
  const physquantity squareme(){
    freefromunitsys();
    if (myError.valincgs != 0){
      myError.valincgs *= 2 * valincgs;
    }
    valincgs *= valincgs;
    caption = cptNULL;
    myDim ^= 2;
    return (*this);
  }
  const physquantity squared () const{
    return physquantity(*this).squareme();
  }

/*  const double operator/(physquantity divs) const{
    divs.chkUnitCompatible(*this);
    if (tunit==NULL) {
      if(divs.tunit!=NULL){
        divs.fixcgs(); divs.tunit=NULL;
      }
      return (valincgs/divs.valincgs);
     }else{
      if(divs.tunit!=NULL){
        if(divs.tunit!=tunit){
          divs.fixcgs(); divs.tunit=tunit; divs.fixtu();}
       }else{
        divs.tunit=tunit; divs.fixtu();
      }
      return (valintu/divs.valintu);
    }
  }*/
  bool operator ==(physquantity cmpv) const{
    if (is_identical_zero()) return cmpv==0;
    if (cmpv.is_identical_zero()) return *this==0;
    cmpv.chkUnitCompatible(*this);
    if (tunit==NULL) {
      if(cmpv.tunit!=NULL){
        cmpv.fixcgs(); cmpv.tunit=NULL;
      }
      return (valincgs==cmpv.valincgs);
     }else{
      if(cmpv.tunit!=NULL){
        if(cmpv.tunit!=tunit){
          cmpv.fixcgs(); cmpv.tunit=tunit; cmpv.fixtu();}
       }else{
        cmpv.tunit=tunit; cmpv.fixtu();
      }
      return (valintu==cmpv.valintu);
    }
  }
  bool conjugates(const physquantity &cmpv) const{
    return Dimension().conjugates(cmpv.Dimension());
  }
  bool compatible(const physquantity &cmpv) const{
    return Dimension()==cmpv.Dimension()
      || cmpv==0 || is_identical_zero();
  }
  bool operator !=(const physquantity &cmpv) const{
    return !(*this == cmpv);
  }
  bool operator ==(const double &cmpv) const{
    if (Dimension() != 1 && !is_identical_zero() && cmpv!=0){
      if (tunit==0) {throw physDimensionException({},{myDim});}
       else throw physDimensionException(tunit);
    }
    if (tunit==NULL) {
      return (valincgs==cmpv);
     }else{
      return (valintu*tunit->cgsPrefactor == cmpv);
    }
  }
  bool operator !=(const double &cmpv) const{
    return !(*this == cmpv);
  }
  bool positivelypositive() const{
    if (tunit==NULL){
      if (valincgs - myError.valincgs > 0) return 1; else return 0;
     }else{
      if (valintu - myError.valintu > 0) return 1; else return 0;
    }
  }
  bool positivelynegative() const{
    if (tunit==NULL){
      if (valincgs + myError.valincgs < 0) return 1; else return 0;
     }else{
      if (valintu + myError.valintu < 0) return 1; else return 0;
    }
  }
  bool positive() const{
    if (tunit==NULL){
      if (valincgs > 0) return 1; else return 0;
     }else{
      if (valintu > 0) return 1; else return 0;
    }
  }
  bool negative() const{
    if (tunit==NULL){
      if (valincgs < 0) return 1; else return 0;
     }else{
      if (valintu < 0) return 1; else return 0;
    }
  }
  physquantity &rectify(){
    if (negative()){
      valintu *= -1;
      valincgs *= -1;
    }
    return (*this);
  }
  const physquantity abs() const{
    return physquantity(*this).rectify();
  }
  const int sgn() const{
    return positive()? 1 : -1;
  }
  bool operator <(const physquantity &cmpv) const{
    return (*this - cmpv).negative();
  }
  bool operator >(const physquantity &cmpv) const{
    return (*this - cmpv).positive();
  }
  bool operator >=(const physquantity &cmpv) const{
    return !(*this - cmpv).negative();
  }
  bool operator <=(const physquantity &cmpv) const{
    return !(*this - cmpv).positive();
  }
  bool equalsRgdErr(physquantity cmpv) const{
    cmpv -= *this;
    return !(cmpv.positivelypositive() || cmpv.positivelynegative());
  }
  bool operator <(const double &cmpv) const{
    if(is_identical_zero()) return 0<cmpv;
    if (Dimension() != 1 && cmpv!=0){
      if (tunit==0) {throw physDimensionException({},{myDim});}
       else throw physDimensionException(tunit);
    }
    if (tunit==NULL) {
      return (valincgs<cmpv);
     }else{
      return (valintu*tunit->cgsPrefactor < cmpv);
    }
  }
  bool operator >(const double &cmpv) const{
    if(is_identical_zero()) return 0>cmpv;
    if (Dimension() != 1 && cmpv!=0){
      if (tunit==0) {throw physDimensionException({},{myDim});}
       else throw physDimensionException(tunit);
    }
    if (tunit==NULL) {
      return (valincgs>cmpv);
     }else{
      return (valintu*tunit->cgsPrefactor > cmpv);
    }
  }
  bool operator >=(const double &cmpv) const{
    return !(*this == cmpv);
  }
  bool operator <=(const double &cmpv) const{
    return !(-*this < -cmpv);
  }
  bool operator <<(const physquantity &cmpv) const{   //Viel kleiner als
    if (positive() && cmpv.positive()){
      if (*this/cmpv < 1./MuchMoreThanOne) return 1; else return 0;
     }else if(negative() && cmpv.negative()){
      if (*this/cmpv > MuchMoreThanOne) return 1; else return 0;
     }else{
      cout << "Warning: \"much less / much more than\" is not really defined for quantities with different sign" << endl;
      return (*this<cmpv);
    }
  }
  bool operator >>(const physquantity &cmpv) const{
    return (-*this << -cmpv);
  }
  
  physquantity &push_upto(const physquantity &upb){
    if (*this < upb) *this = upb;
    return (*this);
  }
  physquantity &push_downto(const physquantity &dnb){
/*    if (dnb.is_identical_zero()) {
      if (positive()) *this = dnb; 
    }else*/ if (*this > dnb) *this = dnb;

    return (*this);
  }
  
  
  friend ostream &operator << (ostream &target, physquantity inputthis);
  friend class phqdata_binstorage;
  //friend class measure;
 /* class LaTeXofstream;
  friend LaTeXofstream::LaTeXofstream &operator << (const physquantity &inputthis);*/
  friend phUnit cgsUnitof(const physquantity &);
  friend const phUnit tmpUnit(const physquantity &, string);
  friend const phUnit &newUnit(const physquantity &, string, const std::vector<unitsscope *> &);
  friend const physquantity phUnit::to(const double &expn) const;
  friend unitsscope::dimrange unitsscope::suitablefor(const physquantity &) const;
};
const physquantity operator *(const double &left, const physquantity &mtpv) {
  return physquantity(mtpv) *= left;
}
const physquantity operator /(const double &left, const physquantity &mtpv) {
  return mtpv.inv() * left;
}

/*    --- better definition in stdtemplatespclz.hpp
physquantity abs(const std::complex<physquantity> &mtpv){
  const phUnit *thatU = mtpv.real().preferredUnit();
  if (thatU==NULL){
    return (mtpv.real()*mtpv.real() + mtpv.imag()*mtpv.imag()).sqrt();
   }else{
    double res = mtpv.real().value_in_unit(*thatU),
           ims = mtpv.imag().value_in_unit(*thatU);
    return sqrt((res*=res) += (ims*=ims)) * *thatU;
  }
}
*/
physquantity abs(const physquantity &mtpv){ return mtpv.abs(); }
int sgn(const physquantity &mtpv){ return mtpv.sgn(); }
physquantity sqrt(const physquantity &mtpv){ return mtpv.sqrt(); }


const physquantity operator *(const phUnit &left, const phUnit &mtpv) {
  return physquantity(1,left) *= physquantity(1,mtpv);}
const physquantity operator /(const phUnit &left, const phUnit &mtpv) {
  return physquantity(1,left) /= physquantity(1,mtpv);}
const physquantity operator /(const phUnit &left, const physquantity &mtpv) {
  return physquantity(1,left) /= mtpv;}
const physquantity operator *(const phUnit &left, const physquantity &mtpv) {
  return physquantity(1,left) *= mtpv;}
const physquantity operator /(const phUnit &left, const double &mtpv) {
  return physquantity(1,left) /= mtpv;}
const physquantity operator *(const phUnit &left, const double &mtpv) {
  return physquantity(1,left) *= mtpv;}
const physquantity operator *(const double &left, const phUnit &mtpv) {
  return physquantity(1,mtpv) *= left;}
const physquantity phUnit::to(const double &expn) const {
  physquantity result(Dimension); result.valincgs = cgsPrefactor;
  return result ^= expn;}


bool isIEEE754nan(double var) {
  volatile double d = var;
  return d != d;
}

  
physquantity exp(const physquantity &expv){
  double dimless = expv.dbl();
  physquantity result = std::exp(dimless) * real1;
  result.seterror(result * expv.error().dbl());
  return result;
}
physquantity ln(const physquantity &expv){
  double dimless = expv.dbl();
  physquantity result = std::log(dimless)*real1;
  result.seterror((expv.error().dbl()/dimless)*real1);
  return result;
}
physquantity sin(const physquantity &expv){
  double dimless = expv.dbl();
  physquantity result = std::sin(dimless) * real1;
  result.seterror(abs(cos(dimless)) * expv.error().dbl() * real1);
  return result;
}
physquantity cos(const physquantity &expv){
  double dimless = expv.dbl();
  physquantity result = std::cos(dimless) * real1;
  result.seterror(abs(sin(dimless)) * expv.error().dbl() * real1);
  return result;
}
physquantity tanh(const physquantity &argv){
  double dimless = argv.dbl(), a=std::exp(dimless), b=(std::exp(-dimless)), s = a+b;
  physquantity result = (a-b)/s * real1;
  if (isIEEE754nan(result.dbl())) return sgn(argv);
  result.seterror(abs(4*argv.error().dbl()/(s*s)) * real1);
     // (id,d/dx)tanh(x)=((e^x-e^-x)/(e^x+e^-x),4/(e^x+e^-x)^2)
  return result;
}


double cosech(double x) { return 2 * std::sinh(x) / (std::cosh(2*x) - 1); }
double cotanh(double x) { return std::sinh(2 * x) / (std::cosh(2*x) - 1); }
physquantity cosech(const physquantity &x){
  double dimless = x.dbl();
  physquantity result = cosech(dimless) * real1;
  result.seterror(abs(x.error().dbl() * cotanh(dimless) * result));
  return result;

}

physquantity conj(const physquantity &x){ return x; } // physquantities are real


/*string dbltostr(const double &src, const double &roundcmp){
  return (tostr)
}*/

ostream &operator << (ostream &target, physquantity inputthis){
  std::stringstream targ;
  inputthis.tryfindUnit();


  auto superscript = [](char c) -> std::string {
         switch (c) {
          case '0': return u8"\u2070";
          case '1': return u8"\u00b9";
          case '2': return u8"\u00b2";
          case '3': return u8"\u00b3";
          case '4': return u8"\u2074";
          case '5': return u8"\u2075";
          case '6': return u8"\u2076";
          case '7': return u8"\u2077";
          case '8': return u8"\u2078";
          case '9': return u8"\u2079";
          case '+': return u8"\u207a";
          case '-': return u8"\u207b";
          case '/': return u8"\u141F"; //actually CANADIAN SYLLABICS FINAL ACUTE,
                                      // for the lack of a proper superscript solidus
          default: return std::string(1,c);
         }
       };

  if (inputthis.tunit!=NULL){
    targ << inputthis.LaTeXval(false, true, true, true, *inputthis.tunit);
   }else{
    std::stringstream ubuild;

    auto expshow = [&](phDimension::cgsSelect sl) {
            std::string rs_n = inputthis.myDim.fractionshow(sl);
//             if(rs_n.find('/')!=std::string::npos || rs_n.find('-')!=std::string::npos)
//               rs_n = "(" + rs_n + ")";
            std::string rs;
            for(auto& c: rs_n) rs += superscript(c);
            return rs;
         };
    if (inputthis.myDim != 0){              // cmⁱ⋅gʲ⋅sᵏ
      ubuild << "cm" << expshow(phDimension::cgsSelect::c)
             << u8"\u22c5g" << expshow(phDimension::cgsSelect::g)
             << u8"\u22c5s" << expshow(phDimension::cgsSelect::s);
    }
    
    phUnit showunit(inputthis.myDim, 1, ubuild.str());

    targ << inputthis.LaTeXval(false, true, true, true, showunit);

  }
/*    if (inputthis.myError.valintu > 0) targ << '(';
    targ << inputthis.valintu;
    if (inputthis.myError.valintu > 0)
      targ << u8" \u00b1 " << inputthis.myError.valintu << ')';
    targ << " " << inputthis.tunit->uName; */

  auto miniTeX = [&](const std::string& str){
         std::stringstream TeXd;
         auto parse_err = [&](const std::string& spec){
                cerr << "Error: in miniTeX-parsing physquantity printout \""<<str<<"\":\n"
                     << spec << std::endl;
                abort();
              };
         for(unsigned i=0; i<str.length(); ++i) {
           if(str[i]=='\\') {
             ++i;
             if(str[i]==':'){
               TeXd << " ";
              }else if(str.substr(i,2)=="pm") {
               TeXd << u8"\u00b1";
               i+=1;
              }else if(str.substr(i,4)=="cdot"){
               TeXd << u8"\u22c5";
               i+=3;
              }else if(str.substr(i,7)=="mathrm{"){
               i+=7;
               for(unsigned brlyr=1; brlyr>0; ++i) {
                 if(str[i]=='{') ++brlyr;
                  else if(str[i]=='}') --brlyr;
                  else TeXd << str[i];
               }
               --i;
              }else{
               parse_err("Backslash unfollowed by a recognized command.");
             }
            }else if(str[i]=='$'){
            }else if(str[i]=='^'){
             ++i;
             if(str[i]=='{') {
               for(++i; str[i]!='}'; ++i) {
                 if(str[i]=='{')
                   parse_err("Nested braces for superscripts not supported.");
                 TeXd << superscript(str[i]);
               }
              }else{
               TeXd << superscript(str[i]);
             }
            }else{
             TeXd << str[i];
           }
         }
         return TeXd.str();
       };

  target << miniTeX(targ.str());

  return target;
}



physquantity soleerror(const physquantity &Errsrc){
  physquantity result = 0*Errsrc;
  return result.seterror(Errsrc).tryconvto(*Errsrc.preferredUnit());
}
physquantity plusminus(const physquantity &Errsrc){
  physquantity result = 0*Errsrc;
  return result.seterror(Errsrc).tryconvto(*Errsrc.preferredUnit());
}

struct errquantity : physquantity{
  errquantity(const double &initvl): physquantity(initvl, real1){}
};

const errquantity lusminus(const double &errvl) {
  errquantity result=0;
  result.error(errvl*real1);
  return result;
}
const physquantity operator +(const double &left, const errquantity &right) {
  return ((left * real1) + (physquantity) right);}



physquantity &physquantity::namestable_access::operator+=(const physquantity &addw){
  return *accessp += addw;
}
physquantity &physquantity::namestable_access::operator-=(const physquantity &addw){
  return *accessp -= addw;
}
physquantity physquantity::namestable_access::operator+(const physquantity &addw){
  return *accessp + addw;
}
physquantity physquantity::namestable_access::operator-(const physquantity &addw){
  return *accessp - addw;
}
bool physquantity::namestable_access::operator<(const physquantity &cmpw){
  return *accessp < cmpw;
}




unitsscope::dimrange unitsscope::suitablefor(const physquantity &dsrd) const{
    return equal_range(dsrd.Dimension()); }


template<typename SrcT>
auto unphysicalize_cast(const SrcT& src) -> double {
  return src;                                      }
auto unphysicalize_cast(const physquantity& src) -> double {
  return src.dbl();                                        }
auto unphysicalize_cast(const std::complex<physquantity>& src)
              -> std::complex<double>;


template<typename tgt_T>
inline tgt_T cast_from_phq(const physquantity &s){
  return s.dbl();
}
template<>
inline physquantity cast_from_phq(const physquantity &s){
  return s;
}


template<typename src_T>
inline physquantity cast_to_phq(const src_T &s){
  return s*real1;
}
template<>
inline physquantity cast_to_phq(const physquantity &s){
  return s;
}





const double doublesafemagntspan = 1e+12;  // somewhat defensive, the actual
                                          //  precision is almost 16 decimals.

inline physquantity uncertainty_forIReal(const physquantity &x) {
  return std::max(x.error(), abs(x/doublesafemagntspan));            }
inline double uncertainty_forIReal(double x) {
  return x/doublesafemagntspan;              }
inline int uncertainty_forIReal(int x) {
  return 1;                            }
#if 0
template<typename rll_T>
inline rll_T uncertainty_forIReal(const rll_T &x) {
  return 1;                                       }
#endif

inline physquantity as_exact_value(const physquantity &x) {
  return x.werror(0);                                     }
template<typename T>
inline T as_exact_value(const T &x) {
  return x;                  }

inline physquantity announce_uncertainty(const physquantity &x, const physquantity &e) {
  return x.werror(e);                                                                  }
template<typename T>
inline T as_exact_value(const T &x, const T &e) {
  return x;                                     }


namespace_cqtxnamespace_CLOSE


#include "stdtemplatespclz.hpp"

namespace lambdalike {
  template<>auto polymorphic_1<cqtxnamespace::physquantity>()
        -> cqtxnamespace::physquantity                         {
    return 1*cqtxnamespace::real1;                             }
  template<>auto polymorphic_1<std::complex<cqtxnamespace::physquantity>>()
       -> std::complex<cqtxnamespace::physquantity>                              {
    return std::complex<cqtxnamespace::physquantity>( 1*cqtxnamespace::real1
                                                    , 0*cqtxnamespace::real1 );
  }
};



namespace_cqtxnamespace_OPEN

auto unphysicalize_cast(const std::complex<physquantity>& src)
                   -> std::complex<double>                     {
  return src.complexity * src.physicality.dbl();               }

namespace_cqtxnamespace_CLOSE
