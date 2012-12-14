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


template <typename vctItT, typename keyfnT>
physquantity meanofElem(vctItT srcbgn,
                        vctItT srcend,
                        const keyfnT &keyme){
  physquantity result = cast_to_phq(keyme(*srcbgn)),
               reserr = cast_to_phq(0 * keyme(*srcbgn).to(2));
  int srcsize = 1;
  vctItT i;
  for (i = (srcbgn+1); i!=srcend; ++i){
    result += cast_to_phq(keyme(*i)); //cout << keyme(*i) << endl;
    ++srcsize;
  }
  result /= srcsize;
  for (i = srcbgn; i!=srcend; ++i)
    reserr += cast_to_phq(keyme(*i) - result).to(2);
  (reserr /= (double(srcsize) * (srcsize-1))).sqrtme();// ^= 1./2;
  result.error(reserr);
  return result;
}


template <typename containerT, typename keyfnT>
physquantity meanofElem(const containerT &src, const keyfnT &keyme){
  return meanofElem(src.begin(), src.end(), keyme);
}
template <typename containerT>
physquantity meanofElem(const containerT &src, const std::string& keymename){
  return meanofElem(src.begin(), src.end(), captfinder(keymename));
}
template <typename containerT>
physquantity meanofElem(const containerT &src, const char *keymename){
  return meanofElem(src.begin(), src.end(), captfinder(keymename));
}

/*template <typename vctItT, typename keyfnT>
physquantity variance(vctItT srcbgn,
                           vctItT srcend,
                           keyfnT keyme){
  struct squareddistto {
    physquantity whatto;
    keyfnT whichone;
    squareddistto(const physquantity &whatt, keyfnT whicho) :
      whatto(whatt), whichone(whicho){}
    template<typename cntnT>
    physquantity operator() (cntnT overc){
      physquantity diffr = whichone(overc); diffr-=whatto;
      return diffr*diffr;
    }
  };
  
  return meanofElems(srcbgn, srcend, squareddistto(meanofElems(srcbgn, srcend, keyme), keyme));
}*/

template <typename vctItT, typename keyfnT>
physquantity meandeviation(vctItT srcbgn,
                           vctItT srcend,
                           keyfnT keyme){
  return meanofElem(srcbgn, srcend, keyme).error() * sqrt(distance(srcbgn, srcend));
}
template <typename cntT, typename keyfnT>
physquantity meandeviation(cntT srccnt,
                           keyfnT keyme){
  return meanofElem(srccnt.begin(), srccnt.end(), keyme).error() * sqrt(srccnt.size());
}
template <typename cntT>
physquantity meandeviation(cntT srccnt,
                           const string &keymename){
  return meanofElem(srccnt.begin(), srccnt.end(), captfinder(keymename)).error() * sqrt(srccnt.size());
}
template <typename cntT>
physquantity meandeviation(cntT srccnt,
                           const char *keymename){
  return meanofElem(srccnt.begin(), srccnt.end(), captfinder(keymename)).error() * sqrt(srccnt.size());
}



template <typename vctItT, typename keyfnT>
physquantity RMSofElem(vctItT srcbgn,
                        vctItT srcend,
                        const keyfnT &keyme){
  physquantity result = cast_to_phq(keyme(*srcbgn).werror(0)).squared();
  int srcsize = 1;
  vctItT i;
  for (i = (srcbgn+1); i!=srcend; ++i){
    result += cast_to_phq(keyme(*i).werror(0)).squared(); //cout << keyme(*i) << endl;
    ++srcsize;
  }
  result /= srcsize;
  return result.sqrt();
}
physquantity RMSofElem(const measureseq &msq, const msq_dereferencer &keyme) {
  return RMSofElem(msq.begin(), msq.end(), keyme);
}



template <typename vctItT, typename keyfnT>
vctItT wheremax(const vctItT &srcbgn,
                     const vctItT &srcend,
                     const keyfnT &keyme){
  vctItT i, mp=srcbgn;
  for (i = srcbgn; i!=srcend; ++i){
    if (keyme(*i) > keyme(*mp)){
      mp = i;
    }
  }
  return mp;
}

/*
template <typename containerT, typename keyfnT>
typename std::vector<containerT>::iterator wheremax(std::vector<containerT> src, const keyfnT &keyme){
  return wheremax(src.begin(), src.end(), keyme);
}
std::vector<physquantity> *wheremax(std::vector<std::vector<physquantity> > &src, const string &tgtcapt){
  return wheremax(src.begin(), src.end(), keyme);
}*/
measureseq::iterator wheremax(measureseq &src, const msq_dereferencer &keyme=measureindexpointer(0)){
  return wheremax(src.begin(), src.end(), keyme);
}
measureseq::const_iterator wheremax(const measureseq &src, const msq_dereferencer &keyme=measureindexpointer(0)){
  return wheremax(src.begin(), src.end(), keyme);
}

template <typename vctItT, typename keyfnT>
vctItT wheremin(vctItT srcbgn,
                const vctItT &srcend,
                const keyfnT &keyme){
  //i;
//  i = srcbgn;
  for (vctItT i = srcbgn; i!=srcend; ++i){
//      cout << keyme(*i) << '<' << keyme(*srcbgn) << ",   ";
    if (keyme(*i) < keyme(*srcbgn))  {
      srcbgn = i;
    }
  }
  return srcbgn;
}

/*template <typename containerT, typename keyfnT>
typename std::vector<containerT>::iterator wheremin(std::vector<containerT> src, keyfnT keyme){
  return wheremin(src.begin(), src.end(), keyme);
}*/

measureseq::iterator wheremin(measureseq &src, const msq_dereferencer &keyme=measureindexpointer(0)){
  return wheremin(src.begin(), src.end(), keyme);
}
measureseq::const_iterator wheremin(const measureseq &src, const msq_dereferencer &keyme=measureindexpointer(0)){
  return wheremin(src.begin(), src.end(), keyme);
}

template <typename containerT, typename keyfnT>
physquantity maxval(const containerT& src, keyfnT keyme){
  return keyme(*wheremax(src.begin(), src.end(), keyme));
}
template <typename containerT, typename keyfnT>
physquantity minval(const containerT& src, keyfnT keyme){
  return keyme(*wheremin(src.begin(), src.end(), keyme));
}
#if 0
physquantity maxval(const measureseq &src, const std::string &keymestr){
  captfinder keyme(keymestr);
  return keyme(*wheremax(src.begin(), src.end(), keyme));
}
physquantity minval(const measureseq &src, const std::string &keymestr){
  captfinder keyme(keymestr);
  return keyme(*wheremin(src.begin(), src.end(), keyme));
}
physquantity maxval(const measureseq &src, const char* keymestr){
  captfinder keyme(keymestr);
  return keyme(*wheremax(src.begin(), src.end(), keyme));
}
physquantity minval(const measureseq &src, const char* keymestr){
  captfinder keyme(keymestr);
  return keyme(*wheremin(src.begin(), src.end(), keyme));
}
#endif

template <typename containerT, typename keyfnT>
containerT *where(std::vector<containerT> src, keyfnT keyme, physquantity compval){
  return &*wheremin(src.begin(), src.end(), distancetothat<keyfnT>(compval, keyme));
}



/*template <typename containerT> class meanofElem(std::vector<containerT> src){
  containerT
}*/



inline float rndround0(){
  return rand() * ( 2 / (float) RAND_MAX) - 1;
}
inline double sfacos(double cosphi){
  if (cosphi>1) return 0; 
  if(cosphi<-1) return pi;
  return acos(cosphi);
}



     // A base class for functions mapping from some interval -- note that these
    //  are NOT necessarily (though they may be) compactly-supported functions in
   //   the sense commonly used in mathematical analysis.
                   // (It might be desirable to make this a seperate base class.
                  //  Or to deprecate the whole class and unify the concept into
                 //   the phqfn hierarchy.)
template <typename arg_T, typename ret_T=arg_T>
struct compactsupportfn {
  virtual ret_T unchecked_eval(const arg_T &) const =0;
  virtual ret_T operator() (const arg_T &x) const;
  virtual interval<arg_T> support() const =0;
  virtual const arg_T &suppleft() const {return support().l();}
  virtual const arg_T &suppright() const {return support().r();}
  virtual ret_T unchecked_derivative_at(const arg_T &x) const;

  class derivative_of_mine: public compactsupportfn<arg_T, ret_T> {
    const compactsupportfn<arg_T, ret_T> *orig;     // unsafe (FIXME), original may be destroyed while this still exists.
   public:
    derivative_of_mine(const compactsupportfn<arg_T, ret_T> &o): orig(&o) {}
    interval<arg_T> support() const{return orig->support();}
    ret_T unchecked_eval(const arg_T &where) const{
      return orig->unchecked_derivative_at(where);
    }
  };

  virtual derivative_of_mine derivative() const { return derivative_of_mine(*this); }
  virtual ret_T derivative_at(const arg_T &x) const{ return derivative()(x); }
//  virtual measureseq quantized(int desirreso);
  virtual ~compactsupportfn() {}
};
template <>   //specialization for physquantity functions, as these need
             // uncertainty-propagation tracking. The class declaration
            //  is an almost exact copy of the templatized version.
struct compactsupportfn<physquantity,physquantity> {
  virtual physquantity unchecked_eval(const physquantity &) const =0;
  virtual physquantity operator() (const physquantity &x) const;
  virtual interval<physquantity> support() const =0;
  virtual const physquantity &suppleft() const {return support().l();}
  virtual const physquantity &suppright() const {return support().r();}
  virtual physquantity unchecked_derivative_at(const physquantity &x) const;

  template <typename arg_T, typename ret_T>
  class derivative_of_mine: public compactsupportfn<arg_T, ret_T> {
    const compactsupportfn<arg_T, ret_T> *orig;
   public:
    derivative_of_mine(const compactsupportfn<arg_T, ret_T> &o): orig(&o) {}
    interval<arg_T> support() const{return orig->support();}
    ret_T unchecked_eval(const arg_T &where) const{
      return orig->unchecked_derivative_at(where);
    }
  };

  virtual derivative_of_mine<physquantity,physquantity> derivative() const {
    return derivative_of_mine<physquantity,physquantity>(*this);           }
  virtual physquantity derivative_at(const physquantity &x) const {
    return derivative()(x);                                       }
//  virtual measureseq quantized(int desirreso);
  virtual ~compactsupportfn() {}
};

    // For physquantity-valued functions, unchecked_eval may represent the 'exact'
   //  result, i.e. without uncertainty computation. operator() is to provide this
  //   computation for those cases. Don't forget to bypass this for functions that
 //    already provide more efficient error calculation of their own!
template <typename arg_T, typename ret_T>
ret_T inline compactsupportfn<arg_T,ret_T>::operator() (const arg_T &x) const {
  return unchecked_eval(x);
}
physquantity inline compactsupportfn<physquantity,physquantity>::
             operator()(const physquantity &x) const{
  physquantity result = unchecked_eval(x);
  return result.werror(sqrt(
                        (unchecked_derivative_at(x)*x.error()).squared()
                       +result.error().squared())
                      );
}

     // Default implementations of derivation by quotient of differences. Of course,
    //  it's not strictly clever to assume a general function should be differentiable
   //   at all, but the more useful ones rather typically are. You should provide an
  //    explicit error overload for non-differentiable functions!
template <typename arg_T, typename ret_T>
ret_T inline compactsupportfn<arg_T,ret_T>::unchecked_derivative_at(const arg_T &x) const{
  ret_T hdelta = support().width()/8192;    //rough estimate over reasonably smallest detail feature
  return (unchecked_eval(x+hdelta) - unchecked_eval(x-hdelta))/(2*hdelta);
}
physquantity inline compactsupportfn<physquantity,physquantity>::unchecked_derivative_at
                     (const physquantity &xprime) const{
  physquantity hdelta = support().width()/ 8192, x=xprime.werror(0);
  //return (unchecked_eval(x+hdelta) - unchecked_eval(x-hdelta))/(2*hdelta);
  hdelta.seterror(0);
  return (unchecked_eval( x.plusexactly(hdelta) )
          .minusexactly( unchecked_eval(x.minusexactly(hdelta)) ) ).werror(0)
             /(2*hdelta);
}




#if 0
QTeXdiagmaster &QTeXdiagmaster::plot_phmsq_function(const compactsupportfn<physquantity, physquantity> &f, const QTeXgrcolor &cl=QTeXgrcolors::undefined) {
  insertCurve(f.quantized(), cl);
  return *this;
}
#endif

template <typename arg_T, typename ret_T=arg_T>
struct normalized_compactsupportfn : compactsupportfn<arg_T, ret_T> {
  virtual ~normalized_compactsupportfn() {}
};
template <typename arg_T, typename ret_T=arg_T>
struct scalable_compactsupportfn {
  virtual ret_T operator() (const arg_T &, const arg_T &) const =0;
  virtual ~scalable_compactsupportfn() {}
};


namespace integration_cIsh {
const long default_intg_steps=8192;

template <typename argT, typename retT, typename intgbfnT>
retT Romberg2 (const intgbfnT &thisfn, interval<argT> Ilims, long steps=default_intg_steps){
  argT x, hby2 = (Ilims.width())/(2*(steps-1)), Iest;
  Iest = thisfn(Ilims.l())/2;
  x = Ilims.l() + hby2;
  Iest += 2*thisfn(x);
  for (x += hby2; x<Ilims.r()-hby2; x += hby2){
    Iest += thisfn(x);
    x += hby2;
    Iest += 2*thisfn(x);
  }
  Iest += thisfn(Ilims.r())/2;
  return(Iest*2*hby2/3);
}

template <typename argT, typename retT>
inline retT Romberg2 (const compactsupportfn<argT, retT> &thisfn, long steps=default_intg_steps){
  return Romberg2<argT, retT, compactsupportfn<argT, retT> >(thisfn, thisfn.support(), steps);
}

}

template <typename argT, typename retT>
inline retT integral (const compactsupportfn<argT, retT> &thisfn, long steps=integration_cIsh::default_intg_steps){
  return integration_cIsh::Romberg2(thisfn, steps);
}


template <typename arg_T, typename ret_T>
phGraphScreen rasterize(const compactsupportfn<arg_T, ret_T> &f, int raster){
  interval<arg_T> support = f.support();
  arg_T h = support.width()/raster;
  phGraphScreen screen;
  for (arg_T x = support.l()+h; x<support.r()-h; x+=h)
//{  cout << x << " = x => f(x) = " << f(x) << endl;
    screen.lineto(x, f(x));
//}
  return screen;
}




struct Friedrichs_mollifierfn: compactsupportfn<double, double>,
                   scalable_compactsupportfn<physquantity, physquantity> {
  double operator() (const double &x)const {
    if (x>-1 && x<1){
      return 2.252283620*exp(1/(x*x - 1));
//      return 2.252283620*exp((x*x+1)/(x*x - 1));
     }else{
      return 0;
    }
  }
  double unchecked_eval(const double&x) const{return operator()(x);}
  double operator() (const physquantity &pos) const {
    return operator()(pos.dbl());
  }

  physquantity operator() (const physquantity &pos, const physquantity &scale) const {
    if (scale.compatible(pos))
      return operator()(pos/scale) / scale;
    return operator()(pos*scale) * scale;
  }
  interval<double> support() const { return interval<double>(-1,1); }
}Friedrichs_mollifier;

struct Friedrichs_mollifierfn_onsquares: compactsupportfn<double, double> {
  double operator() (const double &x)const {
    #if 0//(x > 3456) { 
      cout << x << ": ";
      if (x>-1 && x<1)
        cout <<  2.252283620*exp(1/(x*x - 1)) << endl;
       else
        cout << "0" << endl;
    #endif
    if (x<1){
      return 2.252283620*exp(1/(x - 1));
     }else{
      return 0;
    }
  }
  double unchecked_eval(const double&x) const{return operator()(x);}
  double operator() (const physquantity &pos) const {
    return operator()(pos.dbl());
  }
  physquantity operator() (const physquantity &pos, const physquantity &scale) const {
    if (scale.conjugates(pos))
      return operator()(pos*scale) * scale;
    return operator()(pos/scale) / scale;
  }
  interval<double> support() const { return interval<double>(0,1); }
}Friedrichs_mollifier_onsquares;



struct Gausslikeshaped_Friedrichsmollifierfn: compactsupportfn<double, double>,
                   scalable_compactsupportfn<physquantity, physquantity> {
  double operator() (const double &x) const {
    if (x>-1 && x<1){
      double x2=x; x2*=x;
      return 2.603894883054*exp(1/(x2 - 1) - x2);
//      return 2.252283620*exp((x*x+1)/(x*x - 1));
     }else{
      return 0;
    }
  }
  double unchecked_eval(const double&x) const{return operator()(x);}
  double operator() (const physquantity &pos) const {
    return operator()(pos.dbl());
  }
  physquantity operator() (const physquantity &pos, const physquantity &scale) const {
    if (scale.compatible(pos))
      return operator()(pos/scale) / scale;
    return operator()(pos*scale) * scale;
  }
  interval<double> support() const { return interval<double>(-1,1); }
}Gausslikeshaped_Friedrichsmollifier;



struct Unnormalized_Compactsqueezed_Gaussfn: compactsupportfn<double, double>,
                   scalable_compactsupportfn<physquantity, physquantity> {
 private:
  double s;
 public:
  Unnormalized_Compactsqueezed_Gaussfn(double squeeze_factor): s(squeeze_factor) {}
  double operator() (const double &x) const {
    if (x>-1 && x<1){
      double x2=x; x2*=x;
      return exp(1/(x2 - 1) - s*x2);
//      return 2.252283620*exp((x*x+1)/(x*x - 1));
     }else{
      return 0;
    }
  }
  double unchecked_eval(const double&x) const{return operator()(x);}
  double operator() (const physquantity &pos) const {
    return operator()(pos.dbl());
  }
  physquantity operator() (const physquantity &pos, const physquantity &scale) const {
    if (scale.compatible(pos))
      return operator()(pos/scale) / scale;
    return operator()(pos*scale) * scale;
  }
  interval<double> support() const { return interval<double>(-1,1); }
};



struct Compactsqueezed_Gaussfn: compactsupportfn<double, double>,
                   scalable_compactsupportfn<physquantity, physquantity> {
 private:
  double s, N;
 public:

  Compactsqueezed_Gaussfn(double squeeze_factor):
    s(squeeze_factor),
    N( 1 / integral(Unnormalized_Compactsqueezed_Gaussfn(squeeze_factor)) )
  {}
  
  double operator() (const double &x) const {
    if (x>-1 && x<1){
      double x2=x; x2*=x;
      return N*exp(1/(x2 - 1) - s*x2);
//      return 2.252283620*exp((x*x+1)/(x*x - 1));
     }else{
      return 0;
    }
  }
  double unchecked_eval(const double&x) const{return operator()(x);}
  double operator() (const physquantity &pos) const {
    return operator()(pos.dbl());
  }
  physquantity operator() (const physquantity &pos, const physquantity &scale) const {
    if (scale.compatible(pos))
      return operator()(pos/scale) / scale;
    return operator()(pos*scale) * scale;
  }
  interval<double> support() const { return interval<double>(-1,1); }
};//Compactsqueezed_1_Gauss(1), Compactsqueezed_2_Gauss(2), Compactsqueezed_4_Gauss(4);



struct identityCurve {
  double operator() (const double &pos){
    return 1;
  }
}identityfunction;






struct affinefit_data{
//  affinefit_data()
  physquantity slope, offset;
};
template<typename vctItT, typename keyfnT, typename keyfnTt>
affinefit_data stdleastsquare_affinefit(vctItT srcbgn, vctItT srcend,
                                   keyfnT keyme, keyfnTt keymet){
  affinefit_data result;
  physquantity SumIndys(0), SumIndySQRs(0), SumPends(0), SumIndyPends(0);
  unsigned int nOfPairs=0;
  for (vctItT sumitallup=srcbgn; sumitallup!=srcend; ++sumitallup){
    physquantity indy = keymet(*sumitallup);
    SumIndys.AddAsExactVal(indy);
    SumPends.AddAsExactVal(keyme(*sumitallup));
    SumIndyPends.AddAsExactVal(keyme(*sumitallup) * indy);
    SumIndySQRs.AddAsExactVal(indy*=indy);
    ++nOfPairs;
  }

/*  IF SumIndySQRs = 0 THEN Errormessage "Impossible to calculate slope of data points all at "
+ CellCaption(DataBin(SrcIndy(0)).CaptID) + " = 0!", "AssignCLinearID"*/

//dbgbgn;
  physquantity ThatStupidDelta = nOfPairs*SumIndySQRs - SumIndys*SumIndys;

  result.slope = (nOfPairs*SumIndyPends - SumIndys*SumPends) / ThatStupidDelta;
  result.offset = (SumIndySQRs*SumPends - SumIndys*SumIndyPends) / ThatStupidDelta;
  if (keyme(*srcbgn).preferredUnit() != NULL) result.offset[*keyme(*srcbgn).preferredUnit()];
//dbgend;
  
  physquantity SumSqErrs = 0;
//  cout << "SumSqErrs = " << SumSqErrs << endl;
  for (vctItT sumitallup=srcbgn; sumitallup!=srcend; ++sumitallup){
    physquantity thisdiff = result.offset + result.slope * keymet(*sumitallup) - keyme(*sumitallup);
    SumSqErrs.AddAsExactVal(thisdiff *= thisdiff);
  }
  //cout << "SumSqErrs = " << SumSqErrs << endl;
//  cout << "te: " << (1*seconds) * keymet(*((--srcend)++)) /(1*seconds) << endl;
  physquantity MnSqErrs = (SumSqErrs / (nOfPairs - 1)).sqrt();
  if (keyme(*srcbgn).preferredUnit() != NULL) MnSqErrs[*keyme(*srcbgn).preferredUnit()];
 // cout  << "MnSqErrs = "<< MnSqErrs << endl;

//  cout << "slope: " << result.slope << endl  << "delta: "<< ThatStupidDelta << endl;
  //cout << "sqrt(nOfPairs / ThatStupidDelta): " << (nOfPairs / ThatStupidDelta) << endl;
  //cout << MnSqErrs * (nOfPairs / ThatStupidDelta).sqrt() << endl;
  result.offset.error(MnSqErrs);
  result.slope.error(MnSqErrs * (nOfPairs / ThatStupidDelta).sqrt());
  return result;
}
template<typename cntT, typename keyfnT, typename keyfnTt>
affinefit_data stdleastsquare_affinefit(const cntT &src,
                                   keyfnT keyme, keyfnTt keymet){
  return stdleastsquare_affinefit(src.begin(), src.end(), keyme, keymet);
}




template <typename vctItT, typename keyfnT, typename keyfnTt>
complex<physquantity> esSFTcomponent(vctItT srcbgn, vctItT srcend,
                                   keyfnT keyme, keyfnTt keymet,
                                   physquantity omega){
  const phUnit *u;    // phUnit udummy;
  if (keyme(*srcbgn).preferredUnit()!=NULL){
    u = keyme(*srcbgn).preferredUnit();
   }else{
//    udummy = newUnit(keyme(*srcbgn), "");
    u = new phUnit(newUnit(keyme(*srcbgn), "", NULL));
  }
  complex<double> result=0, twiddle;
  int srcsize = 0;
  for (vctItT i = srcbgn; i!=srcend; ++i){
    twiddle = polar(1., (omega*keymet(*i)).dbl());
    result += keyme(*i)[*u] * twiddle;
    ++srcsize;
  }
  result /= srcsize;
//  result /= (keymet(*srcend) - keymet(*srcbgn));     // 1/T
  complex<physquantity> fresult(result.real() * *u, result.imag() * *u);
  if (keyme(*srcbgn).preferredUnit()!=NULL) delete u;
  return fresult;
}

template <typename vctItT, typename keyfniT, typename keyfnTt>
std::vector<complex<physquantity> > esSFTcomponent(vctItT srcbgn, vctItT srcend,
                                   keyfniT keytst, keyfniT keytnd, keyfnTt keymet,
                                   physquantity omega){
  std::vector<const phUnit *> u;    // phUnit udummy;
  std::vector<keyfniT> kp;
  std::vector<complex<double> > result;
  complex<double> twiddle;
  unsigned int keyringsize;
  keyfniT k; unsigned int j;
  for (k=keytst; k<keytnd; ++k){
    if ((*k)(*srcbgn).preferredUnit()!=NULL){
      u.push_back((*k)(*srcbgn).preferredUnit());
     }else{
      u.push_back(new phUnit(newUnit((*k)(*srcbgn), "")));
    }
    result.push_back(0);
    kp.push_back(k);
  }
  keyringsize = u.size();
  int srcsize = 0;
  for (vctItT i = srcbgn; i!=srcend; ++i){
    twiddle = polar(1., (omega*keymet(*i)).dbl());
    for (j=0; j<keyringsize; ++j)
      result[j] += (*kp[j])(*i)[*u[j]] * twiddle;
    ++srcsize;
  }
  for (j=0; j<keyringsize; ++j)
    result[j] /= srcsize;
    
//  result /= (keymet(*srcend) - keymet(*srcbgn));     // 1/T
  std::vector<complex<physquantity> > fresult;
  for (j=0; j<keyringsize; ++j)
    fresult.push_back(complex<physquantity>(result[j].real() * *u[j], result[j].imag() * *u[j]));
  j=0;
  for (k=keytst; k<keytnd; ++k){
    if ((*k)(*srcbgn).preferredUnit()==NULL){
      delete u[j];
    }
    ++j;
  }
  return fresult;
}

/*
template <typename vctItT, typename keyfnT, typename keyfnTt>
complex<physquantity> esSFTcomponent(vctItT srcbgn, vctItT srcend,
                                   keyfnT keyme, keyfnTt keymet,
                                   physquantity omega){
  const phUnit *u;    // phUnit udummy;
  if (keyme(*srcbgn).preferredUnit()!=NULL){
    u = keyme(*srcbgn).preferredUnit();
   }else{
//    udummy = newUnit(keyme(*srcbgn), "");
    u = new phUnit(newUnit(keyme(*srcbgn), ""));
  }
  complex<double> result=0, twiddle;
  int srcsize = 0;
  for (vctItT i = srcbgn; i!=srcend; ++i){
    twiddle = polar(1., (omega*keymet(*i)).dbl());
    result += keyme(*i)[*u] * twiddle;
    ++srcsize;
  }
  result /= srcsize;
//  result /= (keymet(*srcend) - keymet(*srcbgn));     // 1/T
  return complex<physquantity>(result.real() * *u, result.imag() * *u);
}
*/


/*template <typename vctItT, typename keyfniT, typename keyfnTt>
std::vector<complex<physquantity> > esSFTcomponent(vctItT srcbgn, vctItT srcend,
                                   unsigned int keynum, keyfnTt keymet,
                                   physquantity omega){
  std::vector<const phUnit *> u;    // phUnit udummy;
  std::vector<keyfniT> kp;
  std::vector<complex<double> > result;
  complex<double> twiddle;
  unsigned int keyringsize;
  keyfniT k; unsigned int j;
  for (k=keytst; k<keytnd; ++k){
    if ((*k)(*srcbgn).preferredUnit()!=NULL){
      u.push_back((*k)(*srcbgn).preferredUnit());
     }else{
      u.push_back(new phUnit(newUnit((*k)(*srcbgn), "")));
    }
    result.push_back(0);
    kp.push_back(k);
  }
  keyringsize = u.size();
  int srcsize = 0;
  for (vctItT i = srcbgn; i!=srcend; ++i){
    keymet(*i);
    twiddle = polar(1., (omega*keymet(*i)).dbl());
    for (j=0; j<keyringsize; ++j)
      result[j] += (*kp[j])(*i)[*u[j]] * twiddle;
    ++srcsize;
  }
  for (j=0; j<keyringsize; ++j)
    result[j] /= srcsize;
    
//  result /= (keymet(*srcend) - keymet(*srcbgn));     // 1/T
  std::vector<complex<physquantity> > fresult;
  for (j=0; j<keyringsize; ++j)
    fresult.push_back(complex<physquantity>(result[j].real() * *u[j], result[j].imag() * *u[j]));
  return fresult;
}*/



template <typename containerT, typename keyfnT, typename keyfnTt>
measureseq logscesSFT(typename std::vector<containerT> &src, keyfnT keyme, keyfnTt keymet, float octvDens){
  measureseq result; measure FTcomp;
  FTcomp.resize(2);
  
  class intexpincrementor{unsigned int i; float ovDm;
   public:
    intexpincrementor(const float &novD) {ovDm = pow(2, 1/novD);}
    const intexpincrementor &operator ++() {
      unsigned int ni = i * ovDm;
      i = (ni += (i==ni));
      return *this;
    }
    unsigned int &operator *() {return i;}
  }n(octvDens);

  physquantity omega = keymet(src.back());
  ((omega -= keymet(src.front())) ^= -1) *= 2*pi;
  
  for (*n=1; *n<src.size(); ++n){
    FTcomp[0] = *n * omega;
 //   cout << "omega=" << FTcomp[0];
    FTcomp[1] = abs(esSFTcomponent(src.begin(), src.end(), keyme, keymet, FTcomp[0]));
    result.push_back(FTcomp);
  }
  return result;
}

template <typename containerT, typename keyfniT, typename keyfnTt>
measureseq logscesSFT(typename std::vector<containerT> &srcorg,
                      keyfniT keytst, keyfniT keytnd,
                      keyfnTt keymet, float octvDens){
  //const normalized_compactsupportfn<physquantity, physquantity>&
  auto windowfn = Friedrichs_mollifier;
  
  unsigned int keyringsize=0;
  std::vector<measureindexpointer> newkeyring;
  for(keyfniT k=keytst; k!=keytnd; ++k){
    newkeyring.push_back(measureindexpointer(++keyringsize));
  }

//  physquantity tl = keymet(*wheremin(srcorg, keymet)), tr = keymet(*wheremax(srcorg, keymet)),

  physquantity tl = keymet(srcorg.front()), tr = keymet(srcorg.back());
  physquantity nu_ttl = (tr - tl).inv() * 2,     tmid = (tr + tl)/2;


  measureseq src;
  measure srcmdl;
  srcmdl.resize(keyringsize+1);
  unsigned int j, l, ttlsize=srcorg.size();
  double weight;
  for(j=0; j<ttlsize; ++j){
    srcmdl[0] = keymet(srcorg[j]);
    weight = windowfn((nu_ttl * (srcmdl[0] - tmid)).dbl());
    l=1;
    for(keyfniT k=keytst; k!=keytnd; ++k){

      srcmdl[l] = (*k)(srcorg[j]) * weight;
      ++l;
    }
    src.push_back(srcmdl);
  }
  

  measureseq result; measure FTcomp;
  FTcomp.resize(1+keyringsize);

  
  
  class intexpincrementor{unsigned int i; float ovDm;
   public:
    intexpincrementor(const float &novD) {ovDm = pow(2, 1/novD);}
    const intexpincrementor &operator ++() {
      unsigned int ni = i * ovDm;
      i = (ni += (i==ni));
      return *this;
    }
    unsigned int &operator *() {return i;}
  }n(octvDens);

  physquantity omega = nu_ttl*pi;
  
  std::vector<complex<physquantity> > cpnt;
  
  for (*n=1; *n<src.size(); ++n){
    FTcomp[0] = *n * omega;
 //   cout << "omega=" << FTcomp[0];
    cpnt = esSFTcomponent(src.begin(), src.end(), newkeyring.begin(), newkeyring.end(), measureindexpointer(0), FTcomp[0]);
    for (j=0; j<keyringsize; ++j)
      FTcomp[1+j] = abs(cpnt[j]);
    result.push_back(FTcomp);
  }
  return result;
}



   // performs a "derivative" operation, by calculating the difference quotient
  //  (xᵢ - xᵢ₋₁) / (tᵢ - tᵢ₋₁) and relating this to time (tᵢ + tᵢ₋₁)/2.
template <typename iteratorT, typename keyfnT, typename keyfnTt>
measureseq differentiate(iteratorT srcbgn, iteratorT srcend, keyfnT keyme, keyfnTt keymet){
  measureseq result;
  measure resmsr;
  resmsr.push_back(keymet(*srcbgn));
  resmsr.push_back(keyme(*srcbgn));

  for(iteratorT i = srcbgn+1; i!=srcend; ++i){
    resmsr[0] = (keymet(*i) + keymet(*(i-1)))/2;
    resmsr[1] = (keyme(*i) - keyme(*(i-1)))/(keymet(*i) - keymet(*(i-1)));
    result.push_back(resmsr);
  }

  return result;
}



template <typename containerT, typename keyfnT, typename keyfnTt>
measureseq differentiate(typename std::vector<containerT> &src, keyfnT keyme, keyfnTt keymet){
  return differentiate(src.begin(), src.end(), keyme, keymet);
}



   // A general FIR-kernel convolution calculator.
  //  template versions (deprecated? maybe to do: transform to OOP-style).
 //   convfn is a simple double->double function here.
template <typename iteratorT, typename convolutionT, typename keyfnT, typename keyfnTt>
measureseq orderedFIR(iteratorT srcbgn, iteratorT srcend,
                      convolutionT convfn, keyfnT keyme,
                      keyfnTt keymet, physquantity t_separate){
  measureseq result; measure rspoint;
  rspoint.resize(2);

  physquantity nu = t_separate.inv(), t_mid;
  double convweight, normalizer;
  for (iteratorT sgmtend = srcbgn; sgmtend!=srcend; ++srcbgn){
    while ((keymet(*sgmtend) - keymet(*srcbgn)).abs() < t_separate){
      ++sgmtend;
      if (sgmtend==srcend) return result;
    }
    
    t_mid = (keymet(*sgmtend) + keymet(*srcbgn))/2;
    normalizer = 0;
    rspoint[0] = 0 * t_mid;
    rspoint[1] = 0 * keyme(*srcbgn);
    for (iteratorT sgmi = srcbgn; sgmi!=sgmtend; ++sgmi){
      convweight = convfn(((keymet(*sgmi)-t_mid) * nu).dbl());
      normalizer+=convweight;
      rspoint[0]+=convweight*keymet(*sgmi);
      rspoint[1]+=convweight*keyme(*sgmi);
    }
    rspoint[0]/=normalizer;
    rspoint[1]/=normalizer;
    result.push_back(rspoint);
  }
  
}

template <typename contT, typename convolutionT, typename keyfnT, typename keyfnTt>
measureseq orderedFIR(contT &src,
                      convolutionT convfn, keyfnT keyring,
                      keyfnTt keymet, physquantity t_separate){
  measureseq result; measure rspoint;
//  unsigned int keyringsize = keyring(src.front()).size();
  unsigned int keyringsize = keyring.size();
  rspoint.resize(1 + keyringsize);
  physquantity nu, t_mid;
  if (t_separate.conjugates(keymet(src.front()))){
    nu = (t_separate/=2);
    t_separate ^= -1;
   }else{
    t_separate *= 2;
    nu = t_separate.inv();
  }

  double convweight, normalizer;
  for (typename contT::iterator srcbgn(src.begin()), sgmtend(src.begin()); sgmtend!=src.end(); ++srcbgn){
    while ((keymet(*sgmtend) - keymet(*srcbgn)).abs() < t_separate){
      ++sgmtend;
      if (sgmtend==src.end()) return result;
    }
    
    t_mid = (keymet(*sgmtend) + keymet(*srcbgn))/2;
    normalizer = 0;
    rspoint[0] = 0 * t_mid;
    for (unsigned int i = 0; i<keyringsize; ++i)
      rspoint[i+1] = 0 * keyring[i](*srcbgn);
    for (typename contT::iterator sgmi = srcbgn; sgmi!=sgmtend; ++sgmi){
      convweight = convfn(((keymet(*sgmi)-t_mid) * nu).dbl());
      normalizer+=convweight;
      rspoint[0]+=convweight*keymet(*sgmi);
      for (unsigned int i = 0; i<keyringsize; ++i)
        rspoint[i+1] += convweight*keyring[i](*sgmi);
    }
    rspoint[0]/=normalizer;
    for (unsigned int i = 0; i<keyringsize; ++i)
      rspoint[i+1] /=normalizer;
    result.push_back(rspoint);
  }
  return result;
}


int dither_round(double d) {
  return floor(d + double(rand()) / RAND_MAX);
}


template<typename skeyT=physquantity, typename svalT=physquantity>    //<typename skeyT>
class smooth_unsample;
typedef smooth_unsample<> phys_unsample;

template<typename skeyT, typename svalT>    //<typename skeyT>
class smooth_unsample: public compactsupportfn<skeyT, svalT> {
  struct nodernginfo; friend struct nodernginfo;
  typedef std::multimap<skeyT, nodernginfo> histT;  // <position, (width,height...)> of bell centering around event
  typedef std::vector< std::pair<std::pair<skeyT, nodernginfo>, skeyT> > histdenscvT;
  typedef typename histdenscvT::const_iterator histdenscvT_citer;
//  scalable_compactsupportfn<skeyT, skeyT> *renderfn;
  
  histT eventnodes;
  typedef typename histT::iterator hist_iter;
  typedef typename histT::const_iterator hist_citer;

  struct nodernginfo{
    skeyT w; svalT y,crct; hist_citer ll, rl;
    mutable svalT differsq;
    nodernginfo(const svalT &ny, histT *itdm) : w(0), y(ny), crct(0) { ll = itdm->end(); rl = itdm->end(); }
  };
  
  captT densityname;

  typedef const scalable_compactsupportfn<skeyT, skeyT> kernelT;
  kernelT *stdsmoothener;
  mutable bool stdsmoothener_is_new;
  mutable kernelT *smoothener_of_differcalc;
  
  struct identityfunctional{ skeyT &operator()(skeyT &x){ return x; }
                             const skeyT &operator()(skeyT &x)const{ return x; }   };
                             
//  template<typename iskeyT, typename isvalT>
//  friend QTeXdiagmaster & QTeXdiagmaster::insert_smooth_unsample(const smooth_unsample<iskeyT, isvalT> &H, int max_clutternum, int env_resolution);

  
  skeyT noderngwidthsq(hist_iter lb, hist_iter rb, const skeyT &refto){
    skeyT result=0;
//cout << "[";
    for (hist_iter i = lb; i!=rb; ++i){
  //  DBGBGN;
      skeyT dffrc = i->first - refto;
      result += dffrc*dffrc;
//    DBGEND;
    }
//cout << "]";
    return result;
  }

#if 0
  skeyT minwidth(int i) const { return 1; }
  skeyT minwidth(long i) const { return 1; }
  skeyT minwidth(unsigned i) const { return 1; }
  skeyT minwidth(unsigned long i) const { return 1; }
  skeyT minwidth(float i) const { return 0; }
  skeyT minwidth(double i) const { return 0; }
  skeyT minwidth(physquantity i) const { return i.error(); }
#else
  template <typename srskeyT>
  static srskeyT minwidth(const srskeyT &i) { return uncertainty_forIReal(i); }
#endif
  
  skeyT _mean, _stddeviation, totlbord, totrbord;
  hist_citer midp;
  
  unsigned num_at_minwidth;
  
 public:
  const skeyT &defmean() const { return _mean; }
  const skeyT &defstddeviation() const { return _stddeviation; }
  
  bool worth_smooth_curve() const;
  
 
  template<typename cntTit, typename gkeyT, typename gykeyT>
  void construct_from_container(cntTit vsrcbgn, cntTit vsrcend, const gkeyT &xkeyer, const gykeyT &ykeyer){
    for (cntTit i=vsrcbgn; i!=vsrcend; ++i) {
      if (densityname==cptNULL && xkeyer(*i).caption != cptNULL)
        densityname=globalcaptionsscope.insert("\\rho_{" + *xkeyer(*i).caption + "}").first;
      eventnodes.insert( std::make_pair(xkeyer(*i), nodernginfo(ykeyer(*i),&eventnodes)) );
    }

    physquantity __mean = meanofElem(vsrcbgn, vsrcend, xkeyer);
//    cout << __mean;
    
    _mean = cast_from_phq<skeyT>(__mean);
    midp = eventnodes.lower_bound(defmean());
  //  cout << mean();
    _stddeviation = cast_from_phq<skeyT>(__mean.error() * sqrt(double(eventnodes.size())));

    skeyT meantdist = _stddeviation*_stddeviation * double(eventnodes.size()); // / 4;
    
    totlbord = eventnodes.begin()->first;
    totrbord = eventnodes.rbegin()->first;
    
    num_at_minwidth = 0;

    hist_iter lb=eventnodes.begin(), rb=eventnodes.begin(), tlb, trb;
    int nw=0;

    for (hist_iter i = eventnodes.begin(); i!=eventnodes.end(); ++i){    //for each node, find statistically
                                                                        // significant environment

      for (int g=0; g<2; ++g) if (lb!=i) {++lb; if (lb==rb) --lb; else --nw;}  //

      for (; (nw+1)*noderngwidthsq(lb,rb,i->first) < meantdist
             || (  (rb->first-i->first) < minwidth(i->first)
                 &&(i->first-lb->first) < minwidth(i->first));
           ++nw){
        if (lb==eventnodes.begin())
          ++rb;
         else if (++rb==eventnodes.end()){
          --rb; --lb;
         }else{
          tlb=lb; --tlb;
          trb=rb--;
          if (trb->first - i->first < i->first - tlb->first)
            rb=trb;
           else
            lb=tlb;
        }
      }

   //   cout << rb->first - i->first << " to right, " << i->first - lb->first << " to left -> ";
      i->second.w = std::max(rb->first - i->first, i->first - lb->first);
      if (i->second.w < minwidth(i->first)) {
        i->second.w = minwidth(i->first);
        ++num_at_minwidth;
      }
      //#define USE_EXTENDED_DOMAINS_FOR_SMOOTH_UNSAMPLE_PLOTS
      #ifdef USE_EXTENDED_DOMAINS_FOR_SMOOTH_UNSAMPLE_PLOTS
      if (i->first - i->second.w < totlbord) totlbord = i->first - i->second.w;
      if (i->first + i->second.w > totrbord) totrbord = i->first + i->second.w;
      #else
      if (i->first < totlbord) totlbord = i->first;
      if (i->first > totrbord) totrbord = i->first;
      #endif
  //    cout << i->second.w << " total   (as ";
    //  if (i->second.w == rb->first - i->first) cout << "right)\n";
      // else cout << "left)\n";
//      i->second.ll = lb;
 //     i->second.rl = rb;
//      i->second = sqrt(noderngwidthsq(lb,rb,i->first));
   //   cout << i->first << " : " << i->second << endl;
    }

    
    for (hist_iter i = eventnodes.begin(); i!=eventnodes.end(); ++i){      //make sure the environments
      hist_iter j = eventnodes.upper_bound(i->first - i->second.w);       // are covered
      if (j!=eventnodes.begin()) --j;
      for (; j!=eventnodes.end() && j->first < i->first + i->second.w; ++j){
        if (j->second.ll==eventnodes.end() || j->second.ll->first > i->first)
          j->second.ll = i;
        if (j->second.rl==eventnodes.end() || j->second.rl->first < i->first)
          j->second.rl = i;
      }
    }
    for (hist_iter i = eventnodes.begin(); i!=eventnodes.end(); ++i){      //make sure the environments
      if (i->second.ll!=eventnodes.begin()) --i->second.ll;               // are indeed covered, by adding
      if (i->second.rl!=eventnodes.end()) ++i->second.rl;                //  one more node
    }

  }

      
  skeyT density (skeyT where, kernelT &rfn) const{
    hist_citer m = eventnodes.lower_bound(where), lm=m;
    if (lm!=eventnodes.begin()) --lm;
    if (m==eventnodes.end()) --m;
    skeyT result=0;
    if (m!=eventnodes.end()){
      for (hist_citer r=lm->second.ll; r!=lm->second.rl; ++r)  // && r!=->first-r->second < where
        result += rfn(where - r->first, r->second.w);
    }
    result.caption = densityname;
    return result/eventnodes.size();
  }
  skeyT density (const skeyT &where) const{
    return operator() (where, *stdsmoothener); //Gausslikeshaped_Friedrichsmollifier);
  }
  
  const skeyT &suppleft() const{
    return totlbord;
  }
  const skeyT &suppright() const{
    return totrbord;
  }
  interval<skeyT> support() const{
    return interval<skeyT>(suppleft(),suppright());
  }
 private:
  svalT relevance_of_node(hist_citer i) const{
    return 1/uncertainty_forIReal(i->second.y);
  }
 public:
  svalT eval_with_kernel(const skeyT &where, kernelT &rfn) const{
    hist_citer m = eventnodes.lower_bound(where), lm=m;
    
    svalT accum=0;

    if (lm!=eventnodes.begin()) --lm;
    if (m==eventnodes.end()) --m;
    if (m!=eventnodes.end()){
      histdenscvT consd;   // vector< pair<pair<skeyT, nodernginfo>, skeyT> >
      for (hist_citer r=lm->second.ll; r!=lm->second.rl; ++r) {
        consd.push_back(make_pair(
               *r
             , rfn(where - r->first, r->second.w) * relevance_of_node(r)
        ) );
      }
      std::vector<skeyT> dstymomenta(3,0);   // need density, dipole moment and mean square deviation
      
      for (histdenscvT_citer r=consd.begin(); r!=consd.end(); ++r) {
        skeyT momntc=1;
        for (unsigned i=0; i<dstymomenta.size(); ++i) {
          dstymomenta[i] += momntc * r->second;
          momntc *= r->first.first - where;
        }
      }

      skeyT correcteddsty = 0, one = 1;
      for (histdenscvT_citer r=consd.begin(); r!=consd.end(); ++r) {
        skeyT momntc = dstymomenta[1]/dstymomenta[2];
        correcteddsty += r->second * (one - momntc*(r->first.first - where));
      }
      
      for (histdenscvT_citer r=consd.begin(); r!=consd.end(); ++r) {
        skeyT momntc = dstymomenta[1]/dstymomenta[2];
        accum += (r->first.second.y + r->first.second.crct)
                 * r->second * (one - momntc*(r->first.first - where))/correcteddsty;
      }
    }
    return accum;
  }
  void calc_differences_with_kernel(kernelT &rfn) const{
    if (&rfn!=smoothener_of_differcalc){
      smoothener_of_differcalc = &rfn;
      for (hist_citer r=eventnodes.begin(); r!=eventnodes.end(); ++r) {
        svalT nddif = r->second.y
                          - as_exact_value(eval_with_kernel(r->first, rfn))
            , nduncrtn = uncertainty_forIReal(r->second.y);
        r->second.differsq = nddif*nddif + nduncrtn*nduncrtn/2;
      }
    }
  }
           // TODO: merge eval_with_kernel and evalwkrnl_uncertainty
  svalT evalwkrnl_uncertainty(const skeyT &where, kernelT &rfn) const{
//    std::map<hist_citer, svalT> deviationmmry;
  
    hist_citer m = eventnodes.lower_bound(where), lm=m;
    svalT accum=0;
    
    calc_differences_with_kernel(rfn);
    
    if (lm!=eventnodes.begin()) --lm;
    if (m==eventnodes.end()) --m;
    if (m!=eventnodes.end()){
      typedef
      std::vector<std::pair< std::pair< std::pair< skeyT          // x pos of node
                             , nodernginfo >  // node
                       , svalT >              // deviation to curve
                 , skeyT >                    // weight for this node
            >consdT;
      consdT consd;
      typedef typename consdT::const_iterator consdT_citer;
      for (hist_citer r=lm->second.ll; r!=lm->second.rl; ++r) {
        svalT rfnv = rfn(where - r->first, r->second.w)
         /*   , nddif = r->second.y
                          - as_exact_value(eval_with_kernel(r->first, rfn))
            , nduncrtn = uncertainty_forIReal(r->second.y)*/;
        consd.push_back(make_pair( make_pair( *r
                                            , r->second.differsq )//nddif*nddif + nduncrtn*nduncrtn )
                                 , rfnv * relevance_of_node(r)
        ) );
      }

      std::vector<skeyT> dstymomenta(3,0);   // need density, dipole moment and mean square deviation
      
      for (consdT_citer r=consd.begin(); r!=consd.end(); ++r) {
        skeyT momntc=1;
        for (unsigned i=0; i<dstymomenta.size(); ++i) {
          dstymomenta[i] += momntc * r->second;
          momntc *= r->first.first.first - where;
        }
      }

      skeyT correcteddsty = 0, one = 1;
      for (consdT_citer r=consd.begin(); r!=consd.end(); ++r) {
        skeyT momntc = dstymomenta[1]/dstymomenta[2];
        correcteddsty += r->second * (one - momntc*(r->first.first.first - where));
      }
      
      for (consdT_citer r=consd.begin(); r!=consd.end(); ++r) {
        skeyT momntc = dstymomenta[1]/dstymomenta[2];
        svalT contrb = r->first.second
                           * r->second * (one - momntc*(r->first.first.first - where))
                             ;
        accum += abs(contrb/(correcteddsty));//correcteddsty*r->first.first.second.w));
      }
    }
    svalT finalres = sqrt(accum);
    return finalres;
  }
  svalT unchecked_eval(const skeyT &where) const{
//   if ( globaldebugflag) cout << eval_with_kernel(where, *stdsmoothener) << "  +/-  " << evalwkrnl_uncertainty(where, *stdsmoothener);
    svalT result= announce_uncertainty( eval_with_kernel(where, *stdsmoothener)
                               , evalwkrnl_uncertainty(where, *stdsmoothener)
                               );
    return result;
  }
  svalT eval_derivative_with_kernel(const skeyT &where, kernelT &rfn) const{
    hist_citer m = eventnodes.lower_bound(where);
    if (m==eventnodes.end()) --m;
    skeyT hdelta = m->second.w / 256;
    return ( eval_with_kernel(where+hdelta, rfn) - eval_with_kernel(where-hdelta, rfn) )
                / (2*hdelta);
  }
#if 0         // use the default derivative implementation of compactsupportfn rather than the custom one, as this does not yet support proper uncertainty propagation tracking.
  svalT unchecked_derivative_at(const skeyT &where) const{
    return eval_derivative_with_kernel(where, *stdsmoothener);
  }
#endif

#if 0
class derivative_of_mine: public compactsupportfn<skeyT, svalT> {
    const smooth_unsample orig;
   public:
    derivative_of_mine(const smooth_unsample &o): orig(o) {}
    skeyT suppleft() const{ return orig.suppleft(); }
    skeyT suppright() const{ return orig.suppright(); }
    interval<skeyT> support() const{return orig.support();}
    svalT operator() (const skeyT &where) const{ return orig.derivative_at(where); }
  };
  derivative_of_mine derivative() const { return derivative_of_mine(*this); }
#endif

  void do_correctional_with_kernel(double crctstepsz, kernelT &rfn) {
    std::vector<svalT> allowreldiffrs(eventnodes.size());
    int i=0;
    for (hist_citer j=eventnodes.begin(); j!=eventnodes.end(); ++j,++i) {
      svalT nowdist = eval_with_kernel(j->first, rfn) - j->second.y
          , nuncertain = uncertainty_forIReal(j->second.y)
          , duncertain = uncertainty_forIReal(j->first)
                            * eval_derivative_with_kernel(j->first, rfn);
      allowreldiffrs[i]
        = nowdist*sqrt(nowdist*nowdist
                         / (nuncertain*nuncertain + duncertain*duncertain)
                      );
    }

    i=0;
    for (hist_iter j=eventnodes.begin(); j!=eventnodes.end(); ++j,++i)
      j->second.crct -= allowreldiffrs[i]*crctstepsz;
  }
  void do_correctional(double crctstepsz=.5) {do_correctional_with_kernel(crctstepsz,*stdsmoothener);}
  void do_correctional_disregfreaks_with_kernel(double crctstepsz, kernelT &rfn) {
    std::map<svalT, std::pair<hist_iter,svalT>> allowreldiffrs;
    for (auto j=eventnodes.begin(); j!=eventnodes.end(); ++j) {
      svalT nowdist = eval_with_kernel(j->first, rfn) - j->second.y
          , nuncertain = uncertainty_forIReal(j->second.y)
          , duncertain = uncertainty_forIReal(j->first)
                            * eval_derivative_with_kernel(j->first, rfn);
      allowreldiffrs[ nowdist*nowdist
                           / (nuncertain*nuncertain + duncertain*duncertain)
                    ] = make_pair(j, nowdist);
    }
    auto iit=allowreldiffrs.begin(); for(unsigned k=0; k<eventnodes.size()/2; ++k) ++iit;
    svalT median = iit->first;

    for (auto j=allowreldiffrs.begin(); j!=allowreldiffrs.end(); ++j){
      svalT r = sqrt(j->first / median), one=1;
      j->second.first->second.crct -= j->second.second/(one+r*r);
    }
  }
  void do_correctional_disregfreaks(double crctstepsz=.5) {do_correctional_disregfreaks_with_kernel(crctstepsz,*stdsmoothener);}

  void corrections_it(unsigned crcsteps, double crctstepsz=.125) {
    for(; crcsteps>0; --crcsteps)
      do_correctional_with_kernel(crctstepsz,*stdsmoothener);
  }
  void corrections_disregfreaks_it(unsigned crcsteps, double crctstepsz=.125) {
    for(; crcsteps>0; --crcsteps)
      do_correctional_disregfreaks_with_kernel(crctstepsz,*stdsmoothener);
  }

  measureseq quantized(int desirreso) const{
    measureseq result;

    skeyT l = (*eventnodes.begin()).first, r = (*eventnodes.rbegin()).first, stepw = (r-l)/desirreso;
    for (skeyT x=l; x<r; x+=stepw) {
      measure m(x);
      m.push_back(unchecked_eval(x));
      result.push_back(m);
    }
    
    return result;
  }

  static const int def_autocorrectionalsteps=4;
  template<typename cntT, typename gkeyT, typename gykeyT>
  smooth_unsample( const cntT &vsrc
                 , const gkeyT &xkeyer
                 , const gykeyT &ykeyer
                 , int autocorrectionalsteps=def_autocorrectionalsteps )
    : densityname(cptNULL)
    , stdsmoothener(new Compactsqueezed_Gaussfn(4))
    , stdsmoothener_is_new(true)
    , smoothener_of_differcalc(NULL)                          {
    construct_from_container(vsrc.begin(), vsrc.end(), xkeyer, ykeyer);
    if (densityname==cptNULL) densityname=globalcaptionsscope.insert("\\rho").first;
    for(int i=0; i<autocorrectionalsteps; ++i) do_correctional();
  }
#if 0
  template<typename cntT>
  smooth_unsample(const cntT &vsrc, const char *keyn):
    densityname(globalcaptionsscope.insert(std::string("\\rho_{") + keyn + "}").first),
    stdsmoothener(new Compactsqueezed_Gaussfn(4)),
    stdsmoothener_is_new(true)                          {
    construct_from_container(vsrc.begin(), vsrc.end(), captfinder(keyn));
  }
#endif
  smooth_unsample( const measureseq &vsrc
                 , const msq_dereferencer &xkeyer=measureindexpointer(0)
                 , const msq_dereferencer &ykeyer=measureindexpointer(1)
                 , int autocorrectionalsteps=def_autocorrectionalsteps )
    : densityname(cptNULL)
    , stdsmoothener(new Compactsqueezed_Gaussfn(4))
    , stdsmoothener_is_new(true)
    , smoothener_of_differcalc(NULL)                          {
    construct_from_container(vsrc.begin(), vsrc.end(), xkeyer, ykeyer);
    if (densityname==cptNULL) densityname=globalcaptionsscope.insert("\\rho").first;
    for(int i=0; i<autocorrectionalsteps; ++i) do_correctional();
  }
 private:
    // comparison functor of c_iters by address they dereference to. Would be better done with lambda functions.
  struct cmp_by_tgadr{const histT &d;
    bool operator()(const hist_citer &h, const hist_citer &k){
      if (h==d.end()) return false; if(k==d.end()) return true; return &(*h)<&(*k);
    }
    cmp_by_tgadr(const histT &d): d(d) {}
  };

  void fix_copied_iterators(const smooth_unsample &cpf) {  // fix the internal reference iterators that were invalidated through the copying:
    unsigned lt = eventnodes.size();
    hist_citer itertoloc[lt+1];     // index of the copied map
    unsigned i=0;
    for(hist_citer j=eventnodes.begin(); j!=eventnodes.end(); ++j,++i)
      itertoloc[i] = j;
    itertoloc[lt] = eventnodes.end();

    std::map<hist_citer, unsigned, cmp_by_tgadr> locofiter(cpf.eventnodes);    // back-index of original map
    i=0;
    for(hist_citer j=cpf.eventnodes.begin(); j!=cpf.eventnodes.end(); ++j,++i)
      locofiter[j] = i;
    locofiter[cpf.eventnodes.end()] = lt;
    
    for(hist_iter j=eventnodes.begin(); j!=eventnodes.end(); ++j) {
      j->second.ll = itertoloc[locofiter[j->second.ll]];
      j->second.rl = itertoloc[locofiter[j->second.rl]];
    }
    midp = itertoloc[locofiter[midp]];
  }
 public:
#if 0
  smooth_unsample(smooth_unsample &&cpf)
    : eventnodes(std::move(cpf.eventnodes))
    , densityname(std::move(cpf.densityname))
    , stdsmoothener(cpf.stdsmoothener)
    , stdsmoothener_is_new(std::move(cpf.stdsmoothener_is_new))
    , smoothener_of_differcalc(std::move(cpf.smoothener_of_differcalc))
    , _mean(std::move(cpf._mean)), _stddeviation(std::move(cpf._stddeviation))
    , totlbord(std::move(cpf.totlbord)), totrbord(std::move(cpf.totrbord))
    , midp(std::move(cpf.midp)), num_at_minwidth(std::move(cpf.num_at_minwidth)) {
    cpf.stdsmoothener = nullptr;
  }
#endif
  smooth_unsample(const smooth_unsample &cpf)
    : eventnodes(cpf.eventnodes)
    , densityname(cpf.densityname)
    , stdsmoothener(cpf.stdsmoothener)    // FIXME (See note on cpfstdsmoothener_is_new = false)
    , stdsmoothener_is_new(cpf.stdsmoothener_is_new)
    , smoothener_of_differcalc(cpf.smoothener_of_differcalc)
    , _mean(cpf._mean), _stddeviation(cpf._stddeviation)
    , totlbord(cpf.totlbord), totrbord(cpf.totrbord)
    , midp(cpf.midp), num_at_minwidth(cpf.num_at_minwidth) {
    cpf.stdsmoothener_is_new = false;     // this is dangerous (FIXME): if the copied object is destroyed before the original, the smoothener of the original may remain dangling. TODO: replace with e.g. shared_ptr.
    fix_copied_iterators(cpf);
  }

  smooth_unsample(){}     // Uninitialized objects should be initialized with a copy assignment.
  ~smooth_unsample() {
    if (stdsmoothener_is_new) delete stdsmoothener;
  }
  
  smooth_unsample &operator=(const smooth_unsample &cpf) {
    if (stdsmoothener_is_new) delete stdsmoothener;
    eventnodes = cpf.eventnodes;
    densityname = cpf.densityname;
    stdsmoothener = cpf.stdsmoothener;    // FIXME (See smooth_unsample(const smooth_unsample &cpf)
    stdsmoothener_is_new = cpf.stdsmoothener_is_new;
    smoothener_of_differcalc = cpf.smoothener_of_differcalc;
    _mean=cpf._mean; _stddeviation=cpf._stddeviation;
    totlbord=cpf.totlbord; totrbord=cpf.totrbord;
    midp=cpf.midp; num_at_minwidth=cpf.num_at_minwidth;
    cpf.stdsmoothener_is_new = false;
    fix_copied_iterators(cpf);
    return *this;
  }
  
};



       // At the moment, smooth_unsample is based on a fork of histogram. It
      //  might be good to refactor histogram as a wrapper around smooth_unsample.
template<typename skeyT=physquantity>    //<typename skeyT>
class histogram: public compactsupportfn<skeyT, skeyT> {
  struct nodernginfo; friend struct nodernginfo;
  typedef std::multimap<skeyT, nodernginfo> histT;  // <position, width> of bell centering around event
  
//  scalable_compactsupportfn<skeyT, skeyT> *renderfn;
  
  histT eventnodes;
  typedef typename histT::iterator hist_iter;
  typedef typename histT::const_iterator hist_citer;

  struct nodernginfo{
    skeyT w; hist_citer ll, rl;
    nodernginfo(histT *itdm) : w(0) { ll = itdm->end(); rl = itdm->end(); }
  };
  
  captT densityname;

  const scalable_compactsupportfn<skeyT, skeyT>* stdsmoothener;
  bool stdsmoothener_is_new;
  
  struct identityfunctional{ skeyT &operator()(skeyT &x){ return x; }
                             const skeyT &operator()(skeyT &x)const{ return x; }   };
                             
  template<typename iskeyT>
  friend QTeXdiagmaster & QTeXdiagmaster::insertHistogram(const histogram<iskeyT> &H, int max_clutternum, int env_resolution);

  
  skeyT noderngwidthsq(hist_iter lb, hist_iter rb, const skeyT &refto){
    skeyT result=0;
    for (hist_iter i = lb; i!=rb; ++i){
      skeyT dffrc = i->first - refto;
      result += dffrc*dffrc;
    }
    return result;
  }

  skeyT minwidth(int i) const { return 1; }
  skeyT minwidth(long i) const { return 1; }
  skeyT minwidth(unsigned i) const { return 1; }
  skeyT minwidth(unsigned long i) const { return 1; }
  skeyT minwidth(float i) const { return 0; }
  skeyT minwidth(double i) const { return 0; }
  skeyT minwidth(physquantity i) const { return i.error(); }
  
  skeyT _mean, _stddeviation, totlbord, totrbord;
  hist_iter midp;
  
  unsigned num_at_minwidth;
  
 public:
  const skeyT &mean() const { return _mean; }
  const skeyT &stddeviation() const { return _stddeviation; }
  
  bool worth_smooth_curve() const;
  
 
  template<typename cntTit, typename gkeyT>
  void construct_from_container(cntTit vsrcbgn, cntTit vsrcend, const gkeyT &keyer=identityfunctional()){
    for (cntTit i=vsrcbgn; i!=vsrcend; ++i) {
      if (densityname==cptNULL && keyer(*i).caption != cptNULL)
        densityname=globalcaptionsscope.insert("\\rho_{" + *keyer(*i).caption + "}").first;
      eventnodes.insert(make_pair(keyer(*i), &eventnodes));
    }

    physquantity __mean = meanofElem(vsrcbgn, vsrcend, keyer);
//    cout << __mean;
    
    _mean = cast_from_phq<skeyT>(__mean);
    midp = eventnodes.lower_bound(mean());
  //  cout << mean();
    _stddeviation = cast_from_phq<skeyT>(__mean.error() * sqrt(double(eventnodes.size())));

    skeyT meantdist = _stddeviation*_stddeviation * double(eventnodes.size()) * 4;
    
    totlbord = eventnodes.begin()->first;
    totrbord = eventnodes.rbegin()->first;
    
    num_at_minwidth = 0;

    hist_iter lb=eventnodes.begin(), rb=eventnodes.begin(), tlb, trb;
    int nw=0;

    for (hist_iter i = eventnodes.begin(); i!=eventnodes.end(); ++i){    //for each node, find statistically
      if (lb!=i) {--nw; ++lb;} if (lb!=i) {--nw; ++lb;}                 // significant environment
      for (; (nw+1)*noderngwidthsq(lb,rb,i->first)<meantdist
             || ((rb->first-i->first)<minwidth(i->first) && (i->first-lb->first)<minwidth(i->first));
           ++nw){
        if (lb==eventnodes.begin())
          ++rb;
         else if (++rb==eventnodes.end()){
          --rb; --lb;
         }else{
          tlb=lb; --tlb;
          trb=rb--;
          if (trb->first - i->first < i->first - tlb->first)
            rb=trb;
           else
            lb=tlb;
        }
      }
   //   cout << rb->first - i->first << " to right, " << i->first - lb->first << " to left -> ";
      i->second.w = std::max(rb->first - i->first, i->first - lb->first);
      if (i->second.w < minwidth(i->first)) {
        i->second.w = minwidth(i->first);
        ++num_at_minwidth;
      }
      if (i->first - i->second.w < totlbord) totlbord = i->first - i->second.w;
      if (i->first + i->second.w > totrbord) totrbord = i->first + i->second.w;
  //    cout << i->second.w << " total   (as ";
    //  if (i->second.w == rb->first - i->first) cout << "right)\n";
      // else cout << "left)\n";
//      i->second.ll = lb;
 //     i->second.rl = rb;
//      i->second = sqrt(noderngwidthsq(lb,rb,i->first));
   //   cout << i->first << " : " << i->second << endl;
    }

    
    for (hist_iter i = eventnodes.begin(); i!=eventnodes.end(); ++i){      //make sure the environments
      hist_iter j = eventnodes.upper_bound(i->first - i->second.w);       // are covered
      if (j!=eventnodes.begin()) --j;
      for (; j!=eventnodes.end() && j->first < i->first + i->second.w; ++j){
        if (j->second.ll==eventnodes.end() || j->second.ll->first > i->first)
          j->second.ll = i;
        if (j->second.rl==eventnodes.end() || j->second.rl->first < i->first)
          j->second.rl = i;
      }
    }
    for (hist_iter i = eventnodes.begin(); i!=eventnodes.end(); ++i){      //make sure the environments
      if (i->second.ll!=eventnodes.begin()) --i->second.ll;               // are indeed covered, by adding
      if (i->second.rl!=eventnodes.end()) ++i->second.rl;                //  one more node
    }

  }

  template<typename cntT, typename gkeyT>
  histogram(const cntT &vsrc, const gkeyT &keyer=identityfunctional()):
    densityname(cptNULL),
    stdsmoothener(new Compactsqueezed_Gaussfn(4)),
    stdsmoothener_is_new(true)                          {
    construct_from_container(vsrc.begin(), vsrc.end(), keyer);
    if (densityname==cptNULL) densityname=globalcaptionsscope.insert("\\rho").first;
  }
  template<typename cntT>
  histogram(const cntT &vsrc, const char *keyn):
    densityname(globalcaptionsscope.insert(std::string("\\rho_{") + keyn + "}").first),
    stdsmoothener(new Compactsqueezed_Gaussfn(4)),
    stdsmoothener_is_new(true)                          {
    construct_from_container(vsrc.begin(), vsrc.end(), captfinder(keyn));
  }
  histogram(const measureseq &vsrc, const msq_dereferencer &keyer=measureindexpointer(0)):
    densityname(cptNULL),
    stdsmoothener(new Compactsqueezed_Gaussfn(4)),
    stdsmoothener_is_new(true)                          {
    construct_from_container(vsrc.begin(), vsrc.end(), keyer);
    if (densityname==cptNULL) densityname=globalcaptionsscope.insert("\\rho").first;
  }

  ~histogram() {
    if (stdsmoothener_is_new) delete stdsmoothener;
  }

      
  skeyT operator() (skeyT where, const scalable_compactsupportfn<skeyT, skeyT> &rfn) const{
    hist_citer m = eventnodes.lower_bound(where), lm=m;
    if (lm!=eventnodes.begin()) --lm;
    if (m==eventnodes.end()) --m;
    skeyT result=0;
    if (m!=eventnodes.end()){
//      hist_citer r;
      for (hist_citer r=lm->second.ll; r!=lm->second.rl; ++r)  // && r!=->first-r->second < where
//      for (hist_citer r=eventnodes.begin(); r!=eventnodes.end(); ++r)  // && r!=->first-r->second < where
        result += rfn(where - r->first, r->second.w);
    }
//    if (false && m!=eventnodes.begin()){
     // --m;
   //   for (hist_citer l=m; l!=eventnodes.end() && l->first+l->second > where; --l)
     //   result += rfn(where-l->first, l->second);
  //  }
  //  cout << endl << where << " : " << result << ", lsdkfb: " << rfn(.99) << endl;
    result.caption = densityname;
    return result/eventnodes.size();
  }
  skeyT unchecked_eval(const skeyT &where) const{
    return operator() (where, *stdsmoothener); //Gausslikeshaped_Friedrichsmollifier);
  }
  
  const skeyT &suppleft() const{
    return totlbord;
  }
  const skeyT &suppright() const{
    return totrbord;
  }
  interval<skeyT> support() const{
    return interval<skeyT>(suppleft(),suppright());
  }

//  measureseq quantized(int desirreso) const{    //TODO: actually use desirreso as resolution (or maybe not)
  measureseq quantized() const{
    measureseq result;

    hist_citer pos=midp;
    while (true){
      hist_citer lps = pos, wps;
      pos = eventnodes.upper_bound(lps->first+lps->second.w/2); //lps->second.rl; //
      if (pos==eventnodes.end()) {
        --pos;
        if (pos==lps) break;
/*       }else{
        wps = pos->second.ll;
        pos = eventnodes.upper_bound((pos->first - pos->second.w/2 + lps->first + lps->second.w/2)/2);
  */      
      }
      result.push_back(measure(
          (lps->first + pos->first).plusminus(lps->first - pos->first)/2,
          cast_to_phq(distance(lps,pos))/((pos->first - lps->first) * eventnodes.size())
        ));
      result.back().back().caption = densityname;
    }
    pos=midp;
    while (true){
      hist_citer lps = pos;
      pos = eventnodes.lower_bound(lps->first-lps->second.w/2);  //lps->second.ll; //
//      pos = eventnodes.lower_bound((pos->first + pos->second.w/2 + lps->first - lps->second.w/2)/2);
//      if (!(pos->first < lps->first-lps->second)) --pos;
    //  if (pos==eventnodes.end()) break;
      if (pos==lps) {
        if (pos==eventnodes.begin()) break;
        --pos;
      }
//      if (pos==eventnodes.end() || pos==eventnodes.begin()) break;
      result.push_back(measure(
          (lps->first + pos->first).plusminus(lps->first - pos->first)/2,
          cast_to_phq(distance(pos,lps))/((lps->first - pos->first) * eventnodes.size())
        ));
      result.back().back().caption = densityname;
    }
  //  cout << result.size() << endl;
 //   cout << result.back()[0] << " ,  " << result.back()[1] << endl;
    return result;
  }
  
  
};
template<> inline bool histogram<int>::worth_smooth_curve() const{ return num_at_minwidth < eventnodes.size()/2; }
template<typename skeyT> inline bool histogram<skeyT>::worth_smooth_curve() const{ return true; }

template<typename skeyT>
QTeXdiagmaster & QTeXdiagmaster::insertHistogram(const histogram<skeyT> &H, int max_clutternum, int env_resolution){
  if ( (H.worth_smooth_curve() && env_resolution!=0) || env_resolution>0){
    if (env_resolution<0) env_resolution=default_resolution_for_histogram_envelopes;
    phGraphScreen Envelope(rasterize(H, env_resolution));
    insertCurve(Envelope); //, QTeXgrcolors::blue, "env.qcv");
    int it=0;
    for (typename histogram<skeyT>::hist_citer i = H.eventnodes.begin();
         i!=H.eventnodes.end() && it<max_clutternum;   ++i){
      insertMeasure( (*i).first , rand() * H.unchecked_eval((*i).first) / RAND_MAX );
      ++it;
    }
   }else{
    insertMeasureseq(H.quantized());
  }
  return *this;
}

QTeXdiagmaster & QTeXdiagmaster::insertHistogram(
                  const measureseq &m, const msq_dereferencer &d,
                  int max_clutternum, int env_resolution ){
  histogram<physquantity> H(m, d);
  insertHistogram(H, max_clutternum, env_resolution);

  return *this;
}

namespace_cqtxnamespace_CLOSE


#include "phqfn_a.cpp"

