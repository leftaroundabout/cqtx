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


#ifndef PHYSQUANTITYSPECILIZATIONS_OF_STANDARD_CLASSES
#define PHYSQUANTITYSPECILIZATIONS_OF_STANDARD_CLASSES



namespace std {


     //The physquantity specialization to std::complex only has the absolute
    // value as a physical quantity, while the phase is represented by a
   //  simple complex<double>'s. This is reasonable since the phase can never
  //   have a physical dimension anyway, however it also prevents you from
 //    getting uncertainty-calculations on it. Use a container like measure
//     to get this functionality.
     //Note that this memory layout is NOT conformant with C++11, which requires (§26.4)
    // — the expression reinterpret_cast<cv T(&)[2]>(z) shall be well-formed,
   //  — reinterpret_cast<cv T(&)[2]>(z)[0] shall designate the real part of z, and
  //   — reinterpret_cast<cv T(&)[2]>(z)[1] shall designate the imaginary part of z.
 //    This seems acceptable enough though, as this paragraph is mainly designated
//     to passing arrays to C APIs, which does not make sense for physquantities at all.
template<>
class complex<cqtxnamespace::physquantity> {
  typedef cqtxnamespace::physquantity physquantity;
  physquantity physicality;   //real-proportional to the object's absolute value
  typedef complex<double> rcomplexT;
  rcomplexT complexity;      //proportional to the object's normalized value
  typedef complex<physquantity> physcomplex;

 public:
  auto real()const -> physquantity { return physicality * complexity.real(); }
  auto imag()const -> physquantity { return physicality * complexity.imag(); }

  auto physdim()const -> physquantity {return physicality;}
  auto complexdir()const -> rcomplexT {return complexity;}

  auto negate() -> complex& {complexity = -complexity; return *this;}
  auto invert() -> complex& {
    complexity = 1./complexity; physicality.invme();
    return *this;
  }

  auto operator+=(const physquantity& addl) -> complex&   {   //no proper uncertainty
    complexity += (addl/physicality).dbl(); return *this; }  // handling here!
  auto operator+(const physquantity& addl)const -> complex {
    return complex(*this) += addl;                         }

  auto operator+=(const complex& addl) -> complex&                        {
    complexity += addl.complexity * (addl.physicality/physicality).dbl();
    return *this;
  }
  auto operator+(const complex& addl)const -> complex {
    return complex(*this) += addl;             }

  auto operator-=(const physquantity& addl) -> complex&          {
    complexity -= (physicality/physicality).dbl(); return *this; }
  auto operator-(const physquantity& addl)const -> complex {
    return complex(*this) -= addl;                         }

  auto operator-=(const complex& addl) -> complex&                        {
    complexity -= addl.complexity * double(addl.physicality/physicality);
    return *this;
  }
  auto operator-(const complex& addl)const -> complex {
    return complex(*this) -= addl;                    }

  auto operator*=(const physquantity& mtpl) -> complex& {
    physicality *= mtpl; return *this;                  }
  auto operator*(const physquantity& mtpl)const -> complex {
    return complex(*this) *= mtpl;                         }

  auto operator*=(const complex& mtpl) -> complex& {
    physicality *= mtpl.physicality; complexity *= mtpl.complexity;
    return *this;
  }
  auto operator*(const complex& mtpl)const -> complex {
    return complex(*this) *= mtpl;                    }

  auto operator*=(const rcomplexT& mtpl) -> complex& {
    complexity *= mtpl;
    return *this;
  }
  auto operator*(const rcomplexT& mtpl)const -> complex {
    return complex(*this) *= mtpl;                      }

  auto operator/=(const physquantity& mtpl) -> complex& {
    physicality /= mtpl; return *this;                  }
  auto operator/(const physquantity& mtpl)const -> complex {
    return complex(*this) /= mtpl;                         }

  auto operator/=(const complex& mtpl) -> complex& {
    physicality /= mtpl.physicality; complexity /= mtpl.complexity;
    return *this;
  }
  auto operator/(const complex &mtpl)const -> complex {
    return complex(*this)/=mtpl;                      }

  auto conjugate() -> complex& {complexity = conj(complexity); return *this;}

  friend auto abs<physquantity>(const physcomplex& v) -> physquantity;
  friend auto arg<physquantity>(const physcomplex& v) -> physquantity;
  friend auto polar<physquantity>(const physquantity&, const physquantity&) -> physcomplex;
  friend auto norm<physquantity>(const physcomplex& v) -> physquantity;

  complex(){}
  complex(const physquantity& realp, const physquantity& imagp=0.) {
    if(!realp.compatible(imagp)){
      std::cerr << "Trying to form complex variable from incompatible values "
            << realp << " and " << imagp << ".\n";
      abort();
    }
    physicality = sqrt(realp.squared() + imagp.squared());
    if(!physicality.is_identical_zero())
      complexity = rcomplexT( double(realp / physicality)
                            , double(imagp / physicality) );
     else
      complexity = rcomplexT(1,0);
  }
  complex(const physquantity& absp, const rcomplexT& cmplxp)
    : physicality(absp)
    , complexity(cmplxp)
  {}
  
  explicit operator complex<double>()const {return complexity * double(physicality);}

  friend auto cqtxnamespace::unphysicalize_cast(const physcomplex& src) -> complex<double>;
};

template<>auto
abs(const complex<cqtxnamespace::physquantity>& v) -> cqtxnamespace::physquantity {
  return abs(v.physicality) * abs(v.complexity);                                  }

template<>auto
arg(const complex<cqtxnamespace::physquantity>& v) -> cqtxnamespace::physquantity {
  return arg(v.complexity) * cqtxnamespace::real1;                                         }

auto
conj(complex<cqtxnamespace::physquantity> v) -> complex<cqtxnamespace::physquantity>{
  return v.conjugate();                                                             }

#if 0
template<>auto
exp(const complex<cqtxnamespace::physquantity> &v) -> complex<cqtxnamespace::physquantity>{
  static_assert(false, "exp is not implemented for complex physquantities.");
}
#endif

template<>auto
norm(const complex<cqtxnamespace::physquantity>& v) -> cqtxnamespace::physquantity {
  return v.physicality.squared() * norm(v.complexity);                             }

template<>auto
polar(const cqtxnamespace::physquantity& a, const cqtxnamespace::physquantity& theta)
        -> complex<cqtxnamespace::physquantity>                           {
  return complex<cqtxnamespace::physquantity>(a, polar(1., theta.dbl())); }

template<>auto
operator+( const cqtxnamespace::physquantity& a
         , const complex<cqtxnamespace::physquantity>& b)
              -> complex<cqtxnamespace::physquantity>       {
  return b+a;
}
template<>auto
operator-( const cqtxnamespace::physquantity& a
         , const complex<cqtxnamespace::physquantity>& b)
              -> complex<cqtxnamespace::physquantity>       {
  return complex<cqtxnamespace::physquantity>(a, complex<double>(1,0)) - b;
}
template<>auto
operator*( const cqtxnamespace::physquantity& a
         , const complex<cqtxnamespace::physquantity>& b)
              -> complex<cqtxnamespace::physquantity>       {
  return b*a;
}
template<>auto
operator/( const cqtxnamespace::physquantity& a
         , const complex<cqtxnamespace::physquantity>& b)
              -> complex<cqtxnamespace::physquantity>       {
  return complex<cqtxnamespace::physquantity>(a, complex<double>(1,0)) / b;
}

/*
template<>auto
imag(const complex<cqtxnamespace::physquantity>& v) -> cqtxnamespace::physquantity {
  return v.physicality * imag(v.complexity);
}
*/









template<>
class normal_distribution<cqtxnamespace::physquantity> {
  typedef normal_distribution<double> rgenT;
  rgenT rgen;
  typedef cqtxnamespace::physquantity physquantity;
  physquantity physicalstddev;
 public:

  template<class RandomEngine>
  physquantity operator()(RandomEngine& r) {
    return rgen(r) * physicalstddev;
  }

  normal_distribution( const physquantity& mean = 0
                     , const physquantity& stddev = 1 )
    : physicalstddev(stddev)                             {
    if (mean != 0) assert(mean.isUnitCompatible(stddev));
    rgen = rgenT((mean/stddev).dbl(), 1);
  }
/*  normal_distribution( const normal_distribution<physquantity>& cpy )
    : rgen(cpy.rgen)
    , physicalstddev(cpy.physicalstddev)  {}
*/
};



template<>
class uniform_real_distribution<cqtxnamespace::physquantity> {
  typedef uniform_real_distribution<double> rgenT;
  rgenT rgen;
  typedef cqtxnamespace::physquantity physquantity;
  physquantity physicalmax;
 public:

  template<class RandomEngine>
  physquantity operator()(RandomEngine& r) {
    return rgen(r) * physicalmax;
  }

  uniform_real_distribution( const physquantity& _min = 0
                     , const physquantity& _max = 1 )
    : physicalmax(_max/_min)                             {
    if (_min != 0 && _max != 0) assert(_min.isUnitCompatible(_max));
    rgen = rgenT((_min/_max).dbl(), 1);
  }

};


}


#endif