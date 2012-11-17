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



                                      //template for derived classes that are supposed
template< class Base, class Derived >// to be copyable from only a base class pointer.
class copyable_derived: public Base {
 public:
  virtual Base *clone() const override {
    return new Derived(static_cast<const Derived&>(*this));
  }
  virtual Base *moved() override {
    return new Derived(static_cast<Derived&&>(std::move(*this)));
  }
};



#define COPYABLE_PDERIVED_CLASS(D, B) \
  class D : public copyable_derived< B, D >
#define COPYABLE_DERIVED_STRUCT(D, B) \
  struct D : copyable_derived< B, D >
#define TEMPLATIZED_COPYABLE_PDERIVED_CLASS(T, D, B) \
  template<class T>                                  \
  class D : public copyable_derived< B, D<T> >
#define TEMPLATIZED_COPYABLE_DERIVED_STRUCT(T, D, B) \
  template<class T>                                  \
  struct D : copyable_derived< B, D<T> >
#define TEMPLATIZED2_COPYABLE_PDERIVED_CLASS(T, T2K, T2, D, B) \
  template<class T, T2K T2>                                    \
  class D : public copyable_derived< B, D<T, T2> >
#define TEMPLATIZED2_COPYABLE_DERIVED_STRUCT(T, T2K, T2, D, B) \
  template<class T, T2K T2>                                    \
  struct D : copyable_derived< B, D<T, T2> >
#define TEMPLATIZED3_COPYABLE_PDERIVED_CLASS(T, T2K, T2, T3K, T3, D, B) \
  template<class T, T2K T2, T3K T3>                                     \
  class D : public copyable_derived< B, D<T, T2, T3> >
#define TEMPLATIZED3_COPYABLE_DERIVED_STRUCT(T, T2K, T2, T3K, T3, D, B) \
  template<class T, T2K T2, T3K T3>                                     \
  struct D : copyable_derived< B, D<T, T2, T3> >





template<class Derived>
class lhsMultipliable {
  auto polymorphic()const -> const Derived&    {
    return static_cast<const Derived&>(*this); }
#if VIRTUAL_TEMPLATE_FUNCTIONS_were_allowed
  template<class LHSMultiplier>
  virtual auto
  lefthand_multiplied(const LHSMultiplier& x)const
     -> ...                                         =0;
#endif
};




#include "lambdalike/maybe.hpp"
#include "lambdalike/polyliteral.hpp"
