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


#ifndef PHYSICALQUANTITIES_CONTAINERSFORSUCH_ANDALGORITHMS

#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <typeinfo>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <iterator>
#include <algorithm>
#include <stack>
#include <string>
#include <sstream>
#include <complex>
#include <tr1/random>
#include <cassert>
#include <exception>

//fundamental forward declarations:

#include "fundmeta.h"

// #define USE_NO_NAMESPACE_FOR_CQTX


#ifdef USE_NO_NAMESPACE_FOR_CQTX
#define cqtxnamespace
#define namespace_cqtxnamespace_OPEN
#define namespace_cqtxnamespace_CLOSE


#else

#define cqtxnamespace cqtx
#define namespace_cqtxnamespace_OPEN namespace cqtx{
#define namespace_cqtxnamespace_CLOSE }

#endif

namespace_cqtxnamespace_OPEN


using namespace lambdalike;
using std::string;
using std::stringstream;
using std::ostream;
using std::istream;
using std::ofstream;
using std::ifstream;
using std::cout;
using std::cerr;
using std::endl;

using std::complex;
using ::exp;
using ::sqrt;


class physquantity;

template<typename skeyT>
class histogram;

namespace_cqtxnamespace_CLOSE

     // FIXME: The include hierarchy in these modules is very badly organised.
#include "cqtx_h.cpp"


#define PHYSICALQUANTITIES_CONTAINERSFORSUCH_ANDALGORITHMS

#endif
