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



inline string trim_right(string src , const string& t = " "){
  return src.erase(src.find_last_not_of(t) + 1);           }
inline string trim_left(string src, const string& t = " "){
  return src.erase(0 , src.find_first_not_of(t));        }
inline string trim(string src, const string& t = " "){
  return trim_left(trim_right( src , t) , t);       } 

std::string extract_mathfrom_LaTeXinlinemathenv(string source){
  string::size_type dollar = source.find('$');
  if (dollar == string::npos) return trim(source, " \n\t\b\r");
  source.erase(0, dollar+1);
  dollar = source.find('$');
  if (dollar == string::npos) return string();
  return source.erase(dollar);
}


bool is_closed_LaTeX_block(const std::string s) { return true; } //TODO: should be true for
                                                                // expressions like \left(\frac{a}b\right)

bool is_solid_LaTeX_block(const std::string s) { return s.size()==1; } //TODO: should be true for
                                                                      // expressions like {a^b}


class LaTeX_subscript {
  std::string str;
 public:
  explicit LaTeX_subscript(int i) { std::stringstream s; s<<i; str=s.str(); }
  LaTeX_subscript(const std::string i): str(i) {}
  friend std::string operator+(std::string l, const LaTeX_subscript &i);
  friend bool operator==(std::string l, const LaTeX_subscript &i);
};
std::string operator+(std::string l, const LaTeX_subscript &i) {
  if (!is_closed_LaTeX_block(l)) l = "{" + l + "}";
  if (is_solid_LaTeX_block(i.str))
    return l + "_" + i.str;
   else
    return l+ "_{" + i.str + "}";
}
bool operator==(std::string l, const LaTeX_subscript &i) {
  if (is_solid_LaTeX_block(i.str))
    return l == "_" + i.str;
   else
    return l == "_{" + i.str + "}";
}


class LaTeXindex_itrange;
class LaTeXindex_rangebegin;

class LaTeXindex {
  std::string idstr;
 public:
  LaTeXindex(int i) { std::stringstream s; s<<i; idstr=s.str(); }
  LaTeXindex(const std::string i): idstr(i) {}
  LaTeXindex_rangebegin from(int i) const;
  template<typename IdxCnt>
  LaTeXindex_itrange over(IdxCnt&& is) const;
  std::string &str() { return idstr; }
  const std::string &str() const { return idstr; }
};


class LaTeXindex_rangebegin {
  LaTeXindex rngindex;
  int rngbgn;
  friend class LaTeXindex;
  LaTeXindex_rangebegin(const LaTeXindex &nrngi, int nb): rngindex(nrngi), rngbgn(nb) {}
  LaTeXindex_rangebegin(const LaTeXindex_rangebegin &c): rngindex(c.rngindex) {}
 public:
  LaTeXindex_itrange unto(int) const;
};

LaTeXindex_rangebegin LaTeXindex::from(int i) const{
  LaTeXindex_rangebegin result(*this, i);
  return result;
}

class LaTeXindex_itrange {
  LaTeXindex rngindex;
  std::vector<int> range;
  friend class LaTeXindex;
 public:
  LaTeXindex_itrange(const LaTeXindex& nrngi, int nb, int ne)
    : rngindex(nrngi) { for(;nb<ne; ++nb) range.push_back(nb); }
  template<typename IdxCnt>
  LaTeXindex_itrange(const LaTeXindex& nrngi, const IdxCnt& range)
    : rngindex(nrngi), range(range.begin(), range.end()) {}
  LaTeXindex &index() { return rngindex; }
  const LaTeXindex& index() const{ return rngindex; }
  auto begin()const -> decltype(range.cbegin()) { return range.cbegin(); }
  auto end()const -> decltype(range.cbegin()) { return range.cend(); }
  unsigned size() const{ return range.size(); }
};

LaTeXindex_itrange LaTeXindex_rangebegin::unto(int i) const{
  return LaTeXindex_itrange(rngindex, rngbgn, i);
}

template<typename IdxCnt>
LaTeXindex_itrange LaTeXindex::over(IdxCnt&& is) const{
  return LaTeXindex_itrange(*this, std::forward<IdxCnt>(is));
}



class LaTeXvarnameslist : public std::list<std::string> {
 public:
  using std::list<std::string>::iterator;
  LaTeXvarnameslist() {}
  LaTeXvarnameslist(const LaTeXvarnameslist &cp)
    : std::list<std::string>(cp) {}
  LaTeXvarnameslist(const std::string &init) {
    unsigned i=0, j = init.find(',');
    while(j<init.length()) {
      std::list<std::string>::push_back(init.substr(i, j-i));
      for(i=++j; init[i]==' '; ++i) {}
      if (i<init.size() && init[i]!=',') j=init.find(',', i+1);
       else { cerr<<"Invalid variable names list: \"" << init << "\""; abort(); }
    }
    std::list<std::string>::push_back(init.substr(i));
  }
  LaTeXvarnameslist &expandindextorange(const LaTeXindex_itrange &range) {
    for (iterator i=begin(); i!=end();) {
      if ( i->size() > range.index().str().size()
         && i->substr(i->size()-range.index().str().size()-1) == LaTeX_subscript(range.index().str()) ) {
        for(int j: range) {
          insert(i, i->substr(0, i->size()-range.index().str().size()-1) + LaTeX_subscript(j));
        }
        i=erase(i);
       }else{
        ++i;
      }
    }
    return *this;
  }
  LaTeXvarnameslist operator|(const LaTeXindex_itrange &range) const {
    return LaTeXvarnameslist(*this).expandindextorange(range);
  }
};

LaTeXvarnameslist operator|(const std::string s, const LaTeXindex_itrange &range) {
  return LaTeXvarnameslist(s) | range;
}

std::ostream &operator<<(std::ostream &tgt, const LaTeXvarnameslist &l) {
  LaTeXvarnameslist::const_iterator i=l.begin();
  if (i!=l.end()) tgt << *i;
  for (++i; i!=l.end(); ++i)
    tgt << ", " << *i;
  return tgt;
}


template<typename SepscriptT> auto 
try_splitoff_sepdLaTeXstring(std::string &LaTeXmathexpr, char seperator)
                                     -> maybe<SepscriptT> {
  SepscriptT subscript;
  typedef std::string::size_type idx;
  idx underscorepos = LaTeXmathexpr.rfind(seperator);
  if (underscorepos == std::string::npos) return nothing;
  std::string subsc = LaTeXmathexpr.substr(underscorepos + 1);
  if (subsc.size()==0) return nothing;
  if (subsc.front() == '{' && subsc.back() == '}')
    subsc = subsc.substr(1,subsc.size()-1);
  std::istringstream rdsubscr(subsc);
  try {
    rdsubscr >> subscript;
  }catch(...){return nothing;}
  LaTeXmathexpr.resize(underscorepos);
  return just(subscript);
}

template<typename SubscriptT>
maybe<SubscriptT> try_splitoff_subscript(std::string &LaTeXmathexpr) {
  return try_splitoff_sepdLaTeXstring<SubscriptT>(LaTeXmathexpr, '_');
}
template<typename SupscriptT>
maybe<SupscriptT> try_splitoff_superscript(std::string &LaTeXmathexpr) {
  return try_splitoff_sepdLaTeXstring<SupscriptT>(LaTeXmathexpr, '^');
}

template<typename SubscriptT>
SubscriptT splitoff_subscript(std::string &LaTeXmathexpr) {
  for (auto res : try_splitoff_subscript<SubscriptT>(LaTeXmathexpr))
    return res;
  std::cerr << "Tried to split off subscript from expression \""
            << LaTeXmathexpr << "\" (impossible).\n";
  abort();
}
template<typename SupscriptT>
SupscriptT splitoff_superscript(std::string &LaTeXmathexpr) {
  for (auto res : try_splitoff_superscript<SupscriptT>(LaTeXmathexpr))
    return res;
  std::cerr << "Tried to split off superscript from expression \""
            << LaTeXmathexpr << "\" (impossible).\n";
  abort();
}


namespace_cqtxnamespace_CLOSE