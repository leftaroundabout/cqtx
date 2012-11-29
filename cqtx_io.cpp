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



struct LaTeXofstream: ofstream{

  LaTeXofstream(const string &opstr): ofstream(opstr.c_str()) {}

  struct tablecolumn {
    unsigned int strwidth;
    captT caption;
    const phUnit *tunit;  bool globalunit;
    physquantity err;     bool globalerror;
    std::vector<const physquantity*> resp;
    
    tablecolumn(const physquantity &gen, unsigned int rows, unsigned int sttrow) : resp(rows, NULL) {
      resp[sttrow] = &gen;
      caption=gen.caption;
      tunit=gen.preferredUnit();
      globalunit=globalerror=true;
      err=gen.error();
      strwidth = 0;
    }

    bool compatible(const physquantity &cmpq){
      return (cmpq.compatible(*resp.front()));
    }
    
    std::string titleoutp()const {
      unsigned int i=0;
      while (resp[i]==NULL){++i; if(i==resp.size()) return " ";}
      return resp[i]->LaTeXval(true, false, false, globalunit && *tunit!=real1, *tunit);
    }
    std::string celloutp(unsigned int i)const {
      if (resp[i] == NULL) return "";
      if (resp[i]->preferredUnit()==NULL)
        return resp[i]->LaTeXval(false, true, !globalerror, !globalunit, *tunit);
       else
        return resp[i]->LaTeXval(false, true, !globalerror, !globalunit, *(resp[i]->preferredUnit()));
    }
    std::string footnoteoutp()const {
      if (!globalerror) return " ";
      unsigned int i=0;
      while (resp[i]==NULL){++i; if(i==resp.size()) return " ";}
      std::string result = resp[i]->LaTeXval(false, false, true, !globalunit, *tunit);
      if (result.size()==0) result = " ";
      return result;
    }

    void fetchsomeunit(){
      for (unsigned int i=0; i<resp.size(); ++i){
        if (tunit!=NULL) return;
        if (resp[i]!=NULL) {
    #if 0 
          stringstream targ;
          physquantity &inputthis = *resp[i];
          inputthis.tryfindUnit();
          if (inputthis.tunit==NULL){
            if (inputthis.myError.valincgs > 0) targ << '(';
            targ << inputthis.valincgs;
            if (inputthis.myError.valincgs > 0)
              targ << "\\pm" << inputthis.myError.valincgs << ')';
            if (inputthis.myDim != 0){
              targ << "cm^(" << inputthis.myDim.c()
                  << ")*g^(" << inputthis.myDim.g()
                  << ")*s^(" << inputthis.myDim.s()<< ")" ;
            }    
           }else{
           tunit = inputthis.tunit;
          }
     #endif
           tunit = suitableUnitfor(*resp[i]);
        }
      }
      if (tunit!=NULL) return;
      //If no unit could be found (i.e. still tunit==NULL), we're in trouble.
      cerr << "Unable to find unit for table column {" << endl;
      for (unsigned int i=0; i<resp.size(); ++i){
        if (resp[i]!=NULL)
          cerr << "  " << *resp[i] << endl;
         else
          cerr << "  [empty cell]" << endl;
      }
      cout << "}";
      throw *this;
    }
    
    void clcstrwidth(){
      string cellstr;
      if (tunit == NULL) fetchsomeunit();
      for (unsigned int i=0; i<resp.size(); ++i){
        cellstr = celloutp(i);
        if (cellstr.size() > strwidth) strwidth = cellstr.size();
//          if (resp[i]->preferredUnit() == &megaparsecs) thiscritical = true;
      }
      cellstr = titleoutp();
      if (cellstr.size() > strwidth) strwidth = cellstr.size();
    }
  };

  void inserttable(const measureseq& intable, string TableCaption){

    unsigned int rown=0;

    std::vector<tablecolumn> columns;
    
    for (auto& m: intable) {
      for (auto& q: m) {
        for (auto& k: columns) {
          if (k.resp[rown]==NULL){
            if (k.caption == q.caption){
              if (k.caption == cptNULL){
                if (k.tunit == q.preferredUnit()){
                  if (k.compatible(q)){
                    if (k.err != q.error()) k.globalerror=false;
                    k.resp[rown] = &q;
                    goto skipnewcol;
                  }
                }
               }else{
                if (k.tunit != q.preferredUnit()){
                  if (k.tunit == NULL) k.tunit=q.preferredUnit();
                   else k.globalunit=false;
                }
                if (k.err != q.error()) k.globalerror=false;
                k.resp[rown] = &q;
                goto skipnewcol;
              }
            }
          }
        }
        columns.push_back(tablecolumn(q, intable.size(), rown));
        skipnewcol:
        if(0) std::cout << "After skip-goto-marker. This should never be displayed.";
      }
      ++rown;
    }
    if (columns.size()==0) {return;}


    for (auto& k: columns) k.clcstrwidth();

    stringstream tgt;

    auto forcolumns_print
           = [&]( std::function<void(const tablecolumn&)> col_print
                , const std::string& coldelim                       ) {
               tgt << std::setw(columns.front().strwidth);
               col_print(columns.front());
               for (auto k=columns.begin()+1; k!=columns.end(); ++k) {
                 tgt << coldelim << std::setw(k->strwidth);
                 col_print(*k);
               }
             };
      
    tgt << "\\begin{table}[htbp]\n \\centering\n  \\begin{tabular}{";
    forcolumns_print( [&](const tablecolumn& _) {
                        tgt<<std::setw(1)<<"c"; }
                    , "|"                         );
    tgt << "}\n";

    tgt.setf(left);

    tgt << "   ";
    forcolumns_print( [&](const tablecolumn& k) {
                        tgt << k.titleoutp();   }
                    , " & "                       );

    tgt << "\\\\\\hline";

    for (rown = 0; rown<intable.size(); ++rown){
      if (rown>0) tgt<<"\\\\";
      tgt << "\n   ";
      forcolumns_print( [&](const tablecolumn& k)  {
                          tgt << k.celloutp(rown); }
                      , " & "                        );
      tgt << string(rown+1, ' ') << "";
    }


    bool tsts=false;
    for (auto& k: columns)
      if (k.footnoteoutp().size() > 2) tsts=true;

    if (tsts){
      tgt << "\\\\\n  \\hline\n   ";
      forcolumns_print( [&](const tablecolumn& k)  {
                          tgt << k.footnoteoutp(); }
                      , " & "                        );
    }

    tgt << "\n  \\end{tabular}\n   \\caption{" << TableCaption << "}\n    \\label{tab:id"
        << rand() << "}\n\\end{table}\n";


    *this<<tgt.str();
  }

  LaTeXofstream &operator << (physquantity insq){
    insq.tryfindUnit();
    if (insq.preferredUnit()==NULL && true){
	    phUnit cgsU = cgsUnitof(insq);
      *this << insq.mathLaTeXval(false, true, true, true, cgsU);
	   }else
      *this << insq.mathLaTeXval(false, true, true, true, (*insq.preferredUnit()));
    return *this;
  }
  
  void insertfullvaroutp(physquantity insq){
    *this << "\\[\n  " << insq.cptstr() << " = ";
    operator<<(insq);
    *this << "\n\\]\n";
  }
  
};



template<typename streamT>
struct LaTeXistream;
typedef LaTeXistream<std::istringstream> LaTeXistringstream;
  
// parser for LaTeX expressions
template<typename streamT>
struct LaTeXistream: streamT{
  typedef LaTeXistringstream strs;

  LaTeXistream() : streamT(){}
 // LaTeXistream(const streamT &cpyfrom) {streamT::str((cpyfrom.str()));}
  explicit LaTeXistream(const char *ifname) : streamT(ifname){}
  
  LaTeXistream(string initstr) : streamT(initstr){}
  
  bool eofs() const{return streamT::eof();}
  
  string getbraced(char lprth, char rprth){
    stringstream result;

    unsigned int parentlyr=0;

    do { if(eofs()) return string(); } while(streamT::get()!=lprth);
    while (!eofs() && (streamT::peek()!=rprth || parentlyr)){
      if (streamT::peek()!=lprth) ++parentlyr;
      if (streamT::peek()!=rprth) --parentlyr;
      result << char(streamT::get());
    }

    if(eofs()) return string();
    streamT::get();
    return result.str();
//    return strs(result);
  }
  string getbraced(){return getbraced('{', '}');}
  static string getbraced(string whatfrom, char lprth, char rprth) {return strs(whatfrom).getbraced(lprth, rprth);}
  static string getbraced(string whatfrom) {return strs(whatfrom).getbraced();}
  
  string getcommand(){
    unsigned int parentlyr=0;
    string result;
    do {
//      cout << char(streamT::peek());
  //    for(int i = 0; i<928398; ++i){}
      if ( streamT::fail() || streamT::bad() || eofs()
          || (streamT::peek()=='}' && (parentlyr--) == 0))
        return string();
      if (streamT::peek()=='{') ++parentlyr;
    } while(streamT::get()!='\\' || parentlyr>0);
    bool iswcommand=true;
    while(!eofs() && ((iswcommand && isalpha(streamT::peek())) || (result.size()<1))){  // && (iswcommand=false)
      if (!isalpha(streamT::peek())) iswcommand=false;
      result += streamT::get();
  //    cout << result[result.size()-1];
    }
//    cout << endl;
    return result;
  }
  static string getcommand(string whatfrom) {return strs(whatfrom).getcommand();}

  string getcommand_here(){
    if(streamT::peek()!='\\') return string();
    streamT::get();
    string result;
    bool iswcommand=true;
    while(!eofs() && ((iswcommand && isalpha(streamT::peek())) || (result.size()<1))){  // && (iswcommand=false)
      if (!isalpha(streamT::peek())) iswcommand=false;
      result += streamT::get();
    }
    return result;
  }
  static string getcommand_here(string whatfrom) {return strs(whatfrom).getcommand_here();}

  string peekcommand(){
    if(eofs())return string();
    std::ios::pos_type oldpos=streamT::tellg();
    string result=getcommand();
    if(eofs()) streamT::clear();
    streamT::seekg(oldpos);
    return result;
  }
  string peekcommand_here(){
    if(eofs())return string();
    std::ios::pos_type oldpos=streamT::tellg();
    string result=getcommand_here();
    if(eofs()) streamT::clear();
    streamT::seekg(oldpos);
    return result;
  }
  
  string get_until_command(string delimcmd){
    string result;
    while(!eofs() && peekcommand_here()!=delimcmd)
      result += streamT::get();
    return result;
  }
  string get_until_command(string whatfrom, string delimcmd) const{return strs(whatfrom).get_until_command(delimcmd);}

  bool command_appears(string qcmd) {
    while(!eofs()) { if(getcommand()==qcmd) return true; }
    return false;
  }
  static bool contains_command(string whatshall, string qcmd) {return strs(whatshall).command_appears(qcmd);}

  void ignoreifnextcommandis(string cmdtoign){
    if (peekcommand()==cmdtoign) getcommand();
  }
  
  string getpofcommand(string desiredcmd){
    while(!eofs() && getcommand()!=desiredcmd) {}
    if (eofs() || streamT::peek()==' ') return string();
    if (streamT::peek()=='{') return getbraced();
    return string(1, char(streamT::get()));
  }
  static string getpofcommand(string whatfrom, string desiredcmd) {return strs(whatfrom).getpofcommand(desiredcmd);}
  
  string getenvironment(string desiredenv){
    stringstream result;
    
    while(!eofs() && getpofcommand("begin")!=desiredenv) {}
    do{
      while (!eofs() && streamT::peek()!='\\') result << char(streamT::get());
      if (eofs()) return string();
      string intrpt0 = getcommand();
      if (intrpt0!="end" || streamT::peek()!='{') result << "\\" << intrpt0;
       else{
        string thisendenv = getbraced();
        if (thisendenv!=desiredenv) result << "\\end{" << thisendenv << "}";
         else break;
      }
    } while(1);
    
    return result.str();
  }
  static string getenvironment(string whatfrom, string desiredenv) {return strs(whatfrom).getenvironment(desiredenv);}

  string peekrest(){
    if (eofs()) return string();
    std::ios::pos_type oldpos=streamT::tellg();
    string result;
    while(!eofs()) result.append(1, streamT::get());
    streamT::clear();
    streamT::seekg(oldpos);
    return result;
  }

  string getLaTeXline(){
    string result;
    while(1){
      while (!eofs() && streamT::peek()!='\\') result.append(1, streamT::get());
      if (eofs()) return result;
      string intrpt0 = getcommand();
      if (intrpt0!="\\") {
        result.append("\\").append(intrpt0);
       }else{
        if(!eofs() && streamT::peek()=='\n') streamT::get();
        return result;
      }
    }
  }
  static string getLaTeXline(string whatfrom) {return strs(whatfrom).getLaTeXline();}

  string getLaTeXhcell(){string result; getline(*this, result, '&'); return result;}
  static string getLaTeXhcell(string whatfrom) {return strs(whatfrom).getLaTeXhcell();}

  string getwhitespace() {
    string result;
    while(isspace(streamT::peek())) result.append(1, streamT::get());
    return result;
  }

  double getnextbestnumber() {
    double result;
    string keeprecord;
    int parentlyr=0;
    do {
      if ( streamT::fail() || streamT::bad() || eofs()
          || (streamT::peek()=='}' && (parentlyr--) == 0)) {
        cerr << "Tried to extract number from LaTeX stream, but found only string \""
             << keeprecord << "\".\n";
        abort();
      }
      if (streamT::peek()=='{') ++parentlyr;
      if ((!isdigit(streamT::peek()) && streamT::peek()!='.') || parentlyr>0)
        keeprecord += streamT::get();
       else
        break;
    } while(1);

    int numofdecpoints=0;
    string resultstr;
    while (!eofs() && (isdigit(streamT::peek())
                      || (streamT::peek()=='.' && numofdecpoints++ == 0 )) ){
      resultstr += streamT::get();
  //    cout << result[result.size()-1];
    }
    if (resultstr==".") {
      cerr << "Tried to extract number from LaTeX stream, but found only string \""
           << keeprecord+resultstr << "\".\n";
      abort();
    }
    std::stringstream fetchthenum(resultstr);
    fetchthenum >> result;
    return result;
  }


  bool veof(){
    std::ios::pos_type oldpos=streamT::tellg();
    getwhitespace();
    if(eofs()) return true;
    streamT::clear();
    streamT::seekg(oldpos);
    return false;
  }

  measureseq extracttable(string TableCaption){
    measureseq result;
    
    strs tableenv;
    do{
      tableenv.str(getenvironment("table"));
    }while (tableenv.str().size()>0 && getpofcommand(tableenv.str(), "caption")!=TableCaption);
    
    strs tabenv(tableenv.getenvironment("tabular"));
    
    tabenv.getbraced();             //ignore column align information
    
    std::vector<string> colcapts;
    std::vector<std::pair<const phUnit *, bool> > colunits;
    strs colcaptst(tabenv.getLaTeXline());

    do {
      string colcaptstr = extract_mathfrom_LaTeXinlinemathenv(colcaptst.getLaTeXhcell());
      if (colcaptstr.size()>0){
        string colunitstr = getpofcommand( getbraced(colcaptstr, '[',']'), "mathrm");
        if (colunitstr.size()>0){
          colcapts.push_back(colcaptstr.erase(colcaptstr.find("[")));
          colunits.push_back(std::make_pair(findUnit(colunitstr), true));
          if (colunits.back().first==NULL){
            cout << "In table \"" << TableCaption << "\": unknown unit \"" << colunitstr << "\".\n";
            abort();
          }
         }else{
          colcapts.push_back(colcaptstr);
          colunits.push_back(std::pair<const phUnit*, bool>(NULL, false));
        }
       }else{
        break;
      }
    } while(1);
    
    /*for (unsigned int i=0; i<colcapts.size(); ++i){
      if (colunits[i]==NULL)
        cout << colcapts[i] << " & ";
       else
        cout << colcapts[i] << "[\\mathrm{" << colunits[i]->uName << "}] & ";
    }*/
    
    if (tabenv.getcommand() != "hline"){
      cerr << "Table \"" << TableCaption << "\": bad format; need \\hline after column headers\n";
      abort();
    }
    while (!tabenv.veof() && tabenv.peekcommand()!="hline"){
      strs valcls(tabenv.getLaTeXline());
//      cout << "Reading table from input: - - - - {" << valcls.str() << "\n - - - - - - - - - - - - - - - - }" << endl ;
      unsigned int coln=0;
      measure thisline;
      thisline.clear();
      do {
        physquantity thisthing=0;
        string colvstr = extract_mathfrom_LaTeXinlinemathenv(valcls.getLaTeXhcell());
        
        if (coln>colunits.size() || colunits[coln].second==false){
          string thisunitstr = getpofcommand(colvstr, "mathrm");
          const phUnit *thisunit;
          if (thisunitstr==string()){
            thisunit = &real1;
           }else{
            thisunit = findUnit(thisunitstr);
          }
          if (thisunit==NULL){
            cerr << "In table \"" << TableCaption << "\": unknown unit \"" << thisunitstr << "\".\n";
            abort();
          }
          if (colunits[coln].first!=NULL && colunits[coln].first->Dimension != thisunit->Dimension){
            cerr << "In table \"" << TableCaption << "\": incompatible units \"" << colunits[coln].first->uCaption() << ", "
                                                                                 << thisunit->uCaption() << "\".\n";
            abort();
          }
          colunits[coln].first = thisunit;
          colvstr = get_until_command(colvstr, "mathrm");
        }

        if (contains_command(colvstr, "mathrm")){
          cerr << "In table \"" << TableCaption << "\": redundant unit? \"" << colvstr << "\".\n";
          abort();
        }
        
        if (contains_command(colvstr, "pm")){
          strs tcl(colvstr);
//          double thisval;
          thisthing[*colunits[coln].first] = tcl.getnextbestnumber();
          string shouldbe_pm=tcl.getcommand();
          if (shouldbe_pm!="pm"){
            cerr << "In table \"" << TableCaption << "\": bad cell format \"" << colvstr
                 << "\".\n(Expected a $\\pm$ but found \"" << shouldbe_pm << "\".)";
            abort();
          }
          physquantity thiserror;
          thisthing.seterror(tcl.getnextbestnumber() * *colunits[coln].first);
         }else{
          strs tcl(colvstr);
          tcl >> thisthing[*colunits[coln].first];
        }
//        cout << thisthing << endl;
        thisline.push_back(thisthing.label(colcapts[coln]));
        ++coln;
      }while (!valcls.eofs());
       //cout << thisline;
      result.push_back(thisline);
    }
    return result;
    
 /*   while(!eof()){
      
    }*/
    
    return result;
  }
  
};

typedef LaTeXistream<ifstream> LaTeXifstream;











namespace QTeXgrcolordefault {
  const int colorseqlength = 10;
  const int colornumsseq[colorseqlength] = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12};
  const int red = 4, green = 2, blue = 1,
            i_red=12,i_green=10,i_blue=9,
            teal = 3, lilac = 5, brown = 8,
            i_teal=11,i_lilac=13,i_brown=14,
            grey = 8, i_grey = 7,
            contrast = 15, neutral = 0;
}

class QTeXgrcolor {
  int colorn;
  QTeXgrcolor(int cn): colorn(cn) {}
 public:
  QTeXgrcolor(): colorn(-1) {}
  QTeXgrcolor(const QTeXgrcolor &cn): colorn(cn.colorn) {}
  bool operator==(const QTeXgrcolor &cm) const { return colorn == cm.colorn; }
  bool operator!=(const QTeXgrcolor &cm) const { return colorn != cm.colorn; }
  bool is_undefined() const { return (colorn < 0); }
  friend class QTeXgrcoloraccess;
  int QTeXgrofstreamid() const {
    using namespace QTeXgrcolordefault;
    if (!is_undefined())
     return colorn;
    return colornumsseq[rand()%colorseqlength];
  }
};

class QTeXgrcoloraccess {
 public:
  static std::vector<QTeXgrcolor> defaultcolorssequence() {
    using namespace QTeXgrcolordefault;
    std::vector<QTeXgrcolor> result;
    for (int i = 0; i<colorseqlength; ++i) {
      result.push_back(colornumsseq[i]);
    }
    return result;
  }
  static QTeXgrcolor defaultcolorssequence(int i) {
    using namespace QTeXgrcolordefault;
    return colornumsseq[i%colorseqlength];
  }
  static QTeXgrcolor QTeXgrcolor_w_id(int i) {
    return i;
  }
  static QTeXgrcolor red()      { return QTeXgrcolordefault::red;      }
  static QTeXgrcolor green()    { return QTeXgrcolordefault::green;    }
  static QTeXgrcolor blue()     { return QTeXgrcolordefault::blue;     }
  static QTeXgrcolor i_red()    { return QTeXgrcolordefault::i_red;    }
  static QTeXgrcolor i_green()  { return QTeXgrcolordefault::i_green;  }
  static QTeXgrcolor i_blue()   { return QTeXgrcolordefault::i_blue;   }
  static QTeXgrcolor teal()     { return QTeXgrcolordefault::teal;     }
  static QTeXgrcolor lilac()    { return QTeXgrcolordefault::lilac;    }
  static QTeXgrcolor brown()    { return QTeXgrcolordefault::brown;    }
  static QTeXgrcolor i_teal()   { return QTeXgrcolordefault::i_teal;   }
  static QTeXgrcolor i_lilac()  { return QTeXgrcolordefault::i_lilac;  }
  static QTeXgrcolor i_brown()  { return QTeXgrcolordefault::i_brown;  }
  static QTeXgrcolor grey()     { return QTeXgrcolordefault::grey;     }
  static QTeXgrcolor i_grey()   { return QTeXgrcolordefault::i_grey;   }
  static QTeXgrcolor contrast() { return QTeXgrcolordefault::contrast; }
  static QTeXgrcolor neutral()  { return QTeXgrcolordefault::neutral;  }
  static QTeXgrcolor undefinedcolor(const QTeXgrcolor &lastcl) { return QTeXgrcolor(); }
  static QTeXgrcolor nextbestcolor(const QTeXgrcolor &lastcl) {
    using namespace QTeXgrcolordefault;
    for (int i = 0; i<colorseqlength-1; ++i)
      if (colornumsseq[i]==lastcl.colorn) return colornumsseq[i+1];
    return colornumsseq[0];
  }
};

namespace QTeXgrcolors {
  const std::vector<QTeXgrcolor> defaultcolorssequence = cqtxnamespace::QTeXgrcoloraccess::defaultcolorssequence();
  QTeXgrcolor defaultsequenceat(int i) {
    return cqtxnamespace::QTeXgrcoloraccess::defaultcolorssequence(i);
  }
  QTeXgrcolor defaultsequence[] = { defaultsequenceat(0), defaultsequenceat(1), defaultsequenceat(2), defaultsequenceat(3)
                                  , defaultsequenceat(4), defaultsequenceat(5), defaultsequenceat(6), defaultsequenceat(7)
                                  , defaultsequenceat(8), defaultsequenceat(9) };   //, defaultsequenceat(10), defaultsequenceat(11), defaultsequenceat(12), defaultsequenceat(13), defaultsequenceat(14), defaultsequenceat(15)
  const QTeXgrcolor red     = cqtxnamespace::QTeXgrcoloraccess::red(),
                    green   = cqtxnamespace::QTeXgrcoloraccess::green(),
                    blue    = cqtxnamespace::QTeXgrcoloraccess::blue(),
                    i_red   = cqtxnamespace::QTeXgrcoloraccess::i_red(),
                    i_green = cqtxnamespace::QTeXgrcoloraccess::i_green(),
                    i_blue  = cqtxnamespace::QTeXgrcoloraccess::i_blue(),
                    teal    = cqtxnamespace::QTeXgrcoloraccess::teal(),
                    lilac   = cqtxnamespace::QTeXgrcoloraccess::lilac(),
                    brown   = cqtxnamespace::QTeXgrcoloraccess::brown(),
                    i_teal  = cqtxnamespace::QTeXgrcoloraccess::i_teal(),
                    i_lilac = cqtxnamespace::QTeXgrcoloraccess::i_lilac(),
                    i_brown = cqtxnamespace::QTeXgrcoloraccess::i_brown(),
                    grey    = cqtxnamespace::QTeXgrcoloraccess::grey(),
                    i_grey  = cqtxnamespace::QTeXgrcoloraccess::i_grey(),
                    constrast=cqtxnamespace::QTeXgrcoloraccess::contrast(),
                    neutral  =cqtxnamespace::QTeXgrcoloraccess::neutral(),
                    undefined;
  QTeXgrcolor nextbest(const QTeXgrcolor &lastcl) { return cqtxnamespace::QTeXgrcoloraccess::nextbestcolor(lastcl); }
}


class QTeXgrofstream : protected ofstream {  //Inheritance is handled improperly here.
                                            // Yet to fix!
  union {double dbl; uint16_t intg; char chr;};
 protected:

  QTeXgrofstream &open(const string &nfname){
    ofstream::open(nfname.c_str(), std::ios::binary);
    return *this;
  }
//  QTeXgrofstream(const QTeXgrofstream &src){}
  QTeXgrofstream(){}
 public:
  QTeXgrofstream &iwrite(const int& Numtw){
    intg = Numtw; write(&chr, sizeof intg);
    return(*this);
  }
  QTeXgrofstream &dwrite(const double &Numtw){
    dbl = Numtw; write(&chr, sizeof dbl);
    return(*this);
  }
  QTeXgrofstream &dpwrite(double Numtw0, double Numtw1){
    dbl = Numtw0; write(&chr, sizeof dbl);
    dbl = Numtw1; write(&chr, sizeof dbl);
    return(*this);
  }
  QTeXgrofstream &cwrite(char Numtw){
    chr = Numtw; write(&chr, sizeof chr);
    return(*this);
  }


  QTeXgrofstream(const string &nfname){
    ofstream::open(nfname.c_str(), std::ios::binary);
  }
  



  QTeXgrofstream &operator <<(const physquantity &src){
    physquantity srccpy = src;
    src.tryfindUnit();
    const phUnit *srcunit = src.preferredUnit();
//    cout << 'L' << srccpy << endl;
    if (srcunit != NULL){
      put('f');
      if (srccpy.caption == cptNULL){
        iwrite(0);
       }else{
        iwrite(srccpy.caption->length());
        *this << *srccpy.caption;
      }
      dwrite(srccpy[*srcunit]);
      dwrite(srccpy.error()[*srcunit]);
      iwrite(srcunit->uName.length());
      *this << srcunit->uName;
     }else{
      cout << "Einheit bitte!"; abort();
    }
   return(*this);
  }
  
  QTeXgrofstream &operator <<(const string &src){
    write (src.c_str(), src.length());

    return(*this);
  }
  
  bool operator!() const{
    return fail();
  }

  void finish() {
    close();
  }

  /*QTeXgrofstream &operator <<(const string &src);
*/

//  friend class QTeXdiagmaster;// QTeXdiagmaster::&insertCurve(const phGraphScreen &src, const string &nfname);
};







class phGraphScreen {
  static const int hres=110,vres=56;
  char Pixel[hres][vres];
  char tcolor, bcolor;
  physquantity lcbord,rcbord,ucbord,dcbord;
  physquantity width, height;
  char setcolor(float scolor){
    if (scolor<0) scolor=0;
    if (scolor>1) scolor=1;
    return (int) (scolor * 7);
  }
  struct pntmem{
    double x,y;
    bool lineto;
  };
  std::vector<pntmem> pointmem;
  bool fixedraster, sureaboutxUnit, sureaboutyUnit;
  
  void crcSizes(){
    width=rcbord-lcbord; height=ucbord-dcbord;
  }
  void rescale(physquantity lb, physquantity rb, physquantity ub, physquantity db){
    lcbord=lb; rcbord=rb; ucbord=ub; dcbord=db;
    crcSizes();
    sureaboutxUnit = sureaboutyUnit = fixedraster = true;
  }
  void bgsolid(float scolor){
    bcolor = setcolor (scolor);
    for (int x=0; x<hres; ++x){
      for (int y=0; y<vres; ++y) 
        Pixel[x][y] = bcolor;
    }
  }
 public:
  void paintsolid(float scolor){
    bgsolid(scolor);
    pointmem.clear();
  }

  phGraphScreen(){
    sureaboutxUnit = sureaboutyUnit = fixedraster = false;
  }
  phGraphScreen(physquantity lb, physquantity rb, physquantity ub, physquantity db){
    rescale(lb, rb, ub, db);  tcolor=0;
  }
  phGraphScreen(physquantity lb, physquantity rb, physquantity ub, physquantity db, float initcol){
    rescale(lb, rb, ub, db); tcolor = (initcol>.5)? 0 : 7;
    paintsolid(initcol);
  }
  
  
  void fixraster(physquantity lb, physquantity rb, physquantity ub, physquantity db){
    const phUnit *svxUnit=lcbord.preferredUnit(), *svyUnit=dcbord.preferredUnit();
//    cout << svxUnit->uName << ',';
  //  cout << svyUnit->uName << endl;
    bgsolid(.2); tcolor = 7;
    rescale(lb, rb, ub, db);
    for (std::vector<pntmem>::iterator i = pointmem.begin(); i!=pointmem.end(); ++i){
      i->x = (i->x - lcbord[*svxUnit]) / width[*svxUnit];
      i->y = (i->y - dcbord[*svyUnit]) / height[*svyUnit];
      if (i->x>=0 && i->x<1 && i->y>=0 && i->y<1){
        Pixel[(int) (hres * i->x)][(int) (vres * i->y)] = tcolor;
      }
    }
  }

  const phUnit * xUnit(){
    if (fixedraster)
      return lcbord.preferredUnit();
     else
      return width.preferredUnit();
  }
  const phUnit * yUnit(){
    if (fixedraster)
      return dcbord.preferredUnit();
     else
      return height.preferredUnit();
  }
  void xUnit(const phUnit &thatone){
    fixraster(lcbord(thatone),rcbord(thatone),ucbord,dcbord);
  }
  void yUnit(const phUnit &thatone){
    ucbord[thatone]; dcbord[thatone];}


  void pset(physquantity xc, physquantity yc){
    pntmem newpnt;
    newpnt.lineto = false;
//cout<<xc<<' '<<lcbord<<' '<<xc<<' '<<rcbord <<' '<< yc<<' '<<dcbord <<' '<< yc<<' '<<ucbord;    
    if (fixedraster){
      newpnt.x = ((xc-lcbord)/width).dbl();
      newpnt.y = ((yc-dcbord)/height).dbl();
      if (xc>=lcbord && xc<rcbord && yc>=dcbord && yc<ucbord){
        Pixel[(int) (hres * newpnt.x)][(int) (vres * newpnt.y)] = tcolor;
      }
     }else{
      const phUnit *svxUnit, *svyUnit;
      if (!sureaboutxUnit){
        xc.tryfindUnit();
        svxUnit = xc.preferredUnit();
        if (svxUnit==NULL){cerr << "need unit for diagram x-axis, only got value" << xc << "."; abort();}
        if (!xc.is_identical_zero()) {
          sureaboutxUnit=true;
        }
        if (pointmem.size()>0){
          rcbord.push_upto(xc); lcbord.push_downto(xc);
          rcbord[*svxUnit];     lcbord[*svxUnit];
         }else
          lcbord = rcbord = xc;
       }else{
        svxUnit = lcbord.preferredUnit();   //preserve original borders units, meant to remain
                                           // unchanged since previous points also do!
        if (svxUnit==NULL) { svxUnit = rcbord.preferredUnit(); }
        if (svxUnit==NULL) {cout << "svxUnit==NULL (?\?\?)\n";
          cout << lcbord << endl;}
        rcbord.push_upto(xc); lcbord.push_downto(xc);
        lcbord[*svxUnit];
      }
      if (!sureaboutyUnit){
        yc.tryfindUnit();
        svyUnit = yc.preferredUnit();
        if (svyUnit==NULL){cerr << "need unit for diagram y-axis, only got value" << yc << "."; abort();}
        if (!yc.is_identical_zero()) sureaboutyUnit=true;
        if (pointmem.size()>0){
          ucbord.push_upto(yc); dcbord.push_downto(yc);
          ucbord[*svyUnit];      dcbord[*svyUnit];
         }else
          ucbord = dcbord = yc;
       }else{
        svyUnit = dcbord.preferredUnit();  
        ucbord.push_upto(yc); dcbord.push_downto(yc);
        dcbord[*svyUnit];
      }
      crcSizes();
      newpnt.x = xc[*svxUnit];
      newpnt.y = yc[*svyUnit];
    }
    pointmem.push_back(newpnt);
  }
  void pset(physquantity xc, physquantity yc, float pcol){
    tcolor = setcolor(pcol);
    pset(xc, yc);
  }
  void lineto(physquantity xc, physquantity yc){
    pset(xc, yc);
    if (pointmem.size()>0) pointmem.back().lineto = true;
  }

  void lineto(physquantity xc, physquantity yc, float pcol){
    tcolor = setcolor(pcol);
    pset(xc, yc);
    if (pointmem.size()>0) pointmem.back().lineto = true;
  }

  template<typename itT, typename idpvrptT, typename dptvrptT>
  void drwCurve(itT lItrt, itT rItrt, const idpvrptT &indy, const dptvrptT &pend){
    for (itT Itrt=lItrt; Itrt!=rItrt; ++Itrt)
      lineto(indy(*Itrt), pend(*Itrt));
  }

  template<typename itT, typename idpvrptT, typename dptvrptT>
  void drwCurve(itT lItrt, itT rItrt, const idpvrptT &indy, const dptvrptT &pend, float pcol){
    tcolor = setcolor(pcol);
    drwCurve(lItrt, rItrt, indy, pend);
  }

  template<typename cntT, typename idpvrptT, typename dptvrptT>
  void drwCurve(cntT &msrCnt, const idpvrptT &indy, const dptvrptT &pend){
    drwCurve(msrCnt.begin(), msrCnt.end(), indy, pend);
  }

  template<typename cntT, typename idpvrptT, typename dptvrptT>
  void drwCurve(cntT &msrCnt, const idpvrptT &indy, const dptvrptT &pend, float pcol){
    tcolor = setcolor(pcol);
    drwCurve(msrCnt.begin(), msrCnt.end(), indy, pend);
  }

  template<typename cntT>
  void drwCurve(cntT &msrCnt, float pcol){
    drwCurve(msrCnt.begin(), msrCnt.end(), measureindexpointer(0), measureindexpointer(1), pcol);
  }

  template<typename cntT>
  void drwCurve(cntT &msrCnt){
    drwCurve(msrCnt.begin(), msrCnt.end(), measureindexpointer(0), measureindexpointer(1));
  }
  
  
  

  void pcolor(float pcol){
    tcolor = setcolor(pcol);
  }
  
  float pget(physquantity xc, physquantity yc){
    if (fixedraster){
      if (xc>=lcbord && xc<rcbord && yc>=dcbord && yc<ucbord){
        int x = hres * ((xc-lcbord)/width).dbl(),
            y = vres * ((yc-dcbord)/height).dbl();
        return (Pixel[x][y]/7.);
      }
      return bcolor/7.;
     }else{
      return(0);
    }
  }
  
  void setoutcolor(float ocol){
    bcolor = setcolor (ocol);
  }


/*  void QTeXgrop(string filename){
    union {double dbl; char chr;};
    ofstream outfile(filename.c_str());
    dbl = 
    outfile << 'f';
    
  }*/

  
  friend ostream &operator << (ostream &target, const phGraphScreen &pscr);
  friend class QTeXdiagmaster;
  
  template<typename cntT, typename derefcerT>
  phGraphScreen(const cntT &cnt, const derefcerT &drfx=measureindexpointer(0), const derefcerT &drfy=measureindexpointer(1)){
    sureaboutxUnit = sureaboutyUnit = fixedraster = false;
    drwCurve(cnt.begin(), cnt.end(), drfx, drfy);
  }
};
const char greyscale4bitchars[] = " .'-+|=#";
ostream &operator << (ostream &target, const phGraphScreen &oscr){
  phGraphScreen pscr = oscr;
  if (!pscr.fixedraster){pscr.fixraster(pscr.lcbord,pscr.rcbord,pscr.ucbord,pscr.dcbord);}
  target << endl << '^' << endl << "| " << pscr.ucbord << endl;
  for (int y=pscr.vres-1; y>=0; --y){ 
    target << "|";
    for (int x=0; x<pscr.hres; ++x)
      target << greyscale4bitchars[unsigned(pscr.Pixel[x][y])];
    target << endl;
  }
  target << "| " << pscr.dcbord << endl;
  for (int x=0; x<pscr.hres; ++x)
    target << '-';
  target << '>' << endl << pscr.lcbord;
  for (int x=0; x<3*pscr.hres/4; ++x)
    target << ' ';
  target << pscr.rcbord << endl << endl;
/*  target << "Falls Bild verzerrt, bitte Bildschirmbreite auf min. "
         <<        pscr.hres+1                                    << " Zeichen stellen" << endl
         << "(oder static const int phGraphScreen::hres entsprechend anpassen)";*/
  return target;
}



std::string randomcmpnfilename(std::string ppath = "", const std::string &filenmending = "") {
  if (ppath!="" && ppath[ppath.size()-1]!='/') ppath+="/";
  std::stringstream nms;
  nms << ppath + "cv" << std::hex << std::uppercase << rand()%0x1000000 << filenmending;
  return nms.str();
}

std::string gstr_dirname(std::string path) {
  if (path.size()==0) return path;
  if (path.find('/')==std::string::npos) return "./";
  if (path.rfind('/')==path.size()-1) path.erase(path.size()-1);
  return path.substr(0, path.find_last_of('/'));
}
std::string gstr_basename(std::string path) {
 // cout << "\nFinding basename of: " << path << endl;
  if (path.size()==0) return "";
//  cout << "\nsize is >0: " << path << endl;
  if (path.rfind('/')==path.size()-1) path.erase(path.size()-1);
//  cout << "\nAfter removing trailing '/': " << path << endl;
  if (path.size()==0) return "";
//  cout << "\nsize is >0: " << path << endl;
//  cout << "\nResult to be: " << path.substr(path.find_last_of('/')+1) << endl;
  return path.substr(path.find_last_of('/')+1);
}

std::string safecmpnfilename(const std::string &ppath = "", const std::string &filenmending = "") {
//  static int gflid;
  std::string filename;
  std::ifstream check_existant;
  while (1) {
    check_existant.open( (                       //Try opening file for input,
           filename = randomcmpnfilename(ppath, filenmending)
                         ).c_str() );
    if (check_existant.fail()) break;
    check_existant.close();
//    cout << "File \"" << filename << "\" already exists.\n";
  }
  return filename;                         // return name when file does not exist.
}



     //forward-declarations for objects one might want to insert into diagrammes
template<typename retT>
class phmsq_function_base;
typedef phmsq_function_base<physquantity> phmsq_function;

template <typename arg_T, typename ret_T>
struct compactsupportfn;

template <typename arg_T, typename ret_T>
phGraphScreen rasterize(const compactsupportfn<arg_T, ret_T> &, int raster);


void qdafilecleanup(const std::string& fname) {
  std::string dirname = gstr_dirname(fname);
  std::ifstream qdafile(fname.c_str(), std::ios::binary);
  
  if(!!qdafile) {

    auto read_uint16 = [&]() -> uint16_t {
           uint16_t res; qdafile.read((char*) &res, sizeof(uint16_t)); return res;
         };
    auto read_string = [&]() -> std::string {
           auto strlength = read_uint16();
//           std::cout << "embedded file, name has length " << strlength << std::endl;
           std::vector<char> resbuf(strlength+1, '\0');
           qdafile.read(resbuf.data(), strlength);
           return std::string(resbuf.data());
         };
    auto read_double = [&]() -> double {
           double res; qdafile.read((char*) &res, sizeof(double)); return res;
         };

    while(!qdafile.eof()) {
      char objtypechar = qdafile.get();
      if(!qdafile) break;
      switch(objtypechar) {
       
       case 'g': {
          auto embeddedfile = read_string();
//            std::cout << "found embedded graph file \"" << (dirname + embeddedfile) << "\". Delete.\n";
          remove((dirname + embeddedfile).c_str());
          /*auto clrid =*/read_uint16();
          break;
        }
       case 'c': {
//           std::cout << "found 2d-measure. Ignore.\n"; 
          for(unsigned i=0; i<2; ++i) {
            char pd_objtypechar = qdafile.get();
            if(pd_objtypechar=='f') {
              /*auto vlcpt   =*/read_string();
              /*auto value   =*/read_double();
              /*auto verror  =*/read_double();
              /*auto unitcapt=*/read_string();
            }
          }
          break;
        }
       default: { // apparently, this also reads two measures but disregards caption and unit.
                 //  FIXME
//           std::cout << "found something unexpected (obj-T-ID \'" << objtypechar << "\'). Ignore.\n"; 
          for(unsigned i=0; i<2; ++i) {
            char pd_objtypechar = qdafile.get();
            if(pd_objtypechar=='f') {
              /*auto vlcpt    =*/read_string();
              /*auto value    =*/read_double();
              /*auto verror   =*/read_double();
              /*auto unitcapt =*/read_string();
            }
          }
        }
      }
    }
  
    qdafile.close();
    remove(fname.c_str());
  }
}


class QTeXdiagmaster : QTeXgrofstream {
  physquantity lcbord,rcbord,ucbord,dcbord;
  physquantity width, height;
  std::string myfilename;
  std::string mydirname;
  bool Iamnew;
  unsigned int needbordersets;
  const phUnit *origxUnit, *origyUnit;
  QTeXgrcolor nextcolorcandidate;

  void cleanupandopenfile(const string &nfname) {
    qdafilecleanup(nfname);
    // int donotthrowbutaway = system(("qdafilecleanup " + nfname).c_str());
    // if (donotthrowbutaway);
    QTeXgrofstream::open(nfname);
  }

 public:
  QTeXdiagmaster(const string &nfname)
    : myfilename(nfname)
    , mydirname(gstr_dirname(myfilename))
    , Iamnew(true)
    , needbordersets(2)
    , nextcolorcandidate(QTeXgrcolors::defaultsequence[0]) {
//     std::cout << "New QTeXdiagmaster in file '" << nfname
//               << "' (directory '" << mydirname << "')" << std::endl;
    cleanupandopenfile(nfname);
  }
  
  QTeXdiagmaster &open(const string &nfname){
    cleanupandopenfile(nfname);
    Iamnew = true;
    needbordersets = 2;
    nextcolorcandidate = QTeXgrcolors::defaultsequence[0];
    return *this;
  }

  QTeXdiagmaster &widen_window (const physquantity &xmrngl, const physquantity &xmrngr,
                                const physquantity &ymrngl, const physquantity &ymrngr) {
    if (Iamnew){
      lcbord=xmrngl; rcbord=xmrngr;   ucbord=ymrngr; dcbord=ymrngl;
      width = rcbord - lcbord;
      height = ucbord - dcbord;
      origxUnit = lcbord.tryfindUnit().preferredUnit();
      origyUnit = dcbord.tryfindUnit().preferredUnit();
      assert(origxUnit && origyUnit);
      Iamnew = false;
     }else{
      lcbord.push_downto(xmrngl); rcbord.push_upto(xmrngr);
      width = rcbord - lcbord;
      dcbord.push_downto(ymrngl); ucbord.push_upto(ymrngr);
      height = ucbord - dcbord;
    }
    return *this;
  }
  QTeXdiagmaster &widen_window (const phq_interval &xmrng, const phq_interval &ymrng) {
    if (Iamnew){
      lcbord=xmrng.l(); rcbord=xmrng.r();   ucbord=ymrng.r(); dcbord=ymrng.l();
      width=xmrng.width(); height=ymrng.width();
      origxUnit = lcbord.tryfindUnit().preferredUnit();
      origyUnit = dcbord.tryfindUnit().preferredUnit();
      assert(origxUnit && origyUnit);
      Iamnew = false;
     }else{
      lcbord.push_downto(xmrng.l()); rcbord.push_upto(xmrng.r());
      width = rcbord - lcbord;
      dcbord.push_downto(ymrng.l()); ucbord.push_upto(ymrng.r());
      height = ucbord - dcbord;
    }
    return *this;
  }
  
  QTeXdiagmaster &link_to_graph_file(string nfname, const QTeXgrcolor &cl) {
    cwrite('g');
    if(gstr_dirname(nfname)==mydirname) nfname=gstr_basename(nfname);
    iwrite(nfname.length());
    *this << nfname;

    colorwrite(cl);
    return *this;
  }


 private:
   QTeXdiagmaster &colorwrite(QTeXgrcolor color) {
    if (color.is_undefined()) color = nextcolorcandidate;
    iwrite(color.QTeXgrofstreamid());
    nextcolorcandidate = QTeXgrcolors::nextbest(color);
    return *this;
  }
  void make_safe_qcvfilename(std::string &nfname) const{
    if(nfname=="") nfname = safecmpnfilename(mydirname, ".qcv");
    if(nfname.substr(0,2) == "./")
      nfname = nfname.substr(2);
  }
 public:

  QTeXdiagmaster &nextcolor(QTeXgrcolor color) {
    nextcolorcandidate = color;
    return *this;
  }

  QTeXdiagmaster &insertMeasure(const physquantity &xval, const physquantity &yval){
    if (Iamnew){
      if (xval.is_identical_zero() || yval.is_identical_zero())
        return *this;       // to prevent the diagram to scale itself to degenerate measures do not include zeroes at the beginning. It would be better to store those somewhere seperately and write them before closing the stream.

      lcbord=xval-xval.error(); rcbord=xval+xval.error();
      ucbord=yval+yval.error(); dcbord=yval-yval.error();

      origxUnit = physquantity(xval).tryfindUnit().preferredUnit();
      if (origxUnit==NULL) {
        cerr << "Need some kind of unit to create diagram, but couldn't find any suitable for" << xval
             << ".\nSpecify one manually or declare a units scope containing it as default.\n"; abort(); }
      origyUnit = physquantity(yval).tryfindUnit().preferredUnit();
      if (origyUnit==NULL) {
        cerr << "Need some kind of unit to create diagram, but couldn't find any suitable for" << yval
             << ".\nSpecify one manually or declare a units scope containing it as default.\n"; abort(); }

      Iamnew = false;

     }else{
      lcbord.push_downto(xval); rcbord.push_upto(xval);
      dcbord.push_downto(yval); ucbord.push_upto(yval);
    }
    width = rcbord - lcbord;
    height = ucbord - dcbord;

    cwrite('c');

//    cout << yval;
//    try {
      *this << xval(*origxUnit) << yval(*origyUnit);
//    } catch(...) { cout << "Problem with (" << xval << ", " << yval << ")\n"; }
    
    needbordersets -= (needbordersets>0);
    
    return(*this);
  }

  QTeXdiagmaster &insertMeasureseq(const measureseq &vals,
       const msq_dereferencer &derefx = measureindexpointer(0), const msq_dereferencer &derefy = measureindexpointer(1)){
    for (measureseq::const_iterator i = vals.begin(); i!=vals.end(); ++i){
      insertMeasure(derefx(*i), derefy(*i));
    }
    return *this;
  }
  

  QTeXdiagmaster &insertCurve( const phGraphScreen &src
                             , const QTeXgrcolor &color=QTeXgrcolors::undefined
                             , string nfname=""
                             ) {
    make_safe_qcvfilename(nfname);
    QTeXgrofstream crvfile(nfname);
    if (!crvfile) {cout<<"Bad filename \"" << nfname << "\""; abort();}
    
    widen_window (src.lcbord, src.rcbord, src.dcbord, src.ucbord);

    for (std::vector<phGraphScreen::pntmem>::const_iterator i=src.pointmem.begin(); i!=src.pointmem.end(); ++i){
      if (i->lineto)
        crvfile.cwrite('l');
       else
        crvfile.cwrite('p');

      if (src.fixedraster){
        crvfile.dwrite(((i->x * src.width) + src.lcbord)[*origxUnit]);
        crvfile.dwrite(((i->y * src.height) + src.dcbord)[*origyUnit]);
       }else{
        if (src.lcbord.preferredUnit() == origxUnit){
          crvfile.dwrite(i->x);
         }else{
          crvfile.dwrite(i->x * (*src.lcbord.preferredUnit() / *origxUnit).dbl());
        }
        if (src.dcbord.preferredUnit() == origyUnit){
          crvfile.dwrite(i->y);
         }else{
          crvfile.dwrite(i->y *(*src.dcbord.preferredUnit() / *origyUnit).dbl());
        }
      }
      crvfile.cwrite(' ');
    }
             
    crvfile.finish();
    
    link_to_graph_file(nfname, color);
        
    return(*this);
  }
  QTeXdiagmaster &insertUncertainCurve( const measureseq &src
                                      , const msq_dereferencer &drx=measureindexpointer(0)
                                      , const msq_dereferencer &dry=measureindexpointer(1)
                                      , const QTeXgrcolor &color=QTeXgrcolors::undefined
                                      , string nfname=""
                                      , bool hide_uncertainty=false
                                      ) {
    make_safe_qcvfilename(nfname);
    QTeXgrofstream crvfile(nfname);
    if (!crvfile) {cout<<"Bad filename \"" << nfname << "\""; abort();}
    
    struct schedstack{physquantity x,y; bool lineto; schedstack *last;}schedule;
    
    for ( int outline_errmargins=0;
          outline_errmargins<(hide_uncertainty? 1 : 2);
          ++outline_errmargins ) {
      auto p = src.begin(); if (p==src.end()) return *this;
      if (!outline_errmargins)
        schedule = schedstack{drx(*p), dry(*p), false, NULL};

      for (++p; p!=src.end(); ++p) {
        physquantity fnval = dry(*p);
        if (outline_errmargins) {
          fnval += fnval.error() * ((rand()%2)*2-1);
          schedule = schedstack{drx(*p), fnval, false, new schedstack(schedule)};
         }else{
          if (fnval.couldbe_anything()){
            schedule.lineto=false;
           }else{
            schedule = schedstack{drx(*p), fnval, true, new schedstack(schedule)};
          }
        }
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
             
    crvfile.finish();
    
    link_to_graph_file(nfname, color);
        
    return(*this);
  }
  
  QTeXdiagmaster &insertCurve(const measureseq &src,
                              const msq_dereferencer &drx=measureindexpointer(0),
                              const msq_dereferencer &dry=measureindexpointer(1),
                              const QTeXgrcolor &color=QTeXgrcolors::undefined,
                              const string &nfname=""){
    if (src.size() > 3) {
      if (Iamnew)
        return insertCurve(phGraphScreen(src, drx, dry), color, nfname);
       else
        return insertUncertainCurve(src,drx,dry,color,nfname,true);
     }else{
      ERRMSGSTREAM << "Insufficient data to plot graph: only " << src.size() << " measures in sequence.\n";
      abort();
    }
  }

  QTeXdiagmaster &insertConstant(const physquantity &src, const string &nfname, QTeXgrcolor color=QTeXgrcolors::undefined){
    QTeXgrofstream crvfile(nfname);
    if (!crvfile) {ERRMSGSTREAM<<"Bad filename \"" << nfname << "\""; abort();}
    
    if (!Iamnew){
      dcbord.push_downto(src); ucbord.push_upto(src);
      height = ucbord - dcbord;
     }else{
      origyUnit = src.preferredUnit();
    }
    
    crvfile.cwrite('h');
    crvfile << src(*origyUnit);
             
    crvfile.finish();
    
    link_to_graph_file(nfname, color);
        
    return(*this);
  }

  QTeXdiagmaster &insertline(const std::pair<physquantity,physquantity> &srcst, const std::pair<physquantity,physquantity> &srcfn, const QTeXgrcolor &color=QTeXgrcolors::undefined, string nfname=""){
    make_safe_qcvfilename(nfname);
    QTeXgrofstream crvfile(nfname);
    if (!crvfile) {ERRMSGSTREAM<<"Bad filename \"" << nfname << "\""; abort();}
    
    if (!Iamnew){
      dcbord.push_downto(srcst.second); ucbord.push_upto(srcst.second);
      dcbord.push_downto(srcfn.second); ucbord.push_upto(srcfn.second);
      lcbord.push_downto(srcst.first); rcbord.push_upto(srcst.first);
      lcbord.push_downto(srcfn.first); rcbord.push_upto(srcfn.first);

      height = ucbord - dcbord;
     }else{
       origxUnit = srcst.first.preferredUnit();
      origyUnit = srcst.second.preferredUnit();
    }
    
    crvfile.cwrite('p');
    crvfile.dwrite(srcst.first[*origxUnit]);
    crvfile.dwrite(srcst.second[*origyUnit]);
    crvfile.cwrite('l');
    crvfile.dwrite(srcfn.first[*origxUnit]);
    crvfile.dwrite(srcfn.second[*origyUnit]);
             
    crvfile.finish();
    
    cwrite('g');
    if(gstr_dirname(nfname)==mydirname) nfname=gstr_basename(nfname);
    iwrite(nfname.length());
//    cout << 'P' << lcbord << dcbord << endl;
    *this << nfname;
//    cout << 'G' << lcbord << dcbord << endl;
//    strwrite(nfname);
    colorwrite(color);
        
    return(*this);
  }
  
  
  static const int default_resolution_for_histogram_envelopes=2048;
  static const int default_histogram_max_clutternum=4096;
  template<typename skeyT>
  QTeXdiagmaster &insertHistogram(const histogram<skeyT> &H,
                                  int max_clutternum = default_histogram_max_clutternum,
                                  int env_resolution = -1);
  QTeXdiagmaster &insertHistogram(
                  const measureseq &m, const msq_dereferencer &d=measureindexpointer(0),
                  int max_clutternum = default_histogram_max_clutternum, int env_resolution = -1  );
  QTeXdiagmaster &insertHistogram(
                  const measureseq &m, const string &d,
                  int max_clutternum = default_histogram_max_clutternum, int env_resolution = -1  ){
    return insertHistogram(m, captfinder(d), max_clutternum, env_resolution);
  };
  
  
  static const int default_resolution_for_phfunction_plots=2048;
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, phq_interval rng, const measure &consts=measure(), int res=-1, const QTeXgrcolor &cl=QTeXgrcolors::undefined, std::string nfname="");
     // permutation overloads. This is perhaps not very nice practise.
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, const measure &consts, const QTeXgrcolor &cl, int res=-1, const std::string &nfname="")
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, int res, const measure &consts=measure(), const QTeXgrcolor &cl=QTeXgrcolors::undefined, const std::string &nfname="")
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const measure &consts, const phq_interval &rng, int res=-1, const QTeXgrcolor &cl=QTeXgrcolors::undefined, const std::string &nfname="")
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const measure &consts, const phq_interval &rng, const QTeXgrcolor &cl, int res=-1, const std::string &nfname="")
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, int res, const QTeXgrcolor &cl, const measure &consts=measure(), const std::string &nfname="")
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, int res, const QTeXgrcolor &cl, const std::string &nfname, const measure &consts=measure())
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, int res, const std::string &nfname, const measure &consts=measure(), const QTeXgrcolor &cl=QTeXgrcolors::undefined)
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, int res, const std::string &nfname, const QTeXgrcolor &cl, const measure &consts=measure())
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, const QTeXgrcolor &cl, const measure &consts=measure(), int res=-1, const std::string &nfname="")
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, const QTeXgrcolor &cl, int res, const measure &consts=measure(), const std::string &nfname="")
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, const QTeXgrcolor &cl, int res, const std::string &nfname, const measure &consts=measure())
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, const QTeXgrcolor &cl, const std::string &nfname, const measure &consts=measure(), int res=-1)
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, const QTeXgrcolor &cl, const std::string &nfname, int res, const measure &consts=measure())
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, const std::string &nfname, const measure &consts=measure(), int res=-1, const QTeXgrcolor &cl=QTeXgrcolors::undefined)
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, const std::string &nfname, int res, const measure &consts=measure(), const QTeXgrcolor &cl=QTeXgrcolors::undefined)
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, const std::string &nfname, int res, const QTeXgrcolor &cl, const measure &consts=measure())
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, const std::string &nfname, const QTeXgrcolor &cl, const measure &consts=measure(), int res=-1)
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, const std::string &nfname, const QTeXgrcolor &cl, int res, const measure &consts=measure())
    { return plot_phmsq_function(f, rng, consts, res, cl, nfname); }

  QTeXdiagmaster &plot_phmsq_function(const compactsupportfn<physquantity, physquantity> &f, const QTeXgrcolor &cl=QTeXgrcolors::undefined);/*{
    wrapped_compactsupportfn<physquantity> wfn(f);
    plot_phmsq_function(wfn, f.support(), wfn.example_parameterset[0].cptstr(), cl);
    return *this;
  }*/

  template<class Lambda>
  QTeXdiagmaster &plot_lambda(Lambda f, phq_interval rng, int res=-1, const QTeXgrcolor &cl=QTeXgrcolors::undefined) {
                            //Lambda must be a (physquantity->physquantity)
    std::string nfname;
    make_safe_qcvfilename(nfname);
    QTeXgrofstream crvfile(nfname);
    if (!crvfile) {cout<<"Bad filename \"" << nfname << "\""; abort();}

    if (rng.width()==0) return *this;     // Nothing to plot on a vanishing interval!

    if (res<0) res = default_resolution_for_phfunction_plots;
    physquantity plotstep = rng.width()/res;
    if (rng.l().is_identical_zero()) rng.l() = rng.width()*0;  //pleonastic?..   (this simply ensures that the left border has a proper physical unit, despite being 0)
    
    phq_interval yrng( f(rng.l()), f(rng.r()) );

    
    for (physquantity x = rng.randompoint(); yrng.width()==0; x=rng.randompoint()) {
                                    //search the interval for a function value that
      yrng.widen_to_include(f(x)); // does not equal the one at the borders,
    }                             //  so as to have an estimate over the area the
    widen_window(rng, yrng);     //   graph will need for display


    physquantity x = rng.l();
    crvfile.cwrite('p');
    crvfile.dpwrite(x[*origxUnit], rng.l()[*origyUnit]);
    crvfile.cwrite(' ');
    for (x += plotstep; x<rng.r(); x+=plotstep) {
      physquantity fnval = f(x);
      yrng.widen_to_include(fnval);
      crvfile.cwrite('l');
      crvfile.dpwrite(x[*origxUnit], fnval[*origyUnit]);
      crvfile.cwrite(' ');
    }
    widen_window(rng, yrng);
     
    link_to_graph_file(nfname, cl);
      
    return *this;
    
    
  }

#if 0
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const phq_interval &rng, int res, const string &nfname=safecmpnfilename(".qcv")) {
    return plot_phmsq_function(f, rng, res);
  }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const physquantity &rng, const measure &consts=measure(), int res=default_resolution_for_phfunction_plots) {
    return plot_phmsq_function(f, phq_interval(rng.l_errorbord(), rng.u_errorbord()), consts, res);
  }
  QTeXdiagmaster &plot_phmsq_function(const phmsq_function &f, const physquantity &rng, int res) {
    return plot_phmsq_function(f, phq_interval(rng.l_errorbord(), rng.u_errorbord()), measure(), res);
  }
#endif
  
   
  QTeXdiagmaster &finish(){
    if (Iamnew){
      cwrite('F');
   //   cerr << "Insert something before you close graph stream! Deleting file.\n";
      close();
  //    int donotthrowbutaway = system(("rm " + myfilename).c_str());
    //  if (donotthrowbutaway);
     }else if (needbordersets){
      cwrite('c');
      *this << lcbord(*origxUnit) << dcbord(*origyUnit);
      cwrite('c');
      *this << rcbord(*origxUnit) << ucbord(*origyUnit);
      close();
     }else{
      close();
    }
    myfilename="";
    return *this;
  }
  
  ~QTeXdiagmaster() { try { finish(); } catch(...) {} }
};










class phqdata_binstorage {
  mutable std::fstream ulstream;
  std::string storefilename;
  std::ios::openmode md;
  const unitsscope* const refunitsc;
 public:
  bool fail()const {return ulstream.fail();} bool bad()const {return ulstream.bad();}
  bool good()const {return ulstream.good();} bool eof()const {return ulstream.eof();}
  std::fstream::streampos tellg()const {return ulstream.tellg();}
  phqdata_binstorage& flush(){ulstream.flush(); return *this;}
  bool operator!() {return !ulstream;}
  std::string file_getptrinfo()const{
    std::stringstream o; 
    o << '"' << storefilename <<  "\" w/ stream flags ";
    if (good()) { o << "good"; if (bad() || fail() || eof()) o << "|";}
    if (bad()) { o << "bad"; if (fail() || eof()) o << "|";}
    if (fail()) { o << "bad"; if (eof()) o << "|";}
    if (eof()) { o << "eof";}
    o << " at offset 0x" << std::hex<<tellg();
    return o.str();
  }
  void open(const std::string &stfn, std::ios::openmode nmd) {
    md = nmd;
    ulstream.open(stfn.c_str(), md | std::ios::binary);
    if(fail() || bad()) return;
       //For the time being precede all phqdump files with a 'P', signifying
      // a plainly oulied collection of physics-data objects.
    if (md&std::ios::out && (md&std::ios::trunc || !(md&std::ios::app || md&std::ios::app)))
      ulstream.put('P');
     else if (md == std::ios::in)
      assert (ulstream.get() == 'P');

    if (good())
      storefilename = stfn;
  }
  void close() {
    md = std::ios::in & std::ios::out;  // =0
    ulstream.close();
    storefilename = "";
  }

  phqdata_binstorage(): ulstream(), storefilename(""), md(std::ios::in & std::ios::out)
                      , refunitsc(&all_known_units) {}
  phqdata_binstorage( const std::string &stfn
                    , std::ios::openmode md
                    , const unitsscope* refunitsc=&all_known_units)
    : md(md)
    , refunitsc(refunitsc) {
    open(stfn, md);
  }
  ~phqdata_binstorage() {
    ulstream.close();
  }
 
 private:
  void assert_readaccess() const{
    if (storefilename.size()>0) {
      if (md & std::ios::in) return;
      cerr << "Tried to perform a read operation in file \"" << storefilename << "\", which was not opened in a read mode.\n";
      abort();
    }
    cerr << "Tried to perform a read operation in a stream which was not associated to any file.\n";
    abort();
  }
  void assert_writeaccess() {
    if (storefilename.size()>0) {
      if (md & std::ios::out) return;
      cerr << "Tried to perform a write operation in file \"" << storefilename << "\", which was not opened in a write mode.\n";
      abort();
    }
    cerr << "Tried to perform a write operation in a stream which was not associated to any file.\n";
    abort();
  }

   // the following functions do not perform any checks. Intended for local/auxiliary use only.

  std::string read_string() {                //strings are stored as an integer
    /*assert_readaccess();*/                // length-indicator followed by a
    uint32_t strlen;                       //  char buffer of that length.
    ulstream.read((char*)(&strlen), 4);
    char* cresult = (char*)malloc(strlen+4);
    ulstream.read(cresult, strlen);
    std::string result(cresult, strlen);
    free(cresult);
    return result;
  }
  phqdata_binstorage &write_string(const std::string &str) {
    /*assert_writeaccess();*/
    const char* buf=str.c_str();
    uint32_t strlen=str.size();
    ulstream.write((char*)(&strlen), 4);
    ulstream.write(buf, strlen);
    return *this;
  }

  uint32_t read_uint32_littleendian() {
    /*assert_readaccess();*/
    uint32_t result;
    ulstream.read((char*)(&result), 4);
    return result;
  }
  phqdata_binstorage &write_uint32_littleendian(uint32_t wq) {
    /*assert_writeaccess();*/
    ulstream.write((char*)(&wq), 4);
    return *this;
  }

  int32_t read_int32_littleendian() {
    /*assert_readaccess();*/
    int32_t result;
    ulstream.read((char*)(&result), 4);
    return result;
  }
  phqdata_binstorage &write_int32_littleendian(int32_t wq) {
    /*assert_writeaccess();*/
    ulstream.write((char*)(&wq), 4);
    return *this;
  }

  phq_underlying_float read_phqfloat() {
    /*assert_readaccess();*/
    phq_underlying_float result;
    ulstream.read((char*)(&result), sizeof(phq_underlying_float));
    return result;
  }
  phqdata_binstorage &write_phqfloat(phq_underlying_float wq) {
    /*assert_writeaccess();*/
    ulstream.write((char*)(&wq), sizeof(phq_underlying_float));
    return *this;
  }
  
  int _get() { /*assert_readaccess();*/ return ulstream.get(); }
  phqdata_binstorage &_put(char c) { /*assert_writeaccess();*/ ulstream.put(c); return *this; }

  phqdata_binstorage &_put(physquantity wpq) {
    /*assert_writeaccess();*/
    _put('S'); //  Announce explicitly outwritten string caption.
    write_string(wpq.cptstr());

    if (wpq.tunit!=nullptr && !stdPhysUnitsandConsts::stdUnits.has(wpq.tunit))
      wpq.freefromunitsys();

    wpq.tryfindUnit(&stdPhysUnitsandConsts::stdUnits);
    
    phq_underlying_float val, uncrt;
    if(wpq.tunit!=nullptr) { val=wpq.valintu;  uncrt=wpq.myError.valintu;  }
     else                  { val=wpq.valincgs; uncrt=wpq.myError.valincgs; }
    write_phqfloat(val); write_phqfloat(uncrt);

    if(wpq.tunit!=nullptr) {
      _put('S');
      const phUnit* unit = wpq.tunit;
      write_string(unit->uName);
     }else{
      _put('B');
      _put('v');
      _put('c');
      _put('X');
      write_int32_littleendian(wpq.myDim.cint);
      write_int32_littleendian(wpq.myDim.gint);
      write_int32_littleendian(wpq.myDim.sint);
    }
    return *this;
  }
  phqdata_binstorage &_put(const measure &wpm) {
    /*assert_writeaccess();*/
    _put('E'); //  Announce expandedly outwritten measure.
    uint32_t mslen=wpm.size();               //  size of the measure.
    ulstream.write((char*)(&mslen), 4);

    for (unsigned i=0; i<mslen; ++i) {
      _put(wpm[i]);
    }
    
    return flush();
  }
  phqdata_binstorage &_put(const measureseq &wps) {
    /*assert_writeaccess();*/
    _put('E'); //  Announce expandedly outwritten sequence.
    uint32_t sqlen = wps.size();               //  size of the measure.
    ulstream.write((char*)(&sqlen), 4);

    for (unsigned i=0; i<sqlen; ++i) {
      _put(wps[i]);
    }
    return *this;
  }


 public:
  
  class genDataChunkReader;
  friend class genDataChunkReader;
  genDataChunkReader get();

  enum datachunkKind {
    physquantityChunkT,
    measureChunkT,
    measureseqChunkT,
    invalidChunkT
  };
  class genDataChunkReader {
    friend class phqdata_binstorage;
    union { physquantity *p; measure *m; measureseq *s; };
    datachunkKind chK;
    const std::string *storefnm;
   public:
    const physquantity &from_phq() const;
    const measure &from_measure() const;
    const measureseq &from_msq() const;
    
    void construct_physquantity(phqdata_binstorage &src) {
      unsigned char c = src._get();
      assert(c=='S'); // one byte determining the type of caption information.
                     //  For now assume it is an explicitly outwritten string.
      std::string capt=src.read_string();
      phq_underlying_float val, uncrt;
      val = src.read_phqfloat(); uncrt = src.read_phqfloat();
      
      c = src._get(); switch(c) {  // one byte determining the type of unit information.
       case 'S': {  // explicit name of a standard unit.
        string desired_unit = src.read_string();
        const phUnit* unit = findUnit(desired_unit, src.refunitsc);
        if (unit==NULL) {
          if (desired_unit.length()==0)
            unit = &real1;
           else {
            cerr << desired_unit.length();
            cerr << "Unrecognized unit \"" << desired_unit
                 << "\" in file \"" << *storefnm << "\".";
            abort();
          }
        }
        p = new physquantity(*unit);
       }break;
       case 'B': {  // unit decomposed to some basis (e.g. cgs)
        string unitname="";
        c = src._get();
#if 1
        assert(c=='v');
#else
        switch(c){case 'S':unitname = src.read_string();const phUnit = newUnit (unitname; )break;default:unitname = "";}
#endif
        c = src._get(); switch(c) {
         case 'c': {       // cgs
          phDimension dim;
          c = src._get(); switch(c) {
           case 'X':
            dim.cint = src.read_int32_littleendian();
            dim.gint = src.read_int32_littleendian();
            dim.sint = src.read_int32_littleendian();
            break;
           default:
            cerr << "Unrecognized unit base-representation option '" << c << "' (0x" << std::hex << (int)c << ") in file " << *storefnm << "\".";
          }
          if(unitname!=""){}
          
          p = new physquantity(dim);
         }break;
         default:
          cerr << "Unrecognized unit basis identifier '" << c << "' (0x" << std::hex << (int)c << ") in file " << *storefnm << "\".";
        }
        break;
       }
       default:
        cerr << "Unrecognized option '" << c << "' (0x" << std::hex << unsigned(c)
             << ") in file " << src.file_getptrinfo() << endl;
        abort();
      }

      if(p->tunit!=nullptr) { p->valintu = val;  p->myError.valintu = uncrt;  }
       else                 { p->valincgs = val; p->myError.valincgs = uncrt; }
      
      p->label(capt);
    }
    void construct_measure(phqdata_binstorage &src) {
      assert(char(src._get())=='E'); // For now assume explicitly outwritten meaure.
      uint32_t mslen;          //  size of the measure.
      src.ulstream.read((char*)(&mslen), 4);
      
      m = new measure(mslen);
      for (unsigned i=0; i<mslen; ++i)
        (*m)[i] = genDataChunkReader(physquantityChunkT, src).from_phq();
    }
    void construct_measureseq(phqdata_binstorage &src) {
      assert(char(src._get())=='E'); // For now assume explicitly outwritten meaure.
      uint32_t sqlen;          //  size of the measure.
      src.ulstream.read((char*)(&sqlen), 4);
      
      s = new measureseq(sqlen);
      for (unsigned i=0; i<sqlen; ++i)
        (*s)[i] = genDataChunkReader(measureChunkT, src).from_measure();
    }

    void construct(phqdata_binstorage &src) {
      switch (chK) {
        case physquantityChunkT: construct_physquantity(src); break;
        case measureChunkT: construct_measure(src); break;
        case measureseqChunkT: construct_measureseq(src); break;
        default: cerr << "Trying to construct a genDataChunkReader from an invalid data chunk.\n";
                 abort();
      }
    }
    genDataChunkReader(datachunkKind k, phqdata_binstorage &src)
      : chK(k)
      , storefnm(&src.storefilename) {
      construct(src);
    }
    explicit genDataChunkReader(phqdata_binstorage &src)
      : storefnm(&src.storefilename) {
      if (!src) {chK = invalidChunkT; return;}
      chK = src.peek_typeof_nextchunk(); src.ulstream.get();
      if (!src) {chK = invalidChunkT; return;}
      construct(src);
    }
    
   public:
    genDataChunkReader(): chK(invalidChunkT), storefnm(NULL) {}
    genDataChunkReader(const genDataChunkReader &cpfr)
      : p(cpfr.p)
      , chK(cpfr.chK)
      , storefnm(cpfr.storefnm)
    {}
    operator physquantity() const{ return from_phq(); }
    operator measure() const{ return from_measure(); }
    operator measureseq() const{ return from_msq(); }
    friend std::ostream &operator<<(std::ostream &, const genDataChunkReader &);
#if 0
    genDataChunk(const physquantity &np)
      : chK(physquantityChunkT)
      , p(new physquantity(np))
    {}
    genDataChunk(const measure &nm)
      : chK(measureChunkT)
      , m(new measure(np))
    {}
    genDataChunk(const measureseq &ns)
      : chK(measureseqChunkT)
      , m(new measureseq(ns))
    {}
#endif
    datachunkKind type_of() const {return chK;}


    bool operator!() { return chK==invalidChunkT; }


    ~genDataChunkReader() {
      switch (chK) {
        case physquantityChunkT: delete p; break;
        case measureChunkT: delete m; break;
        case measureseqChunkT: delete s; break;
        default: break;
      }
    }
  };

  datachunkKind peek_typeof_nextchunk() const {
    assert_readaccess();
    switch (char(ulstream.peek())) {
      case 'p': return physquantityChunkT;
      case 'm': return measureChunkT;
      case 's': return measureseqChunkT;
      default: return invalidChunkT;
    }
  }

  phqdata_binstorage &put(const physquantity &wpq) {
    assert_writeaccess();
    _put('p');
    _put(wpq);
    return *this;
  }
  phqdata_binstorage &put(const measure &wpm) {
    assert_writeaccess();
    _put('m');
    _put(wpm);
    return *this;
  }
  phqdata_binstorage &put(const measureseq &wps) {
    assert_writeaccess();
    _put('s');
    _put(wps);
    return *this;
  }

  phqdata_binstorage& operator<<(const physquantity &wpq) {
    assert_writeaccess();
    _put('p');
    _put(wpq);
    return *this;
  }
  phqdata_binstorage& operator<<(const measure &wpm) {
    assert_writeaccess();
    _put('m');
    _put(wpm);
    return *this;
  }
  phqdata_binstorage& operator<<(const measureseq &wps) {
    assert_writeaccess();
    _put('s');
    _put(wps);
    return *this;
  }

};

const physquantity &phqdata_binstorage::genDataChunkReader::from_phq() const {
  if(chK==physquantityChunkT) return *p;
   else if (storefnm!=NULL){
    cerr << "Requesting a physquantity from file \""<<*storefnm<<"\" where none was found.";
    abort();
   }else{
    cerr << "Requesting a physquantity from a stream not associated with any file.";
  }
  abort();
}
const measure &phqdata_binstorage::genDataChunkReader::from_measure() const {
  if(chK==measureChunkT) return *m;
   else if (storefnm!=NULL) {
    cerr << "Requesting a measure from file \""<<*storefnm<<"\" where none was found.";
    abort();
   }else{
    cerr << "Requesting a measure from a stream not associated with any file.";
  }
  abort();
}
const measureseq &phqdata_binstorage::genDataChunkReader::from_msq() const {
  if(chK==measureseqChunkT) return *s;
   else if (storefnm!=NULL) {
    cerr << "Requesting a measureseq from file \""<<*storefnm<<"\" where none was found.";
   }else{
    cerr << "Requesting a measureseq from a stream not associated with any file.";
  }
  abort();
}
phqdata_binstorage::genDataChunkReader phqdata_binstorage::get(){
  assert_readaccess();
  return genDataChunkReader(*this);
}

std::ostream &operator<<(std::ostream &tgt, const phqdata_binstorage::genDataChunkReader &r) {
  switch (r.chK) {
    case phqdata_binstorage::invalidChunkT:
      if (r.storefnm!=NULL)
        tgt << "INVALID phq-DATACHUNK FROM FILE \"" << *r.storefnm << "\"\n";
       else 
        tgt << "NONEXISTANT phq-DATACHUNK\n";
      break;
    case phqdata_binstorage::physquantityChunkT:
      tgt << *r.p; break;
    case phqdata_binstorage::measureChunkT:
      tgt << *r.m; break;
    case phqdata_binstorage::measureseqChunkT:
      tgt << *r.s; break;
  }
  return tgt;
}


namespace_cqtxnamespace_CLOSE