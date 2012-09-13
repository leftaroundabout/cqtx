  // Copyright 2012 Justus Sagemüller.

   //This program is free software: you can redistribute it and/or modify
  // it under the terms of the GNU General Public License as published by
 //  the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
   //This program is distributed in the hope that it will be useful,
  // but WITHOUT ANY WARRANTY; without even the implied warranty of
 //  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
  // You should have received a copy of the GNU General Public License
 //  along with this program.  If not, see <http://www.gnu.org/licenses/>.


#include "../cqtx.h"
#include <iostream>

int main(int argc, char*argv[]){
  try{


  defaultUnitsc = &stdPhysUnits;
  

  fittable_gaussianfn mygauss;
  mygauss.rename_var("x_0",     "t_0")
         .rename_var("A",       "\\rho_0")
//         .var("\\sigma", "\\sigma")
         .rename_var("x",       "t");
  
  unsigned npeaks = 1;
//  cout << "Arguments given: " << argc << endl;
  if (argc == 2) {
    cout << "Requested number of peaks: " << argv[1] << endl;
    npeaks = argv[1][0] - '0';
  }
  
  measureseq ps(npeaks);
  
  QTeXdiagmaster outgr("somegaussian.qda");

  tr1::variate_generator<tr1::mt19937, tr1::normal_distribution<> >
         cdice (std::tr1::mt19937(time(NULL)), std::tr1::normal_distribution<>(0, 1));
  
  physquantity noiselvl = 1./8 * volts, noisenormale = (37*millivolts) (volts),
               peaksheightsortof = 2*volts;
  
  tr1::variate_generator<tr1::mt19937, tr1::normal_distribution<> >
         noise (std::tr1::mt19937(time(NULL)), std::tr1::normal_distribution<>(0, (noiselvl/noisenormale).dbl()));
  
  int c=3;
  measureseq fnplot(0), fnpplot(0), frr, fsum;
  
  physquantity biggestt=0, smallestt=0, smallestrho = 15*tonnes;

//  while (c < 4) {
    for (measureseq::access p = ps.begin(); p!=ps.end(); ++p) {
      p["\\sigma"] = abs((cdice() + 1.6) * .4*seconds);
      p["t_0"] = cdice() * npeaks*seconds;
      p["t_{0p}"] = p["t_0"] + cdice() * p["\\sigma"];
      p["\\rho_0"] = abs(cdice() * peaksheightsortof/2) + abs(cdice() * peaksheightsortof);
      biggestt.push_upto(p["t_0"]);
      smallestt.push_downto(p["t_0"]);
    }
    cout << "Given:\n" << ps << endl;
    fnplot.clear();
    measureseq::access p = ps.begin();
    smallestrho = noiselvl * p["\\rho_0"];
    smallestt -= 13*p["\\sigma"];
    biggestt += 13*p["\\sigma"];
    
    physquantity rastert = (20*milliseconds) (seconds);
    for (p["t"] = smallestt; p["t"]<biggestt; p["t"]+=rastert){
      p["\\rho"] = 0;
      p["\\rho_\\text{noisefree}"] = 0;
      for (measureseq::access q = p; q!=ps.end(); ++q) {
        q["t"] = p["t"];
        p["\\rho"] += q["\\rho_0"]/(1+((q["t"]-q["t_0"])/q["\\sigma"]).squared().dbl()) + noise()*noisenormale;
        p["\\rho_\\text{noisefree}"] += q["\\rho_0"]/(1+((q["t"]-q["t_0"])/q["\\sigma"]).squared().dbl());
//        p["\\rho"] += mygauss(*q) + noise()*noisenormale;
      }
    //  p["\\rho"].push_upto(0);
      fnplot.push_back(measure(p["\\rho"], p["t"]));
      fnpplot.push_back(measure(p["\\rho_\\text{noisefree}"], p["t"]));
    }
    smallestt -= 1*p["\\sigma"];
    biggestt += 1*p["\\sigma"];
    outgr.insertCurve(fnplot, captfinder("t"), captfinder("\\rho"), /*"gaussianssum.qcv",*/ QTeXgrcolors::blue);
    outgr.insertCurve(fnpplot, captfinder("t"), captfinder("\\rho_\\text{noisefree}"), QTeXgrcolors::i_red);
    outgr.nextcolor(QTeXgrcolors::green);
//  cout << "The correct values:\n" << p << endl; 

    fittable_multigaussianfn fgfn(npeaks);  
                                       fgfn//.rename_var("x",     "t")
                                           .rename_var("x, x_i, A_i",   "t, t_i, \\rho_i", LaTeXindex("i").from(0).unto(npeaks))
                                           //.rename_var("A_i",   "\\rho_i", LaTeXindex("i").from(0).unto(npeaks))
                                           ;
    captfinder rhofind("\\rho");
    fitdist_fntomeasures d(&fnplot, &fgfn, &rhofind);
    cout << "t_i in [" << smallestt << ", " << biggestt << "]\n";
    evolution_minimizer fitthisback(d,
                                    enforce_each_difference_smaller_than( "t_i" | LaTeXindex("i").from(0).unto(npeaks),
                                                                  "\\sigma_i" | LaTeXindex("i").from(0).unto(npeaks),
                                                                  maxval(fnplot, captfinder("t")) )
                                 && enforce_each_sum_bigger_than( "t_i" | LaTeXindex("i").from(0).unto(npeaks),
                                                                  "\\sigma_i" | LaTeXindex("i").from(0).unto(npeaks),
                                                                  minval(fnplot, captfinder("t")) )
                                 && enforce_each_smaller_than( "\\sigma_i" | LaTeXindex("i").from(0).unto(npeaks), abs(biggestt-smallestt) )
                                 && enforce_each_bigger_than( "\\sigma_i" | LaTeXindex("i").from(0).unto(npeaks), 2*rastert )
                                 && enforce_each_bigger_than( "\\rho_i" | LaTeXindex("i").from(0).unto(npeaks), noiselvl*2 ),
                                    evolution_minimizer::solutioncertaintycriteria::doublevalue()
                                    );
    measure fttt = fitthisback.result();
    measureseq fttps = fttt.pack_subscripted();
    
    cout << "Result:\n" << fttps << endl;
    
    
    phq_interval totalfitdomain;
    physquantity &ldomb = totalfitdomain.l(), &rdomb=totalfitdomain.r();
    
    for (measureseq::access fttp=fttps.begin(); fttp!=fttps.end(); ++fttp){
      fttp["\\sigma"] = abs(fttp["\\sigma"]);
      fttp["t"].label("t_0");
      fttp["\\rho_0"] = physquantity(fttp["\\rho"]);
      phq_interval thisgaussdomain(
                               centered_about(fttp["t_0"]),
                               interval_size(3*fttp["\\sigma"]),
                               "t" );
      totalfitdomain.widen_to_include(thisgaussdomain);
      outgr.plot_phmsq_function(mygauss, *fttp, thisgaussdomain);
    #if 0
      frr.clear();
      cout << "Plot from " << fttp["t_0"]-3*fttp["\\sigma"] << " to " << fttp["t_0"]+3*fttp["\\sigma"] << endl;
      for (fttp["t"] = fttp["t_0"]-3*fttp["\\sigma"]; fttp["t"]<fttp["t_0"]+3*fttp["\\sigma"]; fttp["t"]+=.01*fttp["\\sigma"]){
        ldomb.push_downto(fttp["t"]);
        rdomb.push_upto(fttp["t"]);
        fttp["\\rho"] = mygauss(*fttp);
        frr.push_back(*fttp);
      }
  /*    outgr.insertCurve(frr, captfinder("t"), captfinder("\\rho"), "gaussianfitted"+std::string(1,char(c+44))+".qcv", */c++/* % 8 + 1)*/;
      outgr.insertCurve(frr, captfinder("t"), captfinder("\\rho")/*, "gaussianfitted"+std::string(1,char(c+44))+".qcv", QTeXgrcolors::defaultsequence(c++)*/);
    #endif  
    }
   
    frr.clear();
    #if 1
  //    outgr.plot_phmsq_function(fgfn, fttt, totalfitdomain, QTeXgrcolors::i_red);
    #else
    cout << "Plot sum from " << ldomb << " to " << rdomb << endl;
    for (physquantity t=ldomb; t<rdomb; t += (rdomb-ldomb)/1024) {
      frr.push_back(measure(t));
      frr.back().push_back(physquantity(0).wlabel("\\sum\\rho"));
      for (measureseq::access fttp=fttps.begin(); fttp!=fttps.end(); ++fttp){
        fttp["t"] = t;
        frr.back()["\\sum\\rho"] += mygauss(*fttp);
      }
    }
    outgr.insertCurve(frr, captfinder("t"), captfinder("\\sum\\rho"), /*"gaussianfittedsum.qcv",*/ QTeXgrcolors::i_red);
    #endif
    
 // }

  }
  catch(unitconversion Utrouble) {
     cerr << "Strange stuff with units " << Utrouble.Dim[0]->uName << " and " << Utrouble.Dim[1]->uName << endl; cerr.flush();
     abort();
  }
  catch(phUnit phU) {
     cerr << "Strange stuff with unit " << phU.uName << endl; cerr.flush();
     abort();
  }
  catch(std::string abscent) {
     cerr << "Name " << abscent << " seems abscent.\n"; cerr.flush();
  }
  return (0);
  
}
