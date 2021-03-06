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

int main(){
  try{


  defaultUnitsc = &stdPhysUnits;
  

  fittable_gaussianfn mygauss;
  mygauss.var("x_0",     "t_0")
         .var("A",       "\\rho_0")
//         .var("\\sigma", "\\sigma")
         .var("x",       "t");
         
  measure p;

  QTeXdiagmaster outgr("somegaussian.qda");

  tr1::variate_generator<tr1::mt19937, tr1::normal_distribution<> >
         cdice (std::tr1::mt19937(time(NULL)), std::tr1::normal_distribution<>(0, 1));
  
  tr1::variate_generator<tr1::mt19937, tr1::normal_distribution<> >
         noise (std::tr1::mt19937(time(NULL)), std::tr1::normal_distribution<>(0, 1./23));
  
  int c=3;
  measureseq fnplot(0), frr;

  while (c < 4) {
    p["\\sigma"] = abs((cdice() + 1.6) * .4*amperes);
    p["t_0"] = cdice() * 4*amperes;
    p["t_{0p}"] = p["t_0"] + cdice() * p["\\sigma"];
    p["\\rho_0"] = cdice() * 1*teslas;
    fnplot.clear();
    for (p["t"] = p["t_{0p}"]-3*p["\\sigma"]; p["t"]<p["t_{0p}"]+3*p["\\sigma"]; p["t"]+=.01*p["\\sigma"]){
//    cout << "Status now:\n" << p << endl;
      p["\\rho"] = mygauss(p) + noise()*p["\\rho_0"];
  //    p["\\rho"].push_upto(0);
      fnplot.push_back(measure(p["\\rho"], p["t"]));//, p["t_0"]));
//    outgr.insertMeasure(p["t"] + p["\\Delta t"], p["\\rho"]);
    }
    outgr.insertCurve(fnplot, captfinder("t"), captfinder("\\rho"), "gaussianno"+std::string(1,char(c+44))+".qcv", c++ % 4 + 1);

//  cout << "The correct values:\n" << p << endl; 
    
    montecarlofit fitthisback
             = montecarlofit ( fnplot, captfinder("\\rho"),
                               fittable_gaussianfn().var("x_0",     "t_0")
                                                    .var("A",       "\\rho_0")
                                                    .var("x",       "t")       );
    measure fttp = fitthisback.result();
    fttp["\\sigma"] = abs(fttp["\\sigma"]);
    frr.clear();
    cout << "Plot from " << fttp["t_0"]-3*fttp["\\sigma"] << " to " << fttp["t_0"]+3*fttp["\\sigma"] << endl;
    for (fttp["t"] = fttp["t_0"]-3*fttp["\\sigma"]; fttp["t"]<fttp["t_0"]+3*fttp["\\sigma"]; fttp["t"]+=.01*fttp["\\sigma"]){
      fttp["\\rho"] = mygauss(fttp);
      frr.push_back(fttp);
    }    
    outgr.insertCurve(frr, captfinder("t"), captfinder("\\rho"), "gaussianfitted.qcv", 2);
  }

  }catch(const phUnit &phU) {
     cout << "Strange stuff with unit " << phU.uName << endl;
     abort();
  }
  
  
  return (0);
  
}
