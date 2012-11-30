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
#include <random>
#include <functional>
#include <iostream>

using namespace cqtx;

const physquantity lambda_alpha1 = 1.5406 * angstroms
                 , lambda_alpha2 = 2.6444 * angstroms
                 , Kalphai_ratio = 2 * real1;

auto xraydoublepeak_relation(const measure& m) -> std::vector<measure> {
  std::vector<measure> result(2, measure());
  result[0].let("s_0") = m["s_0"];
  result[1].let("s_0") = m["s_0"] * lambda_alpha2/lambda_alpha1;
  result[0].let("\\sigma") = m["\\sigma"];
  result[1].let("\\sigma") = m["\\sigma"];
  result[0].let("n") = m["n"];
  result[1].let("n") = m["n"] / Kalphai_ratio;
  return result;
}

struct doublepeak_xrayspectrum : combinedPeaks_fittable_spectrum {
  doublepeak_xrayspectrum(unsigned npeaks)
    : combinedPeaks_fittable_spectrum (
         fittable_multigaussianfn(npeaks*2)
       , 2
       , xraydoublepeak_relation
       , "s_0"
       , "\\sigma"
       , "n"
       , npeaks                       ) {
//     cptof_x("s");
//     for (unsigned i = 0; i<npeaks; ++i) {
//       cptof_x0(   i,    "s"    + LaTeX_subscript(i));
//       cptof_sigma(i, "\\sigma" + LaTeX_subscript(i));
//       cptof_A(    i,    "n"    + LaTeX_subscript(i));
//     }
  }
  virtual combinedPeaks_fittable_spectrum* clone() const override {
    return new doublepeak_xrayspectrum(static_cast<const doublepeak_xrayspectrum&>(*this));
  }
  virtual combinedPeaks_fittable_spectrum* moved() override {
    return new doublepeak_xrayspectrum(static_cast<doublepeak_xrayspectrum&&>(std::move(*this)));
  }
  virtual ~doublepeak_xrayspectrum() override {}
};


int main(){


  defaultUnitsc = &stdPhysUnitsandConsts::stdUnits;
  

  doublepeak_xrayspectrum mygauss(1);
//  fittable_gaussianfn mygauss;
  mygauss.rename_var("x_0",      "t_0")
         .rename_var("A_0",       "\\rho_0")
         .rename_var("\\sigma_0", "\\sigma")
         .rename_var("x",         "t");
         
  measure p;

  QTeXdiagmaster outgr("somedblgaussian/sp0.qda");
  srand(43895367);

  std::normal_distribution<> cdice_distrib(0, 1.);
  std::normal_distribution<> noise_distrib(0, 1./23);
  std::mt19937 random_engine(433487);
  
  auto cdice = std::bind(cdice_distrib, random_engine);
  auto noise = std::bind(noise_distrib, random_engine);
  
  
  int c=3;
  measureseq fnplot(0), frr;

  while (c++ < 4) {
    p.let("\\sigma") = abs((cdice() + 1.6) * .4*amperes);
    p.let("t_0")     = cdice() * 4*amperes;
    p.let("t_{0p}")  = p["t_0"] + cdice() * p["\\sigma"];
    p.let("\\rho_0") = std::sqrt(std::abs(cdice())) * 1*teslas;
    fnplot.clear();
    for (p.let("t") = p["t_{0p}"]-3*p["\\sigma"]; p["t"]<p["t_{0p}"]+3*p["\\sigma"]; p["t"]+=.01*p["\\sigma"]){
  //  cout << "Status now:\n" << p << endl;
      physquantity rho = mygauss(p) + noise()*p["\\rho_0"];
      rho.push_upto(0);
      fnplot.push_back(measure(rho.label("\\rho"), p["t"]));//, p["t_0"]));
//    outgr.insertMeasure(p["t"] + p["\\Delta t"], p["\\rho"]);
    }
    outgr.insertCurve(fnplot, captfinder("t"), captfinder("\\rho")); //, "gaussianno"+std::string(1,char(c+44))+".qcv", c++ % 4 + 1);

  cout << "The correct values:\n" << p << endl; 
    auto fitted
             = fit_phq_to_measureseq( mygauss, fnplot, captfinder("\\rho") );
             
    measure fttp = fitted.result();
    cout << "Result of the fit: \n" << fttp << endl; 
    fttp["\\sigma"] = abs(fttp["\\sigma"]);

    cout << "Plot from " << p["t_0"]-3*p["\\sigma"] << " to " << p["t_0"]+3*p["\\sigma"] << endl;
    for (physquantity t = p["t_0"]-3*p["\\sigma"]; t<p["t_0"]+3*p["\\sigma"]; t+=.01*p["\\sigma"]){
      measure ftpt = fttp;
      ftpt.let("t") = t;
      ftpt.let("\\rho") = fitted(t.label("t")); // mygauss(ftpt); // 
      frr.push_back(ftpt);
    }    
//     measureseq frr2;
//     for(int i=1; i<7; ++i) {
//       measure b;
//       b.let("t") = 3*real1;
//       b.let("\\rho") = 9*real1;
//       frr2.push_back(b);
//     }
//    cout << "measureseq (length " << frr.size() << ") to plot: (...)";// << frr;
     outgr.insertCurve(frr, captfinder("t"), captfinder("\\rho")); // , "gaussianfitted.qcv", 2);
  }

  
  return (0);
  
}
