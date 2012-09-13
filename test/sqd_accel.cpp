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


#define CUDA_ACCELERATION 
#define ACCELERATE_FUNCTIONFITS

#include "cqtx.h"

using namespace cqtx;

int main() {
  measureseq fittgt;
  physquantity sigma0 = 2 * meters, x0 = 1 * meters, A0 = 1*joules;
  physquantity sigma1 = 2 * meters, x1 = 5 * meters, A1 = 0*joules;

  for(unsigned k=0; k<100; ++k) {
    fittgt.push_back(measure()); auto& m = fittgt.back();

    m.let("x") = (k/10. +lusminus(.7*(.5-cos(k*.9232)/5))) * meters;
    m.let("q") = ( A0 * exp(-(m["x"] - x0).squared() / (2*sigma0.squared()))
                 + A1 * exp(-(m["x"] - x1).squared() / (2*sigma1.squared())) )
                    .plusminus(.3 * (1+sin(k)/3)*joules) ;
  }

  fittable_multigaussianfn g(2);

  fitdist_fntomeasures d(fittgt, g, captfinder("q"));

  measure probe;
          probe.let("x_0") = x0 + .1 * meters;
          probe.let("\\sigma_0") = sigma1; //3 * meters;
          probe.let("A_0") = A0;
          probe.let("x_1") = x1 - .1 * meters;
          probe.let("\\sigma_1") = sigma1; //3 * meters;
          probe.let("A_1") = A1;

  (cout << "Square distance: "<< d(probe) << ".\n").flush();

  return 0;
}