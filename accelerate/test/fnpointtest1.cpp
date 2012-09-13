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


#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

#include "interface.h"

struct blockfninfo {
  bool is_usable;

  void (*run_on_device)
        ( int     // characteristic number of static parameters
        , device_concrrcalc_domains // the blocks of static parameters and
                                   //  results for each calculation, as allocated on device
        );
};

blockfninfo phmscdhandle_polynomialfunc(){
  return blockfninfo { true
                     , phmsqfnaccell_polynomial_calc
                     };
}

int main(){
  blockfninfo fn_access = phmscdhandle_polynomialfunc();

  int n = 2048
    , argsblwidth = 5, argsoffs = 0
    , ndyns = 2
    , retsblwidth = 2;
  

  device_concrrcalc_domains devhandle;

  devhandle.ncalcs = n;

  devhandle.stt_argsblocksize = argsblwidth;
  devhandle.stt_argsoffset = argsoffs;
  devhandle.stt_args = phmsqfnaccell_device_allocate(n*argsblwidth);

  std::vector<double>statvals;
  for (double x = -1; statvals.size()<unsigned(n); x+=1./250)
    statvals.push_back(x);
  double *statvals_as_column = statvals.data();
  phmsqfnaccell_device_injectdata( n, argsblwidth
                                 , argsoffs, 1
                                 , &statvals_as_column
                                 , devhandle.stt_args
                                 );

  double *dynargs_devptr = nullptr;
  devhandle.dyn_argsblocksize = ndyns;
  devhandle.dyn_args_mem = &dynargs_devptr;
  //devhandle.dyn_args = nullptr;
  
  devhandle.retsblocksize = retsblwidth;
  devhandle.retsoffset = 0;
  devhandle.returns = phmsqfnaccell_device_allocate(n*argsblwidth);
  

  for (double phi = 0; phi<6.283185; phi+=.785398) {
    std::vector<double>dynvals{cos(phi), sin(phi)};
    devhandle.dyn_args = dynvals.data();
    cout << "For b = " << dynvals.front() << ", a = " << dynvals.back() << ":\n";
    fn_access.run_on_device(1, devhandle);
    //for (int wait = 0; wait < 38759737; ++wait) {}
    std::vector<double>results(n);
    double *resvals_as_column = results.data();
    phmsqfnaccell_device_retrievedata( n, argsblwidth
                                     , argsoffs, 1
                                     , &resvals_as_column
                                     , devhandle.returns
                                     );
    for (int i = 0; i<unsigned(n); i+=64) {
      cout << "#" << i << "   " << statvals[i] << " -> " << results[i] << endl;
    }
  }
  
  phmsqfnaccell_device_freeall(devhandle);
  
  return 0;
}


/*
  device_domains (*prepare_statargsarray)
                  ( int  // block length, i.e. number of calculations to be done
                  , int  // characteristic number of static parameters
                  , int  // characteristic number of results ()
                  );*/
/*
     void (*leave_device)
        ( int   // block length, i.e. number of calculations to be done
        , int   // characteristic number of static parameters
        , device_domains // the blocks of static parameters and
                        //  results for each calculation, as allocated on device
        );

 */

