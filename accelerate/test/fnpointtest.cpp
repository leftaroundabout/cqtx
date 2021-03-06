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

  int n = 3383987
    , argsblwidth = 5, argsoffs = 0
    , retsblwidth = 2;
  

  device_concrrcalc_domains devhandle[4];
  

  std::vector<phmsqfnaccell_fpt>statvals;
  for (phmsqfnaccell_fpt x = -1; statvals.size()<unsigned(n)*4; x+=1./250)
    statvals.push_back(x);

  for (int r = 0; r<1; ++r) {
    devhandle[r].ncalcs = n;

    devhandle[r].stt_argsblocksize = argsblwidth;
    devhandle[r].stt_argsoffset = argsoffs;
    devhandle[r].stt_args = phmsqfnaccell_device_allocate(n*argsblwidth);

    phmsqfnaccell_fpt *statvals_as_column = statvals.data() + n*r;
    phmsqfnaccell_device_injectdata( n, argsblwidth
                                   , argsoffs, 1
                                   , &statvals_as_column
                                   , devhandle[r].stt_args
                                   );
    
    devhandle[r].retsblocksize = retsblwidth;
    devhandle[r].retsoffset = 0;
    devhandle[r].returns = phmsqfnaccell_device_allocate(n*retsblwidth);
  }

  for (phmsqfnaccell_fpt phi = 6.283185; phi>0; phi-=.785398) {
    std::vector<phmsqfnaccell_fpt>dynvals{cos(phi), sin(phi)};
    cout << "For b = " << dynvals.front() << ", a = " << dynvals.back() << ":\n";
    std::vector<phmsqfnaccell_fpt>results(n*4);
    for (int r = 0; r<1; ++r) { cout << "."; cout.flush();
      devhandle[r].dyn_args = dynvals.data();
      fn_access.run_on_device(1, devhandle[r]);
      phmsqfnaccell_fpt *resvals_as_column = results.data() + n*r;
      phmsqfnaccell_device_retrievedata( n, retsblwidth
                                       , 0, 1
                                       , &resvals_as_column
                                       , devhandle[r].returns
                                       );
    }
    cout << '\n';
    for (int i = 0; i<unsigned(n*1); i+=100000) {
      cout << "#" << i << "   " << statvals[i] << " -> " << results[i] << endl;
    }
  }
  
  for (int r = 0; r<4; ++r)
    phmsqfnaccell_device_freeall(devhandle[r]);
  
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

