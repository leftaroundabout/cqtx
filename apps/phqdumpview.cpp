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


#include "cqtx.h"
using namespace cqtx;

#include <iostream>


typedef unsigned viewflags;
viewflags print_flag = 0x1
        , plot_flag  = 0x2;

struct fileopt {
  string filename;
  viewflags flags;
  fileopt(): filename(""), flags(0) {}
};

string tgtfolder = "/tmp/";

void autoplot(const measureseq &plt, viewflags flg = plot_flag) {
  if (!(flg ^ plot_flag)) {
    measureindexpointer graph_x = 0;
    std::vector<measureindexpointer> graph_ys;
    for (unsigned i=1; i<plt.front().size(); ++i)
      graph_ys.push_back(measureindexpointer(i));
    
    QTeXdiagmaster plotview(tgtfolder + "autoplot.qda");

    for (auto yc : graph_ys) {
      cout << "plot graph...";
      plotview.insertCurve(plt, graph_x, yc);
    }
    plotview.finish();
  }
}


int main(int argc, char* argv[]){

  std::vector<fileopt> files(1);
  for (int i = 1; i<argc; ++i) {
    string argument = argv[i];
//    cout << "argument " << argument << "\n";
    if (argument=="-g" || argument=="--graph")
      files.back().flags |= plot_flag;
     else if(argument=="-t" || argument=="--text")
      files.back().flags |= print_flag;
     else {
      files.back().filename = argument;
      files.push_back(fileopt());
    }
  }
  files.resize(files.size()-1);

  phqdata_binstorage reader;

  for (auto filer : files) {
    reader.open(filer.filename, std::ios::in);
    if (!reader) {
      cerr << "Trouble with measureseq file \"" << filer.filename << "\"\n";
      abort();
    }
    measureseq mplotf;
    if(argc>2) cout << "\n----------------File \"" << filer.filename << "\"----------------\n\n";
    while (reader.good()) {
      switch (reader.peek_typeof_nextchunk()) {
       case phqdata_binstorage::invalidChunkT:
        goto nextfile;
       case phqdata_binstorage::measureseqChunkT:
        [&](){
          measureseq seq = reader.get();
          if (filer.flags & print_flag || filer.flags==0);
            cout << seq;
          if (filer.flags & plot_flag);
            autoplot (seq, filer.flags);
        }();
        break;
       default:
        cout << reader.get();
      }
      cout << endl;
    }
    nextfile:
    reader.close();
  }
 


  return 0;
}