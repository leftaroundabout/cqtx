#include "../cqtx.h"

using namespace cqtx;

void explain_usage() {
  std::cout << "Usage:\n    qdafilecleanup 'path1/file1.qda' 'path2/file2.qda' ..." << std::endl;
}

int main(int argc, char* argv[]) {
  if(argc==1) explain_usage();
  for(int i=1; i<argc; ++i) {
    std::string thisarg = argv[i];
    if(thisarg[0]=='-') {
      std::cerr << "This program does not accept any commands." << std::endl;
      explain_usage();
      abort();
    }
    qdafilecleanup(thisarg);
  }
  return 0;
}