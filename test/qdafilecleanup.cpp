#include "../cqtx.h"



int main() {
  int blargh = system("cp somegaussian/* ./");

  cqtx::qdafilecleanup("somegaussian.qda");
  
  return 0;
}