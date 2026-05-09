// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s
// UNSUPPORTED: clang-10, clang-11, clang-12, clang-13, clang-14, clang-15, clang-16
#include <cstdio>
#include "clad/Differentiator/Differentiator.h"

void f2(double i, double j){
    auto _f = [] () {
    {
      double a = 1;
    }
  };
}
int main(){
    auto diff=clad::gradient(f2);
    double di=0,dj=0;
    double i=7,j=7;
    diff.execute(i,j,&di,&dj);

    printf("Execution successful\n");
    // CHECK-EXEC: Execution successful
    return 0;
}