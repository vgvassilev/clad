// RUN: %cladclang %s -I%S/../../include -fsyntax-only 2>&1

//CHECK-NOT: {{.*error|warning|note:.*}}
#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int f(int x) {
  printf("This is f(x).\n");
  return x*x + x - x*x*x*x;;
}

int main () {
  //int x = 4;
  // Here the second arg denotes the differentiation of f with respect to the
  // given arg.
  //diff(f, x);
  // Would it be better if the signature is diff(f, 1), where 1 is the number
  // of the respected parameter? This will help to get rid of the variadic
  // templates (C++11) feature?

  //diff(g, 2);
  clad::CladFunction<decltype(&f)> invalid_func(f,"");
  printf("%s\n", invalid_func.getCode()); //CHECK-EXEC-NEXT: <invalid>
  return 0;
}
