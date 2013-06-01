// RUN: %autodiff %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s

#include "autodiff/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int f(int x) {
  printf("This is f(x).\n");
  return x*x;
}

int g(long y) {
  if (y)
    return 1;
  else
    return 2;
}

int main () {
  int x = 4;
  // Here the second arg denotes the differentiation of f with respect to the 
  // given arg.
  diff(f, x);

  long y = 2;
  diff(g, y);
  // Would it be better if the signature is diff(f, 1), where 1 is the number
  // of the respected parameter? This will help to get rid of the variadic 
  // templates (C++11) feature?
 
  //diff(g, 2);
  return 0;
}

//CHECK: diff(f, x)
//CHECK: diff(g, y)
