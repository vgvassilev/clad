// RUN: %autodiff %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s

#include "autodiff/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int f(int x) {
  printf("This is f(x).\n");
  return x*x;
}

float g(int x, float y) {
  printf("This is g.\n");
  if (x)
    return y;
  else
    return y*y;
}

int main () {
  int x = 4;
  // Here the second arg denotes the differentiation of f with respect to the 
  // given arg.
  diff(f, x);
  // Would it be better if the signature is diff(f, 1), where 1 is the number
  // of the respected parameter? This will help to get rid of the variadic 
  // templates (C++11) feature?
 
  //diff(g, 2);
  return 0;
}

//CHECK: diff(f, x)
