// RUN: %autodiff %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s

#include "autodiff/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int ffffffff(int x) {
  int y =3;
  return x + y + x + 3 + x;
}

int f_simple(int x) {
  //  printf("This is f(x).\n");
  float y, z = 2*(x+9);
  ffffffff(3);
  ffffffff(4);
  return 3*x*x;
}
// CHECK: int f_simple_derived(int x) {
// CHECK-NEXT: printf("This is f(x).\n");
// CHECK-NEXT: return 2 * x;
// CHECK-NEXT: }

int main () {
  int z,x = 4*5;
  // Here the second arg denotes the differentiation of f with respect to the
  // given arg.
  //  diff(f_simple, 1);
  diff(f_simple, x);
  //  diff(f_simple, x);
  //  diff(ffffffff, x);
  // diff(ffffffff, x);
  
  //long y = 2;
  //diff(g, y);
  // Would it be better if the signature is diff(f, 1), where 1 is the number
  // of the respected parameter? This will help to get rid of the variadic
  // templates (C++11) feature?
  
  //diff(g, 2);
  return 0;
}
