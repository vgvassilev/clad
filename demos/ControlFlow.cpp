//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how the high-level user interfaces work in control flow
// consbasic usage of the tool.
//
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang++  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -std=c++11 ControlFlow.C
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang++  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -std=c++11 ControlFlow.C

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

float func(float x) { return 3.14 * x * x; }

int main() {
  // Differentiate pow2. Clad will produce the second derivative of pow2.
  auto d2_func_dx = clad::differentiate<2>(func, 0);

  // Or we can call like this:
  float pow5thOrderDerivative = d2_func_dx.execute(1);
  printf("Result is %f\n", pow5thOrderDerivative);

  // IfStmts
  if (d2_func_dx.execute(1) != 3.14)
    printf("Wrong result!\n");

  // Loops

  // PLEASE NOTE: for (...) clad::differentiate(func, 1); won't give the N-th
  // derivative of func. USE: clad::differentiate<N>(func, 1) instead.

  float sum = 0;
  for (unsigned i = 0; i < 10; ++i) {
    sum += d2_func_dx.execute(i);
  }
  printf("Sum is %f\n", sum);

  return 0;
}
