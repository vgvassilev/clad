//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing the debugging facilities in clad.
//
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang++  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -std=c++11 DebuggingClad.cpp
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang++  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -std=c++11 DebuggingClad.cpp

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

float myFunc(float x) {
   return 3.14 * x * x;
}

int main() {
  // In some cases one needs to make sure the produced code is correct. After
  // the differentiation process finishes, dump can be called to print out some
  // debug information.
  auto cladMyFunc = clad::differentiate(myFunc, 0);
  cladMyFunc.dump();

  // At runtime, this would produce an output similar to:
  // float myFunc_dx(float x) {
  //    return ((3.1400000000000001 * x + 3.1400000000000001 * 1) * x + 3.1400000000000001 * x * 1);
  // }
  //

  return 0;
}
