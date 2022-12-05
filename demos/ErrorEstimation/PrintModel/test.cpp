//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo describing the usage of custom estimation models with the built -in
// error estimator of clad.
//
// author:  Garima Singh
//----------------------------------------------------------------------------//
// For information on how to run this demo, please take a look at the README

#include "clad/Differentiator/Differentiator.h"

#include "test.h"

// Use a trivial dummy example to check if the our custom model is working
// correctly.
float func(float x, float y) {
  float z;
  z = x + y;
  return z;
}

int main() {
  // Call error-estimate on func.
  auto df = clad::estimate_error(func);
  // Finally, dump the generated code.
  df.dump();

  // Calculate the error
  float dx, dy; 
  double error;
  df.execute(2, 3, &dx, &dy, error);
}
