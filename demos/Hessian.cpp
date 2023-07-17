//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to calculate the Hessian matrix of
// a function.
//
// author: Jack Qiu
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang++  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -std=c++11 Hessian.cpp
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang++  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -std=c++11 Hessian.cpp

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

double kinetic_energy(double mass, double velocity) {
  return mass * velocity * velocity * 0.5;
}

int main() {
  // Generates all the second partial derivative columns of a Hessian matrix
  // and stores CallExprs to them inside a single function 
  auto hessian_one = clad::hessian(kinetic_energy);
  
  // Can manually specify independent arguments
  auto hessian_two = clad::hessian(kinetic_energy, "mass, velocity");
  
  // Creates an empty matrix to store the Hessian in
  // Must have enough space, 2 independent variables requires 4 elements (2^2=4)
  double matrix[4];
  
  // Prints the generated Hessian function
  hessian_one.dump();
  hessian_two.dump();
  
  // Substitutes these values into the Hessian function and pipes the result
  // into the matrix variable.
  hessian_one.execute(10, 2, matrix);
  hessian_two.execute(5, 1, matrix);
}
