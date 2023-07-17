//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to calculate gradient/normal vector of
// implicit function.
//
// author:  Alexander Penev <alexander_penev-at-yahoo.com>
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang++  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -std=c++11 Gradient.cpp
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang++  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -std=c++11 Gradient.cpp

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

// Implicit function for sphere
float sphere_implicit_func(float x, float y, float z, float xc, float yc, float zc, float r) {
  return (x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc) - r*r;
}

int main() {
  // Differentiate implicit sphere function. Clad will produce the three partial derivatives
  // of function sphere_implicit_func
  auto sphere_implicit_func_dx = clad::differentiate(sphere_implicit_func, 0);
  auto sphere_implicit_func_dy = clad::differentiate(sphere_implicit_func, 1);
  auto sphere_implicit_func_dz = clad::differentiate(sphere_implicit_func, 2);

  // Point P=(x,y,z) on surface
  float x = 5.0f;
  float y = 0;
  float z = 0;

  // Calculate gradient in point P (Normal vector)
  float Nx = sphere_implicit_func_dx.execute(x, y, z, 0, 0, 0, 5.0f);
  float Ny = sphere_implicit_func_dy.execute(x, y, z, 0, 0, 0, 5.0f);
  float Nz = sphere_implicit_func_dz.execute(x, y, z, 0, 0, 0, 5.0f);
  printf("Result is N=(%f,%f,%f)\n", Nx, Ny, Nz);

  return 0;
}
