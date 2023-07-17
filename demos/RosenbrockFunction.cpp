//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to calculate the Rosenbrock function.
//
// author:  Martin Vasilev <mrtn.vassilev-at-gmail.com>
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang++  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -std=c++11 RosenbrockFunction.cpp
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang++  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -std=c++11 RosenbrockFunction.cpp

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

// Rosenbrock function declaration
double rosenbrock_func(double x, double y) {
  return (x - 1) * (x - 1) + 100 * (y - x * x) * (y - x * x);
}

double rosenbrock(double x[], int size) {
  double sum = 0;
  auto rosenbrockX = clad::differentiate(rosenbrock_func, 0);
  auto rosenbrockY = clad::differentiate(rosenbrock_func, 1);

  for (int i = 0; i < size-1; i++) {
    double one = rosenbrockX.execute(x[i], x[i + 1]);
    double two = rosenbrockY.execute(x[i], x[i + 1]);
    sum = sum + one + two;
  }

  return sum;
}

int main() {
  double Xarray[] = {1.5, 4.5};
  int size = sizeof(Xarray) / sizeof(*Xarray);
  double result = rosenbrock(Xarray, size);
  printf("The result is %f.\n", result);

  return 0;
}
