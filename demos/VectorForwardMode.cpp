//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to use vector forward mode AD to differentiation
// a function with respect to multiple parameters.
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang++  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -x c++ -std=c++11 VectorForwardMode.cpp
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang++  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -x c++ -std=c++11 VectorForwardMode.cpp

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

// A function for weighted sum of array elements.
double weighted_sum(double* arr, double* weights, int n) {
  double res = 0;
  for (int i = 0; i < n; ++i)
    res += weights[i] * arr[i];
  return res;
}

int main() {
  auto weighted_sum_grad =
      clad::differentiate<clad::opts::vector_mode>(weighted_sum, "arr,weights");

  // Initialize array and weights.
  double arr[3] = {3.0, 4.0, 5.0};
  double weights[3] = {0.5, 0.7, 0.9};

  // Allocate memory for derivatives.
  double d_arr[3] = {0.0, 0.0, 0.0};
  double d_weights[3] = {0.0, 0.0, 0.0};
  clad::array_ref<double> d_arr_ref(d_arr, 3);
  clad::array_ref<double> d_weights_ref(d_weights, 3);

  // Calculate gradient.
  weighted_sum_grad.execute(arr, weights, 3, d_arr_ref, d_weights_ref);
  printf("Vector forward mode w.r.t. all:\n darr = {%.2g, %.2g, %.2g}\n "
         "dweights = "
         "{%.2g, %.2g, %.2g}\n",
         d_arr[0], d_arr[1], d_arr[2], d_weights[0], d_weights[1],
         d_weights[2]);

  return 0;
}
