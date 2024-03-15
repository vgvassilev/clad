//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to differentiate functions w.r.t. to arrays in
// forward, reverse and jacobian mode
//
// author:  Baidyanath Kundu <kundubaidya99-at-gmail.com>
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

double weighted_avg(double* arr, double* weights) {
  return (arr[0] * weights[0] + arr[1] * weights[1] + arr[2] * weights[2]) / 3;
}

int main() {
  double arr[3] = {1, 2, 3};
  double weights[3] = {.5, .6, .3};

  // Forward Mode

  // Differentiate weighted_avg w.r.t to each element in arr. For forward mode
  // it is required to specify the array index along with the array to be
  // differentiated w.r.t.
  auto weighted_avg_darr0 = clad::differentiate(weighted_avg, "arr[0]");
  auto weighted_avg_darr1 = clad::differentiate(weighted_avg, "arr[1]");
  auto weighted_avg_darr2 = clad::differentiate(weighted_avg, "arr[2]");

  double res_arr0 = weighted_avg_darr0.execute(arr, weights);
  double res_arr1 = weighted_avg_darr1.execute(arr, weights);
  double res_arr2 = weighted_avg_darr2.execute(arr, weights);

  printf("Forward Mode w.r.t. arr:\n res_arr = %.2g, %.2g, %.2g\n", res_arr0,
         res_arr1, res_arr2);

  // Reverse Mode

  // Differentiate weighted_avg w.r.t. to both arr and weights array.
  auto weighted_avg_dall = clad::gradient(weighted_avg);
  // Differentiate weighted_avg w.r.t. to arr.
  auto weighted_avg_darr = clad::gradient(weighted_avg, "arr");

  double darr[3] = {0, 0, 0};
  double dweights[3] = {0, 0, 0};

  weighted_avg_dall.execute(arr, weights, darr, dweights);
  printf("Reverse Mode w.r.t. all:\n darr = {%.2g, %.2g, %.2g}\n dweights = "
         "{%.2g, %.2g, %.2g}\n",
         darr[0], darr[1], darr[2], dweights[0], dweights[1], dweights[2]);

  darr[0] = darr[1] = darr[2] = 0;
  weighted_avg_darr.execute(arr, weights, darr);
  printf("Reverse Mode w.r.t. arr:\n darr = {%.2g, %.2g, %.2g}\n", darr[0],
         darr[1], darr[2]);

  // Hessian mode

  // Generates the Hessian matrix for weighted_avg w.r.t. to both arr and
  // weights. When dealing with arrays hessian mode requires the user to specify
  // the indexes of the array by using the format arr[0:<last index of arr>]
  auto hessian_all = clad::hessian(weighted_avg, "arr[0:2], weights[0:2]");
  // Generates the Hessian matrix for weighted_avg w.r.t. to arr.
  // auto hessian_arr = clad::hessian(weighted_avg, "arr[0:2]");

  double matrix_all[36] = {0};
  // double matrix_arr[9] = {0};

  hessian_all.execute(arr, weights, matrix_all);
  printf("Hessian Mode w.r.t. to all:\n matrix =\n"
         "  {%.2g, %.2g, %.2g, %.2g, %.2g, %.2g}\n"
         "  {%.2g, %.2g, %.2g, %.2g, %.2g, %.2g}\n"
         "  {%.2g, %.2g, %.2g, %.2g, %.2g, %.2g}\n"
         "  {%.2g, %.2g, %.2g, %.2g, %.2g, %.2g}\n"
         "  {%.2g, %.2g, %.2g, %.2g, %.2g, %.2g}\n"
         "  {%.2g, %.2g, %.2g, %.2g, %.2g, %.2g}\n",
         matrix_all[0], matrix_all[1], matrix_all[2], matrix_all[3],
         matrix_all[4], matrix_all[5], matrix_all[6], matrix_all[7],
         matrix_all[8], matrix_all[9], matrix_all[10], matrix_all[11],
         matrix_all[12], matrix_all[13], matrix_all[14], matrix_all[15],
         matrix_all[16], matrix_all[17], matrix_all[18], matrix_all[19],
         matrix_all[20], matrix_all[21], matrix_all[22], matrix_all[23],
         matrix_all[24], matrix_all[25], matrix_all[26], matrix_all[27],
         matrix_all[28], matrix_all[29], matrix_all[30], matrix_all[31],
         matrix_all[32], matrix_all[33], matrix_all[34], matrix_all[35]);

  /*hessian_arr.execute(arr, weights, matrix_arr);
  printf("Hessian Mode w.r.t. to arr:\n matrix =\n"
         "  {%.2g, %.2g, %.2g}\n"
         "  {%.2g, %.2g, %.2g}\n"
         "  {%.2g, %.2g, %.2g}\n",
         matrix_arr[0], matrix_arr[1], matrix_arr[2], matrix_arr[3],
         matrix_arr[4], matrix_arr[5], matrix_arr[6], matrix_arr[7],
         matrix_arr[8]);
  */
}
