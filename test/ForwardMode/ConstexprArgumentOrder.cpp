// RUN: %cladclang %s -I%S/../../include -std=c++20 \
// RUN:     -oConstexprArgumentOrder.out | %filecheck %s
// RUN: ./ConstexprArgumentOrder.out | %filecheck_exec %s
// UNSUPPORTED: clang-14, clang-15, clang-16

// A derivative evaluated while the program compiles has to read each argument
// at its own position. Constant evaluation goes by a parameter's scope index,
// so generated parameters left at the default index of 0 all read the first
// argument: right at run time, wrong at compile time.
//
// ConstexprTest.C and Documentation/Guide/ImmediateMode.cpp both differentiate
// (x + y) / 2, whose derivative is the same whatever the arguments are, so
// neither of them can tell. This one differentiates a product, where each
// partial derivative is made of the other two arguments.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

constexpr double prod(double a, double b, double c) { return a * b * c; }

//CHECK: constexpr double prod_darg0(double a, double b, double c) {
//CHECK: constexpr double prod_darg1(double a, double b, double c) {
//CHECK: constexpr double prod_darg2(double a, double b, double c) {

constexpr double da() {
  return clad::differentiate<clad::immediate_mode>(prod, "a").execute(2., 3.,
                                                                     5.);
}
constexpr double db() {
  return clad::differentiate<clad::immediate_mode>(prod, "b").execute(2., 3.,
                                                                     5.);
}
constexpr double dc() {
  return clad::differentiate<clad::immediate_mode>(prod, "c").execute(2., 3.,
                                                                     5.);
}

int main() {
  constexpr double ga = da();
  constexpr double gb = db();
  constexpr double gc = dc();

  // Each one is the product of the two arguments it is not taken against, so
  // reading the wrong argument gives the wrong number rather than the same one.
  static_assert(ga == 15., "d(a*b*c)/da at (2,3,5) is b*c");
  static_assert(gb == 10., "d(a*b*c)/db at (2,3,5) is a*c");
  static_assert(gc == 6., "d(a*b*c)/dc at (2,3,5) is a*b");

  printf("%.0f %.0f %.0f\n", ga, gb, gc);
  //CHECK-EXEC: 15 10 6

  // The same calls at run time, which were already right before the parameter
  // index was set and must stay right.
  printf("%.0f %.0f %.0f\n", da(), db(), dc());
  //CHECK-EXEC: 15 10 6
}
