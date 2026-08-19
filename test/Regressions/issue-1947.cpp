// RUN: %cladclang -std=c++17 %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

void vop(double*) {}
void nested_op(double**) {}
void volatile_op(double*) {}
void overload_op(double*) {}
double read_op(double* p) { return *p; }

namespace clad::custom_derivatives {
void vop_pullback(const double*, double* d_p) { *d_p += 5.; }
// The intermediate const makes this safe, unlike double** -> const double**.
void nested_op_pullback(const double* const*, double** d_p) { **d_p += 2.; }
void volatile_op_pullback(volatile double*, double* d_p) { *d_p += 3.; }
void overload_op_pullback(double*, double* d_p) { *d_p += 4.; }
void overload_op_pullback(const double*, double* d_p) { *d_p += 9.; }
ValueAndPushforward<double, double> read_op_pushforward(const double* p,
                                                       double* d_p) {
  return {*p, *d_p + 5.};
}
} // namespace clad::custom_derivatives

double free_function(double x) {
  double value = x;
  vop(&value);
  return value;
}

struct Box {
  void stash(double*) {}
};

namespace clad::custom_derivatives::class_functions {
void stash_pullback(Box*, const double*, Box*, double* d_p) { *d_p += 5.; }
} // namespace clad::custom_derivatives::class_functions

double member_function(double x) {
  Box box;
  double value = x;
  box.stash(&value);
  return value;
}

double nested_pointer(double x) {
  double* p = &x;
  nested_op(&p);
  return x;
}

double volatile_pointer(double x) {
  volatile_op(&x);
  return x;
}

double overload_preference(double x) {
  overload_op(&x);
  return x;
}

double custom_pushforward(double x) { return read_op(&x); }

int main() {
  auto freeGrad = clad::gradient(free_function, "x");
  double freeDx = 0.;
  freeGrad.execute(3., &freeDx);

  auto memberGrad = clad::gradient(member_function, "x");
  double memberDx = 0.;
  memberGrad.execute(3., &memberDx);

  std::printf("{%.2f, %.2f}\n", freeDx, memberDx);

  double nestedDx = 0.;
  clad::gradient(nested_pointer).execute(3., &nestedDx);
  double volatileDx = 0.;
  clad::gradient(volatile_pointer).execute(3., &volatileDx);
  double overloadDx = 0.;
  clad::gradient(overload_preference).execute(3., &overloadDx);
  auto pushforward = clad::differentiate(custom_pushforward, "x");
  std::printf("{%.2f, %.2f, %.2f, %.2f}\n", nestedDx, volatileDx,
              overloadDx, pushforward.execute(3.));
}

// CHECK: void free_function_grad(double x, double *_d_x) {
// CHECK: clad::custom_derivatives::vop_pullback(&value, &_d_value);
// CHECK: void member_function_grad(double x, double *_d_x) {
// CHECK: clad::custom_derivatives::class_functions::stash_pullback(&box, &value, &_d_box, &_d_value);
// CHECK: clad::custom_derivatives::nested_op_pullback(&p, &_d_p);
// CHECK: clad::custom_derivatives::volatile_op_pullback(&x, _d_x);
// CHECK: clad::custom_derivatives::overload_op_pullback(&x, _d_x);
// CHECK: clad::custom_derivatives::read_op_pushforward(&x, &_d_x);

// CHECK-EXEC: {6.00, 6.00}
// CHECK-EXEC-NEXT: {3.00, 4.00, 5.00, 6.00}
