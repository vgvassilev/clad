// RUN: %cladclang %s -I%S/../../include -oLoopCounterType.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./LoopCounterType.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstddef>
#include <cstdio>

// A counted loop's counter takes the induction variable's type -- its
// unsigned counterpart if signed -- spelled as the source does where the
// derivative can name that spelling, and one desugaring step closer to the
// canonical type for each step at which it cannot.

// A function-local alias is out of reach of the derivative, a function of
// its own.
double local(double x) {
  using Idx = unsigned long;
  double r = 0;
  for (Idx i = 0; i < 3; ++i)
    r += x;
  return r;
}
// CHECK-LABEL: void local_grad(double x, double *_d_x) {
// CHECK: unsigned long _t0;

// A member function's derivative is a member too, and names the class's
// private alias.
class Box {
  using Idx = unsigned long;

public:
  double sum(double x) {
    double r = 0;
    for (Idx i = 0; i < 3; ++i)
      r += x;
    return r;
  }
};
// CHECK-LABEL: void sum_grad(double x, Box *_d_this, double *_d_x) {
// CHECK: Idx _t0;

// A public alias of a private class's alias is reachable as written.
class Outer {
  struct Hidden {
    using Idx = unsigned long;
  };

public:
  using Idx = Hidden::Idx;
};
double pub(double x) {
  double r = 0;
  for (Outer::Idx i = 0; i < 3; ++i)
    r += x;
  return r;
}
// CHECK-LABEL: void pub_grad(double x, double *_d_x) {
// CHECK: Outer::Idx _t0;

// decltype is spelled as its expression, whose names may resolve only where
// it was written; the counter takes what it stands for.
std::size_t count = 3; // expected-warning {{gradient uses a global variable 'count'}}
double decl(double x) {
  double r = 0;
  for (decltype(count) i = 0; i < 3; ++i)
    r += x;
  return r;
}
// CHECK-LABEL: void decl_grad(double x, double *_d_x) {
// CHECK: std::size_t _t0;

// Nor can the code name the size_t clang itself gives sizeof, so that one is
// the target's own unsigned type.
double size(double x) {
  double r = 0;
  for (decltype(sizeof(int)) i = 0; i < 3; ++i)
    r += x;
  return r;
}
// CHECK-LABEL: void size_grad(double x, double *_d_x) {
// CHECK: unsigned {{int|long|long long}} _t0;

int main() {
  double d = 0;
  clad::gradient(local).execute(2, &d);
  printf("%.1f\n", d); // CHECK-EXEC: 3.0
  Box b, db;
  d = 0;
  clad::gradient(&Box::sum).execute(b, 2, &db, &d);
  printf("%.1f\n", d); // CHECK-EXEC: 3.0
  d = 0;
  clad::gradient(pub).execute(2, &d);
  printf("%.1f\n", d); // CHECK-EXEC: 3.0
  d = 0;
  clad::gradient(decl).execute(2, &d);
  printf("%.1f\n", d); // CHECK-EXEC: 3.0
  d = 0;
  clad::gradient(size).execute(2, &d);
  printf("%.1f\n", d); // CHECK-EXEC: 3.0
}
