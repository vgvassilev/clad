// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// The parent map of the AST is built the first time clad asks for a parent,
// here while differentiating u. The body of sq_pushforward<double> is only
// instantiated later, when the hessian asks for its pullback, so none of its
// calls has an entry. Asking whether the value of one is used must not read
// past the empty list of parents.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct V {
  double a[2];
  double& operator()(int i) { return a[i]; }
};

template <class T> T sq(T x) { return x * x; }

namespace clad {
namespace custom_derivatives {
template <class T> ValueAndPushforward<T, T> sq_pushforward(T x, T dx) {
  V d{{dx * x, dx * x}};
  for (int i = 1; i < 2; ++i)
    d(0) += d(i);
  return {x * x, d(0)};
}
} // namespace custom_derivatives
} // namespace clad

double g(double x) { return sq(x) * x; }
double f(double x) { return g(x); }
double u(double x) {
  V v{{x, x}};
  v(0) *= x;
  return v(0);
}

// CHECK: void sq_pushforward_pullback(double x, double dx, ValueAndPushforward<double, double> _d_y, double *_d_x, double *_d_dx) {

int main() {
  double du = 0;
  auto G = clad::gradient(u);
  G.execute(2., &du);
  printf("%.2f\n", du); // CHECK-EXEC: 4.00
  double h = 0;
  auto H = clad::hessian(f, "x");
  H.execute(1.5, &h);
  printf("%.2f\n", h); // CHECK-EXEC: 9.00
}
