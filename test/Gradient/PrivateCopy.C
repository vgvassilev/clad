// RUN: %cladclang %s -I%S/../../include -oPrivateCopy.out 2>&1 | %filecheck %s
// RUN: ./PrivateCopy.out | %filecheck_exec %s
// XFAIL: valgrind

// Reverse-mode coverage for utils::isCopyable: a private copy ctor must not
// be treated as copyable, or ReverseModeVisitor copy-inits from
// constructor_reverse_forw and the derivative fails to compile.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

class PrivateCopy {
public:
  double v;

private:
  PrivateCopy(const PrivateCopy&);

public:
  PrivateCopy() : v(0) {}
  PrivateCopy(double x) : v(x) {}
  PrivateCopy(PrivateCopy&& o) noexcept : v(o.v) { o.v = 0; }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
inline clad::ValueAndAdjoint<PrivateCopy, PrivateCopy>
constructor_reverse_forw(clad::Tag<PrivateCopy>, double x, double dx) {
  return {PrivateCopy(x), PrivateCopy(dx)};
}
inline void constructor_pullback(double /*x*/, PrivateCopy* dthis,
                                 double* dx) {
  *dx += dthis->v;
  dthis->v = 0;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

double f_priv(double x) {
  PrivateCopy p(x);
  return p.v;
}

// Without the isCopyable change these would be copy ctors (ill-formed).
// CHECK: void f_priv_grad(double x, double *_d_x) {
// CHECK-NEXT:     clad::ValueAndAdjoint<PrivateCopy, PrivateCopy> _t0 = clad::custom_derivatives::class_functions::constructor_reverse_forw(clad::Tag<PrivateCopy>(), x, 0.);
// CHECK-NEXT:     PrivateCopy p(static_cast<PrivateCopy &&>(_t0.value));
// CHECK-NEXT:     PrivateCopy _d_p(static_cast<PrivateCopy &&>(_t0.adjoint));

int main() {
  double dx = 0;
  auto g = clad::gradient(f_priv);
  g.execute(3.0, &dx);
  printf("priv: %.4f\n", dx); // CHECK-EXEC: priv: 1.0000
  return 0;
}
