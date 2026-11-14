// RUN: %cladclang %s -I%S/../../include -oPrivateCopy.out 2>&1 | %filecheck %s
// RUN: ./PrivateCopy.out | %filecheck_exec %s

// Reverse-mode coverage for utils::isCopyable: a private copy ctor must not
// be treated as copyable, or ReverseModeVisitor copy-inits from
// constructor_reverse_forw and the derivative fails to compile.

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/BuiltinDerivatives.h"

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

struct NoCopyState {
  double value = 3.0;
  NoCopyState() = default;
  NoCopyState(const NoCopyState&) = delete;
};

CLAD_NO_PRIMAL_TAPE_TYPE(NoCopyState)

struct CLAD_NO_PRIMAL_TAPE DirectNoCopyState {
  double value = 5.0;
  DirectNoCopyState() = default;
  DirectNoCopyState(const DirectNoCopyState&) = delete;
};

double advanceAndScale(double x, NoCopyState& state) {
  state.value += 1.0;
  return x * state.value;
}

double f_no_tape(double x) {
  NoCopyState state;
  return advanceAndScale(x, state);
}

double advanceAndScaleDirect(double x, DirectNoCopyState& state) {
  state.value += 1.0;
  return x * state.value;
}

double f_no_tape_direct(double x) {
  DirectNoCopyState state;
  return advanceAndScaleDirect(x, state);
}

// Without the isCopyable change these would be copy ctors (ill-formed).
// CHECK: void f_priv_grad(double x, double *_d_x) {
// CHECK-NEXT:     clad::ValueAndAdjoint<PrivateCopy, PrivateCopy> _t0 = clad::custom_derivatives::class_functions::constructor_reverse_forw(clad::Tag<PrivateCopy>(), x, 0.);
// CHECK-NEXT:     PrivateCopy p(static_cast<PrivateCopy &&>(_t0.value));
// CHECK-NEXT:     PrivateCopy _d_p(static_cast<PrivateCopy &&>(_t0.adjoint));
// CHECK: void f_no_tape_grad(double x, double *_d_x) {
// CHECK: void f_no_tape_direct_grad(double x, double *_d_x) {

int main() {
  double dx = 0;
  auto g = clad::gradient(f_priv);
  g.execute(3.0, &dx);
  printf("priv: %.4f\n", dx); // CHECK-EXEC: priv: 1.0000
  dx = 0;
  auto g_no_tape = clad::gradient(f_no_tape);
  g_no_tape.execute(4.0, &dx);
  printf("no_tape: %.4f\n", dx); // CHECK-EXEC: no_tape: 4.0000
  dx = 0;
  auto g_no_tape_direct = clad::gradient(f_no_tape_direct);
  g_no_tape_direct.execute(4.0, &dx);
  printf("no_tape_direct: %.4f\n", dx); // CHECK-EXEC: no_tape_direct: 6.0000
  return 0;
}
