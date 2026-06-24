// RUN: %cladclang -Wno-unknown-warning-option -Wno-nontrivial-memcall %s -I%S/../../include -oMoveCtorReturnExpr.out 2>&1 | %filecheck %s
// RUN: ./MoveCtorReturnExpr.out
// RUN: %cladclang -Wno-unknown-warning-option -Wno-nontrivial-memcall -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oMoveCtorReturnExpr.out 2>&1 | %filecheck %s
// RUN: ./MoveCtorReturnExpr.out
#include "clad/Differentiator/Differentiator.h"
#include <utility>

struct S {
  double x;
  double* p;
  S() : x(0), p(nullptr) {}
  S(double px) : x(px), p(nullptr) {}
  S(const S&) = default;
  S& operator=(const S&) = default;
  S(S&& other) noexcept : x(other.x), p(other.p) {
    other.x = 0;
    other.p = nullptr;
  }
};

struct M {
  S g(S v) const { return S(std::move(v)); }
};

double loss(const M& m, double x) {
  auto y = m.g(S(x));
  return y.x;
}

// CHECK: clad::ValueAndAdjoint<S, S> g_reverse_forw(S v, M *_d_this, S _d_v, clad::restore_tracker &_tracker0) const {
// CHECK-NEXT:     return {S(std::move(v)), {}};
// CHECK-NEXT: }

// CHECK: void g_pullback(S v, S _d_y, M *_d_this, S *_d_v) const {
// CHECK-NEXT:     S _t0 = v;
// CHECK-NEXT:     {
// CHECK-NEXT:         S::constructor_pullback(std::move(v), &_d_y, _d_v);
// CHECK-NEXT:         v = _t0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
  auto grad = clad::gradient(loss, "x");
  (void)grad;
  return 0;
}
