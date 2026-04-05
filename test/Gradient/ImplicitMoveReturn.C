// RUN: %cladclang -Wno-unknown-warning-option -Wno-nontrivial-memcall -Wno-nontrivial-memaccess %s -I%S/../../include -oImplicitMoveReturn.out 2>&1 | %filecheck %s
// RUN: ./ImplicitMoveReturn.out
// RUN: %cladclang -Wno-unknown-warning-option -Wno-nontrivial-memcall -Wno-nontrivial-memaccess -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oImplicitMoveReturn.out 2>&1 | %filecheck %s
// RUN: ./ImplicitMoveReturn.out
#include "clad/Differentiator/Differentiator.h"

struct S {
  double x;
  S() : x(0) {}
  S(double px) : x(px) {}
  S(const S&) = default;
  S(S&& other) noexcept : x(other.x) { other.x = 0; }

  S operator+(const S& other) const {
    S res(x + other.x);
    return res;
  }
};

struct M {
  S f(const S& a, const S& b) const {
    auto t = a + b;
    return t;
  }
};

double loss(const M& m, const S& a, const S& b) {
  auto y = m.f(a, b);
  return y.x;
}

// CHECK: void f_pullback(const S &a, const S &b, S _d_y, M *_d_this, S *_d_a, S *_d_b) const {
// CHECK-NEXT:     S t = a + b;
// CHECK-NEXT:     S _d_t;
// CHECK-NOT:     S::constructor_reverse_forw
// CHECK-NEXT:     S::constructor_pullback(std::move(t), &_d_y, &_d_t);
// CHECK-NEXT:     a.operator_plus_pullback(b, _d_t, _d_a, _d_b);
// CHECK-NEXT: }

int main() {
  auto grad = clad::gradient(loss, "1");
  (void)grad;
  return 0;
}
