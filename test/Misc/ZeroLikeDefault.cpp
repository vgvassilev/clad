// RUN: %cladclang %s -I%S/../../include -oZeroLikeDefault.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: %cladclang %s -I%S/../../include -oZeroLikeDefault.out 2>&1 \
// RUN:   | %filecheck --check-prefix=PULLBACK %s
// RUN: ./ZeroLikeDefault.out | %filecheck_exec %s

// clad builds the zero of a class with clad::zero_like, whose default copies
// the value and zeroes the copy. A class with no default constructor gets a
// zero this way too: here the zero adjoint activity analysis gives an object
// nothing varies, and the zero tangent useful analysis gives one no output
// depends on.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>
#include <type_traits>
#include <utility>

struct Element {
  explicit Element(double k) : k(k) {}
  double scale() const { return k; }
  double k;
};

// A class whose bytes cannot be zeroed, such as one with a reference member,
// has no default zero_like.
struct HoldsRef {
  explicit HoldsRef(double& r) : r(r) {}
  double& r;
};

template <class T>
auto has_zero_like(int)
    -> decltype(clad::zero_like(std::declval<const T&>()), std::true_type{});
template <class T> std::false_type has_zero_like(...);

static_assert(decltype(has_zero_like<Element>(0))::value,
              "A class with no default constructor is copied and zeroed");
static_assert(!decltype(has_zero_like<HoldsRef>(0))::value,
              "A reference member must not be zeroed");

struct State {
  double c;
  void step(const Element& e) { c = c * e.scale(); }
};

double reverse(double x) {
  Element e(2.0);
  State s{x};
  s.step(e);
  s.step(e);
  return s.c;
}

// CHECK: void reverse_grad(double x, double *_d_x) {
// CHECK: s.step_reverse_forw(e, &_d_s, clad::zero_like(e), _tracker0);
// PULLBACK: void step_pullback(const Element &e, State *_d_this) {
// PULLBACK: Element _r0 = clad::zero_like(e);
// PULLBACK-NEXT: e.scale_pullback(this->c * _r_d0, &_r0);

double ratio(const Element& e, double x) { return x / e.scale(); }

double forward(double x) {
  Element e(2.0);
  double unused = ratio(e, x);
  return x * x;
}

// CHECK: double forward_darg0(double x) {
// CHECK: ratio_pushforward(e, x, clad::zero_like(e), _d_x);

// So is the tangent of a class parameter forward mode does not
// differentiate with respect to.
double seeded(Element e, double x) { return e.scale() * x * x; }

// CHECK: double seeded_darg1(Element e, double x) {
// CHECK-NEXT: Element _d_e = clad::zero_like(e);

int main() {
  double dx = 0;
  clad::gradient<clad::opts::enable_va>(reverse).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 4.00
  auto d = clad::differentiate<clad::opts::enable_ua>(forward, "x");
  printf("%.2f\n", d.execute(3)); // CHECK-EXEC: 6.00
  auto ds = clad::differentiate(seeded, "x");
  printf("%.2f\n", ds.execute(Element(2.0), 3)); // CHECK-EXEC: 12.00
}
