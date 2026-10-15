// RUN: %cladclang -std=c++20 %s -I%S/../../include -o %t -Xclang -verify 2>&1 | %filecheck %s --check-prefixes=CHECK,CXX20 --implicit-check-not="operator_spaceship_pushforward" --implicit-check-not="operator_less_pushforward(const S"
// RUN: %t | %filecheck %s --check-prefixes=CHECK-EXEC,EXEC20
// RUN: %cladclang -std=c++17 %s -I%S/../../include -o %t.17 -Xclang -verify 2>&1 | %filecheck %s --check-prefixes=CHECK,CXX17 --implicit-check-not="operator_less_pushforward(const S"
// RUN: %t.17 | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>
#if __cplusplus >= 202002L
#include <compare>
#endif

// expected-no-diagnostics

struct S {
  double v;
#if __cplusplus >= 202002L
  auto operator<=>(const S&) const = default;
#else
  bool operator<(const S& other) const { return v < other.v; }
  bool operator==(const S& other) const { return v == other.v; }
  bool operator!=(const S& other) const { return v != other.v; }
#endif
};

double defaulted(double x) {
  S a{x}, b{1.0};
  return a < b ? x : 2 * x;
}

double nested(double x) {
  S a{x}, b{1.0};
  if (a < b)
    return x;
  return 2 * x;
}

double stored_result(double x) {
  S a{x}, b{1.0};
#if __cplusplus >= 202002L
  auto c = a <=> b;
  return c < 0 ? x : 2 * x;
#else
  bool c = a < b;
  return c ? x : 2 * x;
#endif
}

double loop_condition(double x) {
  double r = 0;
  for (S s{0.0}; s < S{x}; s.v += 1.0)
    r += x;
  return r;
}

#if __cplusplus >= 202002L
double category_parameter(double x, std::partial_ordering c) {
  auto local = std::partial_ordering::less;
  return c < 0 ? x : 2 * x;
}

double category_references(double x, const std::strong_ordering& s,
                           const std::weak_ordering& w) {
  auto strong = s;
  auto weak = w;
  return strong < 0 && weak < 0 ? x : 2 * x;
}
#endif

double category_call(double x) {
#if __cplusplus >= 202002L
  return category_parameter(x, std::partial_ordering::less);
#else
  return x;
#endif
}

struct Strong {
  int v;
#if __cplusplus >= 202002L
  auto operator<=>(const Strong&) const = default;
#else
  bool operator<(const Strong& other) const { return v < other.v; }
#endif
};

double strong(double x) {
  Strong a{0}, b{1};
#if __cplusplus >= 202002L
  return static_cast<std::partial_ordering>(a <=> b) < 0 ? x : 2 * x;
#else
  return a < b ? x : 2 * x;
#endif
}

struct Written {
  double v;
#if __cplusplus >= 202002L
  std::weak_ordering operator<=>(const Written& other) const {
    if (v < other.v)
      return std::weak_ordering::less;
    if (v > other.v)
      return std::weak_ordering::greater;
    return std::weak_ordering::equivalent;
  }
#else
  bool operator<(const Written& other) const { return v < other.v; }
#endif
};

double written(double x) {
  Written a{x}, b{1.0};
  return a < b ? x : 2 * x;
}

struct Reversed {
  double v;
#if __cplusplus >= 202002L
  auto operator<=>(double other) const { return v <=> other; }
#else
  friend bool operator<(double lhs, const Reversed& rhs) { return lhs < rhs.v; }
#endif
};

double reversed(double x) {
  Reversed a{x};
  return 1.0 < a ? x : 2 * x;
}

double equal(double x) {
  S a{x}, b{1.0};
  return a == b ? x : 2 * x;
}

double unequal(double x) {
  S a{x}, b{1.0};
  return a != b ? x : 2 * x;
}

struct Writable {
  double v;
  bool operator<(double other) {
    v *= 2;
    return v < other;
  }
};

double writable(double x) {
  Writable a{x};
  bool c = a < 10.0;
  return a.v * x;
}

struct Mutable {
  mutable double v;
  bool operator<(double other) const {
    v *= 2;
    return v < other;
  }
};

double mutable_state(double x) {
  Mutable a{x};
  bool c = a < 10.0;
  return a.v * x;
}

namespace pointer_control {
struct Indirect {
  double* p;
};

bool operator<(const Indirect& a, double other) {
  *a.p *= 2;
  return *a.p < other;
}
} // namespace pointer_control

double indirect_state(double x) {
  double y = 3;
  pointer_control::Indirect a{&y};
  bool c = a < 10.0;
  return x * y;
}

struct ReadOnly {
  double v;
  bool operator<(double other) const { return v < other; }
};

double effectful(double x) {
  ReadOnly a{0.0};
  bool c = a < (x = x * 2);
  return x;
}

int status(double& x) {
  x *= 3;
  return 0;
}

bool flag(double& x) {
  x *= 2;
  return true;
}

double ordinary(double x) {
  status(x);
  flag(x);
  return x;
}

#if __cplusplus >= 202002L
namespace my {
struct strong_ordering {
  double v;
};
} // namespace my

double named_category(double x) {
  my::strong_ordering a{x};
  return a.v * x;
}

struct ArrayState {
  double v[2];
  bool operator<(double y) const { return v[0] < y; }
};

double array_state(double x) {
  ArrayState a{{x, 0}};
  return a < 1 ? x : 2 * x;
}

struct Poly {
  double v = 0;
  virtual double get() const { return v; }
  bool operator<(double y) const { return v < y; }
};

double dynamic_state(double x, const Poly& p) {
  return p < x ? x : 2 * x;
}

struct Base {
  double* p = nullptr;
};
struct Derived : Base {
  bool operator<(double y) const { return *p < y; }
};

double inherited_state(double x, const Derived& p) {
  return p < x ? x : 2 * x;
}

struct Flag {
  bool v;
  bool operator==(bool b) const { return v == b; }
};

double rewritten_operand(double x) {
  S a{x}, b{1};
  Flag flag{true};
  bool c = flag == (a != b);
  return c ? x : 2 * x;
}

double unary_operand(double x) {
  ReadOnly a{0};
  bool c = a < x++;
  return c ? x : 2 * x;
}

double stored_unequal(double x) {
  S a{x}, b{1};
  bool ne = a != b;
  return ne ? x : 2 * x;
}
#endif

#define TEST(F, X, Y)                                                          \
  {                                                                          \
    auto d = clad::differentiate(F, "x");                                     \
    auto g = clad::gradient(F);                                               \
    double dx = 0, dy = 0;                                                    \
    g.execute(X, &dx);                                                        \
    g.execute(Y, &dy);                                                        \
    std::printf(#F ": %.1f %.1f / %.1f %.1f\n", d.execute(X), d.execute(Y),    \
                dx, dy);                                                     \
  }

int main() {
  TEST(defaulted, 0.5, 2.0);
  TEST(nested, 0.5, 2.0);
  TEST(stored_result, 0.5, 2.0);
  TEST(loop_condition, 0.5, 2.5);
  TEST(category_call, 0.5, 2.0);
  TEST(strong, 0.5, 2.0);
  TEST(written, 0.5, 2.0);
  TEST(reversed, 0.5, 2.0);
  TEST(equal, 0.5, 1.0);
  TEST(unequal, 0.5, 1.0);
  TEST(writable, 2.0, 3.0);
  TEST(mutable_state, 2.0, 3.0);
  TEST(indirect_state, 2.0, 3.0);
  TEST(effectful, 2.0, 3.0);
  TEST(ordinary, 2.0, 3.0);
#if __cplusplus >= 202002L
  auto dc = clad::differentiate(category_parameter, "x");
  auto gc = clad::gradient(category_parameter);
  double dx = 0;
  gc.execute(0.5, std::partial_ordering::less, &dx);
  std::printf("category_parameter: %.1f %.1f / %.1f\n",
              dc.execute(0.5, std::partial_ordering::less),
              dc.execute(2.0, std::partial_ordering::greater), dx);
  auto dr = clad::differentiate(category_references, "x");
  auto gr = clad::gradient(category_references);
  dx = 0;
  gr.execute(0.5, std::strong_ordering::less, std::weak_ordering::less, &dx);
  std::printf("category_references: %.1f / %.1f\n",
              dr.execute(0.5, std::strong_ordering::less,
                         std::weak_ordering::less), dx);
  TEST(named_category, 2.0, 3.0);
  TEST(array_state, 0.5, 2.0);
  TEST(rewritten_operand, 0.5, 1.0);
  TEST(unary_operand, 0.5, -2.0);
  TEST(stored_unequal, 0.5, 1.0);
  Poly p;
  p.v = 1;
  auto dp = clad::differentiate(dynamic_state, "x");
  std::printf("dynamic_state: %.1f %.1f\n", dp.execute(0.5, p),
              dp.execute(2.0, p));
  double dy = 0;
  double v = 1;
  Derived q{{&v}};
  auto db = clad::differentiate(inherited_state, "x");
  auto gb = clad::gradient(inherited_state, "x");
  dx = dy = 0;
  gb.execute(0.5, q, &dx);
  gb.execute(2.0, q, &dy);
  std::printf("inherited_state: %.1f %.1f / %.1f %.1f\n",
              db.execute(0.5, q), db.execute(2.0, q), dx, dy);
#endif
}

// CHECK-EXEC: defaulted: 1.0 2.0 / 1.0 2.0
// CHECK-EXEC-NEXT: nested: 1.0 2.0 / 1.0 2.0
// CHECK-EXEC-NEXT: stored_result: 1.0 2.0 / 1.0 2.0
// CHECK-EXEC-NEXT: loop_condition: 1.0 3.0 / 1.0 3.0
// CHECK-EXEC-NEXT: category_call: 1.0 1.0 / 1.0 1.0
// CHECK-EXEC-NEXT: strong: 1.0 1.0 / 1.0 1.0
// CHECK-EXEC-NEXT: written: 1.0 2.0 / 1.0 2.0
// CHECK-EXEC-NEXT: reversed: 2.0 1.0 / 2.0 1.0
// CHECK-EXEC-NEXT: equal: 2.0 1.0 / 2.0 1.0
// CHECK-EXEC-NEXT: unequal: 1.0 2.0 / 1.0 2.0
// CHECK-EXEC-NEXT: writable: 8.0 12.0 / 8.0 12.0
// CHECK-EXEC-NEXT: mutable_state: 8.0 12.0 / 8.0 12.0
// CHECK-EXEC-NEXT: indirect_state: 6.0 6.0 / 6.0 6.0
// CHECK-EXEC-NEXT: effectful: 2.0 2.0 / 2.0 2.0
// CHECK-EXEC-NEXT: ordinary: 6.0 6.0 / 6.0 6.0
// EXEC20-NEXT: category_parameter: 1.0 2.0 / 1.0
// EXEC20-NEXT: category_references: 1.0 / 1.0
// EXEC20-NEXT: named_category: 4.0 6.0 / 4.0 6.0
// EXEC20-NEXT: array_state: 1.0 2.0 / 1.0 2.0
// EXEC20-NEXT: rewritten_operand: 1.0 2.0 / 1.0 2.0
// EXEC20-NEXT: unary_operand: 1.0 2.0 / 1.0 2.0
// EXEC20-NEXT: stored_unequal: 1.0 2.0 / 1.0 2.0
// EXEC20-NEXT: dynamic_state: 2.0 1.0
// EXEC20-NEXT: inherited_state: 2.0 1.0 / 2.0 1.0

// CHECK-LABEL: double defaulted_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     S _d_a{_d_x}, _d_b{0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CHECK-NEXT:     bool _t0 = a < b;
// CHECK-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void defaulted_grad(double x, double *_d_x) {
// CHECK-NEXT:     S _d_a = {0.}, _d_b = {0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CXX17-NEXT:     if (a < b)
// CXX20-NEXT:     if (a <=> b < 0)
// CHECK-NEXT:         *_d_x += 1;
// CHECK-NEXT:     else
// CHECK-NEXT:         *_d_x += 2 * 1;
// CHECK-NEXT:     *_d_x += _d_a.v;
// CHECK-NEXT: }

// CHECK-LABEL: double nested_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     S _d_a{_d_x}, _d_b{0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CHECK-NEXT:     if (a < b)
// CHECK-NEXT:         return _d_x;
// CHECK-NEXT:     return 2 * _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void nested_grad(double x, double *_d_x) {
// CHECK-NEXT:     S _d_a = {0.}, _d_b = {0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CHECK-NEXT:     clad::forward_sweep([&] {
// CXX17-NEXT:         if (a < b)
// CXX20-NEXT:         if (a <=> b < 0)
// CHECK-NEXT:             return;
// CHECK-NEXT:         *_d_x += 2 * 1;
// CHECK-NEXT:     });
// CXX17-NEXT:     if (a < b)
// CXX20-NEXT:     if (a <=> b < 0)
// CHECK-NEXT:         *_d_x += 1;
// CHECK-NEXT:     *_d_x += _d_a.v;
// CHECK-NEXT: }

// CHECK-LABEL: double stored_result_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     S _d_a{_d_x}, _d_b{0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CXX17-NEXT:     bool _d_c = 0;
// CXX17-NEXT:     bool c = a < b;
// CXX17-NEXT:     return c ? _d_x : 2 * _d_x;
// CXX20-NEXT:     {{(std::)*}}partial_ordering c = a <=> b;
// CXX20-NEXT:     bool _t0 = c < 0;
// CXX20-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void stored_result_grad(double x, double *_d_x) {
// CHECK-NEXT:     S _d_a = {0.}, _d_b = {0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CXX17-NEXT:     bool _d_c = false;
// CXX17-NEXT:     bool c = a < b;
// CXX17-NEXT:     bool _cond0 = c;
// CXX17-NEXT:     if (_cond0)
// CXX20-NEXT:     {{(std::)*}}partial_ordering c = a <=> b;
// CXX20-NEXT:     if (c < 0)
// CHECK-NEXT:         *_d_x += 1;
// CHECK-NEXT:     else
// CHECK-NEXT:         *_d_x += 2 * 1;
// CHECK-NEXT:     *_d_x += _d_a.v;
// CHECK-NEXT: }

// CHECK-LABEL: double loop_condition_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         S _d_s{0.};
// CHECK-NEXT:         for (S s{0.}; s < S{x}; (_d_s.v += 0.) , (s.v += 1.)) {
// CHECK-NEXT:             _d_r += _d_x;
// CHECK-NEXT:             r += x;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }

// CHECK-LABEL: void loop_condition_grad(double x, double *_d_x) {
// CHECK-NEXT:     S _d_s = {0.};
// CHECK-NEXT:     S s = {0.};
// CHECK-NEXT:     double _d_r = 0.;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = 0;
// CXX17-NEXT:     for (s = {0.}; s < S{x}; s.v += 1.) {
// CXX20-NEXT:     for (s = {0.}; s <=> S{x} < 0; s.v += 1.) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         r += x;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_r += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         double _r_d0 = _d_s.v;
// CHECK-NEXT:         *_d_x += _d_r;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CXX20-LABEL: inline clad::ValueAndPushforward<double, double> category_parameter_pushforward(double x, std::partial_ordering c, double _d_x) {
// CXX20-NEXT:     {{(std::)*}}partial_ordering local = std::partial_ordering::less;
// CXX20-NEXT:     bool _t0 = c < 0;
// CXX20-NEXT:     return {_t0 ? x : 2 * x, _t0 ? _d_x : 2 * _d_x};
// CXX20-NEXT: }

// CHECK-LABEL: double category_call_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CXX17-NEXT:     return _d_x;
// CXX20-NEXT:     clad::ValueAndPushforward<double, double> _t0 = category_parameter_pushforward(x, std::partial_ordering::less, _d_x);
// CXX20-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CXX20-LABEL: inline void category_parameter_pullback(double x, std::partial_ordering c, double _d_y, double *_d_x) {
// CXX20-NEXT:     {{(std::)*}}partial_ordering local = std::partial_ordering::less;
// CXX20-NEXT:     if (c < 0)
// CXX20-NEXT:         *_d_x += _d_y;
// CXX20-NEXT:     else
// CXX20-NEXT:         *_d_x += 2 * _d_y;
// CXX20-NEXT: }

// CHECK-LABEL: void category_call_grad(double x, double *_d_x) {
// CXX17-NEXT:     *_d_x += 1;
// CXX20-NEXT:     {
// CXX20-NEXT:         double _r0 = 0.;
// CXX20-NEXT:         category_parameter_pullback(x, std::partial_ordering::less, 1, &_r0);
// CXX20-NEXT:         *_d_x += _r0;
// CXX20-NEXT:     }
// CHECK-NEXT: }

// CHECK-LABEL: double strong_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Strong _d_a{0}, _d_b{0};
// CHECK-NEXT:     Strong a{0}, b{1};
// CXX17-NEXT:     bool _t0 = a < b;
// CXX20-NEXT:     bool _t0 = static_cast<std::partial_ordering>(a <=> b) < 0;
// CHECK-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void strong_grad(double x, double *_d_x) {
// CHECK-NEXT:     Strong _d_a = {0}, _d_b = {0};
// CHECK-NEXT:     Strong a{0}, b{1};
// CXX17-NEXT:     if (a < b)
// CXX20-NEXT:     if (static_cast<std::partial_ordering>(a <=> b) < 0)
// CHECK-NEXT:         *_d_x += 1;
// CHECK-NEXT:     else
// CHECK-NEXT:         *_d_x += 2 * 1;
// CHECK-NEXT: }

// CHECK-LABEL: double written_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Written _d_a{_d_x}, _d_b{0.};
// CHECK-NEXT:     Written a{x}, b{1.};
// CHECK-NEXT:     bool _t0 = a < b;
// CHECK-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void written_grad(double x, double *_d_x) {
// CHECK-NEXT:     Written _d_a = {0.}, _d_b = {0.};
// CHECK-NEXT:     Written a{x}, b{1.};
// CXX17-NEXT:     if (a < b)
// CXX20-NEXT:     if (a <=> b < 0)
// CHECK-NEXT:         *_d_x += 1;
// CHECK-NEXT:     else
// CHECK-NEXT:         *_d_x += 2 * 1;
// CHECK-NEXT:     *_d_x += _d_a.v;
// CHECK-NEXT: }

// CHECK-LABEL: double reversed_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Reversed _d_a{_d_x};
// CHECK-NEXT:     Reversed a{x};
// CHECK-NEXT:     bool _t0 = 1. < a;
// CHECK-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void reversed_grad(double x, double *_d_x) {
// CHECK-NEXT:     Reversed _d_a = {0.};
// CHECK-NEXT:     Reversed a{x};
// CXX17-NEXT:     if (1. < a)
// CXX20-NEXT:     if (0 < a <=> 1.)
// CHECK-NEXT:         *_d_x += 1;
// CHECK-NEXT:     else
// CHECK-NEXT:         *_d_x += 2 * 1;
// CHECK-NEXT:     *_d_x += _d_a.v;
// CHECK-NEXT: }

// CHECK-LABEL: double equal_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     S _d_a{_d_x}, _d_b{0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CHECK-NEXT:     bool _t0 = a == b;
// CHECK-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void equal_grad(double x, double *_d_x) {
// CHECK-NEXT:     S _d_a = {0.}, _d_b = {0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CHECK-NEXT:     if (a == b)
// CHECK-NEXT:         *_d_x += 1;
// CHECK-NEXT:     else
// CHECK-NEXT:         *_d_x += 2 * 1;
// CHECK-NEXT:     *_d_x += _d_a.v;
// CHECK-NEXT: }

// CHECK-LABEL: double unequal_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     S _d_a{_d_x}, _d_b{0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CHECK-NEXT:     bool _t0 = a != b;
// CHECK-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void unequal_grad(double x, double *_d_x) {
// CHECK-NEXT:     S _d_a = {0.}, _d_b = {0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CXX17-NEXT:     if (a != b)
// CXX20-NEXT:     bool _cond0 = !a == b;
// CXX20-NEXT:     if (_cond0)
// CHECK-NEXT:         *_d_x += 1;
// CHECK-NEXT:     else
// CHECK-NEXT:         *_d_x += 2 * 1;
// CHECK-NEXT:     *_d_x += _d_a.v;
// CHECK-NEXT: }

// CHECK-LABEL: double writable_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Writable _d_a{_d_x};
// CHECK-NEXT:     Writable a{x};
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t0 = a.operator_less_pushforward(10., &_d_a, 0.);
// CHECK-NEXT:     bool _d_c = _t0.pushforward;
// CHECK-NEXT:     bool c = _t0.value;
// CHECK-NEXT:     double &_t1 = a.v;
// CHECK-NEXT:     return _d_a.v * x + _t1 * _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void writable_grad(double x, double *_d_x) {
// CHECK-NEXT:     Writable _d_a = {0.};
// CHECK-NEXT:     Writable a{x};
// CHECK-NEXT:     bool _d_c = false;
// CHECK-NEXT:     bool c = a < 10.;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_a.v += 1 * x;
// CHECK-NEXT:         *_d_x += a.v * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         a.operator_less_pullback(10., _d_c, &_d_a, &_r0);
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_a.v;
// CHECK-NEXT: }

// CHECK-LABEL: double mutable_state_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Mutable _d_a{_d_x};
// CHECK-NEXT:     Mutable a{x};
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t0 = a.operator_less_pushforward(10., &_d_a, 0.);
// CHECK-NEXT:     bool _d_c = _t0.pushforward;
// CHECK-NEXT:     bool c = _t0.value;
// CHECK-NEXT:     double &_t1 = a.v;
// CHECK-NEXT:     return _d_a.v * x + _t1 * _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void mutable_state_grad(double x, double *_d_x) {
// CHECK-NEXT:     Mutable _d_a = {0.};
// CHECK-NEXT:     Mutable a{x};
// CHECK-NEXT:     bool _d_c = false;
// CHECK-NEXT:     bool c = a < 10.;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_a.v += 1 * x;
// CHECK-NEXT:         *_d_x += a.v * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         a.operator_less_pullback(10., _d_c, &_d_a, &_r0);
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_a.v;
// CHECK-NEXT: }

// CHECK-LABEL: double indirect_state_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     double y = 3;
// CHECK-NEXT:     pointer_control::Indirect _d_a{&_d_y};
// CHECK-NEXT:     pointer_control::Indirect a{&y};
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t0 = pointer_control::operator_less_pushforward(a, 10., _d_a, 0.);
// CHECK-NEXT:     bool _d_c = _t0.pushforward;
// CHECK-NEXT:     bool c = _t0.value;
// CHECK-NEXT:     return _d_x * y + x * _d_y;
// CHECK-NEXT: }

// CHECK-LABEL: void indirect_state_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _d_y = 0.;
// CHECK-NEXT:     double y = 3;
// CHECK-NEXT:     pointer_control::Indirect _d_a = {&_d_y};
// CHECK-NEXT:     pointer_control::Indirect a{&y};
// CHECK-NEXT:     bool _d_c = false;
// CHECK-NEXT:     bool c = a < 10.;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * y;
// CHECK-NEXT:         _d_y += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     double _r0 = 0.;
// CHECK-NEXT: }

// CHECK-LABEL: double effectful_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     ReadOnly _d_a{0.};
// CHECK-NEXT:     ReadOnly a{0.};
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t0 = a.operator_less_pushforward((x = x * 2), &_d_a, (_d_x = _d_x * 2));
// CHECK-NEXT:     bool _d_c = _t0.pushforward;
// CHECK-NEXT:     bool c = _t0.value;
// CHECK-NEXT:     return _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void effectful_grad(double x, double *_d_x) {
// CHECK-NEXT:     ReadOnly _d_a = {0.};
// CHECK-NEXT:     ReadOnly a{0.};
// CHECK-NEXT:     double _t0 = x;
// CHECK-NEXT:     bool _d_c = false;
// CHECK-NEXT:     bool c = a < (x = x * 2);
// CHECK-NEXT:     *_d_x += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         x = _t0;
// CHECK-NEXT:         double _r_d0 = *_d_x;
// CHECK-NEXT:         *_d_x = 0.;
// CHECK-NEXT:         *_d_x += _r_d0 * 2;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK-LABEL: double ordinary_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<int, int> _t0 = status_pushforward(x, _d_x);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t1 = flag_pushforward(x, _d_x);
// CHECK-NEXT:     return _d_x;
// CHECK-NEXT: }

// CHECK-LABEL: void ordinary_grad(double x, double *_d_x) {
// CHECK-NEXT:     status(x);
// CHECK-NEXT:     flag(x);
// CHECK-NEXT:     *_d_x += 1;
// CHECK-NEXT:     flag_pullback(x, false, _d_x);
// CHECK-NEXT:     status_pullback(x, 0, _d_x);
// CHECK-NEXT: }

// CXX20-LABEL: double category_parameter_darg0(double x, std::partial_ordering c) {
// CXX20-NEXT:     double _d_x = 1;
// CXX20-NEXT:     {{(std::)*}}partial_ordering local = std::partial_ordering::less;
// CXX20-NEXT:     bool _t0 = c < 0;
// CXX20-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CXX20-NEXT: }

// CXX20-LABEL: void category_parameter_grad(double x, std::partial_ordering c, double *_d_x) {
// CXX20-NEXT:     {{(std::)*}}partial_ordering local = std::partial_ordering::less;
// CXX20-NEXT:     if (c < 0)
// CXX20-NEXT:         *_d_x += 1;
// CXX20-NEXT:     else
// CXX20-NEXT:         *_d_x += 2 * 1;
// CXX20-NEXT: }

// CXX20-LABEL: double category_references_darg0(double x, const std::strong_ordering &s, const std::weak_ordering &w) {
// CXX20-NEXT:     double _d_x = 1;
// CXX20-NEXT:     {{(std::)*}}strong_ordering strong0 = s;
// CXX20-NEXT:     {{(std::)*}}weak_ordering weak = w;
// CXX20-NEXT:     bool _t0 = strong0 < 0 && weak < 0;
// CXX20-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CXX20-NEXT: }

// CXX20-LABEL: void category_references_grad(double x, const std::strong_ordering &s, const std::weak_ordering &w, double *_d_x) {
// CXX20-NEXT:     bool _cond0;
// CXX20-NEXT:     double _d_cond0;
// CXX20-NEXT:     _d_cond0 = 0.;
// CXX20-NEXT:     {{(std::)*}}strong_ordering strong0 = s;
// CXX20-NEXT:     {{(std::)*}}weak_ordering weak = w;
// CXX20-NEXT:     if (strong0 < 0)
// CXX20-NEXT:         _cond0 = weak < 0;
// CXX20-NEXT:     bool _cond1 = strong0 < 0 && _cond0;
// CXX20-NEXT:     {
// CXX20-NEXT:         if (strong0 < 0) {
// CXX20-NEXT:             double _r_d0 = _d_cond0;
// CXX20-NEXT:             _d_cond0 = 0.;
// CXX20-NEXT:         }
// CXX20-NEXT:         if (_cond1)
// CXX20-NEXT:             *_d_x += 1;
// CXX20-NEXT:         else
// CXX20-NEXT:             *_d_x += 2 * 1;
// CXX20-NEXT:     }
// CXX20-NEXT: }

// CXX20-LABEL: double named_category_darg0(double x) {
// CXX20-NEXT:     double _d_x = 1;
// CXX20-NEXT:     my::strong_ordering _d_a{_d_x};
// CXX20-NEXT:     my::strong_ordering a{x};
// CXX20-NEXT:     double &_t0 = a.v;
// CXX20-NEXT:     return _d_a.v * x + _t0 * _d_x;
// CXX20-NEXT: }

// CXX20-LABEL: void named_category_grad(double x, double *_d_x) {
// CXX20-NEXT:     my::strong_ordering _d_a = {0.};
// CXX20-NEXT:     my::strong_ordering a{x};
// CXX20-NEXT:     {
// CXX20-NEXT:         _d_a.v += 1 * x;
// CXX20-NEXT:         *_d_x += a.v * 1;
// CXX20-NEXT:     }
// CXX20-NEXT:     *_d_x += _d_a.v;
// CXX20-NEXT: }

// CXX20-LABEL: double array_state_darg0(double x) {
// CXX20-NEXT:     double _d_x = 1;
// CXX20-NEXT:     ArrayState _d_a{{[{][{]}}_d_x, 0}};
// CXX20-NEXT:     ArrayState a{{[{][{]}}x, 0}};
// CXX20-NEXT:     bool _t0 = a < 1;
// CXX20-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CXX20-NEXT: }

// CXX20-LABEL: void array_state_grad(double x, double *_d_x) {
// CXX20-NEXT:     ArrayState _d_a = {{[{][{]}}0., 0}};
// CXX20-NEXT:     ArrayState a{{[{][{]}}x, 0}};
// CXX20-NEXT:     if (a < 1)
// CXX20-NEXT:         *_d_x += 1;
// CXX20-NEXT:     else
// CXX20-NEXT:         *_d_x += 2 * 1;
// CXX20-NEXT:     *_d_x += _d_a.v[0];
// CXX20-NEXT: }

// CXX20-LABEL: double rewritten_operand_darg0(double x) {
// CXX20-NEXT:     double _d_x = 1;
// CXX20-NEXT:     S _d_a{_d_x}, _d_b{0};
// CXX20-NEXT:     S a{x}, b{1};
// CXX20-NEXT:     Flag _d_flag{0};
// CXX20-NEXT:     Flag flag0{true};
// CXX20-NEXT:     clad::ValueAndPushforward<bool, bool> _t0 = flag0.operator_equal_equal_pushforward(!a == b, &_d_flag, false);
// CXX20-NEXT:     bool _d_c = _t0.pushforward;
// CXX20-NEXT:     bool c = _t0.value;
// CXX20-NEXT:     return c ? _d_x : 2 * _d_x;
// CXX20-NEXT: }

// CXX20-LABEL: void rewritten_operand_grad(double x, double *_d_x) {
// CXX20-NEXT:     S _d_a = {0.}, _d_b = {0};
// CXX20-NEXT:     S a{x}, b{1};
// CXX20-NEXT:     Flag _d_flag = {false};
// CXX20-NEXT:     Flag flag0{true};
// CXX20-NEXT:     bool _d_c = false;
// CXX20-NEXT:     bool c = flag0 == !a == b;
// CXX20-NEXT:     bool _cond0 = c;
// CXX20-NEXT:     if (_cond0)
// CXX20-NEXT:         *_d_x += 1;
// CXX20-NEXT:     else
// CXX20-NEXT:         *_d_x += 2 * 1;
// CXX20-NEXT:     *_d_x += _d_a.v;
// CXX20-NEXT: }

// CXX20-LABEL: double unary_operand_darg0(double x) {
// CXX20-NEXT:     double _d_x = 1;
// CXX20-NEXT:     ReadOnly _d_a{0};
// CXX20-NEXT:     ReadOnly a{0};
// CXX20-NEXT:     clad::ValueAndPushforward<bool, bool> _t0 = a.operator_less_pushforward(x++, &_d_a, _d_x);
// CXX20-NEXT:     bool _d_c = _t0.pushforward;
// CXX20-NEXT:     bool c = _t0.value;
// CXX20-NEXT:     return c ? _d_x : 2 * _d_x;
// CXX20-NEXT: }

// CXX20-LABEL: void unary_operand_grad(double x, double *_d_x) {
// CXX20-NEXT:     ReadOnly _d_a = {0};
// CXX20-NEXT:     ReadOnly a{0};
// CXX20-NEXT:     bool _d_c = false;
// CXX20-NEXT:     bool c = a < x++;
// CXX20-NEXT:     bool _cond0 = c;
// CXX20-NEXT:     if (_cond0)
// CXX20-NEXT:         *_d_x += 1;
// CXX20-NEXT:     else
// CXX20-NEXT:         *_d_x += 2 * 1;
// CXX20-NEXT:     x--;
// CXX20-NEXT: }

// CXX20-LABEL: double stored_unequal_darg0(double x) {
// CXX20-NEXT:     double _d_x = 1;
// CXX20-NEXT:     S _d_a{_d_x}, _d_b{0};
// CXX20-NEXT:     S a{x}, b{1};
// CXX20-NEXT:     bool _d_ne = false;
// CXX20-NEXT:     bool ne = !a == b;
// CXX20-NEXT:     return ne ? _d_x : 2 * _d_x;
// CXX20-NEXT: }

// CXX20-LABEL: void stored_unequal_grad(double x, double *_d_x) {
// CXX20-NEXT:     S _d_a = {0.}, _d_b = {0};
// CXX20-NEXT:     S a{x}, b{1};
// CXX20-NEXT:     bool _d_ne = false;
// CXX20-NEXT:     bool ne = !a == b;
// CXX20-NEXT:     bool _cond0 = ne;
// CXX20-NEXT:     if (_cond0)
// CXX20-NEXT:         *_d_x += 1;
// CXX20-NEXT:     else
// CXX20-NEXT:         *_d_x += 2 * 1;
// CXX20-NEXT:     *_d_x += _d_a.v;
// CXX20-NEXT: }

// CXX20-LABEL: double dynamic_state_darg0(double x, const Poly &p) {
// CXX20-NEXT:     double _d_x = 1;
// CXX20-NEXT:     const Poly _d_p;
// CXX20-NEXT:     bool _t0 = p < x;
// CXX20-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CXX20-NEXT: }

// CXX20-LABEL: double inherited_state_darg0(double x, const Derived &p) {
// CXX20-NEXT:     double _d_x = 1;
// CXX20-NEXT:     const Derived _d_p;
// CXX20-NEXT:     bool _t0 = p < x;
// CXX20-NEXT:     return _t0 ? _d_x : 2 * _d_x;
// CXX20-NEXT: }

// CXX20-LABEL: void inherited_state_grad_0(double x, const Derived &p, double *_d_x) {
// CXX20-NEXT:     Derived _d_p = {};
// CXX20-NEXT:     if (p < x)
// CXX20-NEXT:         *_d_x += 1;
// CXX20-NEXT:     else
// CXX20-NEXT:         *_d_x += 2 * 1;
// CXX20-NEXT: }
