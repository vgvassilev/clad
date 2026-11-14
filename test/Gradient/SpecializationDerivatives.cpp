// RUN: %cladclang %s -I%S/../../include -oSpecializationDerivatives.out 2>&1 | %filecheck %s
// RUN: ./SpecializationDerivatives.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

// pick<0> and pick<1> are two functions to C++, told apart by a template
// argument the call does not deduce. Their derivatives are not templates: they
// take that argument as a tag, as std::in_place_index_t carries an index, and
// are overloads told apart by it.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>
#include <type_traits>
#include <utility>

// expected-no-diagnostics

template <int I> double pick(double x) { return I == 0 ? x * x : 3. * x; }

double usesPicks(double x) { return pick<0>(x) + pick<1>(x); }

// CHECK: inline clad::ValueAndPushforward<double, double> pick_pushforward(std::integral_constant<int, 0>, double x, double _d_x) {
// CHECK: inline clad::ValueAndPushforward<double, double> pick_pushforward(std::integral_constant<int, 1>, double x, double _d_x) {
// CHECK: void usesPicks_grad(double x, double *_d_x) {
// CHECK: _r0 += 1 * pick_pushforward(std::integral_constant<int, 0>(), x, 1.).pushforward;
// CHECK: _r1 += 1 * pick_pushforward(std::integral_constant<int, 1>(), x, 1.).pushforward;

// CHECK: double usesPicks_darg0(double x) {
// CHECK: pick_pushforward(std::integral_constant<int, 0>(), x, _d_x);
// CHECK: pick_pushforward(std::integral_constant<int, 1>(), x, _d_x);

// Two parameters: reverse mode calls a pullback.
template <int I> double mix(double x, double y) { return I * x * y; }

double usesMix(double x, double y) { return mix<1>(x, y) + mix<2>(x, y); }

// CHECK: void mix_pullback(std::integral_constant<int, 1>, double x, double y, double _d_y0, double *_d_x, double *_d_y) {
// CHECK: void mix_pullback(std::integral_constant<int, 2>, double x, double y, double _d_y0, double *_d_x, double *_d_y) {

// A type argument takes a clad::Tag; a member function template takes its
// tags after the object.
struct One { static constexpr int k = 1; };
struct Two { static constexpr int k = 2; };
template <class T> double scale(double x) { return T::k * x * x; }

struct S {
  template <int I> double get(double x) const { return I * x; }
};

double usesScaleAndGet(double x) {
  S s;
  return scale<One>(x) + scale<Two>(x) + s.get<1>(x) + s.get<5>(x);
}

// CHECK: inline clad::ValueAndPushforward<double, double> scale_pushforward(clad::Tag<One>, double x, double _d_x) {
// CHECK: inline clad::ValueAndPushforward<double, double> scale_pushforward(clad::Tag<Two>, double x, double _d_x) {
// CHECK: s.get_pullback(std::integral_constant<int, 1>(), x, 1, &_d_s, &_r2);
// CHECK: s.get_pullback(std::integral_constant<int, 5>(), x, 1, &_d_s, &_r3);

// A reference return: reverse mode calls a reverse_forw, which takes the tag
// too.
template <int I> double& at(double* p) { return p[I]; }
double usesAt(double* p) { return at<0>(p) * at<1>(p); }

// CHECK: clad::ValueAndAdjoint<double &, double &> at_reverse_forw(std::integral_constant<int, 0>, double *p, double *_d_p) {
// CHECK: void at_pullback(std::integral_constant<int, 0>, double *p, double *_d_p) {

// A custom derivative written for every k<N> is found through its tag.
template <int N> double k(double x) { return N * x * x * x; }
namespace clad {
namespace custom_derivatives {
template <int N>
clad::ValueAndPushforward<double, double>
k_pushforward(::std::integral_constant<int, N>, double x, double d_x) {
  return {N * x * x * x, 3 * N * x * x * d_x};
}
} // namespace custom_derivatives
} // namespace clad
double usesK(double x) { return k<1>(x) + k<2>(x); }

// CHECK: clad::custom_derivatives::k_pushforward(std::integral_constant<int, 1>(), x, 1.)

// So is one for a member function template, its tags after the object.
struct M {
  template <int I> double get(double x) const { return I * x; }
};
namespace clad {
namespace custom_derivatives {
namespace class_functions {
template <int I>
clad::ValueAndPushforward<double, double>
get_pushforward(const M* self, ::std::integral_constant<int, I>, double x,
                const M* d_self, double d_x) {
  return {I * x, 100 * I * d_x};
}
template <int I>
void get_pullback(const M* self, ::std::integral_constant<int, I>, double x,
                  double d_y, M* d_self, double* d_x) {
  *d_x += 100 * I * d_y;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad
double usesCustomGet(double x) {
  M m;
  return m.get<1>(x) + m.get<2>(x);
}

// CHECK: clad::custom_derivatives::class_functions::get_pullback(&m, std::integral_constant<int, 1>(), x, 1, &_d_m, &_r0);

// A null pointer and a pack of values take tags too. A pointer to a function
// is covered in NonTypeTemplateCall.cpp.
template <double* P> double orSelf(double x) {
  if constexpr (P == nullptr)
    return x;
  else
    return *P * x;
}
template <int... Is> double sumOf(double x) { return (Is + ... + 0) * x; }
double usesKinds(double x) {
  return orSelf<nullptr>(x) + sumOf<1, 2>(x) + sumOf<12>(x);
}

// CHECK: orSelf_pushforward(std::integral_constant<double *, nullptr>(), x, 1.)
// CHECK: sumOf_pushforward(std::integral_constant<int, 1>(), std::integral_constant<int, 2>(), x, 1.)
// CHECK: sumOf_pushforward(std::integral_constant<int, 12>(), x, 1.)

// A template argument deduced from the parameters takes no tag: the
// derivatives differ by their parameter types. Nor does an argument that is
// its parameter's default, which no call writes: an enable_if parameter, and
// times<>, which is times<2>. A default that depends on another argument is
// compared with what it gives: scaleN<double, 4> takes a tag, scaleN<double>,
// which is scaleN<double, 8>, does not.
template <class T> T sq(T x) { return x * x; }
template <class T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
T cube(T x) { return x * x * x; }
template <int I = 2> double times(double x) { return I * x; }
template <class T, int N = sizeof(T)> T scaleN(T x) { return N * x; }

double usesDeduced(double x) {
  return sq(x) + sq((float)x) + cube(x) + times<>(x) + times<3>(x) +
         scaleN<double, 4>(x) + scaleN(x);
}

// CHECK: inline clad::ValueAndPushforward<double, double> sq_pushforward(double x, double _d_x) {
// CHECK: inline clad::ValueAndPushforward<float, float> sq_pushforward(float x, float _d_x) {
// CHECK: inline clad::ValueAndPushforward<double, double> cube_pushforward(double x, double _d_x) {
// CHECK: inline clad::ValueAndPushforward<double, double> times_pushforward(double x, double _d_x) {
// CHECK: inline clad::ValueAndPushforward<double, double> times_pushforward(std::integral_constant<int, 3>, double x, double _d_x) {
// CHECK: inline clad::ValueAndPushforward<double, double> scaleN_pushforward(std::integral_constant<int, 4>, double x, double _d_x) {
// CHECK: inline clad::ValueAndPushforward<double, double> scaleN_pushforward(double x, double _d_x) {

// The condition of an `if constexpr` is a constant once the class template is
// instantiated; the instantiation keeps only the branch it takes.
template <int Base> struct Offset {
  double v[2];
  double at(int i) const {
    if constexpr (Base == 1)
      --i;
    return v[i];
  }
};
double usesOffset(double x) {
  Offset<1> o{{x, 2 * x}};
  return o.at(2);
}

// The operand of decltype is not evaluated, so std::declval, which has no
// definition, needs no derivative.
template <class U, class V> double product(U u, V v) {
  using P = decltype(std::declval<U>() * std::declval<V>());
  P p = u * v;
  return p;
}
double usesProduct(double x) { return product(x, 2.5); }

int main() {
  double d = 0, dx = 0, dy = 0;
  auto g = clad::gradient(usesPicks);
  auto f = clad::differentiate(usesPicks, "x");
  g.execute(3, &d);
  double dv = 0;
  clad::differentiate<clad::opts::vector_mode>(usesPicks).execute(3, &dv);
  // x^2 + 3x: 2x + 3
  std::printf("%g %g %g\n", d, f.execute(3), dv); // CHECK-EXEC: 9 9 9

  auto gm = clad::gradient(usesMix);
  gm.execute(2, 5, &dx, &dy);
  // 3xy: 3y, 3x
  std::printf("%g %g\n", dx, dy); // CHECK-EXEC: 15 6

  d = 0;
  auto gs = clad::gradient(usesScaleAndGet);
  auto fs = clad::differentiate(usesScaleAndGet, "x");
  gs.execute(2, &d);
  // 3x^2 + 6x: 6x + 6
  std::printf("%g %g\n", d, fs.execute(2)); // CHECK-EXEC: 18 18

  double p[2] = {3, 5}, dp[2] = {0, 0};
  clad::gradient(usesAt).execute(p, dp);
  std::printf("%g %g\n", dp[0], dp[1]); // CHECK-EXEC: 5 3

  d = 0;
  clad::gradient(usesK).execute(1, &d);
  // x^3 + 2x^3: 9x^2
  std::printf("%g %g\n", d, clad::differentiate(usesK, "x").execute(1));
  // CHECK-EXEC: 9 9

  d = 0;
  clad::gradient(usesCustomGet).execute(1, &d);
  std::printf("%g %g\n", d, clad::differentiate(usesCustomGet, "x").execute(1));
  // CHECK-EXEC: 300 300

  d = 0;
  clad::gradient(usesKinds).execute(1, &d);
  // x + 3x + 12x
  std::printf("%g\n", d); // CHECK-EXEC: 16

  d = 0;
  clad::gradient(usesDeduced).execute(1, &d);
  // 2x^2 + x^3 + 5x + 4x + 8x: 4x + 3x^2 + 5 + 12
  std::printf("%g\n", d); // CHECK-EXEC: 24

  d = 0;
  clad::gradient(usesOffset).execute(1, &d);
  std::printf("%g %g\n", d, clad::differentiate(usesOffset, "x").execute(1));
  // CHECK-EXEC: 2 2

  d = 0;
  clad::gradient(usesProduct).execute(1, &d);
  std::printf("%g\n", d); // CHECK-EXEC: 2.5
}

// RUN: not %cladclang %s -I%S/../../include -fsyntax-only -DNOTYET 2>&1 \
// RUN:   | FileCheck --check-prefix=NOTYET %s
#ifdef NOTYET
// A template template argument has no tag yet: an error, rather than one
// derivative for every argument.
template <class T> struct Box { static constexpr int k = 2; };
template <template <class> class B> double boxed(double x) {
  return B<double>::k * x;
}
double usesBoxed(double x) { return boxed<Box>(x); }
auto d_usesBoxed = clad::differentiate(usesBoxed, "x");
// NOTYET: error: differentiating a call to a specialization by a template template argument is not supported yet
#endif
