// RUN: %cladclang %s -I%S/../../include -oLambdas.out 2>&1 | %filecheck %s
// RUN: ./Lambdas.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr -Xclang -plugin-arg-clad -Xclang -disable-va %s -I%S/../../include -oLambdas.out
// RUN: ./Lambdas.out | %filecheck_exec %s
// UNSUPPORTED: clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"

#include <type_traits>
#include <utility>

double f1(double i, double j) {
  auto _f = [] (double t) {
    return t*t + 1.0;
  };
  return i + _f(j);
}

// CHECK: void f1_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     auto _f0 = [](double t) {
// CHECK-NEXT:         return t * t + 1.;
// CHECK-NEXT:     };
// CHECK-NEXT:     auto _d__f = [](double t, double _d_y, double *_d_t) {
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_t += _d_y * t;
// CHECK-NEXT:             *_d_t += t * _d_y;
// CHECK-NEXT:         }
// CHECK-NEXT:     };
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1;
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         _d__f(j, 1, &_r0);
// CHECK-NEXT:         *_d_j += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f2(double i, double j) {
  auto _f = [] (double t, double k) {
    double a = t*k;
    return a;
  };
  double x = _f(i + j, i);
  return x;
}

// CHECK: void f2_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     auto _f0 = [](double t, double k) {
// CHECK-NEXT:         double a = t * k;
// CHECK-NEXT:         return a;
// CHECK-NEXT:     };
// CHECK-NEXT:     auto _d__f = [](double t, double k, double _d_y, double *_d_t, double *_d_k) {
// CHECK-NEXT:         double _d_a = 0.;
// CHECK-NEXT:         double a = t * k;
// CHECK-NEXT:         _d_a += _d_y;
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_t += _d_a * k;
// CHECK-NEXT:             *_d_k += t * _d_a;
// CHECK-NEXT:         }
// CHECK-NEXT:     };
// CHECK-NEXT:     double _d_x = 0.;
// CHECK-NEXT:     double x = _f0(i + j, i);
// CHECK-NEXT:     _d_x += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         _d__f(i + j, i, _d_x, &_r0, &_r1);
// CHECK-NEXT:         *_d_i += _r0;
// CHECK-NEXT:         *_d_j += _r0;
// CHECK-NEXT:         *_d_i += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f3(double i, double j) {
  double c = 3.0;
  auto _f = [c] (double t) {
    return t * t;
  };
  return i + _f(j);
}

// CHECK: void f3_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _d_t = 0.;
// CHECK-NEXT:     double _d_c = 0.;
// CHECK-NEXT:     double c = 3.;
// CHECK-NEXT:     double _t0 = c;
// CHECK-NEXT:     auto _f0 = [c](double t) {
// CHECK-NEXT:         return t * t;
// CHECK-NEXT:     };
// CHECK-NEXT:     auto _d__f = [](double t, double _d_y, double *_d_t0, double c0, double *_d_c0) {
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_t0 += _d_y * t;
// CHECK-NEXT:             *_d_t0 += t * _d_y;
// CHECK-NEXT:         }
// CHECK-NEXT:     };
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1;
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         _d__f(j, 1, &_r0, _t0, &_d_t);
// CHECK-NEXT:         *_d_j += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_c += _d_t;
// CHECK-NEXT: }

double f4(double i, double j) {
  double c = 3.0;
  double d = 4.0;
  auto _f = [c, &d] (double t) {
    return t * 2.0;
  };
  return i + _f(j);
}

// CHECK: void f4_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _d_t = 0.;
// CHECK-NEXT:     double _d_c = 0.;
// CHECK-NEXT:     double c = 3.;
// CHECK-NEXT:     double _d_d = 0.;
// CHECK-NEXT:     double d = 4.;
// CHECK-NEXT:     double _t0 = c;
// CHECK-NEXT:     auto _f0 = [c, &d](double t) {
// CHECK-NEXT:         return t * 2.;
// CHECK-NEXT:     };
// CHECK-NEXT:     auto _d__f = [](double t, double _d_y, double *_d_t0, double c0, double *_d_c0, double d0, double *_d_d0) {
// CHECK-NEXT:         *_d_t0 += _d_y * 2.;
// CHECK-NEXT:     };
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1;
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         _d__f(j, 1, &_r0, _t0, &_d_t, d, &_d_d);
// CHECK-NEXT:         *_d_j += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_c += _d_t;
// CHECK-NEXT: }

const auto _global_f = [](double t) {
  return t * 3.0 + 1.0;
};

// CHECK: inline constexpr void operator_call_grad(double t, double *_d_t) const {
// CHECK-NEXT:     *_d_t += 1 * 3.;
// CHECK-NEXT: }

double f6(double i) {
  auto _outer = [](double x) {
    auto _inner = [](double y) {
      return y * y;
    };
    return _inner(x) + 2.0 * x;
  };
  return _outer(i);
}

// CHECK: void f6_grad(double i, double *_d_i) {
// CHECK-NEXT:     auto _outer0 = [](double x) {
// CHECK-NEXT:         auto _inner0 = [](double y) {
// CHECK-NEXT:             return y * y;
// CHECK-NEXT:         };
// CHECK-NEXT:         return _inner0(x) + 2. * x;
// CHECK-NEXT:     };
// CHECK-NEXT:     auto _d__outer = [](double x, double _d_y, double *_d_x) {
// CHECK-NEXT:         auto _inner0 = [](double y) {
// CHECK-NEXT:             return y * y;
// CHECK-NEXT:         };
// CHECK-NEXT:         auto _d__inner = [](double y, double _d_y0, double *_d_y1) {
// CHECK-NEXT:             {
// CHECK-NEXT:                 *_d_y1 += _d_y0 * y;
// CHECK-NEXT:                 *_d_y1 += y * _d_y0;
// CHECK-NEXT:             }
// CHECK-NEXT:         };
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r0 = 0.;
// CHECK-NEXT:             _d__inner(x, _d_y, &_r0);
// CHECK-NEXT:             *_d_x += _r0;
// CHECK-NEXT:             *_d_x += 2. * _d_y;
// CHECK-NEXT:         }
// CHECK-NEXT:     };
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         _d__outer(i, 1, &_r0);
// CHECK-NEXT:         *_d_i += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }


double f7(double i, double j) {
  // The unnamed parameter needs a synthesized name in the cloned primal.
  auto _f = [](double t, double) {
    return t * t;
  };
  return _f(i, j);
}

// CHECK: void f7_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     auto _f0 = [](double t, double arg) {
// CHECK-NEXT:         return t * t;
// CHECK-NEXT:     };
// CHECK-NEXT:     auto _d__f = [](double t, double arg, double _d_y, double *_d_t, double *_d_arg) {
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_t += _d_y * t;
// CHECK-NEXT:             *_d_t += t * _d_y;
// CHECK-NEXT:         }
// CHECK-NEXT:     };
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         _d__f(i, j, 1, &_r0, &_r1);
// CHECK-NEXT:         *_d_i += _r0;
// CHECK-NEXT:         *_d_j += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f8(double i, double j) {
  double c = 2.0;
  auto _f = [c](double t) { return t * c; };
  return i + _f(j);
}

// CHECK: void f8_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _d_t = 0.;
// CHECK-NEXT:     double _d_c = 0.;
// CHECK-NEXT:     double c = 2.;
// CHECK-NEXT:     double _t0 = c;
// CHECK-NEXT:     auto _f0 = [c](double t) {
// CHECK-NEXT:         return t * c;
// CHECK-NEXT:     };
// CHECK-NEXT:     auto _d__f = [](double t, double _d_y, double *_d_t0, double c0, double *_d_c0) {
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_t0 += _d_y * c0;
// CHECK-NEXT:             *_d_c0 += t * _d_y;
// CHECK-NEXT:         }
// CHECK-NEXT:     };
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1;
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         _d__f(j, 1, &_r0, _t0, &_d_t);
// CHECK-NEXT:         *_d_j += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_c += _d_t;
// CHECK-NEXT: }

double f9(double i, double j) {
  double d = 4.0;
  auto _f = [&d](double t) { return t * d; };
  return i + _f(j);
}

// CHECK: void f9_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _d_d = 0.;
// CHECK-NEXT:     double d = 4.;
// CHECK-NEXT:     auto _f0 = [&d](double t) {
// CHECK-NEXT:         return t * d;
// CHECK-NEXT:     };
// CHECK-NEXT:     auto _d__f = [](double t, double _d_y, double *_d_t, double d0, double *_d_d0) {
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_t += _d_y * d0;
// CHECK-NEXT:             *_d_d0 += t * _d_y;
// CHECK-NEXT:         }
// CHECK-NEXT:     };
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1;
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         _d__f(j, 1, &_r0, d, &_d_d);
// CHECK-NEXT:         *_d_j += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// c is captured twice, with a different value each time.
double f10(double i, double j) {
  double c = i;
  auto f = [c](double t) { return c * t; };
  c = 2 * i;
  auto g = [c](double t) { return c * t; };
  return f(j) + g(j);
}

// CHECK: void f10_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK: double _t0 = c;
// CHECK: double _t1 = c;
// CHECK: _d_f(j, 1, &_r0, _t0, &_d_t);
// CHECK: _d_g(j, 1, &_r1, _t1, &_d_t0);

// d is read through the capture when g is called, after a write to it: TBR
// has to keep that value for the reverse sweep.
double f11(double x, double y) {
  double d = y;
  auto g = [&d](double t) { return d * t; };
  d = 10;
  double r = g(x);
  d = 20;
  return r + d;
}

// Implicit captures, by reference and by copy.
double f12(double x, double y) {
  auto g = [&](double t) { return x * t; };
  auto h = [=](double t) { return y * t; };
  return g(y) + h(x);
}

// The lambda writes through its by-reference capture, once per call.
double f13(double x) {
  double c = 1;
  auto g = [&c](double t) { c *= t; };
  g(x);
  g(x);
  return c;
}

// CHECK: void f13_grad(double x, double *_d_x) {
// CHECK: double _t0 = c;
// CHECK-NEXT: g(x);
// CHECK-NEXT: double _t1 = c;
// CHECK-NEXT: g(x);
// CHECK: c = _t1;
// CHECK-NEXT: _d_g(x, &_r1, c, &_d_c);
// CHECK: c = _t0;
// CHECK-NEXT: _d_g(x, &_r0, c, &_d_c);

struct Acc { double a; };

double f14(double x) {
  Acc p{1};
  auto g = [&p](double t) { p.a = p.a * t + 1; };
  g(x);
  g(x);
  return p.a;
}

// Each way a lambda can write its capture: an increment, a non-const member
// function, an assignment operator, and a non-const reference parameter.
struct Scaled {
  double a;
  void scale(double t) { a *= t; }
};
void scaleBy(double& v, double t) { v *= t; }

double f15(double x) {
  double n = 1;
  auto inc = [&n](double t) {
    ++n;
    return n * t;
  };
  Scaled s{1};
  auto member = [&s](double t) { s.scale(t); };
  Scaled q{1};
  auto assign = [&q](double t) { q = Scaled{q.a * t}; };
  double r = 1;
  auto byRef = [&r](double t) { scaleBy(r, t); };
  double y = inc(x);
  y += inc(x);
  member(x);
  assign(x);
  byRef(x);
  return y + s.a + q.a + r;
}

// A lambda passed to a function template, which calls it in a loop: the
// forward sweep runs in a closure passed in its place, which records each
// call's argument, and the reverse sweep replays the calls last first.
template <class F> void each(int n, const F& f) {
  for (int i = 0; i < n; ++i)
    f(i);
}

double f16(double x, double y) {
  double s = 1;
  each(3, [&](int i) { s *= x + i * y; });
  return s;
}

// CHECK: void f16_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<int> _t1 = {};
// CHECK-NEXT:     clad::tape<double> _t2 = {};
// CHECK-NEXT:     double _d_s = 0.;
// CHECK-NEXT:     double s = 1;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = 0;
// CHECK-NEXT:     each(3, [&](int i0) {
// CHECK-NEXT:         clad::push(_t1, i);
// CHECK-NEXT:         i = i0;
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t2, s);
// CHECK-NEXT:         s *= x + i * y;
// CHECK-NEXT:     });
// CHECK-NEXT:     _d_s += 1;
// CHECK-NEXT:     while (_t0)
// CHECK-NEXT:         {
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     {
// CHECK-NEXT:                         s = clad::pop(_t2);
// CHECK-NEXT:                         double _r_d0 = _d_s;
// CHECK-NEXT:                         _d_s = 0.;
// CHECK-NEXT:                         _d_s += _r_d0 * (x + i * y);
// CHECK-NEXT:                         *_d_x += s * _r_d0;
// CHECK-NEXT:                         *_d_y += i * s * _r_d0;
// CHECK-NEXT:                     }
// CHECK-NEXT:                 }
// CHECK-NEXT:                 _t0--;
// CHECK-NEXT:             }
// CHECK-NEXT:             i = clad::pop(_t1);
// CHECK-NEXT:         }
// CHECK-NEXT: }

// AMReX's constexpr_for, called once per iteration of a loop.
template <int I, int N, class F> void unroll(const F& f) {
  if constexpr (I < N) {
    f(std::integral_constant<int, I>());
    unroll<I + 1, N>(f);
  }
}

double f17(double x) {
  double a[2] = {x, 1};
  for (int k = 0; k < 2; ++k)
    unroll<0, 2>([&](int i) { a[i] = a[i] * a[1 - i] + x; });
  return a[0] + a[1];
}

// The call is made again with the template arguments it wrote, and without
// the arguments it left to their defaults.
template <class T, class F> void eachT(const F& f, T n = 3) {
  for (T i = 0; i < n; ++i)
    f(i);
}

double f18(double x) {
  double s = 0;
  eachT<int>([&](int i) { s += x * i; });
  return s;
}

// CHECK: void f18_grad(double x, double *_d_x) {
// CHECK: eachT<int>([&](int i0) {

// The other arguments, copies of objects that hold only integers here, are
// passed again.
struct Range {
  int n;
};
template <class F> void eachIn(Range r, const F& f) {
  for (int i = 0; i < r.n; ++i)
    f(i);
}

double f19(double x) {
  Range r{3};
  double s = 0;
  eachIn(r, [&](int i) { s += x * i * i; });
  return s;
}

// CHECK: void f19_grad(double x, double *_d_x) {
// CHECK: eachIn(r, [&](int i0) {

// A function that passes its callable on, forwarded, is differentiated as
// the function it passes it to.
template <class F> void eachFwd(int n, F&& f) { each(n, std::forward<F>(f)); }

double f20(double x) {
  double s = 0;
  eachFwd(2, [&](int i) { s += x * (i + 1); });
  return s;
}

// CHECK: void f20_grad(double x, double *_d_x) {
// CHECK: eachFwd(2, [&](int i0) {

// A function that passes its callable on by value, twice: each copy is the
// callable, and each call to it is recorded.
template <class F> void eachVal(int n, F f) {
  for (int i = 0; i < n; ++i)
    f(i);
}
template <class F> void eachTwice(int n, const F& f) {
  eachVal(n, f);
  eachVal(n, f);
}

double f22(double x) {
  double s = 0;
  eachTwice(2, [&](int i) { s += x * (i + 1); });
  return s;
}

// CHECK: void f22_grad(double x, double *_d_x) {
// CHECK: eachTwice(2, [&](int i0) {

// Written through an accessor in the lambda, r is stored element by element,
// as in a loop, not as a whole before each call.
struct Pair {
  double v[2];
  double& operator[](int i) { return v[i]; }
  double operator[](int i) const { return v[i]; }
};
Pair scaled(const Pair& a, double s) {
  Pair r{};
  each(2, [&](int i) { r[i] = a[i] * s; });
  return r;
}
double f21(double x) {
  Pair a{{x, 2}};
  Pair c = scaled(a, x);
  return c[0] + c[1];
}

// CHECK: void scaled_pullback(const Pair &a, double s, Pair _d_y, Pair *_d_a, double *_d_s) {
// CHECK-NOT: clad::push(_t{{[0-9]+}}, r);
// CHECK: constructor_pullback(std::move(r), &_d_y, &_d_r);

// With control flow of its own, the body is not taken for a loop's: TBR stays
// off, and r is stored as a whole before each write.
Pair scaledIf(const Pair& a, double s) {
  Pair r{};
  each(2, [&](int i) { if (i) r[i] = a[i] * s; });
  return r;
}
double f23(double x) {
  Pair a{{x, 2}};
  Pair c = scaledIf(a, x);
  return c[0] + c[1];
}

// CHECK: void scaledIf_pullback(const Pair &a, double s, Pair _d_y, Pair *_d_a, double *_d_s) {
// CHECK: clad::push(_t{{[0-9]+}}, r);

int main() {
  auto df1 = clad::gradient(f1);
  double di = 0, dj = 0;
  df1.execute(3, 4, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 1.00 8.00

  auto df2 = clad::gradient(f2);
  di = 0, dj = 0;
  df2.execute(1, 1, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 3.00 1.00

  auto df3 = clad::gradient(f3);
  di = 0, dj = 0;
  df3.execute(1, 2, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 1.00 4.00

  auto df4 = clad::gradient(f4);
  di = 0, dj = 0;
  df4.execute(1, 2, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 1.00 2.00

  auto df5 = clad::gradient(_global_f);
  double dt = 0; 
  df5.execute(4, &dt);
  printf("%.2f\n", dt);                       // CHECK-EXEC: 3.00

  auto df6 = clad::gradient(f6);
  di = 0;
  df6.execute(3, &di);
  printf("%.2f\n", di);                       // CHECK-EXEC: 8.00

  auto df7 = clad::gradient(f7);
  di = 0, dj = 0;
  df7.execute(3, 4, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 6.00 0.00

  auto df8 = clad::gradient(f8);
  di = 0, dj = 0;
  df8.execute(3, 4, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 1.00 2.00

  auto df9 = clad::gradient(f9);
  di = 0, dj = 0;
  df9.execute(3, 4, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 1.00 4.00

  di = 0, dj = 0;
  auto d_f10 = clad::gradient(f10);
  d_f10.execute(2, 3, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 9.00 6.00

  di = 0, dj = 0;
  auto d_f11 = clad::gradient(f11);
  d_f11.execute(2, 3, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 10.00 0.00

  di = 0, dj = 0;
  auto d_f12 = clad::gradient(f12);
  d_f12.execute(2, 3, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 6.00 4.00

  di = 0;
  auto d_f13 = clad::gradient(f13);
  d_f13.execute(2, &di);
  printf("%.2f\n", di);                       // CHECK-EXEC: 4.00

  di = 0;
  auto d_f14 = clad::gradient(f14);
  d_f14.execute(2, &di);
  printf("%.2f\n", di);                       // CHECK-EXEC: 5.00

  di = 0;
  auto d_f15 = clad::gradient(f15);
  d_f15.execute(2, &di);
  printf("%.2f\n", di);                       // CHECK-EXEC: 8.00

  di = 0, dj = 0;
  auto d_f16 = clad::gradient(f16);
  d_f16.execute(2, 1, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 26.00 20.00

  di = 0;
  auto d_f17 = clad::gradient(f17);
  d_f17.execute(2, &di);
  printf("%.2f\n", di);                       // CHECK-EXEC: 254.00

  di = 0;
  auto d_f18 = clad::gradient(f18);
  d_f18.execute(2, &di);
  printf("%.2f\n", di);                       // CHECK-EXEC: 3.00

  di = 0;
  auto d_f19 = clad::gradient(f19);
  d_f19.execute(2, &di);
  printf("%.2f\n", di);                       // CHECK-EXEC: 5.00

  di = 0;
  auto d_f20 = clad::gradient(f20);
  d_f20.execute(2, &di);
  printf("%.2f\n", di);                       // CHECK-EXEC: 3.00

  di = 0;
  auto d_f22 = clad::gradient(f22);
  d_f22.execute(2, &di);
  printf("%.2f\n", di);                       // CHECK-EXEC: 6.00

  di = 0;
  auto d_f21 = clad::gradient(f21);
  d_f21.execute(3, &di);
  printf("%.2f\n", di);                       // CHECK-EXEC: 8.00

  di = 0;
  auto d_f23 = clad::gradient(f23);
  d_f23.execute(3, &di);
  printf("%.2f\n", di);                       // CHECK-EXEC: 2.00
}
