// RUN: %cladclang %s -I%S/../../include -oLambdas.out 2>&1 | %filecheck %s
// RUN: ./Lambdas.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr -Xclang -plugin-arg-clad -Xclang -disable-va %s -I%S/../../include -oLambdas.out
// RUN: ./Lambdas.out | %filecheck_exec %s
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"

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
// CHECK-NEXT:     double _d_c = 0.;
// CHECK-NEXT:     double c = 3.;
// CHECK-NEXT:     auto _f0 = [c](double t) {
// CHECK-NEXT:         return t * t;
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

double f4(double i, double j) {
  double c = 3.0;
  double d = 4.0;
  auto _f = [c, &d] (double t) {
    return t * 2.0;
  };
  return i + _f(j);
}

// CHECK: void f4_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _d_c = 0.;
// CHECK-NEXT:     double c = 3.;
// CHECK-NEXT:     double _d_d = 0.;
// CHECK-NEXT:     double d = 4.;
// CHECK-NEXT:     auto _f0 = [c, &d](double t) {
// CHECK-NEXT:         return t * 2.;
// CHECK-NEXT:     };
// CHECK-NEXT:     auto _d__f = [](double t, double _d_y, double *_d_t) {
// CHECK-NEXT:         *_d_t += _d_y * 2.;
// CHECK-NEXT:     };
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1;
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         _d__f(j, 1, &_r0);
// CHECK-NEXT:         *_d_j += _r0;
// CHECK-NEXT:     }
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
// CHECK-NEXT:         auto _inner = [](double y) {
// CHECK-NEXT:             return y * y;
// CHECK-NEXT:         };
// CHECK-NEXT:         return _inner(x) + 2. * x;
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
}
