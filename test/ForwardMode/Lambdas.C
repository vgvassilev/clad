// RUN: %cladclang %s -I%S/../../include -oLambdas.out 2>&1 | %filecheck %s
// RUN: ./Lambdas.out | %filecheck_exec %s
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"

double fn0(double x) {
    auto _f = [](double _x) {
        return _x*_x;
    };
    return _f(x) + 1;
}

double fn1(double x, double y) {
    auto _f = [](double _x, double _y) {
        return _x + _y;
    };
    return _f(x*x, x+2) + y;
}

auto fn2_global = [](double _x) {
    return _x * _x * 3.0;
};

double fn2(double x) {
    return fn2_global(x);
}

double fn3(double x, double y) {
    auto _f = [](double _x, double _y) {
        return _x * _x * _y;
    };
    return _f(x, y);
}

double fn4(double x) {
    auto _f = [](double _x) {
        auto _g = [](double _y) {
            return _y + _y;
        };
        return _g(_x) * _x;
    };
    return _f(x);
}

double fn5(double x) {
    auto _f = [](double _x) {
        double s = 0;
        for (int k = 0; k < 3; ++k)
            s += _x * _x;
        return s;
    };
    return _f(x);
}

double fn6(double x) {
    double c = 5.0;
    auto _f = [c](double t) { return t * c; };
    return _f(x);
}

double fn7(double x) {
    double c = 2.0;
    auto _f = [c](double t) { return t * t * c; };
    return _f(x);
}

double fn8(double x) {
    double d = 4.0;
    auto _f = [&d](double t) { return t * d; };
    return _f(x);
}

double fn9(double x) {
    double a = 3.0, b = 7.0;
    auto _f = [a, b](double t) { return a * t + b; };
    return _f(x);
}

double fn10(double x) {
    double c = 2.0;
    auto _f = [c](double t) { return t * c; };
    c = 100.0;
    return _f(x);
}

double fn11(double x) {
    double c = x * x;
    auto _f = [c](double) { return c; };
    return _f(x);
}

double fn12(double x) {
    double c = 2.0;
    auto _f = [c](double t) {
        double s = 0;
        for (int k = 0; k < 3; ++k)
            s += t * c;
        return s;
    };
    return _f(x);
}

// c is captured twice, with a different value each time.
double fn13(double i, double j) {
  double c = i;
  auto f = [c](double t) { return c * t; };
  c = 2 * i;
  auto g = [c](double t) { return c * t; };
  return f(j) + g(j);
}

double fn14(double x) {
  auto f = [x](double t) { return x * t; };
  return f(x);
}

// CHECK: double fn14_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _t0 = x, _d_t = _d_x, auto f = [x](double t) {
// CHECK-NEXT:         return x * t;
// CHECK-NEXT:     }, _d_f = [](double t, double _d_t, double x0, double _d_x0) -> clad::ValueAndPushforward<double, double> {
// CHECK-NEXT:         return {x0 * t, _d_x0 * t + x0 * _d_t};
// CHECK-NEXT:     };
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t1 = _d_f(x, _d_x, _t0, _d_t);
// CHECK-NEXT:     return _t1.pushforward;
// CHECK-NEXT: }

double fn15(double x) {
  auto f = [&x](double t) { return x * t; };
  return f(x);
}

double fn16(double x) {
  if (double c = x * x) {
    auto f = [c](double t) { return c * t; };
    return f(x);
  }
  return 0;
}

int main() {
    auto fn0_dx = clad::differentiate(fn0, 0);
    printf("Result is = %.2f\n", fn0_dx.execute(7)); // CHECK-EXEC: Result is = 14.00
    printf("Result is = %.2f\n", fn0_dx.execute(-1)); // CHECK-EXEC: Result is = -2.00

    auto fn1_dx = clad::differentiate(fn1, 0);
    printf("Result is = %.2f\n", fn1_dx.execute(7, 1)); // CHECK-EXEC: Result is = 15.00
    printf("Result is = %.2f\n", fn1_dx.execute(-1, 1)); // CHECK-EXEC: Result is = -1.00

    auto fn2_dx = clad::differentiate(fn2, 0);
    printf("Result is = %.2f\n", fn2_dx.execute(7)); // CHECK-EXEC: Result is = 42.00
    printf("Result is = %.2f\n", fn2_dx.execute(-1)); // CHECK-EXEC: Result is = -6.00

    auto fn3_dx = clad::differentiate(fn3, 0);
    printf("Result is = %.2f\n", fn3_dx.execute(7, 1)); // CHECK-EXEC: Result is = 14.00
    printf("Result is = %.2f\n", fn3_dx.execute(-1, 1)); // CHECK-EXEC: Result is = -2.00

    auto fn4_dx = clad::differentiate(fn4, 0);
    printf("Result is = %.2f\n", fn4_dx.execute(7)); // CHECK-EXEC: Result is = 28.00
    printf("Result is = %.2f\n", fn4_dx.execute(-1)); // CHECK-EXEC: Result is = -4.00

    auto fn5_dx = clad::differentiate(fn5, 0);
    printf("Result is = %.2f\n", fn5_dx.execute(7)); // CHECK-EXEC: Result is = 42.00
    printf("Result is = %.2f\n", fn5_dx.execute(-1)); // CHECK-EXEC: Result is = -6.00

    auto fn6_dx = clad::differentiate(fn6, 0);
    printf("Result is = %.2f\n", fn6_dx.execute(3)); // CHECK-EXEC: Result is = 5.00

    auto fn7_dx = clad::differentiate(fn7, 0);
    printf("Result is = %.2f\n", fn7_dx.execute(3)); // CHECK-EXEC: Result is = 12.00

    auto fn8_dx = clad::differentiate(fn8, 0);
    printf("Result is = %.2f\n", fn8_dx.execute(3)); // CHECK-EXEC: Result is = 4.00

    auto fn9_dx = clad::differentiate(fn9, 0);
    printf("Result is = %.2f\n", fn9_dx.execute(3)); // CHECK-EXEC: Result is = 3.00

    auto fn10_dx = clad::differentiate(fn10, 0);
    printf("Result is = %.2f\n", fn10_dx.execute(3)); // CHECK-EXEC: Result is = 2.00

    auto fn11_dx = clad::differentiate(fn11, 0);
    printf("Result is = %.2f\n", fn11_dx.execute(3)); // CHECK-EXEC: Result is = 6.00

    auto fn12_dx = clad::differentiate(fn12, 0);
    printf("Result is = %.2f\n", fn12_dx.execute(2)); // CHECK-EXEC: Result is = 6.00

    auto fn13_dx = clad::differentiate(fn13, 0);
    printf("Result is = %.2f\n", fn13_dx.execute(2, 3)); // CHECK-EXEC: Result is = 9.00

    auto fn14_dx = clad::differentiate(fn14, 0);
    printf("Result is = %.2f\n", fn14_dx.execute(1.3)); // CHECK-EXEC: Result is = 2.60

    auto fn15_dx = clad::differentiate(fn15, 0);
    printf("Result is = %.2f\n", fn15_dx.execute(1.3)); // CHECK-EXEC: Result is = 2.60

    auto fn16_dx = clad::differentiate(fn16, 0);
    printf("Result is = %.2f\n", fn16_dx.execute(2)); // CHECK-EXEC: Result is = 12.00
}
