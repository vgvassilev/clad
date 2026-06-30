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
    double c = 5.0;
    auto _f = [c](double t) { return t * c; };
    return _f(x);
}

double fn6(double x) {
    double c = 2.0;
    auto _f = [c](double t) { return t * t * c; };
    return _f(x);
}

double fn7(double x) {
    double d = 4.0;
    auto _f = [&d](double t) { return t * d; };
    return _f(x);
}

double fn8(double x) {
    double a = 3.0, b = 7.0;
    auto _f = [a, b](double t) { return a * t + b; };
    return _f(x);
}

double fn9(double x) {
    double c = 2.0;
    auto _f = [c](double t) { return t * c; };
    c = 100.0;
    return _f(x);
}

double fn10(double x) {
    double c = x * x;
    auto _f = [c](double) { return c; };
    return _f(x);
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
    printf("Result is = %.2f\n", fn5_dx.execute(3)); // CHECK-EXEC: Result is = 5.00

    auto fn6_dx = clad::differentiate(fn6, 0);
    printf("Result is = %.2f\n", fn6_dx.execute(3)); // CHECK-EXEC: Result is = 12.00

    auto fn7_dx = clad::differentiate(fn7, 0);
    printf("Result is = %.2f\n", fn7_dx.execute(3)); // CHECK-EXEC: Result is = 4.00

    auto fn8_dx = clad::differentiate(fn8, 0);
    printf("Result is = %.2f\n", fn8_dx.execute(3)); // CHECK-EXEC: Result is = 3.00

    auto fn9_dx = clad::differentiate(fn9, 0);
    printf("Result is = %.2f\n", fn9_dx.execute(3)); // CHECK-EXEC: Result is = 2.00

    auto fn10_dx = clad::differentiate(fn10, 0);
    printf("Result is = %.2f\n", fn10_dx.execute(3)); // CHECK-EXEC: Result is = 6.00
}
