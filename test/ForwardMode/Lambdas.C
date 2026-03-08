// RUN: %cladclang %s -I%S/../../include -oLambdas.out 2>&1 | %filecheck %s
// RUN: ./Lambdas.out | %filecheck_exec %s

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
}
