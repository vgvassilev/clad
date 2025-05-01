// RUN: %cladclang -std=c++17 %s -I%S/../../include -oVariadicCall.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./VariadicCall.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

template <typename... T>
double fn1(double x, T... y) {
    return (x * ... * y);
}

double fn2(double x, double y) {
    // FIXME: Replace `printf` call with a proper variadic function without template
    printf("x is %f, y is %f\n", x, y); // expected-warning {{function 'printf' was not differentiated because clad failed to differentiate it and no suitable overload was found in namespace 'custom_derivatives'}}
                                    // expected-note@-1 {{fallback to numerical differentiation is disabled by the 'CLAD_NO_NUM_DIFF' macro; considering 'printf' as 0}}
    return fn1(x, y);
}

// CHECK: clad::ValueAndPushforward<double, double> fn1_pushforward(double x, double y, double _d_x, double _d_y) {
// CHECK-NEXT:    return {x * y, _d_x * y + x * _d_y};
// CHECK-NEXT:}

// CHECK: double fn2_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     printf("x is %f, y is %f\n", x, y);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = fn1_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double fn3(double x, double y) {
    return fn1(x, y, y);
}

// CHECK: clad::ValueAndPushforward<double, double> fn1_pushforward(double x, double y, double y2, double _d_x, double _d_y, double _d_y2) {
// CHECK-NEXT:    double _t0 = x * y;
// CHECK-NEXT:    return {_t0 * y2, (_d_x * y + x * _d_y) * y2 + _t0 * _d_y2};
// CHECK-NEXT:}

// CHECK: double fn3_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = fn1_pushforward(x, y, y, _d_x, _d_y, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

int main() {
    auto d_fn2 = clad::differentiate(fn2, "x");
    printf("{%.2f}\n", d_fn2.execute(1.0, 2.0)); // CHECK-EXEC: x is 1.000000, y is 2.000000
    // CHECK-EXEC: {2.00}
    auto d_fn3 = clad::differentiate(fn3, "x");
    printf("{%.2f}\n", d_fn3.execute(1.0, 2.0)); // CHECK-EXEC: {4.00}
    return 0;
}