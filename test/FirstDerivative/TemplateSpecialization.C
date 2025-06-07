// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oTemplateSpecialization.out 2>&1 | %filecheck %s
// RUN: ./TemplateSpecialization.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oTemplateSpecialization.out
// RUN: ./TemplateSpecialization.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

template<int N> double take(std::pair<double, double> a) {return N;}
template<> double take<1>(std::pair<double, double> a) {return a.second;}
template<> double take<0>(std::pair<double, double> a) {return a.first;}

// CHECK: template<> clad::ValueAndPushforward<double, double> take_pushforward<0>(std::pair<double, double> a, std::pair<double, double> _d_a) {
// CHECK-NEXT:     return {a.first, _d_a.first};
// CHECK-NEXT: }
// CHECK: template<> clad::ValueAndPushforward<double, double> take_pushforward<1>(std::pair<double, double> a, std::pair<double, double> _d_a) {
// CHECK-NEXT:     return {a.second, _d_a.second};
// CHECK-NEXT: }


double fn(double u) {
    std::pair<double, double> tmp{u, 2 * u};
    double a = take<0>(tmp);
    double b = take<1>(tmp);
    return a + b;
}

// CHECK: double fn_darg0(double u) {
// CHECK-NEXT:     double _d_u = 1;
// CHECK-NEXT:     std::pair<double, double> _d_tmp{_d_u, 0 * u + 2 * _d_u};
// CHECK-NEXT:     std::pair<double, double> tmp{u, 2 * u};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = take_pushforward(tmp, _d_tmp);
// CHECK-NEXT:     double _d_a = _t0.pushforward;
// CHECK-NEXT:     double a = _t0.value;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t1 = take_pushforward(tmp, _d_tmp);
// CHECK-NEXT:     double _d_b = _t1.pushforward;
// CHECK-NEXT:     double b = _t1.value;
// CHECK-NEXT:     return _d_a + _d_b;
// CHECK-NEXT: }

template<int N> double sum(double A, double B) {return A + B + N;}
template<> double sum<0>(double A, double B) {return A + B * B;}
template<> double sum<1>(double A, double B) {return A * A + B;}


// CHECK: template<> double sum_darg0<0>(double A, double B) {
// CHECK-NEXT:     double _d_A = 1;
// CHECK-NEXT:     double _d_B = 0;
// CHECK-NEXT:     return _d_A + _d_B * B + B * _d_B;
// CHECK-NEXT: }

// CHECK: template<> clad::ValueAndPushforward<double, double> sum_pushforward<0>(double A, double B, double _d_A, double _d_B) {
// CHECK-NEXT:     return {A + B * B, _d_A + _d_B * B + B * _d_B};
// CHECK-NEXT: }

// CHECK: template<> clad::ValueAndPushforward<double, double> sum_pushforward<1>(double A, double B, double _d_A, double _d_B) {
// CHECK-NEXT:     return {A * A + B, _d_A * A + A * _d_A + _d_B};
// CHECK-NEXT: }

// CHECK: template<> clad::ValueAndPushforward<double, double> sum_pushforward<2>(double A, double B, double _d_A, double _d_B) {
// CHECK-NEXT:     return {A + B + 2, _d_A + _d_B + 0};
// CHECK-NEXT: }

double gn(double u) {
    double a = sum<0>(u, u);
    double b = sum<1>(u, u);
    double c = sum<2>(u, u);
    return a + b + c;
}

// CHECK: double gn_darg0(double u) {
// CHECK-NEXT:     double _d_u = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = sum_pushforward(u, u, _d_u, _d_u);
// CHECK-NEXT:     double _d_a = _t0.pushforward;
// CHECK-NEXT:     double a = _t0.value;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t1 = sum_pushforward(u, u, _d_u, _d_u);
// CHECK-NEXT:     double _d_b = _t1.pushforward;
// CHECK-NEXT:     double b = _t1.value;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t2 = sum_pushforward(u, u, _d_u, _d_u);
// CHECK-NEXT:     double _d_c = _t2.pushforward;
// CHECK-NEXT:     double c = _t2.value;
// CHECK-NEXT:     return _d_a + _d_b + _d_c;
// CHECK-NEXT: }

// CHECK: template<> void sum_pullback<0>(double A, double B, double _d_y, double *_d_A, double *_d_B) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_A += _d_y;
// CHECK-NEXT:         *_d_B += _d_y * B;
// CHECK-NEXT:         *_d_B += B * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK-NEXT: template<> void sum_pullback<1>(double A, double B, double _d_y, double *_d_A, double *_d_B) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_A += _d_y * A;
// CHECK-NEXT:         *_d_A += A * _d_y;
// CHECK-NEXT:         *_d_B += _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK-NEXT: template<> void sum_pullback<2>(double A, double B, double _d_y, double *_d_A, double *_d_B) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_A += _d_y;
// CHECK-NEXT:         *_d_B += _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void gn_grad(double u, double *_d_u) {
// CHECK-NEXT:     double _d_a = 0.;
// CHECK-NEXT:     double a = sum<0>(u, u);
// CHECK-NEXT:     double _d_b = 0.;
// CHECK-NEXT:     double b = sum<1>(u, u);
// CHECK-NEXT:     double _d_c = 0.;
// CHECK-NEXT:     double c = sum<2>(u, u);
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_a += 1;
// CHECK-NEXT:         _d_b += 1;
// CHECK-NEXT:         _d_c += 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r4 = 0.;
// CHECK-NEXT:         double _r5 = 0.;
// CHECK-NEXT:         sum_pullback(u, u, _d_c, &_r4, &_r5);
// CHECK-NEXT:         *_d_u += _r4;
// CHECK-NEXT:         *_d_u += _r5;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r2 = 0.;
// CHECK-NEXT:         double _r3 = 0.;
// CHECK-NEXT:         sum_pullback(u, u, _d_b, &_r2, &_r3);
// CHECK-NEXT:         *_d_u += _r2;
// CHECK-NEXT:         *_d_u += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         sum_pullback(u, u, _d_a, &_r0, &_r1);
// CHECK-NEXT:         *_d_u += _r0;
// CHECK-NEXT:         *_d_u += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
    auto a = clad::differentiate(fn, 0);
    printf("f(3)=%.2f; df(3)=%.2f\n", fn(3.0), a.execute(3.0));
    // CHECK-EXEC: f(3)=9.00; df(3)=3.00

    auto b = clad::differentiate(sum<0>, 0);
    printf("g(3, 4)=%.2f; dg(3, 4)=%.2f\n", sum<0>(3.0, 4.0), b.execute(3.0, 4.0));
    // CHECK-EXEC: g(3, 4)=19.00; dg(3, 4)=1.00

    auto c = clad::differentiate(gn);
    printf("g(3)=%.2f; dg(3)=%.2f\n", gn(3.0), c.execute(3.0));
    // CHECK-EXEC: g(3)=32.00; dg(3)=16.00

    double tmp = 0.0;
    auto d = clad::gradient(gn);
    printf("f(3)=%.2f", gn(3.0));
    d.execute(3.0, &tmp);
    printf("; df(3)=%.2f\n", tmp);
    // CHECK-EXEC: f(3)=32.00; df(3)=16.00

    return 0;
}
