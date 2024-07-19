// RUN: %cladclang %s -std=c++14 -I%S/../../include -oSTLCustomDerivatives.out | %filecheck %s
// RUN: ./STLCustomDerivatives.out | %filecheck_exec %s

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <array>
#include <vector>
#include <map>

#include "../TestUtils.h"
#include "../PrintOverloads.h"

double fnVec1(double u, double v) {
    std::vector<double> V1(2);
    std::vector<long double> V2(4);
    V1[0] = u;
    V1[1] = v;
    V2[0] = v;
    V2[1] = u;
    double res = V1[0] * V1[1] + V2[0] * V2[1];
    return res;
}

// CHECK: double fnVec1_darg0(double u, double v) {
// CHECK-NEXT:     double _d_u = 1;
// CHECK-NEXT:     double _d_v = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward< ::std::vector<double>, ::std::vector<double> > _t0 = clad::custom_derivatives::class_functions::constructor_pushforward(clad::ConstructorPushforwardTag<std::vector<double> >(), 2, {{.*}}0{{.*}});
// CHECK-NEXT:     std::vector<double> _d_V1(_t0.pushforward);
// CHECK-NEXT:     std::vector<double> V1(_t0.value);
// CHECK-NEXT:     {{.*}}ValueAndPushforward< ::std::vector<long double>, ::std::vector<long double> > _t1 = clad::custom_derivatives::class_functions::constructor_pushforward(clad::ConstructorPushforwardTag<std::vector<long double> >(), 4, {{.*}}0{{.*}});
// CHECK-NEXT:     std::vector<long double> _d_V2(_t1.pushforward);
// CHECK-NEXT:     std::vector<long double> V2(_t1.value);
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double &, double &> _t2 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V1, 0, &_d_V1, 0);
// CHECK-NEXT:     _t2.pushforward = _d_u;
// CHECK-NEXT:     _t2.value = u;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double &, double &> _t3 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V1, 1, &_d_V1, 0);
// CHECK-NEXT:     _t3.pushforward = _d_v;
// CHECK-NEXT:     _t3.value = v;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<long double &, long double &> _t4 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V2, 0, &_d_V2, 0);
// CHECK-NEXT:     _t4.pushforward = _d_v;
// CHECK-NEXT:     _t4.value = v;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<long double &, long double &> _t5 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V2, 1, &_d_V2, 0);
// CHECK-NEXT:     _t5.pushforward = _d_u;
// CHECK-NEXT:     _t5.value = u;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double &, double &> _t6 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V1, 0, &_d_V1, 0);
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double &, double &> _t7 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V1, 1, &_d_V1, 0);
// CHECK-NEXT:     double &_t8 = _t6.value;
// CHECK-NEXT:     double &_t9 = _t7.value;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<long double &, long double &> _t10 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V2, 0, &_d_V2, 0);
// CHECK-NEXT:     {{.*}}ValueAndPushforward<long double &, long double &> _t11 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V2, 1, &_d_V2, 0);
// CHECK-NEXT:     long double &_t12 = _t10.value;
// CHECK-NEXT:     long double &_t13 = _t11.value;
// CHECK-NEXT:     double _d_res = _t6.pushforward * _t9 + _t8 * _t7.pushforward + _t10.pushforward * _t13 + _t12 * _t11.pushforward;
// CHECK-NEXT:     double res = _t8 * _t9 + _t12 * _t13;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

double fnVec2(double u, double v) {
    const std::vector<double> V1(2, u);
    const std::vector<double> V2(2, v);
    return V1[0] * V2[1];
}

// CHECK: double fnVec2_darg0(double u, double v) {
// CHECK-NEXT:     double _d_u = 1;
// CHECK-NEXT:     double _d_v = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward< ::std::vector<double>, ::std::vector<double> > _t0 = clad::custom_derivatives::class_functions::constructor_pushforward(clad::ConstructorPushforwardTag<std::vector<double> >(), 2, u, {{.*}}0, _d_u{{.*}});
// CHECK-NEXT:     const std::vector<double> _d_V1(_t0.pushforward);
// CHECK-NEXT:     const std::vector<double> V1(_t0.value);
// CHECK-NEXT:     {{.*}}ValueAndPushforward< ::std::vector<double>, ::std::vector<double> > _t1 = clad::custom_derivatives::class_functions::constructor_pushforward(clad::ConstructorPushforwardTag<std::vector<double> >(), 2, v, {{.*}}0, _d_v{{.*}});
// CHECK-NEXT:     const std::vector<double> _d_V2(_t1.pushforward);
// CHECK-NEXT:     const std::vector<double> V2(_t1.value);
// CHECK-NEXT:     {{.*}}ValueAndPushforward<const double &, const double &> _t2 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V1, 0, &_d_V1, 0);
// CHECK-NEXT:     {{.*}}ValueAndPushforward<const double &, const double &> _t3 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V2, 1, &_d_V2, 0);
// CHECK-NEXT:     const double _t4 = _t2.value;
// CHECK-NEXT:     const double _t5 = _t3.value;
// CHECK-NEXT:     return _t2.pushforward * _t5 + _t4 * _t3.pushforward;
// CHECK-NEXT: }

double fnVec3(double u, double v) {
    std::vector<double> V1(2, u);
    std::vector<double> &V2 = V1;
    return V1[0] + V2[1];
}

// CHECK: double fnVec3_darg0(double u, double v) {
// CHECK-NEXT:     double _d_u = 1;
// CHECK-NEXT:     double _d_v = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward< ::std::vector<double>, ::std::vector<double> > _t0 = clad::custom_derivatives::class_functions::constructor_pushforward(clad::ConstructorPushforwardTag<std::vector<double> >(), 2, u, {{.*}}0, _d_u{{.*}});
// CHECK-NEXT:     std::vector<double> _d_V1(_t0.pushforward);
// CHECK-NEXT:     std::vector<double> V1(_t0.value);
// CHECK-NEXT:     std::vector<double> &_d_V2 = _d_V1;
// CHECK-NEXT:     std::vector<double> &V2 = V1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double &, double &> _t1 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V1, 0, &_d_V1, 0);
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double &, double &> _t2 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V2, 1, &_d_V2, 0);
// CHECK-NEXT:     return _t1.pushforward + _t2.pushforward;
// CHECK-NEXT: }

double fnVec4(double u, double v) {
    std::vector<double> V{u, v, u * v};
    return V[0] * V[2];
}

// CHECK: double fnVec4_darg0(double u, double v) {
// CHECK-NEXT:     double _d_u = 1;
// CHECK-NEXT:     double _d_v = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward< ::std::vector<double>, ::std::vector<double> > _t0 = clad::custom_derivatives::class_functions::constructor_pushforward(clad::ConstructorPushforwardTag<std::vector<double> >(), {u, v, u * v}, {{.*}}{_d_u, _d_v, _d_u * v + u * _d_v}{{.*}});
// CHECK-NEXT:     std::vector<double> _d_V_t0.pushforward;
// CHECK-NEXT:     std::vector<double> V_t0.value;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double &, double &> _t1 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V, 0, &_d_V, 0);
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double &, double &> _t2 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V, 2, &_d_V, 0);
// CHECK-NEXT:     double &_t3 = _t1.value;
// CHECK-NEXT:     double &_t4 = _t2.value;
// CHECK-NEXT:     return _t1.pushforward * _t4 + _t3 * _t2.pushforward;
// CHECK-NEXT: }

double fnArr1(double x) {
  std::array<double, 3> a;
  a.fill(x);

  for (size_t i = 0; i < a.size(); ++i) {
    a[i] *= i;
  }

  double res = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    res += a[i];
  }

  return res;
}

//CHECK:     double fnArr1_darg0(double x) {
//CHECK-NEXT:         double _d_x = 1;
//CHECK-NEXT:         std::array<double, 3> _d_a;
//CHECK-NEXT:         std::array<double, 3> a;
//CHECK-NEXT:         clad::custom_derivatives::class_functions::fill_pushforward(&a, x, &_d_a, _d_x);
//CHECK-NEXT:         {
//CHECK-NEXT:             size_t _d_i = 0;
//CHECK-NEXT:             for (size_t i = 0; i < a.size(); ++i) {
//CHECK-NEXT:                 clad::ValueAndPushforward<double &, double &> _t0 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&a, i, &_d_a, _d_i);
//CHECK-NEXT:                 double &_t1 = _t0.pushforward;
//CHECK-NEXT:                 double &_t2 = _t0.value;
//CHECK-NEXT:                 _t1 = _t1 * i + _t2 * _d_i;
//CHECK-NEXT:                 _t2 *= i;
//CHECK-NEXT:             }
//CHECK-NEXT:         }
//CHECK-NEXT:         double _d_res = 0;
//CHECK-NEXT:         double res = 0;
//CHECK-NEXT:         {
//CHECK-NEXT:             size_t _d_i = 0;
//CHECK-NEXT:             for (size_t i = 0; i < a.size(); ++i) {
//CHECK-NEXT:                 clad::ValueAndPushforward<double &, double &> _t3 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&a, i, &_d_a, _d_i);
//CHECK-NEXT:                 _d_res += _t3.pushforward;
//CHECK-NEXT:                 res += _t3.value;
//CHECK-NEXT:             }
//CHECK-NEXT:         }
//CHECK-NEXT:         return _d_res;
//CHECK-NEXT:     }

double fnArr2(double x) {
  std::array<double, 2> a{5, x};
  const std::array<double, 3> b{x, 0, x*x};
  return a.back() * b.front() * b.at(2);
}

//CHECK:     double fnArr2_darg0(double x) {
//CHECK-NEXT:         double _d_x = 1;
//CHECK-NEXT:         std::array<double, 2> _d_a{{[{][{]0, _d_x[}][}]}};
//CHECK-NEXT:         std::array<double, 2> a{{[{][{]5, x[}][}]}};
//CHECK-NEXT:         const std::array<double, 3> _d_b{{[{][{]_d_x, 0, _d_x \* x \+ x \* _d_x[}][}]}};
//CHECK-NEXT:         const std::array<double, 3> b{{[{][{]x, 0, x \* x[}][}]}};
//CHECK-NEXT:         clad::ValueAndPushforward<double &, double &> _t0 = clad::custom_derivatives::class_functions::back_pushforward(&a, &_d_a);
//CHECK-NEXT:         clad::ValueAndPushforward<const double &, const double &> _t1 = clad::custom_derivatives::class_functions::front_pushforward(&b, &_d_b);
//CHECK-NEXT:         double &_t2 = _t0.value;
//CHECK-NEXT:         const double _t3 = _t1.value;
//CHECK-NEXT:         clad::ValueAndPushforward<const double &, const double &> _t4 = clad::custom_derivatives::class_functions::at_pushforward(&b, 2, &_d_b, 0);
//CHECK-NEXT:         double _t5 = _t2 * _t3;
//CHECK-NEXT:         const double _t6 = _t4.value;
//CHECK-NEXT:         return (_t0.pushforward * _t3 + _t2 * _t1.pushforward) * _t6 + _t5 * _t4.pushforward;
//CHECK-NEXT:     }

int main() {
    INIT_DIFFERENTIATE(fnVec1, "u");
    INIT_DIFFERENTIATE(fnVec2, "u");
    INIT_DIFFERENTIATE(fnVec3, "u");
    INIT_DIFFERENTIATE(fnVec4, "u");
    INIT_DIFFERENTIATE(fnArr1, "x");
    INIT_DIFFERENTIATE(fnArr2, "x");

    TEST_DIFFERENTIATE(fnVec1, 3, 5); // CHECK-EXEC: {10.00}
    TEST_DIFFERENTIATE(fnVec2, 3, 5); // CHECK-EXEC: {5.00}
    TEST_DIFFERENTIATE(fnVec3, 3, 5); // CHECK-EXEC: {2.00}
    TEST_DIFFERENTIATE(fnVec4, 3, 5); // CHECK-EXEC: {30.00}
    TEST_DIFFERENTIATE(fnArr1, 3); // CHECK-EXEC: {3.00}
    TEST_DIFFERENTIATE(fnArr2, 3); // CHECK-EXEC: {108.00}
}