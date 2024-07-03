// RUN: %cladclang %s -I%S/../../include -oSTLCustomDerivatives.out | %filecheck %s
// RUN: ./STLCustomDerivatives.out | %filecheck_exec %s

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

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

int main() {
    INIT_DIFFERENTIATE(fnVec1, "u");
    INIT_DIFFERENTIATE(fnVec2, "u");
    INIT_DIFFERENTIATE(fnVec3, "u");
    INIT_DIFFERENTIATE(fnVec4, "u");

    TEST_DIFFERENTIATE(fnVec1, 3, 5); // CHECK-EXEC: {10.00}
    TEST_DIFFERENTIATE(fnVec2, 3, 5); // CHECK-EXEC: {5.00}
    TEST_DIFFERENTIATE(fnVec3, 3, 5); // CHECK-EXEC: {2.00}
    TEST_DIFFERENTIATE(fnVec4, 3, 5); // CHECK-EXEC: {30.00}
}