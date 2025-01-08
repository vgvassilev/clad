// RUN: %cladclang -std=c++14 %s -I%S/../../include -oSTLCustomDerivatives.out | %filecheck %s
// RUN: ./STLCustomDerivatives.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <array>
#include <vector>
#include <map>
#include <tuple>

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
// CHECK-NEXT:     std::vector<double> _d_V(_t0.pushforward);
// CHECK-NEXT:     std::vector<double> V(_t0.value);
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double &, double &> _t1 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V, 0, &_d_V, 0);
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double &, double &> _t2 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&V, 2, &_d_V, 0);
// CHECK-NEXT:     double &_t3 = _t1.value;
// CHECK-NEXT:     double &_t4 = _t2.value;
// CHECK-NEXT:     return _t1.pushforward * _t4 + _t3 * _t2.pushforward;
// CHECK-NEXT: }

double fnVec5(double x, double y) {
    std::vector<double> v;
    
    v.reserve(10);

    double res = x*v.capacity();

    v.push_back(x);
    v.shrink_to_fit();
    res += x*v.capacity();

    return res; // 11x
}

// CHECK:      double fnVec5_darg0(double x, double y) {
// CHECK-NEXT:          double _d_x = 1;
// CHECK-NEXT:          double _d_y = 0;
// CHECK-NEXT:          std::vector<double> _d_v;
// CHECK-NEXT:          std::vector<double> v;
// CHECK-NEXT:          {{.*}}reserve_pushforward(&v, 10, &_d_v, 0);
// CHECK-NEXT:          {{.*}}ValueAndPushforward< ::std::size_t, ::std::size_t> _t0 = {{.*}}capacity_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}} &_t1 = _t0.value;
// CHECK-NEXT:          double _d_res = _d_x * _t1 + x * _t0.pushforward;
// CHECK-NEXT:          double res = x * _t1;
// CHECK-NEXT:          {{.*}}push_back_pushforward(&v, x, &_d_v, _d_x);
// CHECK-NEXT:          {{.*}}shrink_to_fit_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}}ValueAndPushforward< ::std::size_t, ::std::size_t> _t2 = {{.*}}capacity_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}} &_t3 = _t2.value;
// CHECK-NEXT:          _d_res += _d_x * _t3 + x * _t2.pushforward;
// CHECK-NEXT:          res += x * _t3;
// CHECK-NEXT:          return _d_res;
// CHECK-NEXT:      }

double fnVec6(double x, double y) { 
    std::vector<double> v(3, y);
    
    v.pop_back();
    double res = v.size()*x; // res = 2x

    v.erase(v.begin());
    res += v.size()*x; // res = 3x

    std::vector<double> w;
    w = v;
    w.clear();
    res += w.size()*x + v.size()*x; // res = 4x

    w.insert(w.end(), 5);
    res += w.size()*x; // res = 5x

    w.insert(w.end(), {y, x, y});
    w.insert(w.end(), v.begin(), v.end());
    if (w[0] == 5 && w[1] == y && w[2] == x && w[3] == y && v.back() == w.back()) { // should always be true
        res += w[2]; // res = 6x
    }

    w.assign(2, y);
    res += (w[0] == y && w[1] == y)*x; // res = 7x

    v[0] = x;
    w.assign(v.begin(), v.end());
    res += w[0]; // res = 8x;

    w.assign({3*x, 2*x, 4*x});
    res += w[1]; // res = 10x;

    return res; // 10x
}

// CHECK:      double fnVec6_darg0(double x, double y) {
// CHECK-NEXT:          double _d_x = 1;
// CHECK-NEXT:          double _d_y = 0;
// CHECK-NEXT:          {{.*}}ValueAndPushforward< ::std::vector<double>, ::std::vector<double> > _t0 = {{.*}}constructor_pushforward(clad::ConstructorPushforwardTag<std::vector<double> >(), 3, y{{.*}}, 0, _d_y{{.*}});
// CHECK-NEXT:          std::vector<double> _d_v(_t0.pushforward);
// CHECK-NEXT:          std::vector<double> v(_t0.value);
// CHECK-NEXT:          {{.*}}pop_back_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}}ValueAndPushforward< ::std::size_t, ::std::size_t> _t1 = {{.*}}size_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}} &_t2 = _t1.value;
// CHECK-NEXT:          double _d_res = _t1.pushforward * x + _t2 * _d_x;
// CHECK-NEXT:          double res = _t2 * x;
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t3 = {{.*}}begin_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t4 = {{.*}}erase_pushforward(&v, {{.*}}_t3.value{{.*}}, &_d_v, {{.*}}_t3.pushforward{{.*}});
// CHECK-NEXT:          {{.*}}ValueAndPushforward< ::std::size_t, ::std::size_t> _t5 = {{.*}}size_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}} &_t6 = _t5.value;
// CHECK-NEXT:          _d_res += _t5.pushforward * x + _t6 * _d_x;
// CHECK-NEXT:          res += _t6 * x;
// CHECK-NEXT:          std::vector<double> _d_w;
// CHECK-NEXT:          std::vector<double> w;
// CHECK-NEXT:          {{.*}}ValueAndPushforward< ::std::vector<double> &, ::std::vector<double> &> _t7 = {{.*}}operator_equal_pushforward(&w, v, &_d_w, _d_v);
// CHECK-NEXT:          {{.*}}clear_pushforward(&w, &_d_w);
// CHECK-NEXT:          {{.*}}ValueAndPushforward< ::std::size_t, ::std::size_t> _t8 = {{.*}}size_pushforward(&w, &_d_w);
// CHECK-NEXT:          {{.*}} &_t9 = _t8.value;
// CHECK-NEXT:          {{.*}}ValueAndPushforward< ::std::size_t, ::std::size_t> _t10 = {{.*}}size_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}} &_t11 = _t10.value;
// CHECK-NEXT:          _d_res += _t8.pushforward * x + _t9 * _d_x + _t10.pushforward * x + _t11 * _d_x;
// CHECK-NEXT:          res += _t9 * x + _t11 * x;
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t12 = {{.*}}end_pushforward(&w, &_d_w);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t13 = {{.*}}insert_pushforward(&w, {{.*}}_t12.value{{.*}}, 5, &_d_w, {{.*}}_t12.pushforward{{.*}}, 0);
// CHECK-NEXT:          {{.*}}ValueAndPushforward< ::std::size_t, ::std::size_t> _t14 = {{.*}}size_pushforward(&w, &_d_w);
// CHECK-NEXT:          {{.*}} &_t15 = _t14.value;
// CHECK-NEXT:          _d_res += _t14.pushforward * x + _t15 * _d_x;
// CHECK-NEXT:          res += _t15 * x;
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t16 = {{.*}}end_pushforward(&w, &_d_w);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t17 = {{.*}}insert_pushforward(&w, {{.*}}_t16.value{{.*}}, {y, x, y}, &_d_w, {{.*}}_t16.pushforward{{.*}}, {_d_y, _d_x, _d_y});
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t18 = {{.*}}end_pushforward(&w, &_d_w);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t19 = {{.*}}begin_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t20 = {{.*}}end_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t21 = {{.*}}insert_pushforward(&w, {{.*}}_t18.value{{.*}}, {{.*}}_t19.value{{.*}}, {{.*}}_t20.value{{.*}}, &_d_w, {{.*}}_t18.pushforward{{.*}}, {{.*}}_t19.pushforward{{.*}}, {{.*}}_t20.pushforward{{.*}});
// CHECK-NEXT:          if (w[0] == 5 && w[1] == y && w[2] == x && w[3] == y && v.back() == w.back()) {
// CHECK-NEXT:              {{.*}}ValueAndPushforward<{{.*}}> _t22 = {{.*}}operator_subscript_pushforward(&w, 2, &_d_w, 0);
// CHECK-NEXT:              _d_res += _t22.pushforward;
// CHECK-NEXT:              res += _t22.value;
// CHECK-NEXT:          }
// CHECK-NEXT:          {{.*}}assign_pushforward(&w, 2, y, &_d_w, 0, _d_y);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t23 = {{.*}}operator_subscript_pushforward(&w, 0, &_d_w, 0);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t24 = {{.*}}operator_subscript_pushforward(&w, 1, &_d_w, 0);
// CHECK-NEXT:          bool _t25 = ((_t23.value == y) && (_t24.value == y));
// CHECK-NEXT:          _d_res += false * x + _t25 * _d_x;
// CHECK-NEXT:          res += _t25 * x;
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t26 = {{.*}}operator_subscript_pushforward(&v, 0, &_d_v, 0);
// CHECK-NEXT:          _t26.pushforward = _d_x;
// CHECK-NEXT:          _t26.value = x;
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t27 = {{.*}}begin_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t28 = {{.*}}end_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}}assign_pushforward(&w, {{.*}}_t27.value{{.*}}, {{.*}}_t28.value{{.*}}, &_d_w, {{.*}}_t27.pushforward{{.*}}, {{.*}}_t28.pushforward{{.*}});
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t29 = {{.*}}operator_subscript_pushforward(&w, 0, &_d_w, 0);
// CHECK-NEXT:          _d_res += _t29.pushforward;
// CHECK-NEXT:          res += _t29.value;
// CHECK-NEXT:          {{.*}}assign_pushforward(&w, {3 * x, 2 * x, 4 * x}, &_d_w, {0 * x + 3 * _d_x, 0 * x + 2 * _d_x, 0 * x + 4 * _d_x});
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t30 = {{.*}}operator_subscript_pushforward(&w, 1, &_d_w, 0);
// CHECK-NEXT:          _d_res += _t30.pushforward;
// CHECK-NEXT:          res += _t30.value;
// CHECK-NEXT:          return _d_res;
// CHECK-NEXT:      }

double fnVec7(double x, double y) {
    std::vector<double> v;
    for (size_t i = 0; i < 3; ++i) {
        float fx = x;
        v.push_back(fx);
    }
    double res = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = i * v.at(i);
        res += v.at(i);
    }

    const std::vector<double> v2 = v;

    // result is the same as res, that is: 3x
    return res + v.front() + v.back() - v[v.size()-1] + v2.at(0) + v2.front() + v2.back() - v2[v2.size()-1];
}

// CHECK:      double fnVec7_darg0(double x, double y) {
// CHECK-NEXT:          double _d_x = 1;
// CHECK-NEXT:          double _d_y = 0;
// CHECK-NEXT:          std::vector<double> _d_v;
// CHECK-NEXT:          std::vector<double> v;
// CHECK-NEXT:          {
// CHECK-NEXT:              size_t _d_i = 0;
// CHECK-NEXT:              for (size_t i = 0; i < 3; ++i) {
// CHECK-NEXT:                  float _d_fx = _d_x;
// CHECK-NEXT:                  float fx = x;
// CHECK-NEXT:                  {{.*}}push_back_pushforward(&v, static_cast<float &&>(fx), &_d_v, static_cast<float &&>(_d_fx));
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          double _d_res = 0;
// CHECK-NEXT:          double res = 0;
// CHECK-NEXT:          {
// CHECK-NEXT:              size_t _d_i = 0;
// CHECK-NEXT:              for (size_t i = 0; i < v.size(); ++i) {
// CHECK-NEXT:                  {{.*}}ValueAndPushforward<{{.*}}> _t0 = {{.*}}operator_subscript_pushforward(&v, i, &_d_v, _d_i);
// CHECK-NEXT:                  {{.*}}ValueAndPushforward<{{.*}}> _t1 = {{.*}}at_pushforward(&v, i, &_d_v, _d_i);
// CHECK-NEXT:                  double &_t2 = _t1.value;
// CHECK-NEXT:                  _t0.pushforward = _d_i * _t2 + i * _t1.pushforward;
// CHECK-NEXT:                  _t0.value = i * _t2;
// CHECK-NEXT:                  {{.*}}ValueAndPushforward<{{.*}}> _t3 = {{.*}}at_pushforward(&v, i, &_d_v, _d_i);
// CHECK-NEXT:                  _d_res += _t3.pushforward;
// CHECK-NEXT:                  res += _t3.value;
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          const std::vector<double> _d_v2 = _d_v;
// CHECK-NEXT:          const std::vector<double> v2 = v;
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t4 = {{.*}}front_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t5 = {{.*}}back_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}}ValueAndPushforward< ::std::size_t, ::std::size_t> _t6 = {{.*}}size_pushforward(&v, &_d_v);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<{{.*}}> _t7 = {{.*}}operator_subscript_pushforward(&v, _t6.value - 1, &_d_v, _t6.pushforward - 0);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<const double &, const double &> _t8 = {{.*}}at_pushforward(&v2, 0, &_d_v2, 0);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<const double &, const double &> _t9 = {{.*}}front_pushforward(&v2, &_d_v2);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<const double &, const double &> _t10 = {{.*}}back_pushforward(&v2, &_d_v2);
// CHECK-NEXT:          {{.*}}ValueAndPushforward< ::std::size_t, ::std::size_t> _t11 = {{.*}}size_pushforward(&v2, &_d_v2);
// CHECK-NEXT:          {{.*}}ValueAndPushforward<const double &, const double &> _t12 = {{.*}}operator_subscript_pushforward(&v2, _t11.value - 1, &_d_v2, _t11.pushforward - 0);
// CHECK-NEXT:          return _d_res + _t4.pushforward + _t5.pushforward - _t7.pushforward + _t8.pushforward + _t9.pushforward + _t10.pushforward - _t12.pushforward;
// CHECK-NEXT:      }

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

auto pack(double x) {
    return std::make_tuple(x, 2*x, 3*x);
}

double fnTuple1(double x, double y) { 
    double u, v = 288*x, w;

    std::tie(u, v, w) = pack(x+y);

    return v;
} // = 2x + 2y

//CHECK:      clad::ValueAndPushforward<{{.*}}> pack_pushforward({{.*}}) {
//CHECK-NEXT:          clad::ValueAndPushforward<tuple<double, double, double>, tuple<double, double, double> > _t0 = clad::custom_derivatives::std::make_tuple_pushforward(x, 2 * x, 3 * x, _d_x, 0 * x + 2 * _d_x, 0 * x + 3 * _d_x);
//CHECK-NEXT:          return {_t0.value, _t0.pushforward};
//CHECK-NEXT:      }

//CHECK:      double fnTuple1_darg0(double x, double y) {
//CHECK-NEXT:          double _d_x = 1;
//CHECK-NEXT:          double _d_y = 0;
//CHECK-NEXT:          double _d_u, _d_v = 0 * x + 288 * _d_x, _d_w;
//CHECK-NEXT:          double u, v = 288 * x, w;
//CHECK-NEXT:          clad::ValueAndPushforward<tuple<double &, double &, double &>, tuple<double &, double &, double &> > _t0 = clad::custom_derivatives::std::tie_pushforward(u, v, w, _d_u, _d_v, _d_w);
//CHECK-NEXT:          clad::ValueAndPushforward<{{.*}}> _t1 = pack_pushforward(x + y, _d_x + _d_y);
//CHECK-NEXT:          clad::ValueAndPushforward<{{.*}}> _t2 = clad::custom_derivatives::class_functions::operator_equal_pushforward(&_t0.value, static_cast<std::tuple<double, double, double> &&>(_t1.value), &_t0.pushforward, static_cast<std::tuple<double, double, double> &&>(_t1.pushforward));
//CHECK-NEXT:          return _d_v;
//CHECK-NEXT:      }

int main() {
    INIT_DIFFERENTIATE(fnVec1, "u");
    INIT_DIFFERENTIATE(fnVec2, "u");
    INIT_DIFFERENTIATE(fnVec3, "u");
    INIT_DIFFERENTIATE(fnVec4, "u");
    INIT_DIFFERENTIATE(fnVec5, "x");
    INIT_DIFFERENTIATE(fnVec6, "x");
    INIT_DIFFERENTIATE(fnVec7, "x");
    INIT_DIFFERENTIATE(fnArr1, "x");
    INIT_DIFFERENTIATE(fnArr2, "x");
    INIT_DIFFERENTIATE(fnTuple1, "x");

    TEST_DIFFERENTIATE(fnVec1, 3, 5); // CHECK-EXEC: {10.00}
    TEST_DIFFERENTIATE(fnVec2, 3, 5); // CHECK-EXEC: {5.00}
    TEST_DIFFERENTIATE(fnVec3, 3, 5); // CHECK-EXEC: {2.00}
    TEST_DIFFERENTIATE(fnVec4, 3, 5); // CHECK-EXEC: {30.00}
    TEST_DIFFERENTIATE(fnVec5, 3, 4); // CHECK-EXEC: {11.00}
    TEST_DIFFERENTIATE(fnVec6, 3, 4); // CHECK-EXEC: {10.00}
    TEST_DIFFERENTIATE(fnVec7, 3, 4); // CHECK-EXEC: {3.00}
    TEST_DIFFERENTIATE(fnArr1, 3); // CHECK-EXEC: {3.00}
    TEST_DIFFERENTIATE(fnArr2, 3); // CHECK-EXEC: {108.00}
    TEST_DIFFERENTIATE(fnTuple1, 3, 4); // CHECK-EXEC: {2.00}
}
