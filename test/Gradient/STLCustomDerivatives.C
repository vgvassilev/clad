// XFAIL: asserts
// RUN: %cladclang -std=c++14 %s -I%S/../../include -oSTLCustomDerivatives.out 2>&1 | %filecheck %s
// RUN: ./STLCustomDerivatives.out | %filecheck_exec %s
// RUN: %cladclang -std=c++14 -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oSTLCustomDerivativesWithTBR.out
// RUN: ./STLCustomDerivativesWithTBR.out | %filecheck_exec %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"
#include "../TestUtils.h"
#include "../PrintOverloads.h"

#include <vector>

double fn10(double u, double v) {
    std::vector<double> vec;
    vec.push_back(u);
    vec.push_back(v);
    return vec[0] + vec[1];
}

double fn11(double u, double v) {
    std::vector<double> vec;
    vec.push_back(u);
    vec.push_back(v);
    double &ref = vec[0];
    ref += u;
    return vec[0] + vec[1];
}

namespace clad {
    namespace custom_derivatives {
        namespace class_functions {
            ::std::vector<size_t> size_update_stack;

            template <typename T>
            void resize_reverse_forw(::std::vector<T> *v,
                        typename ::std::vector<T>::size_type sz,
                        ::std::vector<T> *d_v,
                        typename ::std::vector<T>::size_type *d_sz) {
              size_update_stack.push_back(v->size());
              v->resize(sz);
              d_v->resize(sz, 0);
            }

            template <typename T>
            void resize_pullback(::std::vector<T> *v,
                                 typename ::std::vector<T>::size_type sz,
                                 ::std::vector<T> *d_v,
                                 typename ::std::vector<T>::size_type *d_sz) {
              size_t prevSz = size_update_stack.back();
              size_update_stack.pop_back();
              d_v->resize(prevSz);
            }

            template<typename T>
            void clear_reverse_forw(::std::vector<T> *v, ::std::vector<T> *d_v) {
                size_update_stack.push_back(v->size());
                v->clear();
                d_v->clear();
            }

            template<typename T>
            void clear_pullback(::std::vector<T> *v, ::std::vector<T> *d_v) {
              size_t prevSz = size_update_stack.back();
              size_update_stack.pop_back();
              d_v->resize(prevSz, 0);
            }
        }
    }
}

double fn12(double u, double v) {
  double res = 0;
  std::vector<double> vec;
  vec.resize(3);
  {
    double &ref0 = vec[0];
    double &ref1 = vec[1];
    double &ref2 = vec[2];
    ref0 = u;
    ref1 = v;
    ref2 = u + v;
  }
  res = vec[0] + vec[1] + vec[2];
  vec.clear();
  vec.resize(2);
  {
    double &ref0 = vec[0];
    double &ref1 = vec[1];
    ref0 = u;
    ref1 = u;
  }
  res += vec[0] + vec[1];
  return res;
}

int main() {
    double d_i, d_j;
    INIT_GRADIENT(fn10);
    INIT_GRADIENT(fn11);
    INIT_GRADIENT(fn12);

    TEST_GRADIENT(fn10, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);  // CHECK-EXEC: {1.00, 1.00}
    TEST_GRADIENT(fn11, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);  // CHECK-EXEC: {2.00, 1.00}
    TEST_GRADIENT(fn12, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);  // CHECK-EXEC: {4.00, 2.00}
}

// CHECK: void fn10_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:     std::vector<double> _d_vec({});
// CHECK-NEXT:     std::vector<double> vec;
// CHECK-NEXT:     double _t0 = u;
// CHECK-NEXT:     std::vector<double> _t1 = vec;
// CHECK-NEXT:     {{.*}}class_functions::push_back_reverse_forw(&vec, u, &_d_vec, &*_d_u);
// CHECK-NEXT:     double _t2 = v;
// CHECK-NEXT:     std::vector<double> _t3 = vec;
// CHECK-NEXT:     {{.*}}class_functions::push_back_reverse_forw(&vec, v, &_d_vec, &*_d_v);
// CHECK-NEXT:     std::vector<double> _t4 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t5 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, &_r0);
// CHECK-NEXT:     std::vector<double> _t6 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t7 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, &_r1);
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r0 = 0;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&_t4, 0, 1, &_d_vec, &_r0);
// CHECK-NEXT:         {{.*}} _r1 = 0;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&_t6, 1, 1, &_d_vec, &_r1);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         v = _t2;
// CHECK-NEXT:         {{.*}}class_functions::push_back_pullback(&_t3, _t2, &_d_vec, &*_d_v);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         u = _t0;
// CHECK-NEXT:         {{.*}}class_functions::push_back_pullback(&_t1, _t0, &_d_vec, &*_d_u);
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK-NEXT: void fn11_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:     std::vector<double> _d_vec({});
// CHECK-NEXT:     std::vector<double> vec;
// CHECK-NEXT:     double _t0 = u;
// CHECK-NEXT:     std::vector<double> _t1 = vec;
// CHECK-NEXT:     {{.*}}class_functions::push_back_reverse_forw(&vec, u, &_d_vec, &*_d_u);
// CHECK-NEXT:     double _t2 = v;
// CHECK-NEXT:     std::vector<double> _t3 = vec;
// CHECK-NEXT:     {{.*}}class_functions::push_back_reverse_forw(&vec, v, &_d_vec, &*_d_v);
// CHECK-NEXT:     std::vector<double> _t4 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t5 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, &_r0);
// CHECK-NEXT:     double &_d_ref = _t5.adjoint;
// CHECK-NEXT:     double &ref = _t5.value;
// CHECK-NEXT:     double _t6 = ref;
// CHECK-NEXT:     ref += u;
// CHECK-NEXT:     std::vector<double> _t7 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t8 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, &_r1);
// CHECK-NEXT:     std::vector<double> _t9 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t10 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, &_r2);
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r1 = 0;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&_t7, 0, 1, &_d_vec, &_r1);
// CHECK-NEXT:         {{.*}} _r2 = 0;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&_t9, 1, 1, &_d_vec, &_r2);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         ref = _t6;
// CHECK-NEXT:         double _r_d0 = _d_ref;
// CHECK-NEXT:         *_d_u += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r0 = 0;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&_t4, 0, 0, &_d_vec, &_r0);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         v = _t2;
// CHECK-NEXT:         {{.*}}class_functions::push_back_pullback(&_t3, _t2, &_d_vec, &*_d_v);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         u = _t0;
// CHECK-NEXT:         {{.*}}class_functions::push_back_pullback(&_t1, _t0, &_d_vec, &*_d_u);
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn12_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:     std::vector<double> _t1;
// CHECK-NEXT:     double *_d_ref0 = 0;
// CHECK-NEXT:     double *ref0 = {};
// CHECK-NEXT:     std::vector<double> _t3;
// CHECK-NEXT:     double *_d_ref1 = 0;
// CHECK-NEXT:     double *ref1 = {};
// CHECK-NEXT:     std::vector<double> _t5;
// CHECK-NEXT:     double *_d_ref2 = 0;
// CHECK-NEXT:     double *ref2 = {};
// CHECK-NEXT:     double _t7;
// CHECK-NEXT:     double _t8;
// CHECK-NEXT:     double _t9;
// CHECK-NEXT:     std::vector<double> _t19;
// CHECK-NEXT:     double *_d_ref00 = 0;
// CHECK-NEXT:     double *ref00 = {};
// CHECK-NEXT:     std::vector<double> _t21;
// CHECK-NEXT:     double *_d_ref10 = 0;
// CHECK-NEXT:     double *ref10 = {};
// CHECK-NEXT:     double _t23;
// CHECK-NEXT:     double _t24;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     std::vector<double> _d_vec({});
// CHECK-NEXT:     std::vector<double> vec;
// CHECK-NEXT:     std::vector<double> _t0 = vec;
// CHECK-NEXT:     {{.*}}class_functions::resize_reverse_forw(&vec, 3, &_d_vec, &_r0);
// CHECK-NEXT:     {
// CHECK-NEXT:         _t1 = vec;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t2 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, &_r1);
// CHECK-NEXT:         _d_ref0 = &_t2.adjoint;
// CHECK-NEXT:         ref0 = &_t2.value;
// CHECK-NEXT:         _t3 = vec;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t4 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, &_r2);
// CHECK-NEXT:         _d_ref1 = &_t4.adjoint;
// CHECK-NEXT:         ref1 = &_t4.value;
// CHECK-NEXT:         _t5 = vec;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t6 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 2, &_d_vec, &_r3);
// CHECK-NEXT:         _d_ref2 = &_t6.adjoint;
// CHECK-NEXT:         ref2 = &_t6.value;
// CHECK-NEXT:         _t7 = *ref0;
// CHECK-NEXT:         *ref0 = u;
// CHECK-NEXT:         _t8 = *ref1;
// CHECK-NEXT:         *ref1 = v;
// CHECK-NEXT:         _t9 = *ref2;
// CHECK-NEXT:         *ref2 = u + v;
// CHECK-NEXT:     }
// CHECK-NEXT:     double _t10 = res;
// CHECK-NEXT:     std::vector<double> _t11 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t12 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, &_r4);
// CHECK-NEXT:     std::vector<double> _t13 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t14 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, &_r5);
// CHECK-NEXT:     std::vector<double> _t15 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t16 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 2, &_d_vec, &_r6);
// CHECK-NEXT:     res = _t12.value + _t14.value + _t16.value;
// CHECK-NEXT:     std::vector<double> _t17 = vec;
// CHECK-NEXT:     {{.*}}class_functions::clear_reverse_forw(&vec, &_d_vec);
// CHECK-NEXT:     std::vector<double> _t18 = vec;
// CHECK-NEXT:     {{.*}}class_functions::resize_reverse_forw(&vec, 2, &_d_vec, &_r7);
// CHECK-NEXT:     {
// CHECK-NEXT:         _t19 = vec;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t20 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, &_r8);
// CHECK-NEXT:         _d_ref00 = &_t20.adjoint;
// CHECK-NEXT:         ref00 = &_t20.value;
// CHECK-NEXT:         _t21 = vec;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t22 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, &_r9);
// CHECK-NEXT:         _d_ref10 = &_t22.adjoint;
// CHECK-NEXT:         ref10 = &_t22.value;
// CHECK-NEXT:         _t23 = *ref00;
// CHECK-NEXT:         *ref00 = u;
// CHECK-NEXT:         _t24 = *ref10;
// CHECK-NEXT:         *ref10 = u;
// CHECK-NEXT:     }
// CHECK-NEXT:     double _t25 = res;
// CHECK-NEXT:     std::vector<double> _t26 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t27 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, &_r10);
// CHECK-NEXT:     std::vector<double> _t28 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t29 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, &_r11);
// CHECK-NEXT:     res += _t27.value + _t29.value;
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         res = _t25;
// CHECK-NEXT:         double _r_d6 = _d_res;
// CHECK-NEXT:         {{.*}} _r10 = 0;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&_t26, 0, _r_d6, &_d_vec, &_r10);
// CHECK-NEXT:         {{.*}} _r11 = 0;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&_t28, 1, _r_d6, &_d_vec, &_r11);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {
// CHECK-NEXT:             *ref10 = _t24;
// CHECK-NEXT:             double _r_d5 = *_d_ref10;
// CHECK-NEXT:             *_d_ref10 = 0;
// CHECK-NEXT:             *_d_u += _r_d5;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             *ref00 = _t23;
// CHECK-NEXT:             double _r_d4 = *_d_ref00;
// CHECK-NEXT:             *_d_ref00 = 0;
// CHECK-NEXT:             *_d_u += _r_d4;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r9 = 0;
// CHECK-NEXT:             {{.*}}class_functions::operator_subscript_pullback(&_t21, 1, 0, &_d_vec, &_r9);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r8 = 0;
// CHECK-NEXT:             {{.*}}class_functions::operator_subscript_pullback(&_t19, 0, 0, &_d_vec, &_r8);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r7 = 0;
// CHECK-NEXT:         {{.*}}class_functions::resize_pullback(&_t18, 2, &_d_vec, &_r7);
// CHECK-NEXT:     }
// CHECK-NEXT:     {{.*}}class_functions::clear_pullback(&_t17, &_d_vec);
// CHECK-NEXT:     {
// CHECK-NEXT:         res = _t10;
// CHECK-NEXT:         double _r_d3 = _d_res;
// CHECK-NEXT:         _d_res = 0;
// CHECK-NEXT:         {{.*}} _r4 = 0;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&_t11, 0, _r_d3, &_d_vec, &_r4);
// CHECK-NEXT:         {{.*}} _r5 = 0;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&_t13, 1, _r_d3, &_d_vec, &_r5);
// CHECK-NEXT:         {{.*}} _r6 = 0;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&_t15, 2, _r_d3, &_d_vec, &_r6);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {
// CHECK-NEXT:             *ref2 = _t9;
// CHECK-NEXT:             double _r_d2 = *_d_ref2;
// CHECK-NEXT:             *_d_ref2 = 0;
// CHECK-NEXT:             *_d_u += _r_d2;
// CHECK-NEXT:             *_d_v += _r_d2;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             *ref1 = _t8;
// CHECK-NEXT:             double _r_d1 = *_d_ref1;
// CHECK-NEXT:             *_d_ref1 = 0;
// CHECK-NEXT:             *_d_v += _r_d1;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             *ref0 = _t7;
// CHECK-NEXT:             double _r_d0 = *_d_ref0;
// CHECK-NEXT:             *_d_ref0 = 0;
// CHECK-NEXT:             *_d_u += _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r3 = 0;
// CHECK-NEXT:             {{.*}}class_functions::operator_subscript_pullback(&_t5, 2, 0, &_d_vec, &_r3);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r2 = 0;
// CHECK-NEXT:             {{.*}}class_functions::operator_subscript_pullback(&_t3, 1, 0, &_d_vec, &_r2);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r1 = 0;
// CHECK-NEXT:             {{.*}}class_functions::operator_subscript_pullback(&_t1, 0, 0, &_d_vec, &_r1);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r0 = 0;
// CHECK-NEXT:         {{.*}}class_functions::resize_pullback(&_t0, 3, &_d_vec, &_r0);
// CHECK-NEXT:     }
// CHECK-NEXT: }