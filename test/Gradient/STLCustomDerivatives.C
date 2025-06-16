// RUN: %cladclang -std=c++14 -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oSTLCustomDerivatives.out 2>&1 | %filecheck %s
// RUN: ./STLCustomDerivatives.out | %filecheck_exec %s
// RUN: %cladclang -std=c++14 %s -I%S/../../include -oSTLCustomDerivativesWithTBR.out
// RUN: ./STLCustomDerivativesWithTBR.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"
#include "../TestUtils.h"
#include "../PrintOverloads.h"

#include <array>
#include <vector>
#include <memory>

double fn1(double u, double v) {
    std::vector<double> vec;
    vec.push_back(u);
    vec.push_back(v);
    return vec[0] + vec[1];
}

double fn2(double u, double v) {
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
                        typename ::std::vector<T>::size_type d_sz) {
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

double fn3(double u, double v) {
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

double fn4(double u, double v) {
  double res = u;
  std::vector<double>::allocator_type allocator;
  typename ::std::vector<double>::size_type count = 3;
  std::vector<double> vec(count, u, allocator);
  return vec[0] + vec[1] + vec[2];
}

double fn5(double x, double y) {
  std::vector<double> a;
  a.push_back(x);
  a.push_back(x);
  a[1] = x*x;
  return a[1];
}

double fn6(double x, double y) {
  std::array<double, 3> a;
  a.fill(x);

  double res = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    res += a.at(i);
  }

  return res;
}

double fn7(double x, double y) {
  std::array<double, 2> a;
  a[0] = 5;
  a[1] = y;
  std::array<double, 3> _b;
  _b[0] = x;
  _b[1] = 0;
  _b[2] = x*x;
  const std::array<double, 3> b = _b;
  return a.back() * b.front() * b.at(2) + b[1];
}

double fn8(double x, double y) {
    std::array<double, 50> a;
    a.fill(y+x+x);
    return a[49]+a[3];
}

double fn9(double x, double y) {
    std::array<double, 2> a;
    a[1] = 2*x;
    return a[1];
}

double fn10(double x, double y) {
    std::vector<double> v;
    for (size_t i = 0; i < 3; ++i) {
        v.push_back(x);
    }
    double res = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        res += v.at(i);
    }

    v.assign(3, 0);
    v.assign(2, y);

    return res + v[0] + v[1] + v[2]; // 3x+2y
}

double fn11(double x, double y) {
    std::vector<double> v;
    
    v.reserve(10);

    double res = x*v.capacity();

    v.push_back(x);
    v.shrink_to_fit();
    res += y*v.capacity() + x*v.size();

    return res; // 11x+y
}

double fn12(double x, double y) {
  std::vector<double> a;
  a.push_back(0);
  a[0] = x*x;
  return a[0];
}

double fn13(double u, double v) {
    std::vector<double>::allocator_type alloc;
    std::vector<double> ls({u, v}, alloc);
    return ls[1] - 2 * ls[0];
}

double fn14(double u, double v) {
    std::vector<double>::allocator_type alloc;
    for (int i = 0; i < 3; ++i) {
      std::vector<double> ls({u, v}, alloc);
      ls[1] += ls[0];
      u = ls[1];
    }
    return u;
}

double fn15(double d, double e) {
  double *p = new double(d);
  std::unique_ptr<double> up(p);
  *up += 5 * e;
  return *up;
}

double fn16(double u, double v) {
    std::vector<double>::allocator_type alloc;
    double prod = 1;
    for (int i = 3; i >= 1; --i) {
        std::vector<double> vec(i, v + u, alloc);
        prod *= vec[i - 1];
    }
    return prod;
}

double fn17(double u, double v) {
    for (int i = 0; i < 3; ++i) {
      std::vector<double> ls{u, v};
      ls[1] += ls[0];
      u = ls[1];
    }
    return u;
}

constexpr int nVals = 2;

struct Session {
   std::vector<float> vec = std::vector<float>(nVals);
   float *arr = vec.data();
};

float fn18(Session const *session, float const *tensor_x, float *tensor_theory_params) {
   Session const &sess = session[0];
   for (int id = 0; id < nVals; id++) {
      sess.arr[id] = tensor_x[id] * tensor_theory_params[0];
   }
   float out = 0.;
   for (int id = 0; id < nVals; id++) {
      out += std::exp(-sess.arr[id]);
   }
   return out;
}

int main() {
    double d_i, d_j;
    INIT_GRADIENT(fn1);
    INIT_GRADIENT(fn2);
    INIT_GRADIENT(fn3);
    INIT_GRADIENT(fn4);
    INIT_GRADIENT(fn5);
    INIT_GRADIENT(fn6);
    INIT_GRADIENT(fn7);
    INIT_GRADIENT(fn8);
    INIT_GRADIENT(fn9);
    INIT_GRADIENT(fn10);
    INIT_GRADIENT(fn11);
    INIT_GRADIENT(fn12);
    INIT_GRADIENT(fn13);
    INIT_GRADIENT(fn14);
    INIT_GRADIENT(fn15);
    INIT_GRADIENT(fn16);
    INIT_GRADIENT(fn17);

    TEST_GRADIENT(fn1, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);  // CHECK-EXEC: {1.00, 1.00}
    TEST_GRADIENT(fn2, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);  // CHECK-EXEC: {2.00, 1.00}
    TEST_GRADIENT(fn3, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);  // CHECK-EXEC: {4.00, 2.00}
    TEST_GRADIENT(fn4, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);  // CHECK-EXEC: {3.00, 0.00}
    TEST_GRADIENT(fn5, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);  // CHECK-EXEC: {6.00, 0.00}
    TEST_GRADIENT(fn6, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);  // CHECK-EXEC: {3.00, 0.00}
    TEST_GRADIENT(fn7, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);  // CHECK-EXEC: {108.00, 27.00}
    TEST_GRADIENT(fn8, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);  // CHECK-EXEC: {4.00, 2.00}
    TEST_GRADIENT(fn9, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);  // CHECK-EXEC: {2.00, 0.00}
    TEST_GRADIENT(fn10, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);  // CHECK-EXEC: {3.00, 2.00}
    TEST_GRADIENT(fn11, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);  // CHECK-EXEC: {11.00, 1.00}
    TEST_GRADIENT(fn12, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);  // CHECK-EXEC: {6.00, 0.00}
    TEST_GRADIENT(fn13, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);  // CHECK-EXEC: {-2.00, 1.00}
    TEST_GRADIENT(fn14, /*numOfDerivativeArgs=*/2, 1, 1, &d_i, &d_j);  // CHECK-EXEC: {1.00, 3.00}
    TEST_GRADIENT(fn15, /*NumOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);  // CHECK-EXEC: {1.00, 5.00}
    TEST_GRADIENT(fn16, /*NumOfDerivativeArgs=*/2, 3, 1, &d_i, &d_j);  // CHECK-EXEC: {48.00, 48.00}
    TEST_GRADIENT(fn17, /*numOfDerivativeArgs=*/2, 1, 1, &d_i, &d_j);  // CHECK-EXEC: {1.00, 3.00}
    auto d_fn18 = clad::gradient(fn18, "tensor_theory_params");
}

// CHECK: void fn1_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:     std::vector<double> vec;
// CHECK-NEXT:     std::vector<double> _d_vec = {};
// CHECK-NEXT:     clad::zero_init(_d_vec);
// CHECK-NEXT:     std::vector<double> _t0 = vec;
// CHECK-NEXT:     {{.*}}class_functions::push_back_reverse_forw(&vec, u, &_d_vec, *_d_u);
// CHECK-NEXT:     std::vector<double> _t1 = vec;
// CHECK-NEXT:     {{.*}}class_functions::push_back_reverse_forw(&vec, v, &_d_vec, *_d_v);
// CHECK-NEXT:     std::vector<double> _t2 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t3 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     std::vector<double> _t4 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t5 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {
// CHECK-NEXT:         size_type _r0 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t2;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&vec, 0, 1, &_d_vec, &_r0);
// CHECK-NEXT:         size_type _r1 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t4;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&vec, 1, 1, &_d_vec, &_r1);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         vec = _t1;
// CHECK-NEXT:         {{.*}}class_functions::push_back_pullback(&vec, v, &_d_vec, &*_d_v);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         vec = _t0;
// CHECK-NEXT:         {{.*}}class_functions::push_back_pullback(&vec, u, &_d_vec, &*_d_u);
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK-NEXT: void fn2_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:     std::vector<double> vec;
// CHECK-NEXT:     std::vector<double> _d_vec = {};
// CHECK-NEXT:     clad::zero_init(_d_vec);
// CHECK-NEXT:     std::vector<double> _t0 = vec;
// CHECK-NEXT:     {{.*}}class_functions::push_back_reverse_forw(&vec, u, &_d_vec, *_d_u);
// CHECK-NEXT:     std::vector<double> _t1 = vec;
// CHECK-NEXT:     {{.*}}class_functions::push_back_reverse_forw(&vec, v, &_d_vec, *_d_v);
// CHECK-NEXT:     std::vector<double> _t2 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t3 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     double &_d_ref = _t3.adjoint;
// CHECK-NEXT:     double &ref = _t3.value;
// CHECK-NEXT:     double _t4 = ref;
// CHECK-NEXT:     ref += u;
// CHECK-NEXT:     std::vector<double> _t5 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t6 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     std::vector<double> _t7 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t8 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r1 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t5;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&vec, 0, 1, &_d_vec, &_r1);
// CHECK-NEXT:         {{.*}} _r2 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t7;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&vec, 1, 1, &_d_vec, &_r2);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         ref = _t4;
// CHECK-NEXT:         double _r_d0 = _d_ref;
// CHECK-NEXT:         *_d_u += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r0 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t2;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&vec, 0, 0., &_d_vec, &_r0);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         vec = _t1;
// CHECK-NEXT:         clad::custom_derivatives::class_functions::push_back_pullback(&vec, v, &_d_vec, &*_d_v);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         vec = _t0;
// CHECK-NEXT:         clad::custom_derivatives::class_functions::push_back_pullback(&vec, u, &_d_vec, &*_d_u);
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn3_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:     std::vector<double> _t1;
// CHECK-NEXT:     double *_d_ref0 = nullptr;
// CHECK-NEXT:     double *ref0 = {};
// CHECK-NEXT:     std::vector<double> _t3;
// CHECK-NEXT:     double *_d_ref1 = nullptr;
// CHECK-NEXT:     double *ref1 = {};
// CHECK-NEXT:     std::vector<double> _t5;
// CHECK-NEXT:     double *_d_ref2 = nullptr;
// CHECK-NEXT:     double *ref2 = {};
// CHECK-NEXT:     double _t7;
// CHECK-NEXT:     double _t8;
// CHECK-NEXT:     double _t9;
// CHECK-NEXT:     std::vector<double> _t19;
// CHECK-NEXT:     double *_d_ref00 = nullptr;
// CHECK-NEXT:     double *ref00 = {};
// CHECK-NEXT:     std::vector<double> _t21;
// CHECK-NEXT:     double *_d_ref10 = nullptr;
// CHECK-NEXT:     double *ref10 = {};
// CHECK-NEXT:     double _t23;
// CHECK-NEXT:     double _t24;
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     std::vector<double> vec;
// CHECK-NEXT:     std::vector<double> _d_vec = {};
// CHECK-NEXT:     clad::zero_init(_d_vec);
// CHECK-NEXT:     std::vector<double> _t0 = vec;
// CHECK-NEXT:     {{.*}}class_functions::resize_reverse_forw(&vec, 3, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {
// CHECK-NEXT:         _t1 = vec;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t2 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:         _d_ref0 = &_t2.adjoint;
// CHECK-NEXT:         ref0 = &_t2.value;
// CHECK-NEXT:         _t3 = vec;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t4 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:         _d_ref1 = &_t4.adjoint;
// CHECK-NEXT:         ref1 = &_t4.value;
// CHECK-NEXT:         _t5 = vec;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t6 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 2, &_d_vec, {{0U|0UL|0}});
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
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t12 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     std::vector<double> _t13 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t14 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     std::vector<double> _t15 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t16 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 2, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     res = _t12.value + _t14.value + _t16.value;
// CHECK-NEXT:     std::vector<double> _t17 = vec;
// CHECK-NEXT:     {{.*}}class_functions::clear_reverse_forw(&vec, &_d_vec);
// CHECK-NEXT:     std::vector<double> _t18 = vec;
// CHECK-NEXT:     {{.*}}class_functions::resize_reverse_forw(&vec, 2, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {
// CHECK-NEXT:         _t19 = vec;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t20 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:         _d_ref00 = &_t20.adjoint;
// CHECK-NEXT:         ref00 = &_t20.value;
// CHECK-NEXT:         _t21 = vec;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t22 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:         _d_ref10 = &_t22.adjoint;
// CHECK-NEXT:         ref10 = &_t22.value;
// CHECK-NEXT:         _t23 = *ref00;
// CHECK-NEXT:         *ref00 = u;
// CHECK-NEXT:         _t24 = *ref10;
// CHECK-NEXT:         *ref10 = u;
// CHECK-NEXT:     }
// CHECK-NEXT:     double _t25 = res;
// CHECK-NEXT:     std::vector<double> _t26 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t27 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     std::vector<double> _t28 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t29 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     res += _t27.value + _t29.value;
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         res = _t25;
// CHECK-NEXT:         double _r_d6 = _d_res;
// CHECK-NEXT:         {{.*}} _r10 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t26;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&vec, 0, _r_d6, &_d_vec, &_r10);
// CHECK-NEXT:         {{.*}} _r11 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t28;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&vec, 1, _r_d6, &_d_vec, &_r11);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {
// CHECK-NEXT:             *ref10 = _t24;
// CHECK-NEXT:             double _r_d5 = *_d_ref10;
// CHECK-NEXT:             *_d_ref10 = 0.;
// CHECK-NEXT:             *_d_u += _r_d5;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             *ref00 = _t23;
// CHECK-NEXT:             double _r_d4 = *_d_ref00;
// CHECK-NEXT:             *_d_ref00 = 0.;
// CHECK-NEXT:             *_d_u += _r_d4;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r9 = {{0U|0UL}};
// CHECK-NEXT:             vec = _t21;
// CHECK-NEXT:             {{.*}}class_functions::operator_subscript_pullback(&vec, 1, 0., &_d_vec, &_r9);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r8 = {{0U|0UL}};
// CHECK-NEXT:             vec = _t19;
// CHECK-NEXT:             {{.*}}class_functions::operator_subscript_pullback(&vec, 0, 0., &_d_vec, &_r8);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r7 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t18;
// CHECK-NEXT:         {{.*}}class_functions::resize_pullback(&vec, 2, &_d_vec, &_r7);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         vec = _t17;
// CHECK-NEXT:         clad::custom_derivatives::class_functions::clear_pullback(&vec, &_d_vec);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         res = _t10;
// CHECK-NEXT:         double _r_d3 = _d_res;
// CHECK-NEXT:         _d_res = 0.;
// CHECK-NEXT:         {{.*}} _r4 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t11;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&vec, 0, _r_d3, &_d_vec, &_r4);
// CHECK-NEXT:         {{.*}} _r5 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t13;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&vec, 1, _r_d3, &_d_vec, &_r5);
// CHECK-NEXT:         {{.*}} _r6 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t15;
// CHECK-NEXT:         {{.*}}class_functions::operator_subscript_pullback(&vec, 2, _r_d3, &_d_vec, &_r6);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {
// CHECK-NEXT:             *ref2 = _t9;
// CHECK-NEXT:             double _r_d2 = *_d_ref2;
// CHECK-NEXT:             *_d_ref2 = 0.;
// CHECK-NEXT:             *_d_u += _r_d2;
// CHECK-NEXT:             *_d_v += _r_d2;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             *ref1 = _t8;
// CHECK-NEXT:             double _r_d1 = *_d_ref1;
// CHECK-NEXT:             *_d_ref1 = 0.;
// CHECK-NEXT:             *_d_v += _r_d1;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             *ref0 = _t7;
// CHECK-NEXT:             double _r_d0 = *_d_ref0;
// CHECK-NEXT:             *_d_ref0 = 0.;
// CHECK-NEXT:             *_d_u += _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r3 = {{0U|0UL}};
// CHECK-NEXT:             vec = _t5;
// CHECK-NEXT:             {{.*}}class_functions::operator_subscript_pullback(&vec, 2, 0., &_d_vec, &_r3);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r2 = {{0U|0UL}};
// CHECK-NEXT:             vec = _t3;
// CHECK-NEXT:             {{.*}}class_functions::operator_subscript_pullback(&vec, 1, 0., &_d_vec, &_r2);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r1 = {{0U|0UL}};
// CHECK-NEXT:             vec = _t1;
// CHECK-NEXT:             {{.*}}class_functions::operator_subscript_pullback(&vec, 0, 0., &_d_vec, &_r1);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r0 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t0;
// CHECK-NEXT:         {{.*}}class_functions::resize_pullback(&vec, 3, &_d_vec, &_r0);
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK-NEXT: void fn4_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = u;
// CHECK-NEXT:     {{.*}}allocator_type allocator;
// CHECK-NEXT:     {{.*}}allocator_type _d_allocator = {};
// CHECK-NEXT:     clad::zero_init(_d_allocator);
// CHECK-NEXT:     {{.*}} _d_count = {{0U|0UL}};
// CHECK-NEXT:     {{.*}} count = 3;
// CHECK-NEXT:     std::vector<double> vec(count, u, allocator);
// CHECK-NEXT:     std::vector<double> _d_vec(vec);
// CHECK-NEXT:     clad::zero_init(_d_vec);
// CHECK-NEXT:     std::vector<double> _t0 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t1 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     std::vector<double> _t2 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t3 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     std::vector<double> _t4 = vec;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t5 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 2, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r1 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t0;
// CHECK-NEXT:         {{.*}}operator_subscript_pullback(&vec, 0, 1, &_d_vec, &_r1);
// CHECK-NEXT:         {{.*}} _r2 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t2;
// CHECK-NEXT:         {{.*}}operator_subscript_pullback(&vec, 1, 1, &_d_vec, &_r2);
// CHECK-NEXT:         {{.*}} _r3 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t4;
// CHECK-NEXT:         {{.*}}operator_subscript_pullback(&vec, 2, 1, &_d_vec, &_r3);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r0 = {{0U|0UL}};
// CHECK-NEXT:         {{.*}}constructor_pullback(count, u, allocator, &_d_vec, &_r0, &*_d_u, &_d_allocator);
// CHECK-NEXT:         _d_count += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_u += _d_res;
// CHECK-NEXT: }

// CHECK:      void fn5_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:          std::vector<double> a;
// CHECK-NEXT:          std::vector<double> _d_a = {};
// CHECK-NEXT:          clad::zero_init(_d_a);
// CHECK-NEXT:          std::vector<double> _t0 = a;
// CHECK-NEXT:          {{.*}}push_back_reverse_forw(&a, x, &_d_a, *_d_x);
// CHECK-NEXT:          std::vector<double> _t1 = a;
// CHECK-NEXT:          {{.*}}push_back_reverse_forw(&a, x, &_d_a, *_d_x);
// CHECK-NEXT:          std::vector<double> _t2 = a;
// CHECK-NEXT:          clad::ValueAndAdjoint<double &, double &> _t3 = {{.*}}operator_subscript_reverse_forw(&a, 1, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:          double _t4 = _t3.value;
// CHECK-NEXT:          _t3.value = x * x;
// CHECK-NEXT:          std::vector<double> _t5 = a;
// CHECK-NEXT:          clad::ValueAndAdjoint<double &, double &> _t6 = {{.*}}operator_subscript_reverse_forw(&a, 1, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:          {
// CHECK-NEXT:              {{.*}} _r1 = {{0U|0UL}};
// CHECK-NEXT:              a = _t5;
// CHECK-NEXT:              {{.*}}operator_subscript_pullback(&a, 1, 1, &_d_a, &_r1);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              _t3.value = _t4;
// CHECK-NEXT:              double _r_d0 = _t3.adjoint;
// CHECK-NEXT:              _t3.adjoint = 0.;
// CHECK-NEXT:              *_d_x += _r_d0 * x;
// CHECK-NEXT:              *_d_x += x * _r_d0;
// CHECK-NEXT:              {{.*}} _r0 = {{0U|0UL}};
// CHECK-NEXT:              a = _t2;
// CHECK-NEXT:              {{.*}}operator_subscript_pullback(&a, 1, 0., &_d_a, &_r0);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              a = _t1;
// CHECK-NEXT:              {{.*}}push_back_pullback(&a, x, &_d_a, &*_d_x);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              a = _t0;
// CHECK-NEXT:              {{.*}}push_back_pullback(&a, x, &_d_a, &*_d_x);
// CHECK-NEXT:          }
// CHECK-NEXT:      }

// CHECK:          void fn6_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:        size_t _d_i = {{0U|0UL}};
// CHECK-NEXT:        size_t i = {{0U|0UL}};
// CHECK-NEXT:        clad::tape<double> _t2 = {};
// CHECK-NEXT:        clad::tape<std::array<double, 3> > _t3 = {};
// CHECK-NEXT:        clad::tape<clad::ValueAndAdjoint<double &, double &> > _t4 = {};
// CHECK-NEXT:        std::array<double, 3> _d_a = {{.*}};
// CHECK-NEXT:        std::array<double, 3> a;
// CHECK-NEXT:        std::array<double, 3> _t0 = a;
// CHECK-NEXT:        {{.*}}fill_reverse_forw(&a, x, &_d_a, *_d_x);
// CHECK-NEXT:        double _d_res = 0.;
// CHECK-NEXT:        double res = 0;
// CHECK-NEXT:        unsigned {{long|int}} _t1 = {{0U|0UL}};
// CHECK-NEXT:        for (i = 0; ; ++i) {
// CHECK-NEXT:            {
// CHECK-NEXT:                if (!(i < a.size()))
// CHECK-NEXT:                    break;
// CHECK-NEXT:            }
// CHECK-NEXT:            _t1++;
// CHECK-NEXT:            clad::push(_t2, res);
// CHECK-NEXT:            clad::push(_t3, a);
// CHECK-NEXT:            clad::push(_t4, {{.*}}at_reverse_forw(&a, i, &_d_a, {{0U|0UL|0}}));
// CHECK-NEXT:            res += clad::back(_t4).value;
// CHECK-NEXT:        }
// CHECK-NEXT:        _d_res += 1;
// CHECK-NEXT:        for (;; _t1--) {
// CHECK-NEXT:            {
// CHECK-NEXT:                if (!_t1)
// CHECK-NEXT:                    break;
// CHECK-NEXT:            }
// CHECK-NEXT:            --i;
// CHECK-NEXT:            {
// CHECK-NEXT:                res = clad::pop(_t2);
// CHECK-NEXT:                double _r_d0 = _d_res;
// CHECK-NEXT:                size_t _r0 = {{0U|0UL}};
// CHECK-NEXT:                a = clad::back(_t3);
// CHECK-NEXT:                {{.*}}at_pullback(&a, i, _r_d0, &_d_a, &_r0);
// CHECK-NEXT:                _d_i += _r0;
// CHECK-NEXT:                clad::pop(_t3);
// CHECK-NEXT:                clad::pop(_t4);
// CHECK-NEXT:            }
// CHECK-NEXT:        }
// CHECK-NEXT:        {
// CHECK-NEXT:            a = _t0;
// CHECK-NEXT:            {{.*}}fill_pullback(&a, x, &_d_a, &*_d_x);
// CHECK-NEXT:        }
// CHECK-NEXT: }

// CHECK:     void fn7_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:         std::array<double, 2> _d_a = {{.*}};
// CHECK-NEXT:         std::array<double, 2> a;
// CHECK-NEXT:         std::array<double, 2> _t0 = a;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t1 = {{.*}}operator_subscript_reverse_forw(&a, 0, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:         double _t2 = _t1.value;
// CHECK-NEXT:         _t1.value = 5;
// CHECK-NEXT:         std::array<double, 2> _t3 = a;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t4 = {{.*}}operator_subscript_reverse_forw(&a, 1, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:         double _t5 = _t4.value;
// CHECK-NEXT:         _t4.value = y;
// CHECK-NEXT:         std::array<double, 3> _d__b = {{.*}};
// CHECK-NEXT:         std::array<double, 3> _b0;
// CHECK-NEXT:         std::array<double, 3> _t6 = _b0;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t7 = {{.*}}operator_subscript_reverse_forw(&_b0, 0, &_d__b, {{0U|0UL|0}});
// CHECK-NEXT:         double _t8 = _t7.value;
// CHECK-NEXT:         _t7.value = x;
// CHECK-NEXT:         std::array<double, 3> _t9 = _b0;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t10 = {{.*}}operator_subscript_reverse_forw(&_b0, 1, &_d__b, {{0U|0UL|0}});
// CHECK-NEXT:         double _t11 = _t10.value;
// CHECK-NEXT:         _t10.value = 0;
// CHECK-NEXT:         std::array<double, 3> _t12 = _b0;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t13 = {{.*}}operator_subscript_reverse_forw(&_b0, 2, &_d__b, {{0U|0UL|0}});
// CHECK-NEXT:         double _t14 = _t13.value;
// CHECK-NEXT:         _t13.value = x * x;
// CHECK-NEXT:         std::array<double, 3> _d_b = {{.*}};
// CHECK-NEXT:         const std::array<double, 3> b = _b0;
// CHECK-NEXT:         std::array<double, 2> _t17 = a;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t18 = {{.*}}back_reverse_forw(&a, &_d_a);
// CHECK-NEXT:         {{.*}}value_type _t16 = b.front();
// CHECK-NEXT:         {{.*}}value_type _t15 = b.at(2);
// CHECK-NEXT:         {
// CHECK-NEXT:             a = _t17;
// CHECK-NEXT:             {{.*}}back_pullback(&a, 1 * _t15 * _t16, &_d_a);
// CHECK-NEXT:             {{.*}}front_pullback(&b, _t18.value * 1 * _t15, &_d_b);
// CHECK-NEXT:             {{.*}}size_type _r5 = {{0U|0UL}};
// CHECK-NEXT:             {{.*}}at_pullback(&b, 2, _t18.value * _t16 * 1, &_d_b, &_r5);
// CHECK-NEXT:             {{.*}}size_type _r6 = {{0U|0UL}};
// CHECK-NEXT:             {{.*}}operator_subscript_pullback(&b, 1, 1, &_d_b, &_r6);
// CHECK-NEXT:         }
// CHECK-NEXT:         {{.*}}constructor_pullback(_b0, &_d_b, &_d__b);
// CHECK-NEXT:         {
// CHECK-NEXT:             _t13.value = _t14;
// CHECK-NEXT:             double _r_d4 = _t13.adjoint;
// CHECK-NEXT:             _t13.adjoint = 0.;
// CHECK-NEXT:             *_d_x += _r_d4 * x;
// CHECK-NEXT:             *_d_x += x * _r_d4;
// CHECK-NEXT:             {{.*}}size_type _r4 = {{0U|0UL}};
// CHECK-NEXT:             _b0 = _t12;
// CHECK-NEXT:             {{.*}}operator_subscript_pullback(&_b0, 2, 0., &_d__b, &_r4);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             _t10.value = _t11;
// CHECK-NEXT:             double _r_d3 = _t10.adjoint;
// CHECK-NEXT:             _t10.adjoint = 0.;
// CHECK-NEXT:             {{.*}}size_type _r3 = {{0U|0UL}};
// CHECK-NEXT:             _b0 = _t9;
// CHECK-NEXT:             {{.*}}operator_subscript_pullback(&_b0, 1, 0., &_d__b, &_r3);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             _t7.value = _t8;
// CHECK-NEXT:             double _r_d2 = _t7.adjoint;
// CHECK-NEXT:             _t7.adjoint = 0.;
// CHECK-NEXT:             *_d_x += _r_d2;
// CHECK-NEXT:             {{.*}}size_type _r2 = {{0U|0UL}};
// CHECK-NEXT:             _b0 = _t6;
// CHECK-NEXT:             {{.*}}operator_subscript_pullback(&_b0, 0, 0., &_d__b, &_r2);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             _t4.value = _t5;
// CHECK-NEXT:             double _r_d1 = _t4.adjoint;
// CHECK-NEXT:             _t4.adjoint = 0.;
// CHECK-NEXT:             *_d_y += _r_d1;
// CHECK-NEXT:             {{.*}}size_type _r1 = {{0U|0UL}};
// CHECK-NEXT:             a = _t3;
// CHECK-NEXT:             {{.*}}operator_subscript_pullback(&a, 1, 0., &_d_a, &_r1);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             _t1.value = _t2;
// CHECK-NEXT:             double _r_d0 = _t1.adjoint;
// CHECK-NEXT:             _t1.adjoint = 0.;
// CHECK-NEXT:             {{.*}}size_type _r0 = {{0U|0UL}};
// CHECK-NEXT:             a = _t0;
// CHECK-NEXT:             {{.*}}operator_subscript_pullback(&a, 0, 0., &_d_a, &_r0);
// CHECK-NEXT:         }
// CHECK-NEXT:     }

// CHECK:     void fn8_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:         std::array<double, 50> _d_a = {{.*}};
// CHECK-NEXT:         std::array<double, 50> a;
// CHECK-NEXT:         std::array<double, 50> _t0 = a;
// CHECK-NEXT:         {{.*}}fill_reverse_forw(&a, y + x + x, &_d_a, 0.);
// CHECK-NEXT:         std::array<double, 50> _t1 = a;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t2 = {{.*}}operator_subscript_reverse_forw(&a, 49, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:         std::array<double, 50> _t3 = a;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t4 = {{.*}}operator_subscript_reverse_forw(&a, 3, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}}size_type _r1 = {{0U|0UL}};
// CHECK-NEXT:             a = _t1;
// CHECK-NEXT:             {{.*}}operator_subscript_pullback(&a, 49, 1, &_d_a, &_r1);
// CHECK-NEXT:             {{.*}}size_type _r2 = {{0U|0UL}};
// CHECK-NEXT:             a = _t3;
// CHECK-NEXT:             {{.*}}operator_subscript_pullback(&a, 3, 1, &_d_a, &_r2);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r0 = 0.;
// CHECK-NEXT:             a = _t0;
// CHECK-NEXT:             {{.*}}fill_pullback(&a, y + x + x, &_d_a, &_r0);
// CHECK-NEXT:             *_d_y += _r0;
// CHECK-NEXT:             *_d_x += _r0;
// CHECK-NEXT:             *_d_x += _r0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }

// CHECK:     void fn9_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:         std::array<double, 2> _d_a = {{.*}};
// CHECK-NEXT:         std::array<double, 2> a;
// CHECK-NEXT:         std::array<double, 2> _t0 = a;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t1 = {{.*}}operator_subscript_reverse_forw(&a, 1, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:         {{.*}} _t2 = _t1.value;
// CHECK-NEXT:         _t1.value = 2 * x;
// CHECK-NEXT:         std::array<double, 2> _t3 = a;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t4 = {{.*}}operator_subscript_reverse_forw(&a, 1, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}}size_type _r1 = {{0U|0UL}};
// CHECK-NEXT:             a = _t3;
// CHECK-NEXT:             {{.*}}operator_subscript_pullback(&a, 1, 1, &_d_a, &_r1);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             _t1.value = _t2;
// CHECK-NEXT:             {{.*}} _r_d0 = _t1.adjoint;
// CHECK-NEXT:             _t1.adjoint = 0.;
// CHECK-NEXT:             *_d_x += 2 * _r_d0;
// CHECK-NEXT:             {{.*}}size_type _r0 = {{0U|0UL}};
// CHECK-NEXT:             a = _t0;
// CHECK-NEXT:             {{.*}}operator_subscript_pullback(&a, 1, 0., &_d_a, &_r0);
// CHECK-NEXT:         }
// CHECK-NEXT:     }

// CHECK:      void fn10_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:          size_t _d_i = {{0U|0UL|0}};
// CHECK-NEXT:          size_t i = {{0U|0UL|0}};
// CHECK-NEXT:          {{.*}}tape<{{.*}}vector<double> > _t1 = {};
// CHECK-NEXT:          size_t _d_i0 = {{0U|0UL|0}};
// CHECK-NEXT:          size_t i0 = {{0U|0UL|0}};
// CHECK-NEXT:          {{.*}}tape<double> _t3 = {};
// CHECK-NEXT:          {{.*}}tape<{{.*}}vector<double> > _t4 = {};
// CHECK-NEXT:          clad::tape<clad::ValueAndAdjoint<double &, double &> > _t5 = {};
// CHECK-NEXT:          {{.*}}vector<double> v;
// CHECK-NEXT:          {{.*}}vector<double> _d_v = {};
// CHECK-NEXT:          clad::zero_init(_d_v);
// CHECK-NEXT:          {{.*}} _t0 = {{0U|0UL|0}};
// CHECK-NEXT:          for (i = 0; ; ++i) {
// CHECK-NEXT:              {
// CHECK-NEXT:                  if (!(i < 3))
// CHECK-NEXT:                      break;
// CHECK-NEXT:              }
// CHECK-NEXT:              _t0++;
// CHECK-NEXT:              {{.*}}push(_t1, v);
// CHECK-NEXT:              {{.*}}push_back_reverse_forw(&v, x, &_d_v, *_d_x);
// CHECK-NEXT:          }
// CHECK-NEXT:          double _d_res = 0.;
// CHECK-NEXT:          double res = 0;
// CHECK-NEXT:          {{.*}} _t2 = {{0U|0UL|0}};
// CHECK-NEXT:          for (i0 = 0; ; ++i0) {
// CHECK-NEXT:              {
// CHECK-NEXT:                  if (!(i0 < v.size()))
// CHECK-NEXT:                      break;
// CHECK-NEXT:              }
// CHECK-NEXT:              _t2++;
// CHECK-NEXT:              {{.*}}push(_t3, res);
// CHECK-NEXT:              {{.*}}push(_t4, v);
// CHECK-NEXT:              clad::push(_t5, {{.*}}at_reverse_forw(&v, i0, &_d_v, {{0U|0UL|0}}));
// CHECK-NEXT:              res += clad::back(_t5).value;
// CHECK-NEXT:          }
// CHECK-NEXT:          {{.*}}vector<double> _t6 = v;
// CHECK-NEXT:          v.assign(3, 0);
// CHECK-NEXT:          {{.*}}vector<double> _t7 = v;
// CHECK-NEXT:          v.assign(2, y);
// CHECK-NEXT:          {{.*}}vector<double> _t8 = v;
// CHECK-NEXT:          {{.*}}ValueAndAdjoint<double &, double &> _t9 = {{.*}}operator_subscript_reverse_forw(&v, 0, &_d_v, {{0U|0UL|0}});
// CHECK-NEXT:          {{.*}}vector<double> _t10 = v;
// CHECK-NEXT:          {{.*}}ValueAndAdjoint<double &, double &> _t11 = {{.*}}operator_subscript_reverse_forw(&v, 1, &_d_v, {{0U|0UL|0}});
// CHECK-NEXT:          {{.*}}vector<double> _t12 = v;
// CHECK-NEXT:          {{.*}}ValueAndAdjoint<double &, double &> _t13 = {{.*}}operator_subscript_reverse_forw(&v, 2, &_d_v, {{0U|0UL|0}});
// CHECK-NEXT:          {
// CHECK-NEXT:              _d_res += 1;
// CHECK-NEXT:              {{.*}}size_type _r4 = {{0U|0UL|0}};
// CHECK-NEXT:              v = _t8;
// CHECK-NEXT:              {{.*}}operator_subscript_pullback(&v, 0, 1, &_d_v, &_r4);
// CHECK-NEXT:              {{.*}}size_type _r5 = {{0U|0UL|0}};
// CHECK-NEXT:              v = _t10;
// CHECK-NEXT:              {{.*}}operator_subscript_pullback(&v, 1, 1, &_d_v, &_r5);
// CHECK-NEXT:              {{.*}}size_type _r6 = {{0U|0UL|0}};
// CHECK-NEXT:              v = _t12;
// CHECK-NEXT:              {{.*}}operator_subscript_pullback(&v, 2, 1, &_d_v, &_r6);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              {{.*}}size_type _r3 = {{0U|0UL|0}};
// CHECK-NEXT:              v = _t7;
// CHECK-NEXT:              {{.*}}assign_pullback(&v, 2, y, &_d_v, &_r3, &*_d_y);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              {{.*}}size_type _r1 = {{0U|0UL|0}};
// CHECK-NEXT:              {{.*}}value_type _r2 = 0.;
// CHECK-NEXT:              v = _t6;
// CHECK-NEXT:              {{.*}}assign_pullback(&v, 3, 0, &_d_v, &_r1, &_r2);
// CHECK-NEXT:          }
// CHECK-NEXT:          for (;; _t2--) {
// CHECK-NEXT:              {
// CHECK-NEXT:                  if (!_t2)
// CHECK-NEXT:                      break;
// CHECK-NEXT:              }
// CHECK-NEXT:              --i0;
// CHECK-NEXT:              {
// CHECK-NEXT:                  res = {{.*}}pop(_t3);
// CHECK-NEXT:                  double _r_d0 = _d_res;
// CHECK-NEXT:                  size_t _r0 = {{0U|0UL|0}};
// CHECK-NEXT:                  v = {{.*}}back(_t4);
// CHECK-NEXT:                  {{.*}}at_pullback(&v, i0, _r_d0, &_d_v, &_r0);
// CHECK-NEXT:                  _d_i0 += _r0;
// CHECK-NEXT:                  {{.*}}pop(_t4);
// CHECK-NEXT:                  {{.*}}pop(_t5);
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          for (;; _t0--) {
// CHECK-NEXT:              {
// CHECK-NEXT:                  if (!_t0)
// CHECK-NEXT:                      break;
// CHECK-NEXT:              }
// CHECK-NEXT:              --i;
// CHECK-NEXT:              {
// CHECK-NEXT:                  v = {{.*}}back(_t1);
// CHECK-NEXT:                  {{.*}}push_back_pullback(&v, x, &_d_v, &*_d_x);
// CHECK-NEXT:                  {{.*}}pop(_t1);
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      }

// CHECK:      void fn11_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:          {{.*}}vector<double> v;
// CHECK-NEXT:          {{.*}}vector<double> _d_v = {};
// CHECK-NEXT:          clad::zero_init(_d_v);
// CHECK-NEXT:          {{.*}}vector<double> _t0 = v;
// CHECK-NEXT:          v.reserve(10);
// CHECK-NEXT:          double _t1 = v.capacity();
// CHECK-NEXT:          double _d_res = 0.;
// CHECK-NEXT:          double res = x * _t1;
// CHECK-NEXT:          {{.*}}vector<double> _t2 = v;
// CHECK-NEXT:          {{.*}}push_back_reverse_forw(&v, x, &_d_v, *_d_x);
// CHECK-NEXT:          {{.*}}vector<double> _t3 = v;
// CHECK-NEXT:          v.shrink_to_fit();
// CHECK-NEXT:          double _t4 = res;
// CHECK-NEXT:          double _t5 = v.capacity();
// CHECK-NEXT:          double _t6 = v.size();
// CHECK-NEXT:          res += y * _t5 + x * _t6;
// CHECK-NEXT:          _d_res += 1;
// CHECK-NEXT:          {
// CHECK-NEXT:              res = _t4;
// CHECK-NEXT:              double _r_d0 = _d_res;
// CHECK-NEXT:              *_d_y += _r_d0 * _t5;
// CHECK-NEXT:              {{.*}}capacity_pullback(&v, y * _r_d0, &_d_v);
// CHECK-NEXT:              *_d_x += _r_d0 * _t6;
// CHECK-NEXT:              {{.*}}size_pullback(&v, x * _r_d0, &_d_v);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              v = _t3;
// CHECK-NEXT:              {{.*}}shrink_to_fit_pullback(&v, &_d_v);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              v = _t2;
// CHECK-NEXT:              {{.*}}push_back_pullback(&v, x, &_d_v, &*_d_x);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              *_d_x += _d_res * _t1;
// CHECK-NEXT:              {{.*}}capacity_pullback(&v, x * _d_res, &_d_v);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              {{.*}}size_type _r0 = {{0U|0UL|0}};
// CHECK-NEXT:              v = _t0;
// CHECK-NEXT:              {{.*}}reserve_pullback(&v, 10, &_d_v, &_r0);
// CHECK-NEXT:          }
// CHECK-NEXT:      }

// CHECK:      void fn12_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:          std::vector<double> a;
// CHECK-NEXT:          std::vector<double> _d_a = {};
// CHECK-NEXT:          clad::zero_init(_d_a);
// CHECK-NEXT:          std::vector<double> _t0 = a;
// CHECK-NEXT:          {{.*}}push_back_reverse_forw(&a, 0{{.*}}, &_d_a, 0.);
// CHECK-NEXT:          std::vector<double> _t1 = a;
// CHECK-NEXT:          {{.*}}ValueAndAdjoint<double &, double &> _t2 = {{.*}}operator_subscript_reverse_forw(&a, 0, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:          double _t3 = _t2.value;
// CHECK-NEXT:          _t2.value = x * x;
// CHECK-NEXT:          std::vector<double> _t4 = a;
// CHECK-NEXT:          {{.*}}ValueAndAdjoint<double &, double &> _t5 = {{.*}}operator_subscript_reverse_forw(&a, 0, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:          {
// CHECK-NEXT:              {{.*}}size_type _r2 = 0{{.*}};
// CHECK-NEXT:              a = _t4;
// CHECK-NEXT:              {{.*}}operator_subscript_pullback(&a, 0, 1, &_d_a, &_r2);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              _t2.value = _t3;
// CHECK-NEXT:              double _r_d0 = _t2.adjoint;
// CHECK-NEXT:              _t2.adjoint = 0{{.*}};
// CHECK-NEXT:              *_d_x += _r_d0 * x;
// CHECK-NEXT:              *_d_x += x * _r_d0;
// CHECK-NEXT:              {{.*}}size_type _r1 = 0{{.*}};
// CHECK-NEXT:              a = _t1;
// CHECK-NEXT:              {{.*}}operator_subscript_pullback(&a, 0, 0{{.*}}, &_d_a, &_r1);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              {{.*}}value_type _r0 = 0.;
// CHECK-NEXT:              a = _t0;
// CHECK-NEXT:              {{.*}}push_back_pullback(&a, 0{{.*}}, &_d_a, &_r0);
// CHECK-NEXT:          }
// CHECK-NEXT:      }

// CHECK:      void fn13_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:      std::vector<double>::allocator_type alloc;
// CHECK-NEXT:      std::vector<double>::allocator_type _d_alloc = {};
// CHECK-NEXT:      clad::zero_init(_d_alloc);
// CHECK-NEXT:      std::vector<double> ls({u, v}, alloc);
// CHECK-NEXT:      std::vector<double> _d_ls(ls);
// CHECK-NEXT:      clad::zero_init(_d_ls);
// CHECK-NEXT:      std::vector<double> _t0 = ls;
// CHECK-NEXT:      clad::ValueAndAdjoint<double &, double &> _t1 = clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 1, &_d_ls, {{0U|0UL|0}});
// CHECK-NEXT:      std::vector<double> _t3 = ls;
// CHECK-NEXT:      clad::ValueAndAdjoint<double &, double &> _t4 = clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 0, &_d_ls, {{0U|0UL|0}});
// CHECK-NEXT:      {{.*}}value_type _t2 = _t4.value;
// CHECK-NEXT:      {
// CHECK-NEXT:          {{.*}}size_type _r1 = 0{{.*}};
// CHECK-NEXT:          ls = _t0;
// CHECK-NEXT:          clad::custom_derivatives::class_functions::operator_subscript_pullback(&ls, 1, 1, &_d_ls, &_r1);
// CHECK-NEXT:          {{.*}}size_type _r2 = 0{{.*}};
// CHECK-NEXT:          ls = _t3;
// CHECK-NEXT:          clad::custom_derivatives::class_functions::operator_subscript_pullback(&ls, 0, 2 * -1, &_d_ls, &_r2);
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          clad::array<double> _r0 = {{2U|2UL|2ULL}};
// CHECK-NEXT:          {{.*}}::class_functions::constructor_pullback({{.*u, v.*}}, alloc, &_d_ls, &_r0, &_d_alloc);
// CHECK-NEXT:          *_d_u += _r0[0];
// CHECK-NEXT:          *_d_v += _r0[1];
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:      void fn14_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:      int _d_i = 0;
// CHECK-NEXT:      int i = 0;
// CHECK-NEXT:      clad::tape<std::vector<double> > _t1 = {};
// CHECK-NEXT:      clad::tape<std::vector<double> > _t2 = {};
// CHECK-NEXT:      std::vector<double> ls = {};
// CHECK-NEXT:      std::vector<double> _d_ls{};
// CHECK-NEXT:      clad::tape<std::vector<double> > _t3 = {};
// CHECK-NEXT:      clad::tape<clad::ValueAndAdjoint<double &, double &> > _t4 = {};
// CHECK-NEXT:      clad::tape<double> _t5 = {};
// CHECK-NEXT:      clad::tape<std::vector<double> > _t6 = {};
// CHECK-NEXT:      clad::tape<clad::ValueAndAdjoint<double &, double &> > _t7 = {};
// CHECK-NEXT:      clad::tape<double> _t8 = {};
// CHECK-NEXT:      clad::tape<std::vector<double> > _t9 = {};
// CHECK-NEXT:      clad::tape<clad::ValueAndAdjoint<double &, double &> > _t10 = {};
// CHECK-NEXT:      {{.*}}allocator_type alloc;
// CHECK-NEXT:      {{.*}}allocator_type _d_alloc = {};
// CHECK-NEXT:      clad::zero_init(_d_alloc);
// CHECK-NEXT:      unsigned {{int|long|long long}} _t0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:      for (i = 0; ; ++i) {
// CHECK-NEXT:          {
// CHECK-NEXT:              if (!(i < 3))
// CHECK-NEXT:                  break;
// CHECK-NEXT:          }
// CHECK-NEXT:          _t0++;
// CHECK-NEXT:          clad::push(_t1, std::move(_d_ls));
// CHECK-NEXT:          clad::push(_t2, std::move(ls)) , ls = {u, v}, alloc;
// CHECK-NEXT:          _d_ls = ls;
// CHECK-NEXT:          clad::zero_init(_d_ls);
// CHECK-NEXT:          clad::push(_t3, ls);
// CHECK-NEXT:          clad::push(_t4, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 1, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:          clad::push(_t5, clad::back(_t4).value); 
// CHECK-NEXT:          clad::push(_t6, ls);
// CHECK-NEXT:          clad::push(_t7, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 0, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:          clad::back(_t4).value += clad::back(_t7).value;
// CHECK-NEXT:          clad::push(_t8, u);
// CHECK-NEXT:          clad::push(_t9, ls);
// CHECK-NEXT:          clad::push(_t10, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 1, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:          u = clad::back(_t10).value;
// CHECK-NEXT:      }
// CHECK-NEXT:      *_d_u += 1;
// CHECK-NEXT:      for (;; _t0--) {
// CHECK-NEXT:          {
// CHECK-NEXT:              if (!_t0)
// CHECK-NEXT:                  break;
// CHECK-NEXT:          }
// CHECK-NEXT:          --i;
// CHECK-NEXT:          {
// CHECK-NEXT:              u = clad::pop(_t8);
// CHECK-NEXT:              double _r_d1 = *_d_u;
// CHECK-NEXT:              *_d_u = 0.;
// CHECK-NEXT:              {{.*}}size_type _r3 = 0{{.*}};
// CHECK-NEXT:              ls = clad::back(_t9);
// CHECK-NEXT:              clad::custom_derivatives::class_functions::operator_subscript_pullback(&ls, 1, _r_d1, &_d_ls, &_r3);
// CHECK-NEXT:              clad::pop(_t9);
// CHECK-NEXT:              clad::pop(_t10);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              clad::back(_t4).value = clad::pop(_t5);
// CHECK-NEXT:              double _r_d0 = clad::back(_t4).adjoint;
// CHECK-NEXT:              {{.*}}size_type _r2 = 0{{.*}};
// CHECK-NEXT:              ls = clad::back(_t6);
// CHECK-NEXT:              clad::custom_derivatives::class_functions::operator_subscript_pullback(&ls, 0, _r_d0, &_d_ls, &_r2);
// CHECK-NEXT:              clad::pop(_t6);
// CHECK-NEXT:              clad::pop(_t7);
// CHECK-NEXT:              {{.*}}size_type _r1 = 0{{.*}};
// CHECK-NEXT:              ls = clad::back(_t3);
// CHECK-NEXT:              clad::custom_derivatives::class_functions::operator_subscript_pullback(&ls, 1, 0., &_d_ls, &_r1);
// CHECK-NEXT:              clad::pop(_t3);
// CHECK-NEXT:              clad::pop(_t4);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              clad::array<double> _r0 = {{2U|2UL|2ULL}};
// CHECK-NEXT:              {{.*}}::class_functions::constructor_pullback({u, v}, alloc, &_d_ls, &_r0, &_d_alloc);
// CHECK-NEXT:              *_d_u += _r0[0];
// CHECK-NEXT:              *_d_v += _r0[1];
// CHECK-NEXT:              _d_ls = clad::pop(_t1);
// CHECK-NEXT:              ls = clad::pop(_t2);
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK: void fn15_grad(double d, double e, double *_d_d, double *_d_e) {
// CHECK-NEXT:     double *_d_p = new double(*_d_d);
// CHECK-NEXT:     double *p = new double(d);
// CHECK-NEXT:     clad::ValueAndAdjoint< {{.*}}, {{.*}} > _t0 = {{.*}}class_functions::constructor_reverse_forw(clad::ConstructorReverseForwTag<unique_ptr{{.*}}, p, _d_p);
// CHECK-NEXT:     std::unique_ptr{{.*}} up(static_cast<std::unique_ptr{{.*}}(_t0.value));
// CHECK-NEXT:     std::unique_ptr{{.*}} _d_up = static_cast<std::unique_ptr{{.*}}(_t0.adjoint);
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t1 = {{.*}}class_functions::operator_star_reverse_forw(&up, &_d_up);
// CHECK-NEXT:     double _t2 = _t1.value;
// CHECK-NEXT:     _t1.value += 5 * e;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t3 = {{.*}}class_functions::operator_star_reverse_forw(&up, &_d_up);
// CHECK-NEXT:     {{.*}}class_functions::operator_star_pullback(&up, 1, &_d_up);
// CHECK-NEXT:     {
// CHECK-NEXT:         _t1.value = _t2;
// CHECK-NEXT:         double _r_d0 = _t1.adjoint;
// CHECK-NEXT:         *_d_e += 5 * _r_d0;
// CHECK-NEXT:         {{.*}}class_functions::operator_star_pullback(&up, 0., &_d_up);
// CHECK-NEXT:     }
// Not check next because on some implementations the unique_ptr ctor is trivial and we generate a pullback.
// CHECK:     *_d_d += *_d_p;
// CHECK-NEXT: }

// CHECK-NEXT: void fn16_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<std::vector<double> > _t1 = {};
// CHECK-NEXT:     clad::tape<std::vector<double> > _t2 = {};
// CHECK-NEXT:     std::vector<double> vec = {};
// CHECK-NEXT:     std::vector<double> _d_vec{};
// CHECK-NEXT:     clad::tape<double> _t3 = {};
// CHECK-NEXT:     clad::tape<std::vector<double> > _t4 = {};
// CHECK-NEXT:     clad::tape<clad::ValueAndAdjoint<double &, double &> > _t5 = {};
// CHECK-NEXT:     {{.*}}allocator_type alloc;
// CHECK-NEXT:     {{.*}}allocator_type _d_alloc = {};
// CHECK-NEXT:     clad::zero_init(_d_alloc);
// CHECK-NEXT:     double _d_prod = 0.;
// CHECK-NEXT:     double prod = 1;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:     for (i = 3; ; --i) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i >= 1))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, std::move(_d_vec));
// CHECK-NEXT:         clad::push(_t2, std::move(vec)) , vec = i, v + u, alloc;
// CHECK-NEXT:         _d_vec = vec;
// CHECK-NEXT:         clad::zero_init(_d_vec);
// CHECK-NEXT:         clad::push(_t3, prod);
// CHECK-NEXT:         clad::push(_t4, vec);
// CHECK-NEXT:         clad::push(_t5, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&vec, i - 1, &_d_vec, {{0U|0UL|0}}));
// CHECK-NEXT:         prod *= clad::back(_t5).value;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_prod += 1;
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         ++i;
// CHECK-NEXT:         {
// CHECK-NEXT:             prod = clad::pop(_t3);
// CHECK-NEXT:             double _r_d0 = _d_prod;
// CHECK-NEXT:             _d_prod = 0.;
// CHECK-NEXT:             _d_prod += _r_d0 * clad::back(_t5).value;
// CHECK-NEXT:             {{.*}}size_type _r2 = 0{{.*}};
// CHECK-NEXT:             vec = clad::back(_t4);
// CHECK-NEXT:             clad::custom_derivatives::class_functions::operator_subscript_pullback(&vec, i - 1, prod * _r_d0, &_d_vec, &_r2);
// CHECK-NEXT:             _d_i += _r2;
// CHECK-NEXT:             clad::pop(_t4);
// CHECK-NEXT:             clad::pop(_t5);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}}size_type _r0 = 0{{.*}};
// CHECK-NEXT:             {{.*}}value_type _r1 = 0.;
// CHECK-NEXT:             clad::custom_derivatives::class_functions::constructor_pullback(i, v + u, alloc, &_d_vec, &_r0, &_r1, &_d_alloc);
// CHECK-NEXT:             _d_i += _r0;
// CHECK-NEXT:             *_d_v += _r1;
// CHECK-NEXT:             *_d_u += _r1;
// CHECK-NEXT:             _d_vec = clad::pop(_t1);
// CHECK-NEXT:             vec = clad::pop(_t2);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }



// CHECK: void fn17_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<std::vector<double> > _t1 = {};
// CHECK-NEXT:     clad::tape<std::vector<double> > _t2 = {};
// CHECK-NEXT:     std::vector<double> ls = {};
// CHECK-NEXT:     std::vector<double> _d_ls{};
// CHECK-NEXT:     clad::tape<std::vector<double> > _t3 = {};
// CHECK-NEXT:     clad::tape<clad::ValueAndAdjoint<double &, double &> > _t4 = {};
// CHECK-NEXT:     clad::tape<double> _t5 = {};
// CHECK-NEXT:     clad::tape<std::vector<double> > _t6 = {};
// CHECK-NEXT:     clad::tape<clad::ValueAndAdjoint<double &, double &> > _t7 = {};
// CHECK-NEXT:     clad::tape<double> _t8 = {};
// CHECK-NEXT:     clad::tape<std::vector<double> > _t9 = {};
// CHECK-NEXT:     clad::tape<clad::ValueAndAdjoint<double &, double &> > _t10 = {};
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:     for (i = 0; ; ++i) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 3))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, std::move(_d_ls));
// CHECK-NEXT:         clad::push(_t2, std::move(ls)) , ls = {{.*{u, v}.*}};
// CHECK-NEXT:         _d_ls = ls;
// CHECK-NEXT:         clad::zero_init(_d_ls);
// CHECK-NEXT:         clad::push(_t3, ls);
// CHECK-NEXT:         clad::push(_t4, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 1, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:         clad::push(_t5, clad::back(_t4).value);
// CHECK-NEXT:         clad::push(_t6, ls);
// CHECK-NEXT:         clad::push(_t7, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 0, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:         clad::back(_t4).value += clad::back(_t7).value;
// CHECK-NEXT:         clad::push(_t8, u);
// CHECK-NEXT:         clad::push(_t9, ls);
// CHECK-NEXT:         clad::push(_t10, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 1, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:         u = clad::back(_t10).value;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_u += 1;
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         --i;
// CHECK-NEXT:         {
// CHECK-NEXT:             u = clad::pop(_t8);
// CHECK-NEXT:             double _r_d1 = *_d_u;
// CHECK-NEXT:             *_d_u = 0.;
// CHECK-NEXT:             {{.*}}size_type _r{{3|4}} = {{0U|0UL|0ULL}};
// CHECK-NEXT:             ls = clad::back(_t9);
// CHECK-NEXT:             clad::custom_derivatives::class_functions::operator_subscript_pullback(&ls, 1, _r_d1, &_d_ls, &_r{{3|4}});
// CHECK-NEXT:             clad::pop(_t9);
// CHECK-NEXT:             clad::pop(_t10);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             clad::back(_t4).value = clad::pop(_t5);
// CHECK-NEXT:             double _r_d0 = clad::back(_t4).adjoint;
// CHECK-NEXT:             {{.*}}size_type _r{{2|3}} = {{0U|0UL|0ULL}};
// CHECK-NEXT:             ls = clad::back(_t6);
// CHECK-NEXT:             clad::custom_derivatives::class_functions::operator_subscript_pullback(&ls, 0, _r_d0, &_d_ls, &_r{{2|3}});
// CHECK-NEXT:             clad::pop(_t6);
// CHECK-NEXT:             clad::pop(_t7);
// CHECK-NEXT:             {{.*}}size_type _r{{1|2}} = {{0U|0UL|0ULL}};
// CHECK-NEXT:             ls = clad::back(_t3);
// CHECK-NEXT:             clad::custom_derivatives::class_functions::operator_subscript_pullback(&ls, 1, 0., &_d_ls, &_r{{1|2}});
// CHECK-NEXT:             clad::pop(_t3);
// CHECK-NEXT:             clad::pop(_t4);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             clad::array<double> _r0 = {{2U|2UL|2ULL}};
// CHECK:                  clad::custom_derivatives::class_functions::constructor_pullback({u, v},{{.*}} &_d_ls, &_r0{{.*}});
// CHECK-NEXT:             *_d_u += _r0[0];
// CHECK-NEXT:             *_d_v += _r0[1];
// CHECK-NEXT:             _d_ls = clad::pop(_t1);
// CHECK-NEXT:             ls = clad::pop(_t2);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn18_grad_2(const Session *session, const float *tensor_x, float *tensor_theory_params, float *_d_tensor_theory_params) {
// CHECK-NEXT:     int _d_id = 0;
// CHECK-NEXT:     int id = 0;
// CHECK-NEXT:     clad::tape<float> _t1 = {};
// CHECK-NEXT:     int _d_id0 = 0;
// CHECK-NEXT:     int id0 = 0;
// CHECK-NEXT:     clad::tape<float> _t3 = {};
// CHECK-NEXT:     Session _d_sess = {{.*}}, nullptr};
// CHECK-NEXT:     const Session &sess = session[0];
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:     for (id = 0; ; id++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(id < nVals))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, sess.arr[id]);
// CHECK-NEXT:         sess.arr[id] = tensor_x[id] * tensor_theory_params[0];
// CHECK-NEXT:     }
// CHECK-NEXT:     float _d_out = 0.F;
// CHECK-NEXT:     float out = 0.;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t2 = {{0U|0UL|0ULL}};
// CHECK-NEXT:     for (id0 = 0; ; id0++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(id0 < nVals))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t2++;
// CHECK-NEXT:         clad::push(_t3, out);
// CHECK-NEXT:         out += std::exp(-sess.arr[id0]);
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_out += 1;
// CHECK-NEXT:     for (;; _t2--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t2)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         id0--;
// CHECK-NEXT:         {
// CHECK-NEXT:             out = clad::pop(_t3);
// CHECK-NEXT:             float _r_d1 = _d_out;
// CHECK-NEXT:             float _r0 = 0.F;
// CHECK-NEXT:             _r0 += _r_d1 * clad::custom_derivatives::std::exp_pushforward(-sess.arr[id0], 1.F).pushforward;
// CHECK-NEXT:             _d_sess.arr[id0] += -_r0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         id--;
// CHECK-NEXT:         {
// CHECK-NEXT:             sess.arr[id] = clad::pop(_t1);
// CHECK-NEXT:             float _r_d0 = _d_sess.arr[id];
// CHECK-NEXT:             _d_sess.arr[id] = 0.F;
// CHECK-NEXT:             _d_tensor_theory_params[0] += tensor_x[id] * _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

