// RUN: %cladclang -std=c++14 %s -I%S/../../include -oSTLCustomDerivatives.out 2>&1 | %filecheck %s
// RUN: ./STLCustomDerivatives.out | %filecheck_exec %s
// RUN: %cladclang -std=c++14 -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oSTLCustomDerivativesWithTBR.out
// RUN: ./STLCustomDerivativesWithTBR.out | %filecheck_exec %s
// XFAIL: valgrind

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

double fn19(Session const *session, float *tensor_theory_params) {
   Session const &sess = session[0];
   auto &arr = sess.arr;
   float out = 0.;
   for (int id = 0; id < nVals; id++) {
      out += arr[id] * tensor_theory_params[0];
   }
   return out;
}

double fn20(std::vector<double>::iterator start, std::vector<double>::iterator end) {
  double sum = 0;
  int u = 1;
  for (auto it = start; it != end; it++) {
    sum += u * *it;
    u += 2;
  }
  return sum;
}

/// This is the function we want to differentiate.
/// It uses a shared_ptr to demonstrate safe ownership in a differentiable context.
double simple_func(std::shared_ptr<double> x_ptr) {
    double x = *x_ptr;
    return x * x + 2.0 * x;
}

double fn21(double x) {
    std::shared_ptr<double> x_ptr = std::make_shared<double>(x);
    return simple_func(x_ptr);
}

double weak_fn(std::weak_ptr<double> x_ptr) {
  std::shared_ptr<double> s = x_ptr.lock();
  double x = *s;
  return x * x + 2.0 * x;
}

double fn22(double x) {
    std::shared_ptr<double> s_ptr = std::make_shared<double>(x);
    std::weak_ptr<double> w_ptr = s_ptr;
    return weak_fn(w_ptr);
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
    INIT_GRADIENT(fn20);

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
    auto d_fn19 = clad::gradient(fn19, "tensor_theory_params");

    std::vector<double> v{1, 3, 5, 7, 9};
    std::vector<double> dv(5, 0);
    auto dbegin = dv.begin();
    auto dend = dv.end();
    fn20_grad.execute(v.begin(), v.end(), &dbegin, &dend);
    // CHECK-EXEC: {1.00, 3.00, 5.00, 7.00, 9.00}
    printf("{");
    for (auto i = 0; i < dv.size(); ++i) {
      printf("%.2f", dv[i]);
      if (i != dv.size() - 1)
        printf(", ");
    }
    printf("}\n");

    INIT_GRADIENT(fn21);
    TEST_GRADIENT(fn21, /*numOfDerivativeArgs=*/1, 3, &d_i);  // CHECK-EXEC: {8.00}
    INIT_GRADIENT(fn22);
    TEST_GRADIENT(fn22, /*numOfDerivativeArgs=*/1, 3, &d_i);  // CHECK-EXEC: {8.00}
}

// CHECK: void fn1_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:     std::vector<double> vec;
// CHECK-NEXT:     std::vector<double> _d_vec = {};
// CHECK-NEXT:     clad::zero_init(_d_vec);
// CHECK-NEXT:     std::vector<double> _t0 = vec;
// CHECK-NEXT:     {{.*}}class_functions::push_back_reverse_forw(&vec, u, &_d_vec, *_d_u);
// CHECK-NEXT:     std::vector<double> _t1 = vec;
// CHECK-NEXT:     {{.*}}class_functions::push_back_reverse_forw(&vec, v, &_d_vec, *_d_v);
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t2 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t3 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {
// CHECK-NEXT:         _t2.adjoint += 1;
// CHECK-NEXT:         {{.*size_type|size_t}} _r0 = {{0U|0UL}};
// CHECK-NEXT:         _t3.adjoint += 1;
// CHECK-NEXT:         {{.*size_type|size_t}} _r1 = {{0U|0UL}};
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
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t2 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     double &_d_ref = _t2.adjoint;
// CHECK-NEXT:     double &ref = _t2.value;
// CHECK-NEXT:     ref += u;
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t3 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t4 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {
// CHECK-NEXT:         _t3.adjoint += 1;
// CHECK-NEXT:         {{.*}} _r1 = {{0U|0UL}};
// CHECK-NEXT:         _t4.adjoint += 1;
// CHECK-NEXT:         {{.*}} _r2 = {{0U|0UL}};
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_ref;
// CHECK-NEXT:         *_d_u += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {{.*}} _r0 = {{0U|0UL}};
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
// CHECK-NEXT:     double *_d_ref0 = nullptr;
// CHECK-NEXT:     double *ref0 = {};
// CHECK-NEXT:     double *_d_ref1 = nullptr;
// CHECK-NEXT:     double *ref1 = {};
// CHECK-NEXT:     double *_d_ref2 = nullptr;
// CHECK-NEXT:     double *ref2 = {};
// CHECK-NEXT:     double *_d_ref00 = nullptr;
// CHECK-NEXT:     double *ref00 = {};
// CHECK-NEXT:     double *_d_ref10 = nullptr;
// CHECK-NEXT:     double *ref10 = {};
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     std::vector<double> vec;
// CHECK-NEXT:     std::vector<double> _d_vec = {};
// CHECK-NEXT:     clad::zero_init(_d_vec);
// CHECK-NEXT:     std::vector<double> _t0 = vec;
// CHECK-NEXT:     {{.*}}class_functions::resize_reverse_forw(&vec, 3, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t1 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:         _d_ref0 = &_t1.adjoint;
// CHECK-NEXT:         ref0 = &_t1.value;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t2 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:         _d_ref1 = &_t2.adjoint;
// CHECK-NEXT:         ref1 = &_t2.value;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t3 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 2, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:         _d_ref2 = &_t3.adjoint;
// CHECK-NEXT:         ref2 = &_t3.value;
// CHECK-NEXT:         *ref0 = u;
// CHECK-NEXT:         *ref1 = v;
// CHECK-NEXT:         *ref2 = u + v;
// CHECK-NEXT:     }
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t4 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t5 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t6 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 2, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     res = _t4.value + _t5.value + _t6.value;
// CHECK-NEXT:     std::vector<double> _t7 = vec;
// CHECK-NEXT:     {{.*}}class_functions::clear_reverse_forw(&vec, &_d_vec);
// CHECK-NEXT:     std::vector<double> _t8 = vec;
// CHECK-NEXT:     {{.*}}class_functions::resize_reverse_forw(&vec, 2, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t9 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:         _d_ref00 = &_t9.adjoint;
// CHECK-NEXT:         ref00 = &_t9.value;
// CHECK-NEXT:         {{.*}}ValueAndAdjoint<double &, double &> _t10 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:         _d_ref10 = &_t10.adjoint;
// CHECK-NEXT:         ref10 = &_t10.value;
// CHECK-NEXT:         *ref00 = u;
// CHECK-NEXT:         *ref10 = u;
// CHECK-NEXT:     }
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t11 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t12 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     res += _t11.value + _t12.value;
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d6 = _d_res;
// CHECK-NEXT:         _t11.adjoint += _r_d6;
// CHECK-NEXT:         {{.*}} _r10 = {{0U|0UL}};
// CHECK-NEXT:         _t12.adjoint += _r_d6;
// CHECK-NEXT:         {{.*}} _r11 = {{0U|0UL}};
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d5 = *_d_ref10;
// CHECK-NEXT:             *_d_ref10 = 0.;
// CHECK-NEXT:             *_d_u += _r_d5;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d4 = *_d_ref00;
// CHECK-NEXT:             *_d_ref00 = 0.;
// CHECK-NEXT:             *_d_u += _r_d4;
// CHECK-NEXT:         }
// CHECK-NEXT:         {{.*}} _r9 = {{0U|0UL}};
// CHECK-NEXT:         {{.*}} _r8 = {{0U|0UL}};
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r7 = {{0U|0UL}};
// CHECK-NEXT:         vec = _t8;
// CHECK-NEXT:         {{.*}}class_functions::resize_pullback(&vec, 2, &_d_vec, &_r7);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         vec = _t7;
// CHECK-NEXT:         clad::custom_derivatives::class_functions::clear_pullback(&vec, &_d_vec);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d3 = _d_res;
// CHECK-NEXT:         _d_res = 0.;
// CHECK-NEXT:         _t4.adjoint += _r_d3;
// CHECK-NEXT:         {{.*}} _r4 = {{0U|0UL}};
// CHECK-NEXT:         _t5.adjoint += _r_d3;
// CHECK-NEXT:         {{.*}} _r5 = {{0U|0UL}};
// CHECK-NEXT:         _t6.adjoint += _r_d3;
// CHECK-NEXT:         {{.*}} _r6 = {{0U|0UL}};
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d2 = *_d_ref2;
// CHECK-NEXT:             *_d_ref2 = 0.;
// CHECK-NEXT:             *_d_u += _r_d2;
// CHECK-NEXT:             *_d_v += _r_d2;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d1 = *_d_ref1;
// CHECK-NEXT:             *_d_ref1 = 0.;
// CHECK-NEXT:             *_d_v += _r_d1;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d0 = *_d_ref0;
// CHECK-NEXT:             *_d_ref0 = 0.;
// CHECK-NEXT:             *_d_u += _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:         {{.*}} _r3 = {{0U|0UL}};
// CHECK-NEXT:         {{.*}} _r2 = {{0U|0UL}};
// CHECK-NEXT:         {{.*}} _r1 = {{0U|0UL}};
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
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t0 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 0, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t1 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 1, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t2 = {{.*}}class_functions::operator_subscript_reverse_forw(&vec, 2, &_d_vec, {{0U|0UL|0}});
// CHECK-NEXT:     {
// CHECK-NEXT:         _t0.adjoint += 1;
// CHECK-NEXT:         {{.*}} _r1 = {{0U|0UL}};
// CHECK-NEXT:         _t1.adjoint += 1;
// CHECK-NEXT:         {{.*}} _r2 = {{0U|0UL}};
// CHECK-NEXT:         _t2.adjoint += 1;
// CHECK-NEXT:         {{.*}} _r3 = {{0U|0UL}};
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
// CHECK-NEXT:          clad::ValueAndAdjoint<double &, double &> _t2 = {{.*}}operator_subscript_reverse_forw(&a, 1, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:          _t2.value = x * x;
// CHECK-NEXT:          clad::ValueAndAdjoint<double &, double &> _t3 = clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&a, 1, &_d_a, 0);
// CHECK-NEXT:          {
// CHECK-NEXT:              _t3.adjoint += 1;
// CHECK-NEXT:              {{.*}} _r1 = {{0U|0UL}};
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              double _r_d0 = _t2.adjoint;
// CHECK-NEXT:              _t2.adjoint = 0.;
// CHECK-NEXT:              *_d_x += _r_d0 * x;
// CHECK-NEXT:              *_d_x += x * _r_d0;
// CHECK-NEXT:              {{.*}} _r0 = {{0U|0UL}};
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
// CHECK-NEXT:        clad::tape<clad::ValueAndAdjoint<double &, double &> > _t2 = {};
// CHECK-NEXT:        std::array<double, 3> _d_a = {{.*}};
// CHECK-NEXT:        std::array<double, 3> a;
// CHECK-NEXT:        std::array<double, 3> _t0 = a;
// CHECK-NEXT:        {{.*}}fill_reverse_forw(&a, x, &_d_a, *_d_x);
// CHECK-NEXT:        double _d_res = 0.;
// CHECK-NEXT:        double res = 0;
// CHECK-NEXT:        unsigned {{long|int}} _t1 = {{0U|0UL}};
// CHECK-NEXT:        for (i = 0; i < a.size(); ++i) {
// CHECK-NEXT:            _t1++;
// CHECK-NEXT:            clad::push(_t2, {{.*}}at_reverse_forw(&a, i, &_d_a, _d_i));
// CHECK-NEXT:            res += clad::back(_t2).value;
// CHECK-NEXT:        }
// CHECK-NEXT:        _d_res += 1;
// CHECK-NEXT:        for (; _t1; _t1--) {
// CHECK-NEXT:            --i;
// CHECK-NEXT:            {
// CHECK-NEXT:                double _r_d0 = _d_res;
// CHECK-NEXT:                clad::back(_t2).adjoint += _r_d0;
// CHECK-NEXT:                size_t _r0 = {{0U|0UL}};
// CHECK-NEXT:                _d_i += _r0;
// CHECK-NEXT:                clad::pop(_t2);
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
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t0 = {{.*}}operator_subscript_reverse_forw(&a, 0, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:         _t0.value = 5;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t1 = {{.*}}operator_subscript_reverse_forw(&a, 1, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:         _t1.value = y;
// CHECK-NEXT:         std::array<double, 3> _d__b = {{.*}};
// CHECK-NEXT:         std::array<double, 3> _b0;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t2 = {{.*}}operator_subscript_reverse_forw(&_b0, 0, &_d__b, {{0U|0UL|0}});
// CHECK-NEXT:         _t2.value = x;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t3 = {{.*}}operator_subscript_reverse_forw(&_b0, 1, &_d__b, {{0U|0UL|0}});
// CHECK-NEXT:         _t3.value = 0;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t4 = {{.*}}operator_subscript_reverse_forw(&_b0, 2, &_d__b, {{0U|0UL|0}});
// CHECK-NEXT:         _t4.value = x * x;
// CHECK-NEXT:         std::array<double, 3> _d_b = {{.*}};
// CHECK-NEXT:         const std::array<double, 3> b = _b0;
// CHECK:              clad::ValueAndAdjoint<double &, double &> _t{{7|8}} = {{.*}}back_reverse_forw(&a, &_d_a);
// CHECK-NEXT:         {{.*}}value_type _t6 = b.front();
// CHECK-NEXT:         {{.*}}value_type _t5 = b.at(2);
// CHECK-NEXT:         {
// CHECK-NEXT:             _t{{7|8}}.adjoint += 1 * _t5 * _t6;
// CHECK:                  {{.*}}front_pullback(&b, _t{{7|8}}.value * 1 * _t5, &_d_b);
// CHECK-NEXT:             {{.*size_type|size_t}} _r5 = {{0U|0UL}};
// CHECK-NEXT:             {{.*}}at_pullback(&b, 2, _t{{7|8}}.value * _t6 * 1, &_d_b, &_r5);
// CHECK-NEXT:             {{.*size_type|size_t}} _r6 = {{0U|0UL}};
// CHECK-NEXT:             {{.*}}operator_subscript_pullback(&b, 1, 1, &_d_b, &_r6);
// CHECK-NEXT:         }
// CHECK-NEXT:         {{.*}}constructor_pullback(_b0, &_d_b, &_d__b);
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d4 = _t4.adjoint;
// CHECK-NEXT:             _t4.adjoint = 0.;
// CHECK-NEXT:             *_d_x += _r_d4 * x;
// CHECK-NEXT:             *_d_x += x * _r_d4;
// CHECK-NEXT:             {{.*size_type|size_t}} _r4 = {{0U|0UL}};
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d3 = _t3.adjoint;
// CHECK-NEXT:             _t3.adjoint = 0.;
// CHECK-NEXT:             {{.*size_type|size_t}} _r3 = {{0U|0UL}};
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d2 = _t2.adjoint;
// CHECK-NEXT:             _t2.adjoint = 0.;
// CHECK-NEXT:             *_d_x += _r_d2;
// CHECK-NEXT:             {{.*size_type|size_t}} _r2 = {{0U|0UL}};
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d1 = _t1.adjoint;
// CHECK-NEXT:             _t1.adjoint = 0.;
// CHECK-NEXT:             *_d_y += _r_d1;
// CHECK-NEXT:             {{.*size_type|size_t}} _r1 = {{0U|0UL}};
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d0 = _t0.adjoint;
// CHECK-NEXT:             _t0.adjoint = 0.;
// CHECK-NEXT:             {{.*size_type|size_t}} _r0 = {{0U|0UL}};
// CHECK-NEXT:         }
// CHECK-NEXT:     }

// CHECK:     void fn8_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:         std::array<double, 50> _d_a = {{.*}};
// CHECK-NEXT:         std::array<double, 50> a;
// CHECK-NEXT:         std::array<double, 50> _t0 = a;
// CHECK-NEXT:         {{.*}}fill_reverse_forw(&a, y + x + x, &_d_a, 0.);
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t1 = {{.*}}operator_subscript_reverse_forw(&a, 49, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t2 = {{.*}}operator_subscript_reverse_forw(&a, 3, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:         {
// CHECK-NEXT:             _t1.adjoint += 1;
// CHECK-NEXT:             {{.*size_type|size_t}} _r1 = {{0U|0UL}};
// CHECK-NEXT:             _t2.adjoint += 1;
// CHECK-NEXT:             {{.*size_type|size_t}} _r2 = {{0U|0UL}};
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
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t0 = {{.*}}operator_subscript_reverse_forw(&a, 1, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:         _t0.value = 2 * x;
// CHECK-NEXT:         clad::ValueAndAdjoint<double &, double &> _t1 = clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&a, 1, &_d_a, 0);
// CHECK-NEXT:         {
// CHECK-NEXT:             _t1.adjoint += 1;
// CHECK-NEXT:             {{.*size_type|size_t}} _r1 = {{0U|0UL}};
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*}} _r_d0 = _t0.adjoint;
// CHECK-NEXT:             _t0.adjoint = 0.;
// CHECK-NEXT:             *_d_x += 2 * _r_d0;
// CHECK-NEXT:             {{.*size_type|size_t}} _r0 = {{0U|0UL}};
// CHECK-NEXT:         }
// CHECK-NEXT:     }

// CHECK:      void fn10_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:          size_t _d_i = {{0U|0UL|0}};
// CHECK-NEXT:          size_t i = {{0U|0UL|0}};
// CHECK-NEXT:          {{.*}}tape<{{.*}}vector<double> > _t1 = {};
// CHECK-NEXT:          size_t _d_i0 = {{0U|0UL|0}};
// CHECK-NEXT:          size_t i0 = {{0U|0UL|0}};
// CHECK-NEXT:          clad::tape<clad::ValueAndAdjoint<double &, double &> > _t3 = {};
// CHECK-NEXT:          {{.*}}vector<double> v;
// CHECK-NEXT:          {{.*}}vector<double> _d_v = {};
// CHECK-NEXT:          clad::zero_init(_d_v);
// CHECK-NEXT:          {{.*}} _t0 = {{0U|0UL|0}};
// CHECK-NEXT:          for (i = 0; i < 3; ++i) {
// CHECK-NEXT:              _t0++;
// CHECK-NEXT:              {{.*}}push(_t1, v);
// CHECK-NEXT:              {{.*}}push_back_reverse_forw(&v, x, &_d_v, *_d_x);
// CHECK-NEXT:          }
// CHECK-NEXT:          double _d_res = 0.;
// CHECK-NEXT:          double res = 0;
// CHECK-NEXT:          {{.*}} _t2 = {{0U|0UL|0}};
// CHECK-NEXT:          for (i0 = 0; i0 < v.size(); ++i0) {
// CHECK-NEXT:              _t2++;
// CHECK-NEXT:              clad::push(_t3, {{.*}}at_reverse_forw(&v, i0, &_d_v, _d_i0));
// CHECK-NEXT:              res += clad::back(_t3).value;
// CHECK-NEXT:          }
// CHECK-NEXT:          {{.*}}vector<double> _t4 = v;
// CHECK-NEXT:          v.assign(3, 0);
// CHECK-NEXT:          {{.*}}vector<double> _t5 = v;
// CHECK-NEXT:          v.assign(2, y);
// CHECK-NEXT:          {{.*}}ValueAndAdjoint<double &, double &> _t6 = {{.*}}operator_subscript_reverse_forw(&v, 0, &_d_v, {{0U|0UL|0}});
// CHECK-NEXT:          {{.*}}ValueAndAdjoint<double &, double &> _t7 = {{.*}}operator_subscript_reverse_forw(&v, 1, &_d_v, {{0U|0UL|0}});
// CHECK-NEXT:          {{.*}}ValueAndAdjoint<double &, double &> _t8 = {{.*}}operator_subscript_reverse_forw(&v, 2, &_d_v, {{0U|0UL|0}});
// CHECK-NEXT:          {
// CHECK-NEXT:              _d_res += 1;
// CHECK-NEXT:              _t6.adjoint += 1;
// CHECK-NEXT:              {{.*size_type|size_t}} _r4 = {{0U|0UL|0}};
// CHECK-NEXT:              _t7.adjoint += 1;
// CHECK-NEXT:              {{.*size_type|size_t}} _r5 = {{0U|0UL|0}};
// CHECK-NEXT:              _t8.adjoint += 1;
// CHECK-NEXT:              {{.*size_type|size_t}} _r6 = {{0U|0UL|0}};
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              {{.*size_type|size_t}} _r3 = {{0U|0UL|0}};
// CHECK-NEXT:              v = _t5;
// CHECK-NEXT:              {{.*}}assign_pullback(&v, 2, y, &_d_v, &_r3, &*_d_y);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              {{.*size_type|size_t}} _r1 = {{0U|0UL|0}};
// CHECK-NEXT:              {{.*}}value_type _r2 = 0.;
// CHECK-NEXT:              v = _t4;
// CHECK-NEXT:              {{.*}}assign_pullback(&v, 3, 0, &_d_v, &_r1, &_r2);
// CHECK-NEXT:          }
// CHECK-NEXT:          for (; _t2; _t2--) {
// CHECK-NEXT:              --i0;
// CHECK-NEXT:              {
// CHECK-NEXT:                  double _r_d0 = _d_res;
// CHECK-NEXT:                  clad::back(_t3).adjoint += _r_d0;
// CHECK-NEXT:                  size_t _r0 = {{0U|0UL|0}};
// CHECK-NEXT:                  _d_i0 += _r0;
// CHECK-NEXT:                  {{.*}}pop(_t3);
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          for (; _t0; _t0--) {
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
// CHECK-NEXT:          double _t4 = v.capacity();
// CHECK-NEXT:          double _t5 = v.size();
// CHECK-NEXT:          res += y * _t4 + x * _t5;
// CHECK-NEXT:          _d_res += 1;
// CHECK-NEXT:          {
// CHECK-NEXT:              double _r_d0 = _d_res;
// CHECK-NEXT:              *_d_y += _r_d0 * _t4;
// CHECK-NEXT:              *_d_x += _r_d0 * _t5;
// CHECK-NEXT:          }
// CHECK-NEXT:          v = _t3;
// CHECK-NEXT:          {
// CHECK-NEXT:              v = _t2;
// CHECK-NEXT:              {{.*}}push_back_pullback(&v, x, &_d_v, &*_d_x);
// CHECK-NEXT:          }
// CHECK-NEXT:          *_d_x += _d_res * _t1;
// CHECK-NEXT:          {
// CHECK-NEXT:              {{.*size_type|size_t}} _r0 = {{0U|0UL|0}};
// CHECK-NEXT:              v = _t0;
// CHECK-NEXT:          }
// CHECK-NEXT:      }

// CHECK:      void fn12_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:          std::vector<double> a;
// CHECK-NEXT:          std::vector<double> _d_a = {};
// CHECK-NEXT:          clad::zero_init(_d_a);
// CHECK-NEXT:          std::vector<double> _t0 = a;
// CHECK-NEXT:          {{.*}}push_back_reverse_forw(&a, 0{{.*}}, &_d_a, 0{{.*}});
// CHECK-NEXT:          {{.*}}ValueAndAdjoint<double &, double &> _t1 = {{.*}}operator_subscript_reverse_forw(&a, 0, &_d_a, {{0U|0UL|0}});
// CHECK-NEXT:          _t1.value = x * x;
// CHECK-NEXT:          clad::ValueAndAdjoint<double &, double &> _t2 = clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&a, 0, &_d_a, 0);
// CHECK-NEXT:          {
// CHECK-NEXT:              _t2.adjoint += 1;
// CHECK-NEXT:              {{.*size_type|size_t}} _r2 = 0{{.*}};
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              double _r_d0 = _t1.adjoint;
// CHECK-NEXT:              _t1.adjoint = 0{{.*}};
// CHECK-NEXT:              *_d_x += _r_d0 * x;
// CHECK-NEXT:              *_d_x += x * _r_d0;
// CHECK-NEXT:              {{.*size_type|size_t}} _r1 = 0{{.*}};
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
// CHECK-NEXT:      clad::ValueAndAdjoint<double &, double &> _t0 = clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 1, &_d_ls, {{0U|0UL|0}});
// CHECK-NEXT:      clad::ValueAndAdjoint<double &, double &> _t2 = clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 0, &_d_ls, {{0U|0UL|0}});
// CHECK-NEXT:      {{.*}}value_type _t1 = _t2.value;
// CHECK-NEXT:      {
// CHECK-NEXT:          _t0.adjoint += 1;
// CHECK-NEXT:          {{.*size_type|size_t}} _r1 = 0{{.*}};
// CHECK-NEXT:          _t2.adjoint += 2 * -1;
// CHECK-NEXT:          {{.*size_type|size_t}} _r2 = 0{{.*}};
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          clad::array<{{double|std::vector<double, std::allocator<double> >::value_type}}> _r0 = {{2U|2UL|2ULL}};
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
// CHECK-NEXT:      clad::tape<clad::ValueAndAdjoint<double &, double &> > _t3 = {};
// CHECK-NEXT:      clad::tape<clad::ValueAndAdjoint<double &, double &> > _t4 = {};
// CHECK-NEXT:      clad::tape<clad::ValueAndAdjoint<double &, double &> > _t5 = {};
// CHECK-NEXT:      {{.*}}allocator_type alloc;
// CHECK-NEXT:      {{.*}}allocator_type _d_alloc = {};
// CHECK-NEXT:      clad::zero_init(_d_alloc);
// CHECK-NEXT:      unsigned {{int|long|long long}} _t0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:      for (i = 0; i < 3; ++i) {
// CHECK-NEXT:          _t0++;
// CHECK-NEXT:          clad::push(_t1, std::move(_d_ls));
// CHECK-NEXT:          clad::push(_t2, std::move(ls)) , ls = {u, v}, alloc;
// CHECK-NEXT:          _d_ls = ls;
// CHECK-NEXT:          clad::zero_init(_d_ls);
// CHECK-NEXT:          clad::push(_t3, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 1, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:          clad::push(_t4, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 0, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:          clad::back(_t3).value += clad::back(_t4).value;
// CHECK-NEXT:          clad::push(_t5, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 1, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:          u = clad::back(_t5).value;
// CHECK-NEXT:      }
// CHECK-NEXT:      *_d_u += 1;
// CHECK-NEXT:      for (; _t0; _t0--) {
// CHECK-NEXT:          {
// CHECK-NEXT:              double _r_d1 = *_d_u;
// CHECK-NEXT:              *_d_u = 0.;
// CHECK-NEXT:              clad::back(_t5).adjoint += _r_d1;
// CHECK-NEXT:              {{.*size_type|size_t}} _r3 = 0{{.*}};
// CHECK-NEXT:              clad::pop(_t5);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              double _r_d0 = clad::back(_t3).adjoint;
// CHECK-NEXT:              clad::back(_t4).adjoint += _r_d0;
// CHECK-NEXT:              {{.*size_type|size_t}} _r2 = 0{{.*}};
// CHECK-NEXT:              clad::pop(_t4);
// CHECK-NEXT:              {{.*size_type|size_t}} _r1 = 0{{.*}};
// CHECK-NEXT:              clad::pop(_t3);
// CHECK-NEXT:          }
// CHECK-NEXT:          {
// CHECK-NEXT:              clad::array<{{double|std::vector<double, std::allocator<double> >::value_type}}> _r0 = {{2U|2UL|2ULL}};
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
// CHECK-NEXT:     clad::ValueAndAdjoint< {{.*}}, {{.*}} > _t0 = {{.*}}class_functions::constructor_reverse_forw(clad::ConstructorReverseForwTag<{{(std::)?}}unique_ptr{{.*}}(), p, _d_p);
// CHECK-NEXT:     std::unique_ptr{{.*}} up(static_cast<std::unique_ptr{{.*}}(_t0.value));
// CHECK-NEXT:     std::unique_ptr{{.*}} _d_up = static_cast<std::unique_ptr{{.*}}(_t0.adjoint);
// CHECK-NEXT:     {{.*}}ValueAndAdjoint<double &, double &> _t1 = {{.*}}class_functions::operator_star_reverse_forw(&up, &_d_up);
// CHECK-NEXT:     _t1.value += 5 * e;
// CHECK-NEXT:     clad::ValueAndAdjoint<double &, double &> _t2 = clad::custom_derivatives::class_functions::operator_star_reverse_forw(&up, &_d_up);
// CHECK-NEXT:     _t2.adjoint += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _t1.adjoint;
// CHECK-NEXT:         *_d_e += 5 * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_d += *_d_p;
// CHECK-NEXT: }

// CHECK-NEXT: void fn16_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<std::vector<double> > _t1 = {};
// CHECK-NEXT:     clad::tape<std::vector<double> > _t2 = {};
// CHECK-NEXT:     std::vector<double> vec = {};
// CHECK-NEXT:     std::vector<double> _d_vec{};
// CHECK-NEXT:     clad::tape<double> _t3 = {};
// CHECK-NEXT:     clad::tape<clad::ValueAndAdjoint<double &, double &> > _t4 = {};
// CHECK-NEXT:     {{.*}}allocator_type alloc;
// CHECK-NEXT:     {{.*}}allocator_type _d_alloc = {};
// CHECK-NEXT:     clad::zero_init(_d_alloc);
// CHECK-NEXT:     double _d_prod = 0.;
// CHECK-NEXT:     double prod = 1;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:     for (i = 3; i >= 1; --i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, std::move(_d_vec));
// CHECK-NEXT:         clad::push(_t2, std::move(vec)) , vec = i, v + u, alloc;
// CHECK-NEXT:         _d_vec = vec;
// CHECK-NEXT:         clad::zero_init(_d_vec);
// CHECK-NEXT:         clad::push(_t3, prod);
// CHECK-NEXT:         clad::push(_t4, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&vec, i - 1, &_d_vec, {{0U|0UL|0}}));
// CHECK-NEXT:         prod *= clad::back(_t4).value;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_prod += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         ++i;
// CHECK-NEXT:         {
// CHECK-NEXT:             prod = clad::pop(_t3);
// CHECK-NEXT:             double _r_d0 = _d_prod;
// CHECK-NEXT:             _d_prod = 0.;
// CHECK-NEXT:             _d_prod += _r_d0 * clad::back(_t4).value;
// CHECK-NEXT:             clad::back(_t4).adjoint += prod * _r_d0;
// CHECK-NEXT:             {{.*size_type|size_t}} _r2 = 0{{.*}};
// CHECK-NEXT:             _d_i += _r2;
// CHECK-NEXT:             clad::pop(_t4);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {{.*size_type|size_t}} _r0 = 0{{.*}};
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
// CHECK-NEXT:     clad::tape<clad::ValueAndAdjoint<double &, double &> > _t3 = {};
// CHECK-NEXT:     clad::tape<clad::ValueAndAdjoint<double &, double &> > _t4 = {};
// CHECK-NEXT:     clad::tape<clad::ValueAndAdjoint<double &, double &> > _t5 = {};
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:     for (i = 0; i < 3; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, std::move(_d_ls));
// CHECK-NEXT:         clad::push(_t2, std::move(ls)) , ls = {{.*{u, v}.*}};
// CHECK-NEXT:         _d_ls = ls;
// CHECK-NEXT:         clad::zero_init(_d_ls);
// CHECK-NEXT:         clad::push(_t3, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 1, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:         clad::push(_t4, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 0, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:         clad::back(_t3).value += clad::back(_t4).value;
// CHECK-NEXT:         clad::push(_t5, clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&ls, 1, &_d_ls, {{0U|0UL|0}}));
// CHECK-NEXT:         u = clad::back(_t5).value;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_u += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d1 = *_d_u;
// CHECK-NEXT:             *_d_u = 0.;
// CHECK-NEXT:             clad::back(_t5).adjoint += _r_d1;
// CHECK-NEXT:             {{.*size_type|size_t}} _r{{3|4}} = {{0U|0UL|0ULL}};
// CHECK-NEXT:             clad::pop(_t5);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d0 = clad::back(_t3).adjoint;
// CHECK-NEXT:             clad::back(_t4).adjoint += _r_d0;
// CHECK-NEXT:             {{.*size_type|size_t}} _r{{2|3}} = {{0U|0UL|0ULL}};
// CHECK-NEXT:             clad::pop(_t4);
// CHECK-NEXT:             {{.*size_type|size_t}} _r{{1|2}} = {{0U|0UL|0ULL}};
// CHECK-NEXT:             clad::pop(_t3);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             clad::array<{{double|std::vector<double, std::allocator<double> >::value_type}}> _r0 = {{2U|2UL|2ULL}};
// CHECK:                  clad::custom_derivatives::class_functions::constructor_pullback({u, v},{{.*}} &_d_ls, &_r0{{.*}});
// CHECK-NEXT:             *_d_u += _r0[0];
// CHECK-NEXT:             *_d_v += _r0[1];
// CHECK-NEXT:             _d_ls = clad::pop(_t1);
// CHECK-NEXT:             ls = clad::pop(_t2);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn20_grad({{.*}}::iterator start, {{.*}}::iterator end, {{.*}}::iterator *_d_start, {{.*}}::iterator *_d_end) {
// CHECK-NEXT:     {{.*}} it = {};
// CHECK-NEXT:     {{.*}} _d_it{};
// CHECK-NEXT:     clad::tape<{{.*}}> _t2 = {};
// CHECK-NEXT:     clad::tape<double> _t3 = {};
// CHECK-NEXT:     clad::tape<clad::ValueAndAdjoint<{{.*}}> _t4 = {};
// CHECK-NEXT:     clad::tape<int> _t5 = {};
// CHECK-NEXT:     double _d_sum = 0.;
// CHECK-NEXT:     double sum = 0;
// CHECK-NEXT:     int _d_u = 0;
// CHECK-NEXT:     int u = 1;
// CHECK-NEXT:     {{.*}} _t0 = 0{{.*}};
// CHECK-NEXT:     clad::ValueAndAdjoint<{{.*}}> _t1 = clad::custom_derivatives::class_functions::constructor_reverse_forw(clad::ConstructorReverseForwTag<{{.*}}>(), start, (*_d_start));
// CHECK-NEXT:     it = _t1.value;
// CHECK-NEXT:     _d_it = _t1.adjoint;
// CHECK-NEXT:     for (; it != end; clad::push(_t2, it) , {{.*}}class_functions::operator_plus_plus_reverse_forw(&it, 0, &_d_it, 0)) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t4, {{.*}}class_functions::operator_star_reverse_forw(&it, &_d_it));
// CHECK-NEXT:         sum += u * clad::push(_t3, clad::back(_t4).value);
// CHECK-NEXT:         clad::push(_t5, u);
// CHECK-NEXT:         u += 2;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_sum += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             int _r0 = 0;
// CHECK-NEXT:             it = clad::back(_t2);
// CHECK-NEXT:             {{.*}}class_functions::operator_plus_plus_pullback(&it, 0, {}, &_d_it, &_r0);
// CHECK-NEXT:             clad::pop(_t2);
// CHECK-NEXT:         }
// CHECK-NEXT:         u = clad::pop(_t5);
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d0 = _d_sum;
// CHECK-NEXT:             _d_u += _r_d0 * clad::pop(_t3);
// CHECK-NEXT:             clad::back(_t4).adjoint += u * _r_d0;
// CHECK-NEXT:             clad::pop(_t4);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn18_grad_2(const Session *session, const float *tensor_x, float *tensor_theory_params, float *_d_tensor_theory_params) {
// CHECK-NEXT:     int _d_id = 0;
// CHECK-NEXT:     int id = 0;
// CHECK-NEXT:     int _d_id0 = 0;
// CHECK-NEXT:     int id0 = 0;
// CHECK-NEXT:     Session _d_sess = {{.*}}, nullptr};
// CHECK-NEXT:     const Session &sess = session[0];
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:     for (id = 0; id < nVals; id++) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         sess.arr[id] = tensor_x[id] * tensor_theory_params[0];
// CHECK-NEXT:     }
// CHECK-NEXT:     float _d_out = 0.F;
// CHECK-NEXT:     float out = 0.;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t1 = {{0U|0UL|0ULL}};
// CHECK-NEXT:     for (id0 = 0; id0 < nVals; id0++) {
// CHECK-NEXT:         _t1++;
// CHECK-NEXT:         out += std::exp(-sess.arr[id0]);
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_out += 1;
// CHECK-NEXT:     for (; _t1; _t1--) {
// CHECK-NEXT:         id0--;
// CHECK-NEXT:         {
// CHECK-NEXT:             float _r_d1 = _d_out;
// CHECK-NEXT:             float _r0 = 0.F;
// CHECK-NEXT:             _r0 += _r_d1 * clad::custom_derivatives::std::exp_pushforward(-sess.arr[id0], 1.F).pushforward;
// CHECK-NEXT:             _d_sess.arr[id0] += -_r0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         id--;
// CHECK-NEXT:         {
// CHECK-NEXT:             float _r_d0 = _d_sess.arr[id];
// CHECK-NEXT:             _d_sess.arr[id] = 0.F;
// CHECK-NEXT:             _d_tensor_theory_params[0] += tensor_x[id] * _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn19_grad_1(const Session *session, float *tensor_theory_params, float *_d_tensor_theory_params) {
// CHECK-NEXT:     int _d_id = 0;
// CHECK-NEXT:     int id = 0;
// CHECK-NEXT:     Session _d_sess = {{[{][{][}]}}, nullptr};
// CHECK-NEXT:     const Session &sess = session[0];
// CHECK-NEXT:     float *&_d_arr = _d_sess.arr;
// CHECK-NEXT:     float *const &arr = sess.arr;
// CHECK-NEXT:     float _d_out = 0.F;
// CHECK-NEXT:     float out = 0.;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:     for (id = 0; id < nVals; id++) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         out += arr[id] * tensor_theory_params[0];
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_out += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         id--;
// CHECK-NEXT:         {
// CHECK-NEXT:             float _r_d0 = _d_out;
// CHECK-NEXT:             _d_arr[id] += _r_d0 * tensor_theory_params[0];
// CHECK-NEXT:             _d_tensor_theory_params[0] += arr[id] * _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK-NEXT: void simple_func_pullback(std::shared_ptr<double> x_ptr, double _d_y, std::shared_ptr<double> *_d_x_ptr) {
// CHECK-NEXT:     clad::ValueAndAdjoint<{{.*}} &, {{.*}} &> _t0 = clad::custom_derivatives::class_functions::operator_star_reverse_forw(&x_ptr, &(*_d_x_ptr));
// CHECK-NEXT:     double _d_x = 0.;
// CHECK-NEXT:     double x = _t0.value;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_x += _d_y * x;
// CHECK-NEXT:         _d_x += x * _d_y;
// CHECK-NEXT:         _d_x += 2. * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT:     _t0.adjoint += _d_x;
// CHECK-NEXT: }

// CHECK: void fn21_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _t0 = x;
// CHECK-NEXT:     clad::ValueAndAdjoint< ::std::shared_ptr<double>, ::std::shared_ptr<double> > _t1 = clad::custom_derivatives::std::make_shared_reverse_forw(x, *_d_x);
// CHECK-NEXT:     clad::ValueAndAdjoint< ::std::shared_ptr<double>, ::std::shared_ptr<double> > _t2 = clad::custom_derivatives::class_functions::constructor_reverse_forw(clad::ConstructorReverseForwTag<{{(std::)?}}shared_ptr<double> >(), std::move(_t1.value), std::move(_t1.adjoint));
// CHECK-NEXT:     std::shared_ptr<double> x_ptr = _t2.value;
// CHECK-NEXT:     std::shared_ptr<double> _d_x_ptr = _t2.adjoint;
// CHECK-NEXT:     clad::ValueAndAdjoint< ::std::shared_ptr<double>, ::std::shared_ptr<double> > _t3 = clad::custom_derivatives::class_functions::constructor_reverse_forw(clad::ConstructorReverseForwTag<{{(std::)?}}shared_ptr<double> >(), x_ptr, _d_x_ptr);
// CHECK-NEXT:     {
// CHECK-NEXT:         std::shared_ptr<double> _r1 = _t3.adjoint;
// CHECK-NEXT:         simple_func_pullback(_t3.value, 1, &_r1);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         shared_ptr<{{.*}}double{{.*}}> _r0 = _t1.adjoint;
// CHECK-NEXT:         x = _t0;
// CHECK-NEXT:         clad::custom_derivatives::std::make_shared_pullback(x, _r0, &*_d_x);
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void weak_fn_pullback(std::weak_ptr<double> x_ptr, double _d_y, std::weak_ptr<double> *_d_x_ptr) {
// CHECK-NEXT:     clad::ValueAndAdjoint< ::std::shared_ptr<double>, ::std::shared_ptr<double> > _t0 = clad::custom_derivatives::class_functions::lock_reverse_forw(&x_ptr, &(*_d_x_ptr));
// CHECK-NEXT:     clad::ValueAndAdjoint< ::std::shared_ptr<double>, ::std::shared_ptr<double> > _t1 = clad::custom_derivatives::class_functions::constructor_reverse_forw(clad::ConstructorReverseForwTag<{{(std::)?}}shared_ptr<double> >(), std::move(_t0.value), std::move(_t0.adjoint));
// CHECK-NEXT:     std::shared_ptr<double> s = _t1.value;
// CHECK-NEXT:     std::shared_ptr<double> _d_s = _t1.adjoint;
// CHECK-NEXT:     clad::ValueAndAdjoint<{{.*}} &, {{.*}} &> _t2 = clad::custom_derivatives::class_functions::operator_star_reverse_forw(&s, &_d_s);
// CHECK-NEXT:     double _d_x = 0.;
// CHECK-NEXT:     double x = _t2.value;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_x += _d_y * x;
// CHECK-NEXT:         _d_x += x * _d_y;
// CHECK-NEXT:         _d_x += 2. * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT:     _t2.adjoint += _d_x;
// CHECK-NEXT:     shared_ptr<double> _r0 = _t0.adjoint;
// CHECK-NEXT: }

// CHECK: void fn22_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _t0 = x;
// CHECK-NEXT:     clad::ValueAndAdjoint< ::std::shared_ptr<double>, ::std::shared_ptr<double> > _t1 = clad::custom_derivatives::std::make_shared_reverse_forw(x, *_d_x);
// CHECK-NEXT:     clad::ValueAndAdjoint< ::std::shared_ptr<double>, ::std::shared_ptr<double> > _t2 = clad::custom_derivatives::class_functions::constructor_reverse_forw(clad::ConstructorReverseForwTag<{{(std::)?}}shared_ptr<double> >(), std::move(_t1.value), std::move(_t1.adjoint));
// CHECK-NEXT:     std::shared_ptr<double> s_ptr = _t2.value;
// CHECK-NEXT:     std::shared_ptr<double> _d_s_ptr = _t2.adjoint;
// CHECK-NEXT:     clad::ValueAndAdjoint< ::std::weak_ptr<double>, ::std::weak_ptr<double> > _t3 = clad::custom_derivatives::class_functions::constructor_reverse_forw(clad::ConstructorReverseForwTag<{{(std::)?}}weak_ptr<double> >(), s_ptr, _d_s_ptr);
// CHECK-NEXT:     clad::ValueAndAdjoint< ::std::weak_ptr<double>, ::std::weak_ptr<double> > _t4 = clad::custom_derivatives::class_functions::constructor_reverse_forw(clad::ConstructorReverseForwTag<{{(std::)?}}weak_ptr<double> >(), std::move(_t3.value), std::move(_t3.adjoint));
// CHECK-NEXT:     std::weak_ptr<double> w_ptr = _t4.value;
// CHECK-NEXT:     std::weak_ptr<double> _d_w_ptr = _t4.adjoint;
// CHECK-NEXT:     clad::ValueAndAdjoint< ::std::weak_ptr<double>, ::std::weak_ptr<double> > _t5 = clad::custom_derivatives::class_functions::constructor_reverse_forw(clad::ConstructorReverseForwTag<{{(std::)?}}weak_ptr<double> >(), w_ptr, _d_w_ptr);
// CHECK-NEXT:     {
// CHECK-NEXT:         std::weak_ptr<double> _r2 = _t5.adjoint;
// CHECK-NEXT:         weak_fn_pullback(_t5.value, 1, &_r2);
// CHECK-NEXT:     }
// CHECK-NEXT:     std::weak_ptr<double> _r1 = _t3.adjoint;
// CHECK-NEXT:     {
// CHECK-NEXT:         shared_ptr<{{.*}}double{{.*}}> _r0 = _t1.adjoint;
// CHECK-NEXT:         x = _t0;
// CHECK-NEXT:         clad::custom_derivatives::std::make_shared_pullback(x, _r0, &*_d_x);
// CHECK-NEXT:     }
// CHECK-NEXT: }

