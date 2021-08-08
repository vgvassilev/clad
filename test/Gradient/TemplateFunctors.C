// RUN: %cladclang %s -I%S/../../include -oTemplateFunctors.out 2>&1 | FileCheck %s 
// RUN: ./TemplateFunctors.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

template <typename T> struct Experiment {
  mutable T x, y;
  Experiment(T p_x, T p_y) : x(p_x), y(p_y) {}
  T operator()(T i, T j) { return x * i * i + y * j; }
  void setX(T val) { x = val; }
};

// CHECK: void operator_call_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     double _t5;
// CHECK-NEXT:     _t2 = this->x;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     _t3 = _t2 * _t1;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     _t5 = this->y;
// CHECK-NEXT:     _t4 = j;
// CHECK-NEXT:     double operator_call_return = _t3 * _t0 + _t5 * _t4;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 1 * _t0;
// CHECK-NEXT:         double _r1 = _r0 * _t1;
// CHECK-NEXT:         double _r2 = _t2 * _r0;
// CHECK-NEXT:         * _d_i += _r2;
// CHECK-NEXT:         double _r3 = _t3 * 1;
// CHECK-NEXT:         * _d_i += _r3;
// CHECK-NEXT:         double _r4 = 1 * _t4;
// CHECK-NEXT:         double _r5 = _t5 * 1;
// CHECK-NEXT:         * _d_j += _r5;
// CHECK-NEXT:     }
// CHECK-NEXT: }

template <> struct Experiment<long double> {
  mutable long double x, y;
  Experiment(long double p_x, long double p_y) : x(p_x), y(p_y) {}
  long double operator()(long double i, long double j) {
    return x * i * i * j + y * j * i;
  }
  void setX(long double val) { x = val; }
};

// CHECK: void operator_call_grad(long double i, long double j, clad::array_ref<long double> _d_i, clad::array_ref<long double> _d_j) {
// CHECK-NEXT:     long double _t0;
// CHECK-NEXT:     long double _t1;
// CHECK-NEXT:     long double _t2;
// CHECK-NEXT:     long double _t3;
// CHECK-NEXT:     long double _t4;
// CHECK-NEXT:     long double _t5;
// CHECK-NEXT:     long double _t6;
// CHECK-NEXT:     long double _t7;
// CHECK-NEXT:     long double _t8;
// CHECK-NEXT:     long double _t9;
// CHECK-NEXT:     _t3 = this->x;
// CHECK-NEXT:     _t2 = i;
// CHECK-NEXT:     _t4 = _t3 * _t2;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     _t5 = _t4 * _t1;
// CHECK-NEXT:     _t0 = j;
// CHECK-NEXT:     _t8 = this->y;
// CHECK-NEXT:     _t7 = j;
// CHECK-NEXT:     _t9 = _t8 * _t7;
// CHECK-NEXT:     _t6 = i;
// CHECK-NEXT:     long double operator_call_return = _t5 * _t0 + _t9 * _t6;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         long double _r0 = 1 * _t0;
// CHECK-NEXT:         long double _r1 = _r0 * _t1;
// CHECK-NEXT:         long double _r2 = _r1 * _t2;
// CHECK-NEXT:         long double _r3 = _t3 * _r1;
// CHECK-NEXT:         * _d_i += _r3;
// CHECK-NEXT:         long double _r4 = _t4 * _r0;
// CHECK-NEXT:         * _d_i += _r4;
// CHECK-NEXT:         long double _r5 = _t5 * 1;
// CHECK-NEXT:         * _d_j += _r5;
// CHECK-NEXT:         long double _r6 = 1 * _t6;
// CHECK-NEXT:         long double _r7 = _r6 * _t7;
// CHECK-NEXT:         long double _r8 = _t8 * _r6;
// CHECK-NEXT:         * _d_j += _r8;
// CHECK-NEXT:         long double _r9 = _t9 * 1;
// CHECK-NEXT:         * _d_i += _r9;
// CHECK-NEXT:     }
// CHECK-NEXT: }

template <typename T> struct ExperimentConstVolatile {
  mutable T x, y;
  ExperimentConstVolatile(T p_x, T p_y) : x(p_x), y(p_y) {}
  T operator()(T i, T j) const volatile { return x * i * i + y * j; }
  void setX(T val) const volatile { x = val; }
};

// CHECK: void operator_call_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const volatile {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     volatile double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     volatile double _t5;
// CHECK-NEXT:     _t2 = this->x;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     _t3 = _t2 * _t1;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     _t5 = this->y;
// CHECK-NEXT:     _t4 = j;
// CHECK-NEXT:     double operator_call_return = _t3 * _t0 + _t5 * _t4;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 1 * _t0;
// CHECK-NEXT:         double _r1 = _r0 * _t1;
// CHECK-NEXT:         double _r2 = _t2 * _r0;
// CHECK-NEXT:         * _d_i += _r2;
// CHECK-NEXT:         double _r3 = _t3 * 1;
// CHECK-NEXT:         * _d_i += _r3;
// CHECK-NEXT:         double _r4 = 1 * _t4;
// CHECK-NEXT:         double _r5 = _t5 * 1;
// CHECK-NEXT:         * _d_j += _r5;
// CHECK-NEXT:     }
// CHECK-NEXT: }

template <> struct ExperimentConstVolatile<long double> {
  mutable long double x, y;
  ExperimentConstVolatile(long double p_x, long double p_y) : x(p_x), y(p_y) {}
  long double operator()(long double i, long double j) const volatile {
    return x * i * i * j + y * j * i;
  }
  void setX(long double val) const volatile { x = val; }
};

// CHECK: void operator_call_grad(long double i, long double j, clad::array_ref<long double> _d_i, clad::array_ref<long double> _d_j) const volatile {
// CHECK-NEXT:     long double _t0;
// CHECK-NEXT:     long double _t1;
// CHECK-NEXT:     long double _t2;
// CHECK-NEXT:     volatile long double _t3;
// CHECK-NEXT:     long double _t4;
// CHECK-NEXT:     long double _t5;
// CHECK-NEXT:     long double _t6;
// CHECK-NEXT:     long double _t7;
// CHECK-NEXT:     volatile long double _t8;
// CHECK-NEXT:     long double _t9;
// CHECK-NEXT:     _t3 = this->x;
// CHECK-NEXT:     _t2 = i;
// CHECK-NEXT:     _t4 = _t3 * _t2;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     _t5 = _t4 * _t1;
// CHECK-NEXT:     _t0 = j;
// CHECK-NEXT:     _t8 = this->y;
// CHECK-NEXT:     _t7 = j;
// CHECK-NEXT:     _t9 = _t8 * _t7;
// CHECK-NEXT:     _t6 = i;
// CHECK-NEXT:     long double operator_call_return = _t5 * _t0 + _t9 * _t6;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         long double _r0 = 1 * _t0;
// CHECK-NEXT:         long double _r1 = _r0 * _t1;
// CHECK-NEXT:         long double _r2 = _r1 * _t2;
// CHECK-NEXT:         long double _r3 = _t3 * _r1;
// CHECK-NEXT:         * _d_i += _r3;
// CHECK-NEXT:         long double _r4 = _t4 * _r0;
// CHECK-NEXT:         * _d_i += _r4;
// CHECK-NEXT:         long double _r5 = _t5 * 1;
// CHECK-NEXT:         * _d_j += _r5;
// CHECK-NEXT:         long double _r6 = 1 * _t6;
// CHECK-NEXT:         long double _r7 = _r6 * _t7;
// CHECK-NEXT:         long double _r8 = _t8 * _r6;
// CHECK-NEXT:         * _d_j += _r8;
// CHECK-NEXT:         long double _r9 = _t9 * 1;
// CHECK-NEXT:         * _d_i += _r9;
// CHECK-NEXT:     }
// CHECK-NEXT: }

#define INIT(E)                                                                \
  auto d_##E = clad::gradient(&E);                                             \
  auto d_##E##Ref = clad::gradient(E);

#define TEST_DOUBLE(E, ...)                                                    \
  res[0] = res[1] = 0;                                                         \
  d_##E.execute(__VA_ARGS__, di_ref, dj_ref);                                  \
  printf("{%.2f, %.2f} ", res[0], res[1]);                                     \
  res[0] = res[1] = 0;                                                         \
  d_##E##Ref.execute(__VA_ARGS__, di_ref, dj_ref);                             \
  printf("{%.2f, %.2f}\n", res[0], res[1]);

#define TEST_LONG_DOUBLE(E, ...)                                               \
  res_ld[0] = res_ld[1] = 0;                                                   \
  d_##E.execute(__VA_ARGS__, di_ref_ld, dj_ref_ld);                            \
  printf("{%.2Lf, %.2Lf} ", res_ld[0], res_ld[1]);                             \
  res_ld[0] = res_ld[1] = 0;                                                   \
  d_##E##Ref.execute(__VA_ARGS__, di_ref_ld, dj_ref_ld);                       \
  printf("{%.2Lf, %.2Lf}\n", res_ld[0], res_ld[1]);

int main() {
  double res[2];
  long double res_ld[2];
  clad::array_ref<double> di_ref(res, 1), dj_ref(res + 1, 1);
  clad::array_ref<long double> di_ref_ld(res_ld, 1), dj_ref_ld(res_ld + 1, 1);

  Experiment<double> E_double(3, 5);
  Experiment<long double> E_ld(3, 5);
  ExperimentConstVolatile<double> EConstVolatile_double(3, 5);
  ExperimentConstVolatile<long double> EConstVolatile_ld(3, 5);

  INIT(E_double);
  INIT(E_ld);
  INIT(EConstVolatile_double);
  INIT(EConstVolatile_ld);

  TEST_DOUBLE(E_double, 7, 9);                  // CHECK-EXEC: {42.00, 5.00} {42.00, 5.00}
  TEST_LONG_DOUBLE(E_ld, 7, 9);                 // CHECK-EXEC: {423.00, 182.00} {423.00, 182.00}
  TEST_DOUBLE(EConstVolatile_double, 7, 9);     // CHECK-EXEC: {42.00, 5.00} {42.00, 5.00}
  TEST_LONG_DOUBLE(EConstVolatile_ld, 7, 9);    // CHECK-EXEC: {423.00, 182.00} {423.00, 182.00}

  E_double.setX(5);
  E_ld.setX(5);
  EConstVolatile_double.setX(5);
  EConstVolatile_ld.setX(5);

  TEST_DOUBLE(E_double, 7, 9);                  // CHECK-EXEC: {70.00, 5.00} {70.00, 5.00}
  TEST_LONG_DOUBLE(E_ld, 7, 9);                 // CHECK-EXEC: {675.00, 280.00} {675.00, 280.00}
  TEST_DOUBLE(EConstVolatile_double, 7, 9);     // CHECK-EXEC: {70.00, 5.00} {70.00, 5.00}
  TEST_LONG_DOUBLE(EConstVolatile_ld, 7, 9);    // CHECK-EXEC: {675.00, 280.00} {675.00, 280.00}
}