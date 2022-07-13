// RUN: %cladclang %s -I%S/../../include -oTemplateFunctors.out  
// RUN: ./TemplateFunctors.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

template <typename T> struct Experiment {
  mutable T x, y;
  Experiment() : x(), y() {}
  Experiment(T p_x, T p_y) : x(p_x), y(p_y) {}
  T operator()(T i, T j) { return x * i * i + y * j; }
  void setX(T val) { x = val; }
};

// CHECK: double operator_call_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     Experiment<double> _d_this_obj;
// CHECK-NEXT:     Experiment<double> *_d_this = &_d_this_obj;
// CHECK-NEXT:     double _d_x = 0;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     double &_t0 = this->x;
// CHECK-NEXT:     double _t1 = _t0 * i;
// CHECK-NEXT:     double &_t2 = this->y;
// CHECK-NEXT:     return (_d_x * i + _t0 * _d_i) * i + _t1 * _d_i + _d_y * j + _t2 * _d_j;
// CHECK-NEXT: }

template <> struct Experiment<long double> {
  mutable long double x, y;
  Experiment() : x(), y() {}
  Experiment(long double p_x, long double p_y) : x(p_x), y(p_y) {}
  long double operator()(long double i, long double j) { return x * i * i * j + y * j; }
  void setX(long double val) { x = val; }
};

// CHECK: long double operator_call_darg0(long double i, long double j) {
// CHECK-NEXT:     long double _d_i = 1;
// CHECK-NEXT:     long double _d_j = 0;
// CHECK-NEXT:     Experiment<long double> _d_this_obj;
// CHECK-NEXT:     Experiment<long double> *_d_this = &_d_this_obj;
// CHECK-NEXT:     long double _d_x = 0;
// CHECK-NEXT:     long double _d_y = 0;
// CHECK-NEXT:     long double &_t0 = this->x;
// CHECK-NEXT:     long double _t1 = _t0 * i;
// CHECK-NEXT:     long double _t2 = _t1 * i;
// CHECK-NEXT:     long double &_t3 = this->y;
// CHECK-NEXT:     return ((_d_x * i + _t0 * _d_i) * i + _t1 * _d_i) * j + _t2 * _d_j + _d_y * j + _t3 * _d_j;
// CHECK-NEXT: }

template <typename T> struct ExperimentConstVolatile {
  mutable T x, y;
  ExperimentConstVolatile() : x(), y() {}
  ExperimentConstVolatile(T p_x, T p_y) : x(p_x), y(p_y) {}
  T operator()(T i, T j) const volatile { return x * i * i + y * j; }
  void setX(T val) const volatile { x = val; }
};

// CHECK: double operator_call_darg0(double i, double j) const volatile {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     const volatile ExperimentConstVolatile<double> _d_this_obj;
// CHECK-NEXT:     const volatile ExperimentConstVolatile<double> *_d_this = &_d_this_obj;
// CHECK-NEXT:     double _d_x = 0;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     volatile double &_t0 = this->x;
// CHECK-NEXT:     double _t1 = _t0 * i;
// CHECK-NEXT:     volatile double &_t2 = this->y;
// CHECK-NEXT:     return (_d_x * i + _t0 * _d_i) * i + _t1 * _d_i + _d_y * j + _t2 * _d_j;
// CHECK-NEXT: }

template <> struct ExperimentConstVolatile<long double> {
  mutable long double x, y;
  ExperimentConstVolatile() : x(), y() {}
  ExperimentConstVolatile(long double p_x, long double p_y) : x(p_x), y(p_y) {}
  long double operator()(long double i, long double j) const volatile {
    return x * i * i * j + y * j;
  }
  void setX(long double val) const volatile { x = val; }
};

// CHECK: long double operator_call_darg0(long double i, long double j) const volatile {
// CHECK-NEXT:     long double _d_i = 1;
// CHECK-NEXT:     long double _d_j = 0;
// CHECK-NEXT:     const volatile ExperimentConstVolatile<long double> _d_this_obj;
// CHECK-NEXT:     const volatile ExperimentConstVolatile<long double> *_d_this = &_d_this_obj;
// CHECK-NEXT:     long double _d_x = 0;
// CHECK-NEXT:     long double _d_y = 0;
// CHECK-NEXT:     volatile long double &_t0 = this->x;
// CHECK-NEXT:     long double _t1 = _t0 * i;
// CHECK-NEXT:     long double _t2 = _t1 * i;
// CHECK-NEXT:     volatile long double &_t3 = this->y;
// CHECK-NEXT:     return ((_d_x * i + _t0 * _d_i) * i + _t1 * _d_i) * j + _t2 * _d_j + _d_y * j + _t3 * _d_j;
// CHECK-NEXT: }

#define INIT(E)                                                                \
  auto d_##E = clad::differentiate(&E, "i");                                   \
  auto d_##E##Ref = clad::differentiate(E, "i");

#define TEST_DOUBLE(E, ...)                                                    \
  printf("{%.2f, %.2f}\n", d_##E.execute(__VA_ARGS__),                         \
         d_##E##Ref.execute(__VA_ARGS__));

#define TEST_LONG_DOUBLE(E, ...)                                               \
  printf("{%.2Lf, %.2Lf}\n", d_##E.execute(__VA_ARGS__),                       \
         d_##E##Ref.execute(__VA_ARGS__));

int main() {
  Experiment<double> E_double(3, 5);
  Experiment<long double> E_ld(3, 5);
  ExperimentConstVolatile<double> EConstVolatile_double(3, 5);
  ExperimentConstVolatile<long double> EConstVolatile_ld(3, 5);

  INIT(E_double);
  INIT(E_ld);
  INIT(EConstVolatile_double);
  INIT(EConstVolatile_ld);

  TEST_DOUBLE(E_double, 7, 9);               // CHECK-EXEC: {42.00, 42.00}
  TEST_LONG_DOUBLE(E_ld, 7, 9);              // CHECK-EXEC: {378.00, 378.00}
  TEST_DOUBLE(EConstVolatile_double, 7, 9);  // CHECK-EXEC: {42.00, 42.00}
  TEST_LONG_DOUBLE(EConstVolatile_ld, 7, 9); // CHECK-EXEC: {378.00, 378.00}

  E_double.setX(5);
  E_ld.setX(5);
  EConstVolatile_double.setX(5);
  EConstVolatile_ld.setX(5);

  TEST_DOUBLE(E_double, 7, 9);               // CHECK-EXEC: {70.00, 70.00}
  TEST_LONG_DOUBLE(E_ld, 7, 9);              // CHECK-EXEC: {630.00, 630.00}
  TEST_DOUBLE(EConstVolatile_double, 7, 9);  // CHECK-EXEC: {70.00, 70.00}
  TEST_LONG_DOUBLE(EConstVolatile_ld, 7, 9); // CHECK-EXEC: {630.00, 630.00}
}