// RUN: %cladclang %s -I%S/../../include -oTemplateFunctors.out 2>&1 | FileCheck %s
// RUN: ./TemplateFunctors.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oTemplateFunctors.out
// RUN: ./TemplateFunctors.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

template <typename T> struct Experiment {
  mutable T x, y;
  Experiment(T p_x = 0, T p_y = 0) : x(p_x), y(p_y) {}
  T operator()(T i, T j) { return x * i * i + y * j; }
  void setX(T val) { x = val; }
  Experiment& operator=(const Experiment& E) = default;
};

// CHECK: void operator_call_grad(double i, double j, Experiment<double> *_d_this, double *_d_i, double *_d_j) {
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         (*_d_this).x += 1 * i * i;
// CHECK-NEXT:         *_d_i += this->x * 1 * i;
// CHECK-NEXT:         *_d_i += this->x * i * 1;
// CHECK-NEXT:         (*_d_this).y += 1 * j;
// CHECK-NEXT:         *_d_j += this->y * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

template <> struct Experiment<long double> {
  mutable long double x, y;
  Experiment(long double p_x = 0, long double p_y = 0) : x(p_x), y(p_y) {}
  long double operator()(long double i, long double j) {
    return x * i * i * j + y * j * i;
  }
  void setX(long double val) { x = val; }
  Experiment& operator=(const Experiment& E) = default;
};

// CHECK: void operator_call_grad(long double i, long double j, Experiment<long double>  *_d_this, long double  *_d_i, long double  *_d_j) {
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         (*_d_this).x += 1 * j * i * i;
// CHECK-NEXT:         *_d_i += this->x * 1 * j * i;
// CHECK-NEXT:         *_d_i += this->x * i * 1 * j;
// CHECK-NEXT:         *_d_j += this->x * i * i * 1;
// CHECK-NEXT:         (*_d_this).y += 1 * i * j;
// CHECK-NEXT:         *_d_j += this->y * 1 * i;
// CHECK-NEXT:         *_d_i += this->y * j * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

template <typename T> struct ExperimentConstVolatile {
  mutable T x, y;
  ExperimentConstVolatile(T p_x = 0, T p_y = 0) : x(p_x), y(p_y) {}
  T operator()(T i, T j) const volatile { return x * i * i + y * j; }
  void setX(T val) const volatile { x = val; }
  ExperimentConstVolatile& operator=(const ExperimentConstVolatile& E) = default;
};

// CHECK: void operator_call_grad(double i, double j, volatile ExperimentConstVolatile<double> *_d_this, double *_d_i, double *_d_j) const volatile {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     _t0 = this->x * i;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         (*_d_this).x += 1 * i * i;
// CHECK-NEXT:         *_d_i += this->x * 1 * i;
// CHECK-NEXT:         *_d_i += _t0 * 1;
// CHECK-NEXT:         (*_d_this).y += 1 * j;
// CHECK-NEXT:         *_d_j += this->y * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

template <> struct ExperimentConstVolatile<long double> {
  mutable long double x, y;
  ExperimentConstVolatile(long double p_x = 0, long double p_y = 0) : x(p_x), y(p_y) {}
  long double operator()(long double i, long double j) const volatile {
    return x * i * i * j + y * j * i;
  }
  void setX(long double val) const volatile { x = val; }
  ExperimentConstVolatile& operator=(const ExperimentConstVolatile& E) = default;
};

// CHECK: void operator_call_grad(long double i, long double j, volatile ExperimentConstVolatile<long double> *_d_this, long double  *_d_i, long double  *_d_j) const volatile {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     _t0 = this->x * i;
// CHECK-NEXT:     _t1 = this->y * j;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         (*_d_this).x += 1 * j * i * i;
// CHECK-NEXT:         *_d_i += this->x * 1 * j * i;
// CHECK-NEXT:         *_d_i += _t0 * 1 * j;
// CHECK-NEXT:         *_d_j += _t0 * i * 1;
// CHECK-NEXT:         (*_d_this).y += 1 * i * j;
// CHECK-NEXT:         *_d_j += this->y * 1 * i;
// CHECK-NEXT:         *_d_i += _t1 * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

#define INIT(E)                                                                \
  auto d_##E = clad::gradient(&E);                                             \
  auto d_##E##Ref = clad::gradient(E);

#define TEST_DOUBLE(E, dE, ...)                                                \
  di = dj = 0;                                                                 \
  dE = decltype(dE)();                                                         \
  d_##E.execute(__VA_ARGS__, &dE, &di, &dj);                                   \
  printf("{%.2f, %.2f} ", di, dj);                                             \
  di = dj = 0;                                                                 \
  dE = decltype(dE)();                                                         \
  d_##E##Ref.execute(__VA_ARGS__, &dE, &di, &dj);                              \
  printf("{%.2f, %.2f}\n", di, dj);

#define TEST_LONG_DOUBLE(E, dE, ...)                                           \
  di_ld = dj_ld = 0;                                                           \
  dE = decltype(dE)();                                                         \
  d_##E.execute(__VA_ARGS__, &dE, &di_ld, &dj_ld);                             \
  printf("{%.2Lf, %.2Lf} ", di_ld, dj_ld);                                     \
  di_ld = dj_ld = 0;                                                           \
  dE = decltype(dE)();                                                         \
  d_##E##Ref.execute(__VA_ARGS__, &dE, &di_ld, &dj_ld);                        \
  printf("{%.2Lf, %.2Lf}\n", di_ld, dj_ld);

int main() {
  double di, dj;
  long double di_ld, dj_ld;

  Experiment<double> E_double(3, 5), dE_double;
  Experiment<long double> E_ld(3, 5), dE_ld;
  ExperimentConstVolatile<double> EConstVolatile_double(3, 5), dEConstVolatile_double;
  ExperimentConstVolatile<long double> EConstVolatile_ld(3, 5), dEConstVolatile_ld;

  INIT(E_double);
  INIT(E_ld);
  INIT(EConstVolatile_double);
  INIT(EConstVolatile_ld);

  TEST_DOUBLE(E_double, dE_double, 7, 9);                               // CHECK-EXEC: {42.00, 5.00} {42.00, 5.00}
  TEST_LONG_DOUBLE(E_ld, dE_ld, 7, 9);                                  // CHECK-EXEC: {423.00, 182.00} {423.00, 182.00}
  TEST_DOUBLE(EConstVolatile_double, dEConstVolatile_double, 7, 9);     // CHECK-EXEC: {42.00, 5.00} {42.00, 5.00}
  TEST_LONG_DOUBLE(EConstVolatile_ld, dEConstVolatile_ld, 7, 9);        // CHECK-EXEC: {423.00, 182.00} {423.00, 182.00}

  E_double.setX(5);
  E_ld.setX(5);
  EConstVolatile_double.setX(5);
  EConstVolatile_ld.setX(5);

  TEST_DOUBLE(E_double, dE_double, 7, 9);                               // CHECK-EXEC: {70.00, 5.00} {70.00, 5.00}
  TEST_LONG_DOUBLE(E_ld, dE_ld, 7, 9);                                  // CHECK-EXEC: {675.00, 280.00} {675.00, 280.00}
  TEST_DOUBLE(EConstVolatile_double, dEConstVolatile_double, 7, 9);     // CHECK-EXEC: {70.00, 5.00} {70.00, 5.00}
  TEST_LONG_DOUBLE(EConstVolatile_ld, dEConstVolatile_ld, 7, 9);        // CHECK-EXEC: {675.00, 280.00} {675.00, 280.00}
}
