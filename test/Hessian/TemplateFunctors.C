// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oTemplateFunctors.out 2>&1 | %filecheck %s
// RUN: ./TemplateFunctors.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oTemplateFunctors.out
// RUN: ./TemplateFunctors.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

template <typename T> struct Experiment {
  mutable T x, y;
  Experiment(T p_x=0, T p_y=0) : x(p_x), y(p_y) {}
  T operator()(T i, T j) { return x * i * i * j + y * j * j * i; }
  void setX(T val) { x = val; }
};

// CHECK: void operator_call_hessian(double i, double j, double *hessianMatrix) {
// CHECK-NEXT:     Experiment<double> _d_this;
// CHECK-NEXT:     this->operator_call_darg0_grad(i, j, &_d_this, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
// CHECK-NEXT:     Experiment<double> _d_this0;
// CHECK-NEXT:     this->operator_call_darg1_grad(i, j, &_d_this0, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
// CHECK-NEXT: }

template <> struct Experiment<long double> {
  mutable long double x, y;
  Experiment(long double p_x=0, long double p_y=0) : x(p_x), y(p_y) {}
  long double operator()(long double i, long double j) {
    return i * i * j + j * j * i;
  }
  void setX(long double val) { x = val; }
};

// CHECK: void operator_call_hessian(long double i, long double j, long double *hessianMatrix) {
// CHECK-NEXT:     Experiment<long double> _d_this;
// CHECK-NEXT:     this->operator_call_darg0_grad(i, j, &_d_this, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
// CHECK-NEXT:     Experiment<long double> _d_this0;
// CHECK-NEXT:     this->operator_call_darg1_grad(i, j, &_d_this0, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
// CHECK-NEXT: }

#define INIT(E)                   \
  auto d_##E = clad::hessian(&E); \
  auto d_##E##Ref = clad::hessian(E);

#define TEST_DOUBLE(E, ...)                                            \
  res[0] = res[1] = res[2] = res[3] = 0;                               \
  d_##E.execute(__VA_ARGS__, res);                                     \
  printf("{%.2f, %.2f, %.2f, %.2f} ", res[0], res[1], res[2], res[3]); \
  res[0] = res[1] = res[2] = res[3] = 0;                               \
  d_##E##Ref.execute(__VA_ARGS__, res);                                \
  printf("{%.2f, %.2f, %.2f, %.2f}\n", res[0], res[1], res[2], res[3]);

#define TEST_LONG_DOUBLE(E, ...)                                                       \
  res_ld[0] = res_ld[1] = res_ld[2] = res_ld[3] = 0;                                   \
  d_##E.execute(__VA_ARGS__, res_ld);                                                  \
  printf("{%.2Lf, %.2Lf, %.2Lf, %.2Lf} ", res_ld[0], res_ld[1], res_ld[2], res_ld[3]); \
  res_ld[0] = res_ld[1] = res_ld[2] = res_ld[3] = 0;                                   \
  d_##E##Ref.execute(__VA_ARGS__, res_ld);                                             \
  printf("{%.2Lf, %.2Lf, %.2Lf, %.2Lf}\n", res_ld[0], res_ld[1], res_ld[2], res_ld[3]);

int main() {
  double res[4];
  long double res_ld[4];
  Experiment<double> E(3, 5);
  Experiment<long double> E_ld(3, 5);

  INIT(E);
  INIT(E_ld);
  
  TEST_DOUBLE(E, 7, 9);  // CHECK-EXEC: {54.00, 132.00, 132.00, 70.00} {54.00, 132.00, 132.00, 70.00}
  TEST_LONG_DOUBLE(E_ld, 7, 9); // CHECK-EXEC: {18.00, 32.00, 32.00, 14.00} {18.00, 32.00, 32.00, 14.00}
}

