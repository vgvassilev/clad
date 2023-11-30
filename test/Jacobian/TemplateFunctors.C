// RUN: %cladclang %s -I%S/../../include -oTemplateFunctors.out 2>&1 | FileCheck %s 
// RUN: ./TemplateFunctors.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oTemplateFunctors.out
// RUN: ./TemplateFunctors.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

template <typename T> struct Experiment {
  mutable T x, y;
  Experiment(T p_x, T p_y) : x(p_x), y(p_y) {}
  void operator()(T i, T j, T *output) {
    output[0] = x*y*i*j;
    output[1] = 2*x*y*i*j;
  }
  void setX(T val) { x = val; }
};

// CHECK: void operator_call_jac(double i, double j, double *output, double *jacobianMatrix) {
// CHECK-NEXT:     output[0] = this->x * this->y * i * j;
// CHECK-NEXT:     output[1] = 2 * this->x * this->y * i * j;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r6 = 1 * j;
// CHECK-NEXT:         double _r7 = _r6 * i;
// CHECK-NEXT:         double _r8 = _r7 * this->y;
// CHECK-NEXT:         double _r9 = _r8 * this->x;
// CHECK-NEXT:         double _r10 = 2 * _r8;
// CHECK-NEXT:         double _r11 = 2 * this->x * _r7;
// CHECK-NEXT:         double _r12 = 2 * this->x * this->y * _r6;
// CHECK-NEXT:         jacobianMatrix[2UL] += _r12;
// CHECK-NEXT:         double _r13 = 2 * this->x * this->y * i * 1;
// CHECK-NEXT:         jacobianMatrix[3UL] += _r13;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 1 * j;
// CHECK-NEXT:         double _r1 = _r0 * i;
// CHECK-NEXT:         double _r2 = _r1 * this->y;
// CHECK-NEXT:         double _r3 = this->x * _r1;
// CHECK-NEXT:         double _r4 = this->x * this->y * _r0;
// CHECK-NEXT:         jacobianMatrix[0UL] += _r4;
// CHECK-NEXT:         double _r5 = this->x * this->y * i * 1;
// CHECK-NEXT:         jacobianMatrix[1UL] += _r5;
// CHECK-NEXT:     }
// CHECK-NEXT: }

template <> struct Experiment<long double> {
  mutable long double x, y;
  Experiment(long double p_x, long double p_y) : x(p_x), y(p_y) {}
  void operator()(long double i, long double j, long double *output) {
    output[0] = x*y*i*i*j;
    output[1] = 2*x*y*i*i*j;
  }
  void setX(long double val) { x = val; }
};

// CHECK: void operator_call_jac(long double i, long double j, long double *output, long double *jacobianMatrix) {
// CHECK-NEXT:     output[0] = this->x * this->y * i * i * j;
// CHECK-NEXT:     output[1] = 2 * this->x * this->y * i * i * j;
// CHECK-NEXT:     {
// CHECK-NEXT:         long double _r8 = 1 * j;
// CHECK-NEXT:         long double _r9 = _r8 * i;
// CHECK-NEXT:         long double _r10 = _r9 * i;
// CHECK-NEXT:         long double _r11 = _r10 * this->y;
// CHECK-NEXT:         long double _r12 = _r11 * this->x;
// CHECK-NEXT:         long double _r13 = 2 * _r11;
// CHECK-NEXT:         long double _r14 = 2 * this->x * _r10;
// CHECK-NEXT:         long double _r15 = 2 * this->x * this->y * _r9;
// CHECK-NEXT:         jacobianMatrix[2UL] += _r15;
// CHECK-NEXT:         long double _r16 = 2 * this->x * this->y * i * _r8;
// CHECK-NEXT:         jacobianMatrix[2UL] += _r16;
// CHECK-NEXT:         long double _r17 = 2 * this->x * this->y * i * i * 1;
// CHECK-NEXT:         jacobianMatrix[3UL] += _r17;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         long double _r0 = 1 * j;
// CHECK-NEXT:         long double _r1 = _r0 * i;
// CHECK-NEXT:         long double _r2 = _r1 * i;
// CHECK-NEXT:         long double _r3 = _r2 * this->y;
// CHECK-NEXT:         long double _r4 = this->x * _r2;
// CHECK-NEXT:         long double _r5 = this->x * this->y * _r1;
// CHECK-NEXT:         jacobianMatrix[0UL] += _r5;
// CHECK-NEXT:         long double _r6 = this->x * this->y * i * _r0;
// CHECK-NEXT:         jacobianMatrix[0UL] += _r6;
// CHECK-NEXT:         long double _r7 = this->x * this->y * i * i * 1;
// CHECK-NEXT:         jacobianMatrix[1UL] += _r7;
// CHECK-NEXT:     }
// CHECK-NEXT: }

#define INIT(E)                                                                \
  auto d_##E = clad::jacobian(&E);                                             \
  auto d_##E##Ref = clad::jacobian(E);

#define TEST_DOUBLE(E, ...)                                                    \
  result[0] = result[1] = result[2] = result[3] = 0;                           \
  output[0] = output[1] = 0;                                                   \
  d_##E.execute(__VA_ARGS__, output, result);                                  \
  printf("{%.2f, %.2f, %.2f, %.2f} ", result[0], result[1], result[2],        \
         result[3]);                                                           \
  result[0] = result[1] = result[2] = result[3] = 0;                           \
  output[0] = output[1] = 0;                                                   \
  d_##E##Ref.execute(__VA_ARGS__, output, result);                             \
  printf("{%.2f, %.2f, %.2f, %.2f}\n", result[0], result[1], result[2],        \
         result[3]);

#define TEST_LONG_DOUBLE(E, ...)                                               \
  result_ld[0] = result_ld[1] = result_ld[2] = result_ld[3] = 0;               \
  output_ld[0] = output_ld[1] = 0;                                             \
  d_##E.execute(__VA_ARGS__, output_ld, result_ld);                            \
  printf("{%.2Lf, %.2Lf, %.2Lf, %.2Lf} ", result_ld[0], result_ld[1],         \
         result_ld[2], result_ld[3]);                                          \
  result_ld[0] = result_ld[1] = result_ld[2] = result_ld[3] = 0;               \
  output_ld[0] = output_ld[1] = 0;                                             \
  d_##E##Ref.execute(__VA_ARGS__, output_ld, result_ld);                       \
  printf("{%.2Lf, %.2Lf, %.2Lf, %.2Lf}\n", result_ld[0], result_ld[1],         \
         result_ld[2], result_ld[3]);

int main() {
  double result[4], output[2];
  long double result_ld[4], output_ld[2];
  Experiment<double> E(3, 5);
  Experiment<long double> E_ld(3, 5);

  INIT(E);
  INIT(E_ld);

  TEST_DOUBLE(E, 7, 9);           // CHECK-EXEC: {135.00, 105.00, 270.00, 210.00} {135.00, 105.00, 270.00, 210.00}
  TEST_LONG_DOUBLE(E_ld, 7, 9);   // CHECK-EXEC: {1890.00, 735.00, 3780.00, 1470.00} {1890.00, 735.00, 3780.00, 1470.00}

  E.setX(5);
  E_ld.setX(5);

  TEST_DOUBLE(E, 7, 9);           // CHECK-EXEC: {225.00, 175.00, 450.00, 350.00} {225.00, 175.00, 450.00, 350.00}
  TEST_LONG_DOUBLE(E_ld, 7, 9);   // CHECK-EXEC: {3150.00, 1225.00, 6300.00, 2450.00} {3150.00, 1225.00, 6300.00, 2450.00}
}