// RUN: %cladclang %s -I%S/../../include -oArrays.out 2>&1 | FileCheck %s
// RUN: ./Arrays.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oArrays.out
// RUN: ./Arrays.out | FileCheck -check-prefix=CHECK-EXEC %s

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f(double i, double j[2]) { return i * j[0] * j[1]; }
// CHECK: void f_hessian(double i, double j[2], double *hessianMatrix) {
// CHECK-NEXT:     f_darg0_grad(i, j, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
// CHECK-NEXT:     f_darg1_0_grad(i, j, hessianMatrix + {{3U|3UL}}, hessianMatrix + {{4U|4UL}});
// CHECK-NEXT:     f_darg1_1_grad(i, j, hessianMatrix + {{6U|6UL}}, hessianMatrix + {{7U|7UL}});
// CHECK-NEXT: }

double g(double i, double j[2]) { return i * (j[0] + j[1]); }
// CHECK: void g_hessian(double i, double j[2], double *hessianMatrix) {
// CHECK-NEXT:   g_darg0_grad(i, j, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
// CHECK-NEXT:   g_darg1_0_grad(i, j, hessianMatrix + {{3U|3UL}}, hessianMatrix + {{4U|4UL}});
// CHECK-NEXT:   g_darg1_1_grad(i, j, hessianMatrix + {{6U|6UL}}, hessianMatrix + {{7U|7UL}});
// CHECK-NEXT: }

#define TEST(var, i, j)                                                        \
  result[0] = result[1] = result[2] = result[3] = result[4] = result[5] =      \
      result[6] = result[7] = result[8] = 0;                                   \
  var.execute(i, j, result_ref);                                               \
  printf("Result = {%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f}\n",          \
         result[0],                                                            \
         result[1],                                                            \
         result[2],                                                            \
         result[3],                                                            \
         result[4],                                                            \
         result[5],                                                            \
         result[6],                                                            \
         result[7],                                                            \
         result[8]);

int main() {
  double result[9];
  clad::array_ref<double> result_ref(result, 9);
  double x[] = {3, 4};

  auto h1 = clad::hessian(f, "i, j[0:1]");
  TEST(h1, 2, x); // CHECK-EXEC: Result = {0.00 4.00 3.00 4.00 0.00 2.00 3.00 2.00 0.00}
  auto h2 = clad::hessian(g, "i, j[0:1]");
  TEST(h2, 2, x); // CHECK-EXEC: Result = {0.00 1.00 1.00 1.00 0.00 0.00 1.00 0.00 0.00}
}
