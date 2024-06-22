// RUN: %cladclang %s -I%S/../../include -oArrays.out 2>&1 | %filecheck %s
// RUN: ./Arrays.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oArrays.out
// RUN: ./Arrays.out | %filecheck_exec %s

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

double h(double arr[3], double weights[3], double multiplier) {
  // return square of weighted sum.
  double weightedSum = arr[0] * weights[0];
  weightedSum += arr[1] * weights[1];
  weightedSum += arr[2] * weights[2];
  weightedSum *= multiplier;
  return weightedSum * weightedSum;
}
// CHECK: void h_hessian_diagonal(double arr[3], double weights[3], double multiplier, double *diagonalHessianVector) {
// CHECK-NEXT:   *(diagonalHessianVector + 0{{U|UL}}) = h_d2arg0_0(arr, weights, multiplier);
// CHECK-NEXT:   *(diagonalHessianVector + 1{{U|UL}}) = h_d2arg0_1(arr, weights, multiplier);
// CHECK-NEXT:   *(diagonalHessianVector + 2{{U|UL}}) = h_d2arg0_2(arr, weights, multiplier);
// CHECK-NEXT:   *(diagonalHessianVector + 3{{U|UL}}) = h_d2arg1_0(arr, weights, multiplier);
// CHECK-NEXT:   *(diagonalHessianVector + 4{{U|UL}}) = h_d2arg1_1(arr, weights, multiplier);
// CHECK-NEXT:   *(diagonalHessianVector + 5{{U|UL}}) = h_d2arg1_2(arr, weights, multiplier);
// CHECK-NEXT:   *(diagonalHessianVector + 6{{U|UL}}) = h_d2arg2(arr, weights, multiplier);
// CHECK-NEXT: }

#define TEST(var, i, j)                                                        \
  result[0] = result[1] = result[2] = result[3] = result[4] = result[5] =      \
      result[6] = result[7] = result[8] = 0;                                   \
  var.execute(i, j, result);                                               \
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
  double x[] = {3, 4};

  auto h1 = clad::hessian(f, "i, j[0:1]");
  TEST(h1, 2, x); // CHECK-EXEC: Result = {0.00 4.00 3.00 4.00 0.00 2.00 3.00 2.00 0.00}
  auto h2 = clad::hessian(g, "i, j[0:1]");
  TEST(h2, 2, x); // CHECK-EXEC: Result = {0.00 1.00 1.00 1.00 0.00 0.00 1.00 0.00 0.00}

  double arr[] = {1, 2, 3};
  double weights[] = {4, 5, 6};
  double diag[7]; // result will be the diagonal of the Hessian matrix.
  double multiplier = 2.0;
  auto h3 = clad::hessian<clad::opts::diagonal_only>(h, "arr[0:2], weights[0:2], multiplier");
  h3.execute(arr, weights, multiplier, diag);
  printf("Diagonal (arr) = {%.2f %.2f %.2f},\n", diag[0], diag[1], diag[2]); // CHECK-EXEC: Diagonal (arr) = {128.00 200.00 288.00},
  printf("Diagonal (weights) = {%.2f %.2f %.2f}\n", diag[3], diag[4], diag[5]); // CHECK-EXEC: Diagonal (weights) = {8.00 32.00 72.00}
  printf("Diagonal (multiplier) = %.2f\n", diag[6]); // CHECK-EXEC: Diagonal (multiplier) = 2048.00
}
