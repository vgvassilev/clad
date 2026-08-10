// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// A nested product keeps the outer operand's placeholder live while the inner
// one is finalized. The two used to be value-equal, so the inner Finalize
// spliced its operand into the outer placeholder's occurrences as well,
// yielding `_d_x += 1 * c[i * 2] * c[i * 2]` for compound_index below. A plain
// `c[i]` subscript is stored rather than recomputed, so it never hit this.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double compound_index(double x, const double* c, int i) {
  return (x + 1.) * c[i * 2] * (x + 1.);
}

// CHECK: void compound_index_grad_0(double x, const double *c, int i, double *_d_x) {
// CHECK: *_d_x += 1 * (x + 1.) * c[i * 2];
// CHECK-NEXT: *_d_x += (x + 1.) * c[i * 2] * 1;

double plain_index(double x, const double* c, int i) {
  return (x + 1.) * c[i] * (x + 1.);
}

// Placeholders are literals of the operand type, so they must stay far enough
// apart to survive `float`'s 24-bit mantissa.
float compound_index_f(float x, const float* c, int i) {
  return (x + 1.F) * c[i * 2] * (x + 1.F);
}

double nested_products(double x, const double* c, int i) {
  return ((x * x + 1.) * c[i * 2]) * ((x + 3.) * c[i * 2 + 1]) * (x + 1.);
}

double mixed_div(double x, const double* c, int i) {
  return (x + 1.) / c[i * 2] * (x + 1.) / c[i * 2 + 1];
}

// The RooFit::Detail::MathFuncs::multiVarGaussian() shape this was found with.
double multi_var(double x, const double* c, int n) {
  double result = 0.;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      result += (x - c[i]) * c[i * n + j] * (x - c[j]);
  return result;
}

#define CHECK_GRAD(F, ARG)                                                     \
  do {                                                                         \
    auto grad = clad::gradient(F, "x");                                        \
    double d = 0.;                                                             \
    grad.execute(5., c, ARG, &d);                                              \
    printf(#F " %.6f\n", d);                                                   \
  } while (0)

int main() {
  double c[8];
  for (int k = 0; k < 8; ++k)
    c[k] = 1.5 + 0.25 * k;

  // d/dx of (x+1)^2 * c[2] at x = 5 is 2 * 6 * 2 == 24.
  CHECK_GRAD(compound_index, 1); // CHECK-EXEC: compound_index 24.000000
  CHECK_GRAD(plain_index, 2);    // CHECK-EXEC: plain_index 24.000000
  CHECK_GRAD(nested_products, 1); // CHECK-EXEC: nested_products 3798.000000
  CHECK_GRAD(mixed_div, 1);       // CHECK-EXEC: mixed_div 2.666667
  CHECK_GRAD(multi_var, 2);       // CHECK-EXEC: multi_var 50.437500

  {
    float cf[8];
    for (int k = 0; k < 8; ++k)
      cf[k] = 1.5F + 0.25F * k;
    auto grad = clad::gradient(compound_index_f, "x");
    float d = 0.F;
    grad.execute(5.F, cf, 1, &d);
    printf("compound_index_f %.6f\n", (double)d);
    // CHECK-EXEC: compound_index_f 24.000000
  }
}
