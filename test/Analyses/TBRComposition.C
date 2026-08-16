// RUN: %cladclang %s -I%S/../../include -oTBRComposition.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./TBRComposition.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oTBRComposition.out
// RUN: ./TBRComposition.out | %filecheck_exec %s

// The GradBench/ADBench GMM objective in miniature: a differentiated function
// with an out-parameter (so clad builds a reverse_forw carrying a restore
// tracker), caller-owned local buffers, a non-differentiated data pointer,
// and nested helpers that overwrite those buffers inside a loop whose
// iterations are each read back non-linearly. Only that composition exercises the
// per-call-instance state protocol; each mechanism in isolation stays correct
// without it.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// Both mutating calls must record their pre-call state, including the one
// whose data pointer carries no adjoint.
// CHECK: void objective_pullback(
// CHECK: center_reverse_forw(
// CHECK: scale_reverse_forw(

// out[i] = x[i] - m[i]: overwrites caller storage through a pointer, with a
// non-differentiated data pointer as one input.
void center(int d, const double* x, const double* m, double* out) {
  for (int i = 0; i < d; i++)
    out[i] = x[i] - m[i];
}

// out[i] = s[i] * v[i]: reads the buffer written above and overwrites a
// second one.
void scale(int d, const double* s, const double* v, double* out) {
  for (int i = 0; i < d; i++)
    out[i] = s[i] * v[i];
}

// Non-linear read: its pullback needs the primal values of v.
double sqnorm(int d, const double* v) {
  double t = 0;
  for (int i = 0; i < d; i++)
    t += v[i] * v[i];
  return t;
}

// err = sum_k (|diag_k * (x - m_k)|^2 + a_k), with x the fixed data.
void objective(int d, int k, const double* a, const double* m,
               const double* diag, const double* x, double* err) {
  double centered[2];
  double scaled[2];
  double term[2];
  for (int ik = 0; ik < k; ik++) {
    center(d, x, &m[ik * d], &centered[0]);
    scale(d, &diag[ik * d], &centered[0], &scaled[0]);
    term[ik] = a[ik] + sqnorm(d, &scaled[0]);
  }
  double total = 0;
  for (int ik = 0; ik < k; ik++)
    total += term[ik];
  *err = total;
}

double wrapper(const double* a, const double* m, const double* diag,
               const double* x) {
  double err = 0;
  objective(2, 2, a, m, diag, x, &err);
  return err;
}

int main() {
  double a[2] = {0.5, -1.5};
  double m[4] = {0.25, 0.5, -1., 2.};
  double diag[4] = {2., 3., 0.5, 1.5};
  double x[2] = {1., 2.};
  double da[2] = {}, dm[4] = {}, dd[4] = {};

  auto grad = clad::gradient(wrapper, "a, m, diag");
  grad.execute(a, m, diag, x, da, dm, dd);

  // f = sum_k sum_i (diag_ki * (x_i - m_ki))^2 + a_k
  //   df/da_k     = 1
  //   df/dm_ki    = -2 diag_ki^2 (x_i - m_ki)
  //   df/ddiag_ki = 2 diag_ki (x_i - m_ki)^2
  printf("da={%.2f, %.2f}\n", da[0], da[1]);
  // CHECK-EXEC: da={1.00, 1.00}
  printf("dm={%.2f, %.2f, %.2f, %.2f}\n", dm[0], dm[1], dm[2], dm[3]);
  // CHECK-EXEC: dm={-6.00, -27.00, -1.00, 0.00}
  printf("dd={%.2f, %.2f, %.2f, %.2f}\n", dd[0], dd[1], dd[2], dd[3]);
  // CHECK-EXEC: dd={2.25, 13.50, 4.00, 0.00}
}
