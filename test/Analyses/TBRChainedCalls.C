// RUN: %cladclang %s -I%S/../../include -oTBRChainedCalls.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./TBRChainedCalls.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oTBRChainedCalls.out
// RUN: ./TBRChainedCalls.out | %filecheck_exec %s

// Two mutating calls chained through caller-owned buffers inside a loop,
// where the second callee also reads a differentiated parameter it does not
// write (`diag`). Its pullback recomputes from `centered`, so that buffer
// must hold the values of the iteration being reversed; with only the second
// call recorded, every reverse iteration reads the last iteration's values
// and the read-only parameter's adjoint comes out wrong while the adjoints
// flowing through the buffers still look right. This is the ADBench/GradBench
// GMM inner loop in miniature.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// CHECK: void chained_grad_0_1(
// CHECK: sub_reverse_forw(

void sub(int d, const double* x, const double* y, double* out) {
  for (int i = 0; i < d; i++)
    out[i] = x[i] - y[i];
}

void scale(int d, const double* diag, const double* x, double* out) {
  for (int i = 0; i < d; i++)
    out[i] = diag[i] * x[i];
}

// Non-linear read: its pullback needs the primal values of v.
double sqnorm(int d, const double* v) {
  double t = 0;
  for (int i = 0; i < d; i++)
    t += v[i] * v[i];
  return t;
}

double chained(const double* diag, const double* mu, const double* x) {
  double centered[2];
  double scaled[2];
  double s = 0;
  for (int k = 0; k < 2; k++) {
    sub(2, &x[k * 2], mu, centered);
    scale(2, diag, centered, scaled);
    s += sqnorm(2, scaled);
  }
  return s;
}


// The same defect without any chaining: `scale`'s output buffer is not
// snapshotted per iteration because its `x` argument is not differentiated,
// so the reverse sweep reads the last iteration's values. This is the shape
// the GradBench GMM objective hits through its q/l blocks.
double scaled_only(const double* q, const double* x) {
  double g[2];
  double out[2];
  double s = 0;
  for (int i = 0; i < 2; i++)
    g[i] = q[i] * q[i];
  for (int k = 0; k < 2; k++) {
    scale(2, g, &x[k * 2], out);
    s += sqnorm(2, out);
  }
  return s;
}

int main() {
  double diag[2] = {1.5, 0.5};
  double mu[2] = {0.25, 0.5};
  double x[4] = {1., 2., 3., -1.};
  double d_diag[2] = {};
  double d_mu[2] = {};

  auto grad = clad::gradient(chained, "diag, mu");
  grad.execute(diag, mu, x, d_diag, d_mu);

  // s = sum_k sum_i (diag_i * (x_ki - mu_i))^2
  //   ds/ddiag_i = sum_k 2 diag_i (x_ki - mu_i)^2
  //   ds/dmu_i   = sum_k -2 diag_i^2 (x_ki - mu_i)
  printf("d_diag={%.4f, %.4f}\n", d_diag[0], d_diag[1]);
  // CHECK-EXEC: d_diag={24.3750, 4.5000}
  printf("d_mu={%.4f, %.4f}\n", d_mu[0], d_mu[1]);
  // CHECK-EXEC: d_mu={-15.7500, 0.0000}

  double q[2] = {0.5, 1.5};
  double d_q[2] = {};
  auto grad2 = clad::gradient(scaled_only, "q");
  grad2.execute(q, x, d_q);
  // s = sum_k sum_i (q_i^2 x_ki)^2 => ds/dq_i = sum_k 4 q_i^3 x_ki^2
  printf("d_q={%.4f, %.4f}\n", d_q[0], d_q[1]);
  // CHECK-EXEC: d_q={5.0000, 67.5000}
}
