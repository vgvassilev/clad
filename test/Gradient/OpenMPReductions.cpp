// An adjoint the threads of a parallel loop all add to.
//
// A read at an index the loop never moves is a broadcast, so its adjoint is a
// sum over the loop. Accumulated where it stands, every thread stores to the
// same address and the gradient loses whatever they overwrite. Each thread
// sums its share into an accumulator of its own instead, and adds that to the
// adjoint once.
//
// RUN: %cladclang %s -I%S/../../include -fopenmp -oOpenMPReductions.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./OpenMPReductions.out | %filecheck_exec %s
// REQUIRES: OpenMP

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double weighted(const double* x, const double* w, int n) {
  double total = 0;
  #pragma omp parallel for reduction(+:total)
  for (int j = 0; j < n; j++)
    total += x[0] * w[j];
  return total;
}

// CHECK: void weighted_grad_0(const double *x, const double *w, int n, double *_d_x) {
// The accumulator is declared inside the region, so each thread has one.
// CHECK: #pragma omp parallel private(total) firstprivate(_d_total)
// CHECK-NEXT: {
// CHECK-NEXT: double _acc0 = 0.;
// CHECK: _acc0 += _r_d0 * w[j];
// One update per thread, in place of one racing store per iteration.
// CHECK: #pragma omp atomic
// CHECK-NEXT: _d_x[0] += _acc0;

// `w[j]` moves with the loop, so each iteration owns its element and its
// adjoint is stored where it stands.
// CHECK-NOT: _acc1

// The same loop inside a sequential one. The outer loop reads `x[0]` at an
// index it never moves too, but it would keep that sum outside the region,
// where every thread shares it, so only the region's own accumulator is used.
double repeated(const double* x, const double* w, int m, int n) {
  double total = 0;
  for (int i = 0; i < m; i++) {
    #pragma omp parallel for reduction(+:total)
    for (int j = 0; j < n; j++)
      total += x[0] * w[j];
  }
  return total;
}

// CHECK: void repeated_grad_0(const double *x, const double *w, int m, int n, double *_d_x) {
// CHECK: for (_t0 = {{.*}}; _t0; _t0--) {
// CHECK-NEXT: #pragma omp parallel private(total) firstprivate(_d_total)
// CHECK-NEXT: {
// CHECK-NEXT: double _acc0 = 0.;
// CHECK: #pragma omp atomic
// CHECK-NEXT: _d_x[0] += _acc0;

int main() {
  const int n = 100000;
  double x[2] = {2., 5.};
  double* w = new double[n];
  for (int i = 0; i < n; i++)
    w[i] = 1.;
  auto g = clad::gradient(weighted, "x");
  double dx[2] = {0, 0};
  g.execute(x, w, n, dx);
  // d/dx[0] is the sum of w, which is n; nothing reads x[1].
  printf("%.1f %.1f\n", dx[0], dx[1]);
  // CHECK-EXEC: 100000.0 0.0

  auto h = clad::gradient(repeated, "x");
  double dr[2] = {0, 0};
  h.execute(x, w, 7, n, dr);
  printf("%.1f %.1f\n", dr[0], dr[1]);
  // CHECK-EXEC: 700000.0 0.0
  delete[] w;
}
