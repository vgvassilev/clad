// RUN: %cladclang %s -I%S/../../include -fopenmp -fsyntax-only -oOpenMP.out 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

double sum_scaled_parallel_for(const double* x, int n, double scale) {
  double total = 0.0;      // reduction target
  double tmp = 0.0;        // private
  double last = 0.0;       // lastprivate

  #pragma omp parallel for shared(n) firstprivate(scale) private(tmp) \
      lastprivate(last) reduction(+:total)
  for (int i = 0; i < n; ++i) {
    tmp = x[i] * scale;
    total += tmp;
    if (i == n - 1)
      last = tmp; // lastprivate variable is assigned on the last iteration
  }

  // Touch 'last' to avoid unused warnings without affecting results
  total += 0 * last;
  // Use only total in the return to keep math simple for derivatives
  return total;
}

// CHECK: double sum_scaled_parallel_for_darg0_1(const double *x, int n, double scale) {
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     double _d_scale = 0;
// CHECK-NEXT:     double _d_total = 0.;
// CHECK-NEXT:     double total = 0.;
// CHECK-NEXT:     double _d_tmp = 0.;
// CHECK-NEXT:     double tmp = 0.;
// CHECK-NEXT:     double _d_last = 0.;
// CHECK-NEXT:     double last = 0.;
// CHECK-NEXT:     #pragma omp parallel for shared(_d_n,n) firstprivate(_d_scale,scale) private(_d_tmp,tmp) lastprivate(_d_last,last) reduction(+: _d_total,total)
// CHECK-NEXT:         for (int i = 0; i < n; ++i) {
// CHECK-NEXT:             _d_tmp = (i == 1.) * scale + x[i] * _d_scale;
// CHECK-NEXT:             tmp = x[i] * scale;
// CHECK-NEXT:             _d_total += _d_tmp;
// CHECK-NEXT:             total += tmp;
// CHECK-NEXT:             if (i == n - 1) {
// CHECK-NEXT:                 _d_last = _d_tmp;
// CHECK-NEXT:                 last = tmp;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     _d_total += 0 * last + 0 * _d_last;
// CHECK-NEXT:     total += 0 * last;
// CHECK-NEXT:     return _d_total;
// CHECK-NEXT: }

double accumulate_scale_parallel(double scale, int n) {

  double sum = 0.0;  // reduction target
  double tmp = 0.0;  // private

  #pragma omp parallel shared(n) firstprivate(scale) private(tmp) reduction(+:sum)
  {
    // Each thread contributes exactly once; with 1 thread this is sum += scale
    tmp = scale;
    sum += tmp;
  }

  return sum;
}

// CHECK: double accumulate_scale_parallel_darg0(double scale, int n) {
// CHECK-NEXT:     double _d_scale = 1;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     double _d_sum = 0.;
// CHECK-NEXT:     double sum = 0.;
// CHECK-NEXT:     double _d_tmp = 0.;
// CHECK-NEXT:     double tmp = 0.;
// CHECK-NEXT:     #pragma omp parallel shared(_d_n,n) firstprivate(_d_scale,scale) private(_d_tmp,tmp) reduction(+: _d_sum,sum)
// CHECK-NEXT:         {
// CHECK-NEXT:             _d_tmp = _d_scale;
// CHECK-NEXT:             tmp = scale;
// CHECK-NEXT:             _d_sum += _d_tmp;
// CHECK-NEXT:             sum += tmp;
// CHECK-NEXT:         }
// CHECK-NEXT:     return _d_sum;
// CHECK-NEXT: }

int main() {
  double x[5] = {1, 2, 3, 4, 5};
  int n = 5;
  double scale = 2.0;

  auto d_sum_wrt_x1 = clad::differentiate(sum_scaled_parallel_for, "x[1]");
  double d2 = d_sum_wrt_x1.execute(x, n, scale);

  auto d_accum_wrt_scale = clad::differentiate(accumulate_scale_parallel, "scale");
  double d3 = d_accum_wrt_scale.execute(3.5, 8);
  return 0;
}
