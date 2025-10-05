// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oThrustAdjacentDifference.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: ./ThrustAdjacentDifference.out | %filecheck_exec %s
//
// REQUIRES: cuda-runtime
//
// expected-no-diagnostics

#include <iostream>
#include <vector>
#include <iomanip>

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/ThrustDerivatives.h"
#include "../TestUtils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/adjacent_difference.h>
#include <thrust/functional.h>

void adj_diff_default(const thrust::device_vector<double>& vec,
                      thrust::device_vector<double>& out) {
  thrust::adjacent_difference(vec.begin(), vec.end(), out.begin(), thrust::minus<double>());
}
// CHECK: void adj_diff_default_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> &out, thrust::device_vector<double> *_d_vec, thrust::device_vector<double> *_d_out) {
// CHECK-NEXT: {{.*}}thrust::adjacent_difference_reverse_forw(std::begin(vec), std::end(vec), std::begin(out), thrust::minus<double>(), std::begin((*_d_vec)), std::end((*_d_vec)), std::begin((*_d_out)), {});
// CHECK-NEXT: {
// CHECK-NEXT:     thrust::detail::vector_base<double, thrust::device_allocator<double> >::const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     thrust::detail::vector_base<double, thrust::device_allocator<double> >::const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     thrust::detail::vector_base<double, thrust::device_allocator<double> >::iterator _r2 = std::begin((*_d_out));
// CHECK-NEXT:   thrust::minus<double> _r3 = {};
// CHECK-NEXT:   clad::custom_derivatives::thrust::adjacent_difference_pullback(std::begin(vec), std::end(vec), std::begin(out), thrust::minus<double>(), {}, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT: }
// CHECK-NEXT: }

void adj_diff_plus(const thrust::device_vector<double>& vec,
                   thrust::device_vector<double>& out) {
  thrust::adjacent_difference(vec.begin(), vec.end(), out.begin(), thrust::plus<double>());
}
// CHECK: void adj_diff_plus_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> &out, thrust::device_vector<double> *_d_vec, thrust::device_vector<double> *_d_out) {
// CHECK-NEXT: {{.*}}thrust::adjacent_difference_reverse_forw(std::begin(vec), std::end(vec), std::begin(out), thrust::plus<double>(), std::begin((*_d_vec)), std::end((*_d_vec)), std::begin((*_d_out)), {});
// CHECK-NEXT: {
// CHECK-NEXT:     thrust::detail::vector_base<double, thrust::device_allocator<double> >::const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     thrust::detail::vector_base<double, thrust::device_allocator<double> >::const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     thrust::detail::vector_base<double, thrust::device_allocator<double> >::iterator _r2 = std::begin((*_d_out));
// CHECK-NEXT:   thrust::plus<double> _r3 = {};
// CHECK-NEXT:   clad::custom_derivatives::thrust::adjacent_difference_pullback(std::begin(vec), std::end(vec), std::begin(out), thrust::plus<double>(), {}, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT: }
// CHECK-NEXT: }

int main() {
  std::vector<double> h = {1.0, 3.0, 6.0, 10.0};
  thrust::device_vector<double> x = h;
  thrust::device_vector<double> y(h.size());

  thrust::device_vector<double> d_x(h.size(), 0.0);
  thrust::device_vector<double> d_y(h.size(), 0.0);

  // default (minus)
  thrust::fill(d_y.begin(), d_y.end(), 1.0);
  INIT_GRADIENT(adj_diff_default);
  adj_diff_default_grad.execute(x, y, &d_x, &d_y);
  thrust::host_vector<double> hx = d_x;
  printf("AdjDiff minus d_x: %.3f %.3f %.3f %.3f\n", hx[0], hx[1], hx[2], hx[3]);
  // With y = [x0, x1-x0, x2-x1, x3-x2] and d_y=1 ⇒ d_x = [0,0,0,1]
  // CHECK-EXEC: AdjDiff minus d_x: 0.000 0.000 0.000 1.000

  // plus op
  thrust::fill(d_x.begin(), d_x.end(), 0.0);
  thrust::fill(d_y.begin(), d_y.end(), 1.0);
  INIT_GRADIENT(adj_diff_plus);
  adj_diff_plus_grad.execute(x, y, &d_x, &d_y);
  hx = d_x;
  printf("AdjDiff plus d_x: %.3f %.3f %.3f %.3f\n", hx[0], hx[1], hx[2], hx[3]);
  // With y = [x0, x1+x0, x2+x1, x3+x2] and d_y=1 ⇒ d_x = [2,2,2,1]
  // CHECK-EXEC: AdjDiff plus d_x: 2.000 2.000 2.000 1.000

  return 0;
}


