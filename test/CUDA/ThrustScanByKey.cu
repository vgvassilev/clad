// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oThrustScanByKey.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: ./ThrustScanByKey.out | %filecheck_exec %s
//
// REQUIRES: cuda-runtime
//
// expected-no-diagnostics

#include <iostream>
#include <vector>

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/ThrustDerivatives.h"
#include "../TestUtils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>

void inclusive_scan_by_key_plus(const thrust::device_vector<int>& keys,
                                const thrust::device_vector<double>& values,
                                thrust::device_vector<double>& output) {
  thrust::inclusive_scan_by_key(keys.begin(), keys.end(), values.begin(),
                                output.begin(), thrust::equal_to<int>(),
                                thrust::plus<double>());
}
// CHECK: void inclusive_scan_by_key_plus_grad(const thrust::device_vector<int> &keys, const thrust::device_vector<double> &values, thrust::device_vector<double> &output, thrust::device_vector<int> *_d_keys, thrust::device_vector<double> *_d_values, thrust::device_vector<double> *_d_output) {
// CHECK-NEXT:     clad::custom_derivatives::thrust::inclusive_scan_by_key_reverse_forw(std::begin(keys), std::end(keys), std::begin(values), std::begin(output), thrust::equal_to<int>(), thrust::plus<double>(), std::begin((*_d_keys)), std::end((*_d_keys)), std::begin((*_d_values)), std::begin((*_d_output)), {}, {});
// CHECK-NEXT:     {
// CHECK-NEXT:         thrust::detail::vector_base<int, thrust::device_allocator<int> >::const_iterator _r0 = std::begin((*_d_keys));
// CHECK-NEXT:         thrust::detail::vector_base<int, thrust::device_allocator<int> >::const_iterator _r1 = std::end((*_d_keys));
// CHECK-NEXT:         thrust::detail::vector_base<double, thrust::device_allocator<double> >::const_iterator _r2 = std::begin((*_d_values));
// CHECK-NEXT:         thrust::detail::vector_base<double, thrust::device_allocator<double> >::iterator _r3 = std::begin((*_d_output));
// CHECK-NEXT:         thrust::equal_to<int> _r4 = {};
// CHECK-NEXT:         thrust::plus<double> _r5 = {};
// CHECK-NEXT:         clad::custom_derivatives::thrust::inclusive_scan_by_key_pullback(std::begin(keys), std::end(keys), std::begin(values), std::begin(output), thrust::equal_to<int>(), thrust::plus<double>(), {}, &_r0, &_r1, &_r2, &_r3, &_r4, &_r5);
// CHECK-NEXT:     }
// CHECK-NEXT: }

void exclusive_scan_by_key_plus(const thrust::device_vector<int>& keys,
                                const thrust::device_vector<double>& values,
                                thrust::device_vector<double>& output,
                                double init) {
  thrust::exclusive_scan_by_key(keys.begin(), keys.end(), values.begin(),
                                output.begin(), init, thrust::equal_to<int>(),
                                thrust::plus<double>());
}
// CHECK: void exclusive_scan_by_key_plus_grad(const thrust::device_vector<int> &keys, const thrust::device_vector<double> &values, thrust::device_vector<double> &output, double init, thrust::device_vector<int> *_d_keys, thrust::device_vector<double> *_d_values, thrust::device_vector<double> *_d_output, double *_d_init) {
// CHECK-NEXT:     clad::custom_derivatives::thrust::exclusive_scan_by_key_reverse_forw(std::begin(keys), std::end(keys), std::begin(values), std::begin(output), init, thrust::equal_to<int>(), thrust::plus<double>(), std::begin((*_d_keys)), std::end((*_d_keys)), std::begin((*_d_values)), std::begin((*_d_output)), 0., {}, {});
// CHECK-NEXT:     {
// CHECK-NEXT:         thrust::detail::vector_base<int, thrust::device_allocator<int> >::const_iterator _r0 = std::begin((*_d_keys));
// CHECK-NEXT:         thrust::detail::vector_base<int, thrust::device_allocator<int> >::const_iterator _r1 = std::end((*_d_keys));
// CHECK-NEXT:         thrust::detail::vector_base<double, thrust::device_allocator<double> >::const_iterator _r2 = std::begin((*_d_values));
// CHECK-NEXT:         thrust::detail::vector_base<double, thrust::device_allocator<double> >::iterator _r3 = std::begin((*_d_output));
// CHECK-NEXT:         double _r4 = 0.;
// CHECK-NEXT:         thrust::equal_to<int> _r5 = {};
// CHECK-NEXT:         thrust::plus<double> _r6 = {};
// CHECK-NEXT:         clad::custom_derivatives::thrust::exclusive_scan_by_key_pullback(std::begin(keys), std::end(keys), std::begin(values), std::begin(output), init, thrust::equal_to<int>(), thrust::plus<double>(), {}, &_r0, &_r1, &_r2, &_r3, &_r4, &_r5, &_r6);
// CHECK-NEXT:         *_d_init += _r4;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
  std::vector<int> host_keys = {0, 0, 1, 1, 2, 2, 2};
  std::vector<double> host_vals = {1, 2, 3, 4, 5, 6, 7};
  const size_t n = host_keys.size();

  thrust::device_vector<int> d_keys = host_keys;
  thrust::device_vector<double> d_vals = host_vals;
  thrust::device_vector<double> d_out(n);

  thrust::device_vector<double> d_output_adj(n);
  thrust::fill(d_output_adj.begin(), d_output_adj.end(), 1.0);

  // Inclusive scan by key
  INIT_GRADIENT(inclusive_scan_by_key_plus);
  thrust::device_vector<int> d_keys_grad(n);
  thrust::device_vector<double> d_vals_grad(n);
  inclusive_scan_by_key_plus_grad.execute(d_keys, d_vals, d_out, &d_keys_grad,
                                          &d_vals_grad, &d_output_adj);
  thrust::host_vector<double> hv_incl = d_vals_grad;
  printf("Inclusive ByKey Gradients: ");
  for (size_t i = 0; i < n; ++i) printf("%.3f%s", hv_incl[i], i + 1 == n ? "\n" : " ");
  // CHECK-EXEC: Inclusive ByKey Gradients: 2.000 1.000 2.000 1.000 3.000 2.000 1.000

  // Exclusive scan by key
  thrust::fill(d_output_adj.begin(), d_output_adj.end(), 1.0);
  INIT_GRADIENT(exclusive_scan_by_key_plus);
  thrust::fill(d_vals_grad.begin(), d_vals_grad.end(), 0.0);
  double d_init = 1.0;
  exclusive_scan_by_key_plus_grad.execute(d_keys, d_vals, d_out, 0.0,
                                          &d_keys_grad, &d_vals_grad,
                                          &d_output_adj, &d_init);
  thrust::host_vector<double> hv_excl = d_vals_grad;
  printf("Exclusive ByKey Gradients: ");
  for (size_t i = 0; i < n; ++i) printf("%.3f%s", hv_excl[i], i + 1 == n ? "\n" : " ");
  // CHECK-EXEC: Exclusive ByKey Gradients: 1.000 0.000 1.000 0.000 2.000 1.000 0.000
  printf("Exclusive ByKey Init Gradient: %.3f\n", d_init);
  // Sum of adjoints (n) + initial seed (1.0) = 8.000
  // CHECK-EXEC: Exclusive ByKey Init Gradient: 8.000

  return 0;
}


