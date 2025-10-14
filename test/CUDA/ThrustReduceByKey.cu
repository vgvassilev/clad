// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oThrustReduceByKey.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: ./ThrustReduceByKey.out | %filecheck_exec %s
//
// REQUIRES: cuda-runtime
//
// expected-no-diagnostics

#include <vector>
#include <cstdio>

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/ThrustDerivatives.h"
#include "../TestUtils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

// Simple reduce_by_key (equal_to, plus)
void reduce_by_key_simple(const thrust::device_vector<int>& keys,
                          const thrust::device_vector<double>& vals,
                          thrust::device_vector<int>& keys_out,
                          thrust::device_vector<double>& vals_out) {
  thrust::reduce_by_key(keys.begin(), keys.end(), vals.begin(),
                        keys_out.begin(), vals_out.begin());
}

// CHECK: void reduce_by_key_simple_grad(const thrust::device_vector<int> &keys, const thrust::device_vector<double> &vals, thrust::device_vector<int> &keys_out, thrust::device_vector<double> &vals_out, thrust::device_vector<int> *_d_keys, thrust::device_vector<double> *_d_vals, thrust::device_vector<int> *_d_keys_out, thrust::device_vector<double> *_d_vals_out) {
// CHECK-NEXT: {{.*}}clad::custom_derivatives::thrust::reduce_by_key_reverse_forw(std::begin(keys), std::end(keys), std::begin(vals), std::begin(keys_out), std::begin(vals_out), std::begin((*_d_keys)), std::end((*_d_keys)), std::begin((*_d_vals)), std::begin((*_d_keys_out)), std::begin((*_d_vals_out)));
// CHECK-NEXT: {
// CHECK-NEXT:     thrust::detail::vector_base<int, thrust::device_allocator<int> >::const_iterator _r0 = std::begin((*_d_keys));
// CHECK-NEXT:     thrust::detail::vector_base<int, thrust::device_allocator<int> >::const_iterator _r1 = std::end((*_d_keys));
// CHECK-NEXT:     thrust::detail::vector_base<double, thrust::device_allocator<double> >::const_iterator _r2 = std::begin((*_d_vals));
// CHECK-NEXT:     thrust::detail::vector_base<int, thrust::device_allocator<int> >::iterator _r3 = std::begin((*_d_keys_out));
// CHECK-NEXT:     thrust::detail::vector_base<double, thrust::device_allocator<double> >::iterator _r4 = std::begin((*_d_vals_out));
// CHECK-NEXT:     clad::custom_derivatives::thrust::reduce_by_key_pullback(std::begin(keys), std::end(keys), std::begin(vals), std::begin(keys_out), std::begin(vals_out), {}, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT: }

// reduce_by_key with explicit predicate/op (equal_to, plus)
void reduce_by_key_custom_op(const thrust::device_vector<int>& keys,
                             const thrust::device_vector<double>& vals,
                             thrust::device_vector<int>& keys_out,
                             thrust::device_vector<double>& vals_out) {
  thrust::reduce_by_key(keys.begin(), keys.end(), vals.begin(),
                        keys_out.begin(), vals_out.begin(),
                        thrust::equal_to<int>(), thrust::plus<double>());
}

// CHECK: void reduce_by_key_custom_op_grad(const thrust::device_vector<int> &keys, const thrust::device_vector<double> &vals, thrust::device_vector<int> &keys_out, thrust::device_vector<double> &vals_out, thrust::device_vector<int> *_d_keys, thrust::device_vector<double> *_d_vals, thrust::device_vector<int> *_d_keys_out, thrust::device_vector<double> *_d_vals_out) {
// CHECK-NEXT: {{.*}}clad::custom_derivatives::thrust::reduce_by_key_reverse_forw(std::begin(keys), std::end(keys), std::begin(vals), std::begin(keys_out), std::begin(vals_out), thrust::equal_to<int>(), thrust::plus<double>(), std::begin((*_d_keys)), std::end((*_d_keys)), std::begin((*_d_vals)), std::begin((*_d_keys_out)), std::begin((*_d_vals_out)), {}, {});
// CHECK-NEXT: {
// CHECK-NEXT:     thrust::detail::vector_base<int, thrust::device_allocator<int> >::const_iterator _r0 = std::begin((*_d_keys));
// CHECK-NEXT:     thrust::detail::vector_base<int, thrust::device_allocator<int> >::const_iterator _r1 = std::end((*_d_keys));
// CHECK-NEXT:     thrust::detail::vector_base<double, thrust::device_allocator<double> >::const_iterator _r2 = std::begin((*_d_vals));
// CHECK-NEXT:     thrust::detail::vector_base<int, thrust::device_allocator<int> >::iterator _r3 = std::begin((*_d_keys_out));
// CHECK-NEXT:     thrust::detail::vector_base<double, thrust::device_allocator<double> >::iterator _r4 = std::begin((*_d_vals_out));
// CHECK-NEXT:     thrust::equal_to<int> _r5 = {};
// CHECK-NEXT:     thrust::plus<double> _r6 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::reduce_by_key_pullback(std::begin(keys), std::end(keys), std::begin(vals), std::begin(keys_out), std::begin(vals_out), thrust::equal_to<int>(), thrust::plus<double>(), {}, &_r0, &_r1, &_r2, &_r3, &_r4, &_r5, &_r6);
// CHECK-NEXT: }

int main() {
  // keys: [1,1,2,2,2], vals: [10,20,3,4,5] => out: keys=[1,2], vals=[30,12]
  std::vector<int> hkeys{1, 1, 2, 2, 2};
  std::vector<double> hvals{10., 20., 3., 4., 5.};

  thrust::device_vector<int> keys(hkeys.begin(), hkeys.end());
  thrust::device_vector<double> vals(hvals.begin(), hvals.end());
  thrust::device_vector<int> keys_out(hkeys.size());
  thrust::device_vector<double> vals_out(hvals.size());

  // Adjoint buffers
  thrust::device_vector<int> d_keys(hkeys.size());
  thrust::device_vector<double> d_vals(hvals.size());
  thrust::device_vector<int> d_keys_out(hkeys.size());
  thrust::device_vector<double> d_vals_out(hvals.size());
  thrust::fill(d_keys.begin(), d_keys.end(), 0);
  thrust::fill(d_vals.begin(), d_vals.end(), 0.);
  thrust::fill(d_keys_out.begin(), d_keys_out.end(), 0);
  thrust::fill(d_vals_out.begin(), d_vals_out.end(), 0.);

  // Gradient on first reduced group (key==1)
  d_vals_out[0] = 1.0;
  INIT_GRADIENT(reduce_by_key_simple);
  reduce_by_key_simple_grad.execute(keys, vals, keys_out, vals_out,
                                    &d_keys, &d_vals, &d_keys_out, &d_vals_out);
  thrust::host_vector<double> hdv = d_vals;
  std::printf("Group1 d_vals: %.1f %.1f %.1f %.1f %.1f\n",
              hdv[0], hdv[1], hdv[2], hdv[3], hdv[4]);
  // CHECK-EXEC: Group1 d_vals: 1.0 1.0 0.0 0.0 0.0

  // Reset, gradient on second reduced group (key==2)
  thrust::fill(d_vals.begin(), d_vals.end(), 0.);
  thrust::fill(d_vals_out.begin(), d_vals_out.end(), 0.);
  d_vals_out[1] = 2.0;
  reduce_by_key_simple_grad.execute(keys, vals, keys_out, vals_out,
                                    &d_keys, &d_vals, &d_keys_out, &d_vals_out);
  hdv = d_vals;
  std::printf("Group2 d_vals: %.1f %.1f %.1f %.1f %.1f\n",
              hdv[0], hdv[1], hdv[2], hdv[3], hdv[4]);
  // CHECK-EXEC: Group2 d_vals: 0.0 0.0 2.0 2.0 2.0

  // Explicit predicate/op overload (equal_to, plus)
  thrust::fill(d_vals.begin(), d_vals.end(), 0.);
  thrust::fill(d_vals_out.begin(), d_vals_out.end(), 0.);
  d_vals_out[0] = 1.0;
  INIT_GRADIENT(reduce_by_key_custom_op);
  reduce_by_key_custom_op_grad.execute(keys, vals, keys_out, vals_out,
                                       &d_keys, &d_vals, &d_keys_out, &d_vals_out);
  hdv = d_vals;
  std::printf("Custom Group1 d_vals: %.1f %.1f %.1f %.1f %.1f\n",
              hdv[0], hdv[1], hdv[2], hdv[3], hdv[4]);
  // CHECK-EXEC: Custom Group1 d_vals: 1.0 1.0 0.0 0.0 0.0

  // Reset and test second group
  thrust::fill(d_vals.begin(), d_vals.end(), 0.);
  thrust::fill(d_vals_out.begin(), d_vals_out.end(), 0.);
  d_vals_out[1] = 2.0;
  reduce_by_key_custom_op_grad.execute(keys, vals, keys_out, vals_out,
                                       &d_keys, &d_vals, &d_keys_out, &d_vals_out);
  hdv = d_vals;
  std::printf("Custom Group2 d_vals: %.1f %.1f %.1f %.1f %.1f\n",
              hdv[0], hdv[1], hdv[2], hdv[3], hdv[4]);
  // CHECK-EXEC: Custom Group2 d_vals: 0.0 0.0 2.0 2.0 2.0

  return 0;
}


