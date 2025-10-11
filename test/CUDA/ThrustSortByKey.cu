// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oThrustSortByKey.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: ./ThrustSortByKey.out | %filecheck_exec %s
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
#include <thrust/sort.h>
#include <thrust/functional.h>

// Simple sort_by_key (no comparator)
void sort_by_key_simple(thrust::device_vector<int>& keys,
                        thrust::device_vector<double>& vals) {
    thrust::sort_by_key(keys.begin(), keys.end(), vals.begin());
}
// CHECK: void sort_by_key_simple_grad(thrust::device_vector<int> &keys, thrust::device_vector<double> &vals, thrust::device_vector<int> *_d_keys, thrust::device_vector<double> *_d_vals) {
// CHECK-NEXT: {{.*}}clad::custom_derivatives::thrust::sort_by_key_reverse_forw(std::begin(keys), std::end(keys), std::begin(vals), std::begin((*_d_keys)), std::end((*_d_keys)), std::begin((*_d_vals)));
// CHECK-NEXT: {
// CHECK-NEXT:     {{.*}}iterator _r0 = std::begin((*_d_keys));
// CHECK-NEXT:     {{.*}}iterator _r1 = std::end((*_d_keys));
// CHECK-NEXT:     {{.*}}iterator _r2 = std::begin((*_d_vals));
// CHECK-NEXT:     clad::custom_derivatives::thrust::sort_by_key_pullback(std::begin(keys), std::end(keys), std::begin(vals), &_r0, &_r1, &_r2);
// CHECK-NEXT: }
// CHECK-NEXT: }

// sort_by_key with comparator
void sort_by_key_comp(thrust::device_vector<int>& keys,
                      thrust::device_vector<double>& vals) {
    thrust::sort_by_key(keys.begin(), keys.end(), vals.begin(), thrust::greater<int>());
}
// CHECK: void sort_by_key_comp_grad(thrust::device_vector<int> &keys, thrust::device_vector<double> &vals, thrust::device_vector<int> *_d_keys, thrust::device_vector<double> *_d_vals) {
// CHECK-NEXT: {{.*}}clad::custom_derivatives::thrust::sort_by_key_reverse_forw(std::begin(keys), std::end(keys), std::begin(vals), thrust::greater<int>(), std::begin((*_d_keys)), std::end((*_d_keys)), std::begin((*_d_vals)), {});
// CHECK-NEXT: {
// CHECK-NEXT:     {{.*}}iterator _r0 = std::begin((*_d_keys));
// CHECK-NEXT:     {{.*}}iterator _r1 = std::end((*_d_keys));
// CHECK-NEXT:     {{.*}}iterator _r2 = std::begin((*_d_vals));
// CHECK-NEXT:     thrust::greater<int> _r3 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::sort_by_key_pullback(std::begin(keys), std::end(keys), std::begin(vals), thrust::greater<int>(), &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT: }
// CHECK-NEXT: }

int main() {
    std::vector<int> hkeys_simple{3, 1, 4, 2};
    std::vector<double> hvals_simple{30., 10., 40., 20.};
    thrust::device_vector<int> keys_simple(hkeys_simple.begin(), hkeys_simple.end());
    thrust::device_vector<double> vals_simple(hvals_simple.begin(), hvals_simple.end());
    thrust::device_vector<int> d_keys_simple(4);
    thrust::device_vector<double> d_vals_simple(4);
    thrust::fill(d_keys_simple.begin(), d_keys_simple.end(), 0);
    thrust::fill(d_vals_simple.begin(), d_vals_simple.end(), 0.);

    d_vals_simple[2] = 1.0;

    INIT_GRADIENT(sort_by_key_simple);
    sort_by_key_simple_grad.execute(keys_simple, vals_simple, &d_keys_simple, &d_vals_simple);

    thrust::host_vector<double> h_d_vals_simple = d_vals_simple;
    thrust::host_vector<int> h_d_keys_simple = d_keys_simple;
    std::printf("Simple vals d: %.1f %.1f %.1f %.1f\n", h_d_vals_simple[0], h_d_vals_simple[1], h_d_vals_simple[2], h_d_vals_simple[3]);
    // CHECK-EXEC: Simple vals d: 1.0 0.0 0.0 0.0
    std::printf("Simple keys d: %d %d %d %d\n", h_d_keys_simple[0], h_d_keys_simple[1], h_d_keys_simple[2], h_d_keys_simple[3]);
    // CHECK-EXEC: Simple keys d: 0 0 0 0

    std::vector<int> hkeys_comp{3, 1, 4, 2};
    std::vector<double> hvals_comp{30., 10., 40., 20.};
    thrust::device_vector<int> keys_comp(hkeys_comp.begin(), hkeys_comp.end());
    thrust::device_vector<double> vals_comp(hvals_comp.begin(), hvals_comp.end());
    thrust::device_vector<int> d_keys_comp(4);
    thrust::device_vector<double> d_vals_comp(4);
    thrust::fill(d_keys_comp.begin(), d_keys_comp.end(), 0);
    thrust::fill(d_vals_comp.begin(), d_vals_comp.end(), 0.);

    d_vals_comp[1] = 1.0;

    INIT_GRADIENT(sort_by_key_comp);
    sort_by_key_comp_grad.execute(keys_comp, vals_comp, &d_keys_comp, &d_vals_comp);

    thrust::host_vector<double> h_d_vals_comp = d_vals_comp;
    thrust::host_vector<int> h_d_keys_comp = d_keys_comp;
    std::printf("Comp vals d: %.1f %.1f %.1f %.1f\n", h_d_vals_comp[0], h_d_vals_comp[1], h_d_vals_comp[2], h_d_vals_comp[3]);
    // CHECK-EXEC: Comp vals d: 1.0 0.0 0.0 0.0
    std::printf("Comp keys d: %d %d %d %d\n", h_d_keys_comp[0], h_d_keys_comp[1], h_d_keys_comp[2], h_d_keys_comp[3]);
    // CHECK-EXEC: Comp keys d: 0 0 0 0

    return 0;
}


