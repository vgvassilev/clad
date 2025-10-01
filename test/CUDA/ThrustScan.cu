// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oThrustScan.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: ./ThrustScan.out | %filecheck_exec %s
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
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/memory.h>
#include <thrust/scan.h>

void inclusive_scan_plus(const thrust::device_vector<double>& vec, thrust::device_vector<double>& output) {
    thrust::inclusive_scan(vec.begin(), vec.end(), output.begin(), thrust::plus<double>());
}
// CHECK: void inclusive_scan_plus_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> &output, thrust::device_vector<double> *_d_vec, thrust::device_vector<double> *_d_output) {
// CHECK-NEXT: {{.*}}thrust::inclusive_scan_reverse_forw(std::begin(vec), std::end(vec), std::begin(output), thrust::plus<double>(), std::begin((*_d_vec)), std::end((*_d_vec)), std::begin((*_d_output)), {});
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     iterator _r2 = std::begin((*_d_output));
// CHECK-NEXT:     thrust::plus<double> _r3 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::inclusive_scan_pullback(std::begin(vec), std::end(vec), std::begin(output), thrust::plus<double>(), {}, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT: }
// CHECK-NEXT: }

void exclusive_scan_plus(const thrust::device_vector<double>& vec, thrust::device_vector<double>& output, double init) {
    thrust::exclusive_scan(vec.begin(), vec.end(), output.begin(), init, thrust::plus<double>());
}
// CHECK: void exclusive_scan_plus_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> &output, double init, thrust::device_vector<double> *_d_vec, thrust::device_vector<double> *_d_output, double *_d_init) {
// CHECK-NEXT: {{.*}}thrust::exclusive_scan_reverse_forw(std::begin(vec), std::end(vec), std::begin(output), init, thrust::plus<double>(), std::begin((*_d_vec)), std::end((*_d_vec)), std::begin((*_d_output)), 0., {});
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     iterator _r2 = std::begin((*_d_output));
// CHECK-NEXT:     double _r3 = 0.;
// CHECK-NEXT:     thrust::plus<double> _r4 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::exclusive_scan_pullback(std::begin(vec), std::end(vec), std::begin(output), init, thrust::plus<double>(), {}, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT:     *_d_init += _r3;
// CHECK-NEXT: }
// CHECK-NEXT: }

int main() {
    std::vector<double> host_input = {1.0, 2.0, 3.0, 4.0};
    thrust::device_vector<double> device_input = host_input;
    thrust::device_vector<double> device_output(host_input.size());

    thrust::device_vector<double> d_output(host_input.size());
    thrust::fill(d_output.begin(), d_output.end(), 1.0);

    // Test Inclusive Scan
    INIT_GRADIENT(inclusive_scan_plus);
    thrust::device_vector<double> d_input_inclusive(host_input.size());
    inclusive_scan_plus_grad.execute(device_input, device_output, &d_input_inclusive, &d_output);
    thrust::host_vector<double> host_d_input_inclusive = d_input_inclusive;
    printf("Inclusive Scan Gradients: %.3f %.3f %.3f %.3f\n", host_d_input_inclusive[0], host_d_input_inclusive[1], host_d_input_inclusive[2], host_d_input_inclusive[3]);
    // CHECK-EXEC: Inclusive Scan Gradients: 4.000 3.000 2.000 1.000

    // Test Exclusive Scan
    thrust::fill(d_output.begin(), d_output.end(), 1.0);
    INIT_GRADIENT(exclusive_scan_plus);
    thrust::device_vector<double> d_input_exclusive(host_input.size());
    double d_init_exclusive = 1.0;
    exclusive_scan_plus_grad.execute(device_input, device_output, 0.0, &d_input_exclusive, &d_output, &d_init_exclusive);
    thrust::host_vector<double> host_d_input_exclusive = d_input_exclusive;
    printf("Exclusive Scan Gradients: %.3f %.3f %.3f %.3f\n", host_d_input_exclusive[0], host_d_input_exclusive[1], host_d_input_exclusive[2], host_d_input_exclusive[3]);
    // CHECK-EXEC: Exclusive Scan Gradients: 3.000 2.000 1.000 0.000
    printf("Exclusive Scan Init Gradient: %.3f\n", d_init_exclusive);
    // CHECK-EXEC: Exclusive Scan Init Gradient: 5.000

    return 0;
}
