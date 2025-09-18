// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oThrustTransform.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: ./ThrustTransform.out | %filecheck_exec %s
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
#include <thrust/reduce.h>
#include <thrust/transform.h>

void transform_negate(const thrust::device_vector<double>& vec, thrust::device_vector<double>& output) {
    thrust::transform(vec.begin(), vec.end(), output.begin(), thrust::negate<double>());
}
// CHECK: void transform_negate_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> &output, thrust::device_vector<double> *_d_vec, thrust::device_vector<double> *_d_output) {
// CHECK-NEXT: {{.*}}thrust::transform_reverse_forw(std::begin(vec), std::end(vec), std::begin(output), thrust::negate<double>(), std::begin((*_d_vec)), std::end((*_d_vec)), std::begin((*_d_output)), {});
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     iterator _r2 = std::begin((*_d_output));
// CHECK-NEXT:     thrust::negate<double> _r3 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_pullback(std::begin(vec), std::end(vec), std::begin(output), thrust::negate<double>(), {}, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT: }
// CHECK-NEXT: }

void transform_identity(const thrust::device_vector<double>& vec, thrust::device_vector<double>& output) {
    thrust::transform(vec.begin(), vec.end(), output.begin(), thrust::identity<double>());
}
// CHECK: void transform_identity_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> &output, thrust::device_vector<double> *_d_vec, thrust::device_vector<double> *_d_output) {
// CHECK-NEXT: {{.*}}thrust::transform_reverse_forw(std::begin(vec), std::end(vec), std::begin(output), thrust::identity<double>(), std::begin((*_d_vec)), std::end((*_d_vec)), std::begin((*_d_output)), {});
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     iterator _r2 = std::begin((*_d_output));
// CHECK-NEXT:     thrust::identity<double> _r3 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_pullback(std::begin(vec), std::end(vec), std::begin(output), thrust::identity<double>(), {}, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT: }
// CHECK-NEXT: }

void transform_plus(const thrust::device_vector<double>& vec1, const thrust::device_vector<double>& vec2, thrust::device_vector<double>& output) {
    thrust::transform(vec1.begin(), vec1.end(), vec2.begin(), output.begin(), thrust::plus<double>());
}
// CHECK: void transform_plus_grad(const thrust::device_vector<double> &vec1, const thrust::device_vector<double> &vec2, thrust::device_vector<double> &output, thrust::device_vector<double> *_d_vec1, thrust::device_vector<double> *_d_vec2, thrust::device_vector<double> *_d_output) {
// CHECK-NEXT: {{.*}}thrust::transform_reverse_forw(std::begin(vec1), std::end(vec1), std::begin(vec2), std::begin(output), thrust::plus<double>(), std::begin((*_d_vec1)), std::end((*_d_vec1)), std::begin((*_d_vec2)), std::begin((*_d_output)), {});
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec1));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec1));
// CHECK-NEXT:     const_iterator _r2 = std::begin((*_d_vec2));
// CHECK-NEXT:     iterator _r3 = std::begin((*_d_output));
// CHECK-NEXT:     thrust::plus<double> _r4 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_pullback(std::begin(vec1), std::end(vec1), std::begin(vec2), std::begin(output), thrust::plus<double>(), {}, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT: }
// CHECK-NEXT: }

void transform_minus(const thrust::device_vector<double>& vec1, const thrust::device_vector<double>& vec2, thrust::device_vector<double>& output) {
    thrust::transform(vec1.begin(), vec1.end(), vec2.begin(), output.begin(), thrust::minus<double>());
}
// CHECK: void transform_minus_grad(const thrust::device_vector<double> &vec1, const thrust::device_vector<double> &vec2, thrust::device_vector<double> &output, thrust::device_vector<double> *_d_vec1, thrust::device_vector<double> *_d_vec2, thrust::device_vector<double> *_d_output) {
// CHECK-NEXT: {{.*}}thrust::transform_reverse_forw(std::begin(vec1), std::end(vec1), std::begin(vec2), std::begin(output), thrust::minus<double>(), std::begin((*_d_vec1)), std::end((*_d_vec1)), std::begin((*_d_vec2)), std::begin((*_d_output)), {});
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec1));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec1));
// CHECK-NEXT:     const_iterator _r2 = std::begin((*_d_vec2));
// CHECK-NEXT:     iterator _r3 = std::begin((*_d_output));
// CHECK-NEXT:     thrust::minus<double> _r4 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_pullback(std::begin(vec1), std::end(vec1), std::begin(vec2), std::begin(output), thrust::minus<double>(), {}, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT: }
// CHECK-NEXT: }

void transform_multiplies(const thrust::device_vector<double>& vec1, const thrust::device_vector<double>& vec2, thrust::device_vector<double>& output) {
    thrust::transform(vec1.begin(), vec1.end(), vec2.begin(), output.begin(), thrust::multiplies<double>());
}
// CHECK: void transform_multiplies_grad(const thrust::device_vector<double> &vec1, const thrust::device_vector<double> &vec2, thrust::device_vector<double> &output, thrust::device_vector<double> *_d_vec1, thrust::device_vector<double> *_d_vec2, thrust::device_vector<double> *_d_output) {
// CHECK-NEXT: {{.*}}thrust::transform_reverse_forw(std::begin(vec1), std::end(vec1), std::begin(vec2), std::begin(output), thrust::multiplies<double>(), std::begin((*_d_vec1)), std::end((*_d_vec1)), std::begin((*_d_vec2)), std::begin((*_d_output)), {});
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec1));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec1));
// CHECK-NEXT:     const_iterator _r2 = std::begin((*_d_vec2));
// CHECK-NEXT:     iterator _r3 = std::begin((*_d_output));
// CHECK-NEXT:     thrust::multiplies<double> _r4 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_pullback(std::begin(vec1), std::end(vec1), std::begin(vec2), std::begin(output), thrust::multiplies<double>(), {}, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT: }
// CHECK-NEXT: }

void transform_divides(const thrust::device_vector<double>& vec1, const thrust::device_vector<double>& vec2, thrust::device_vector<double>& output) {
    thrust::transform(vec1.begin(), vec1.end(), vec2.begin(), output.begin(), thrust::divides<double>());
}
// CHECK: void transform_divides_grad(const thrust::device_vector<double> &vec1, const thrust::device_vector<double> &vec2, thrust::device_vector<double> &output, thrust::device_vector<double> *_d_vec1, thrust::device_vector<double> *_d_vec2, thrust::device_vector<double> *_d_output) {
// CHECK-NEXT: {{.*}}thrust::transform_reverse_forw(std::begin(vec1), std::end(vec1), std::begin(vec2), std::begin(output), thrust::divides<double>(), std::begin((*_d_vec1)), std::end((*_d_vec1)), std::begin((*_d_vec2)), std::begin((*_d_output)), {});
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec1));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec1));
// CHECK-NEXT:     const_iterator _r2 = std::begin((*_d_vec2));
// CHECK-NEXT:     iterator _r3 = std::begin((*_d_output));
// CHECK-NEXT:     thrust::divides<double> _r4 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_pullback(std::begin(vec1), std::end(vec1), std::begin(vec2), std::begin(output), thrust::divides<double>(), {}, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT: }
// CHECK-NEXT: }

int main() {
    std::vector<double> host_input1 = {10.0, 5.0, 2.0, 20.0};
    thrust::device_vector<double> device_input1 = host_input1;

    std::vector<double> host_input2 = {1.0, 2.0, 3.0, 4.0};
    thrust::device_vector<double> device_input2 = host_input2;

    thrust::device_vector<double> device_output(host_input1.size());
    thrust::device_vector<double> d_output(host_input1.size());
    thrust::fill(d_output.begin(), d_output.end(), 1.0);

    // Test Negate
    INIT_GRADIENT(transform_negate);
    thrust::device_vector<double> d_input_negate(host_input1.size());
    transform_negate_grad.execute(device_input1, device_output, &d_input_negate, &d_output);
    thrust::host_vector<double> host_d_input_negate = d_input_negate;
    printf("Negate Gradients: %.3f %.3f %.3f %.3f\n", host_d_input_negate[0], host_d_input_negate[1], host_d_input_negate[2], host_d_input_negate[3]);
    // CHECK-EXEC: Negate Gradients: -1.000 -1.000 -1.000 -1.000

    // Test Identity
    thrust::fill(d_output.begin(), d_output.end(), 1.0);
    INIT_GRADIENT(transform_identity);
    thrust::device_vector<double> d_input_identity(host_input1.size());
    transform_identity_grad.execute(device_input1, device_output, &d_input_identity, &d_output);
    thrust::host_vector<double> host_d_input_identity = d_input_identity;
    printf("Identity Gradients: %.3f %.3f %.3f %.3f\n", host_d_input_identity[0], host_d_input_identity[1], host_d_input_identity[2], host_d_input_identity[3]);
    // CHECK-EXEC: Identity Gradients: 1.000 1.000 1.000 1.000

    // Test Plus
    thrust::fill(d_output.begin(), d_output.end(), 1.0);
    INIT_GRADIENT(transform_plus);
    thrust::device_vector<double> d_input1_plus(host_input1.size());
    thrust::device_vector<double> d_input2_plus(host_input2.size());
    transform_plus_grad.execute(device_input1, device_input2, device_output, &d_input1_plus, &d_input2_plus, &d_output);
    thrust::host_vector<double> host_d_input1_plus = d_input1_plus;
    thrust::host_vector<double> host_d_input2_plus = d_input2_plus;
    printf("Plus Gradients 1: %.3f %.3f %.3f %.3f\n", host_d_input1_plus[0], host_d_input1_plus[1], host_d_input1_plus[2], host_d_input1_plus[3]);
    // CHECK-EXEC: Plus Gradients 1: 1.000 1.000 1.000 1.000
    printf("Plus Gradients 2: %.3f %.3f %.3f %.3f\n", host_d_input2_plus[0], host_d_input2_plus[1], host_d_input2_plus[2], host_d_input2_plus[3]);
    // CHECK-EXEC: Plus Gradients 2: 1.000 1.000 1.000 1.000

    // Test Minus
    thrust::fill(d_output.begin(), d_output.end(), 1.0);
    INIT_GRADIENT(transform_minus);
    thrust::device_vector<double> d_input1_minus(host_input1.size());
    thrust::device_vector<double> d_input2_minus(host_input2.size());
    transform_minus_grad.execute(device_input1, device_input2, device_output, &d_input1_minus, &d_input2_minus, &d_output);
    thrust::host_vector<double> host_d_input1_minus = d_input1_minus;
    thrust::host_vector<double> host_d_input2_minus = d_input2_minus;
    printf("Minus Gradients 1: %.3f %.3f %.3f %.3f\n", host_d_input1_minus[0], host_d_input1_minus[1], host_d_input1_minus[2], host_d_input1_minus[3]);
    // CHECK-EXEC: Minus Gradients 1: 1.000 1.000 1.000 1.000
    printf("Minus Gradients 2: %.3f %.3f %.3f %.3f\n", host_d_input2_minus[0], host_d_input2_minus[1], host_d_input2_minus[2], host_d_input2_minus[3]);
    // CHECK-EXEC: Minus Gradients 2: -1.000 -1.000 -1.000 -1.000

    // Test Multiplies
    thrust::fill(d_output.begin(), d_output.end(), 1.0);
    INIT_GRADIENT(transform_multiplies);
    thrust::device_vector<double> d_input1_multiplies(host_input1.size());
    thrust::device_vector<double> d_input2_multiplies(host_input2.size());
    transform_multiplies_grad.execute(device_input1, device_input2, device_output, &d_input1_multiplies, &d_input2_multiplies, &d_output);
    thrust::host_vector<double> host_d_input1_multiplies = d_input1_multiplies;
    thrust::host_vector<double> host_d_input2_multiplies = d_input2_multiplies;
    printf("Multiplies Gradients 1: %.3f %.3f %.3f %.3f\n", host_d_input1_multiplies[0], host_d_input1_multiplies[1], host_d_input1_multiplies[2], host_d_input1_multiplies[3]);
    // CHECK-EXEC: Multiplies Gradients 1: 1.000 2.000 3.000 4.000
    printf("Multiplies Gradients 2: %.3f %.3f %.3f %.3f\n", host_d_input2_multiplies[0], host_d_input2_multiplies[1], host_d_input2_multiplies[2], host_d_input2_multiplies[3]);
    // CHECK-EXEC: Multiplies Gradients 2: 10.000 5.000 2.000 20.000

    // Test Divides
    thrust::fill(d_output.begin(), d_output.end(), 1.0);
    INIT_GRADIENT(transform_divides);
    thrust::device_vector<double> d_input1_divides(host_input1.size());
    thrust::device_vector<double> d_input2_divides(host_input2.size());
    transform_divides_grad.execute(device_input1, device_input2, device_output, &d_input1_divides, &d_input2_divides, &d_output);
    thrust::host_vector<double> host_d_input1_divides = d_input1_divides;
    thrust::host_vector<double> host_d_input2_divides = d_input2_divides;
    printf("Divides Gradients 1: %.3f %.3f %.3f %.3f\n", host_d_input1_divides[0], host_d_input1_divides[1], host_d_input1_divides[2], host_d_input1_divides[3]);
    // CHECK-EXEC: Divides Gradients 1: 1.000 0.500 0.333 0.250
    printf("Divides Gradients 2: %.3f %.3f %.3f %.3f\n", host_d_input2_divides[0], host_d_input2_divides[1], host_d_input2_divides[2], host_d_input2_divides[3]);
    // CHECK-EXEC: Divides Gradients 2: -10.000 -1.250 -0.222 -1.250

    return 0;
}
