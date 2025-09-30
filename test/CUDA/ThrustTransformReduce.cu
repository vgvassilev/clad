// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oThrustTransformReduce.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: ./ThrustTransformReduce.out | %filecheck_exec %s
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

double transform_reduce_plus_negate(const thrust::device_vector<double>& vec) {
    return thrust::transform_reduce(vec.begin(), vec.end(), thrust::negate<double>(), 0.0, thrust::plus<double>());
}
// CHECK: void transform_reduce_plus_negate_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> *_d_vec) {
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     thrust::negate<double> _r2 = {};
// CHECK-NEXT:     double _r3 = 0.;
// CHECK-NEXT:     thrust::plus<double> _r4 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_reduce_pullback(std::begin(vec), std::end(vec), thrust::negate<double>(), 0., thrust::plus<double>(), 1, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT: }
// CHECK-NEXT: }

double transform_reduce_max_negate(const thrust::device_vector<double>& vec) {
    return thrust::transform_reduce(vec.begin(), vec.end(), thrust::negate<double>(), 0.0, thrust::maximum<double>());
}
// CHECK: void transform_reduce_max_negate_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> *_d_vec) {
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     thrust::negate<double> _r2 = {};
// CHECK-NEXT:     double _r3 = 0.;
// CHECK-NEXT:     thrust::maximum<double> _r4 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_reduce_pullback(std::begin(vec), std::end(vec), thrust::negate<double>(), 0., thrust::maximum<double>(), 1, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT: }
// CHECK-NEXT: }

double transform_reduce_plus_identity(const thrust::device_vector<double>& vec) {
    return thrust::transform_reduce(vec.begin(), vec.end(), thrust::identity<double>(), 0.0, thrust::plus<double>());
}
// CHECK: void transform_reduce_plus_identity_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> *_d_vec) {
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     thrust::identity<double> _r2 = {};
// CHECK-NEXT:     double _r3 = 0.;
// CHECK-NEXT:     thrust::plus<double> _r4 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_reduce_pullback(std::begin(vec), std::end(vec), thrust::identity<double>(), 0., thrust::plus<double>(), 1, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT: }
// CHECK-NEXT: }

double transform_reduce_max_identity(const thrust::device_vector<double>& vec) {
    return thrust::transform_reduce(vec.begin(), vec.end(), thrust::identity<double>(), 0.0, thrust::maximum<double>());
}
// CHECK: void transform_reduce_max_identity_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> *_d_vec) {
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     thrust::identity<double> _r2 = {};
// CHECK-NEXT:     double _r3 = 0.;
// CHECK-NEXT:     thrust::maximum<double> _r4 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_reduce_pullback(std::begin(vec), std::end(vec), thrust::identity<double>(), 0., thrust::maximum<double>(), 1, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT: }
// CHECK-NEXT: }

double transform_reduce_min_negate(const thrust::device_vector<double>& vec) {
    return thrust::transform_reduce(vec.begin(), vec.end(), thrust::negate<double>(), 100.0, thrust::minimum<double>());
}
// CHECK: void transform_reduce_min_negate_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> *_d_vec) {
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     thrust::negate<double> _r2 = {};
// CHECK-NEXT:     double _r3 = 0.;
// CHECK-NEXT:     thrust::minimum<double> _r4 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_reduce_pullback(std::begin(vec), std::end(vec), thrust::negate<double>(), 100., thrust::minimum<double>(), 1, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT: }
// CHECK-NEXT: }


int main() {
    std::vector<double> host_input = {1.0, 2.0, -3.0, 4.0, -5.0};
    thrust::device_vector<double> device_input = host_input;

    // Test Plus-Negate
    INIT_GRADIENT(transform_reduce_plus_negate);
    thrust::device_vector<double> d_input_plus(host_input.size());
    transform_reduce_plus_negate_grad.execute(device_input, &d_input_plus);

    thrust::host_vector<double> host_d_input_plus = d_input_plus;
    printf("Plus-Negate Gradients: ");
    for(size_t i = 0; i < host_d_input_plus.size(); ++i) {
        printf("%.3f ", host_d_input_plus[i]);
    }
    printf("\n");
    // CHECK-EXEC: Plus-Negate Gradients: -1.000 -1.000 -1.000 -1.000 -1.000 

    // Test Max-Negate
    INIT_GRADIENT(transform_reduce_max_negate);
    thrust::device_vector<double> d_input_max(host_input.size());
    transform_reduce_max_negate_grad.execute(device_input, &d_input_max);
    
    thrust::host_vector<double> host_d_input_max = d_input_max;
    printf("Max-Negate Gradients: ");
    for(size_t i = 0; i < host_d_input_max.size(); ++i) {
        printf("%.3f ", host_d_input_max[i]);
    }
    printf("\n");
    // CHECK-EXEC: Max-Negate Gradients: 0.000 0.000 0.000 0.000 -1.000 

    // Test Plus-Identity
    INIT_GRADIENT(transform_reduce_plus_identity);
    thrust::device_vector<double> d_input_plus_id(host_input.size());
    transform_reduce_plus_identity_grad.execute(device_input, &d_input_plus_id);
    thrust::host_vector<double> host_d_input_plus_id = d_input_plus_id;
    printf("Plus-Identity Gradients: ");
    for(size_t i = 0; i < host_d_input_plus_id.size(); ++i) {
        printf("%.3f ", host_d_input_plus_id[i]);
    }
    printf("\n");
    // CHECK-EXEC: Plus-Identity Gradients: 1.000 1.000 1.000 1.000 1.000 

    // Test Max-Identity
    INIT_GRADIENT(transform_reduce_max_identity);
    thrust::device_vector<double> d_input_max_id(host_input.size());
    transform_reduce_max_identity_grad.execute(device_input, &d_input_max_id);
    thrust::host_vector<double> host_d_input_max_id = d_input_max_id;
    printf("Max-Identity Gradients: ");
    for(size_t i = 0; i < host_d_input_max_id.size(); ++i) {
        printf("%.3f ", host_d_input_max_id[i]);
    }
    printf("\n");
    // CHECK-EXEC: Max-Identity Gradients: 0.000 0.000 0.000 1.000 0.000 

    // Test Min-Negate
    INIT_GRADIENT(transform_reduce_min_negate);
    thrust::device_vector<double> d_input_min_neg(host_input.size());
    transform_reduce_min_negate_grad.execute(device_input, &d_input_min_neg);
    thrust::host_vector<double> host_d_input_min_neg = d_input_min_neg;
    printf("Min-Negate Gradients: ");
    for(size_t i = 0; i < host_d_input_min_neg.size(); ++i) {
        printf("%.3f ", host_d_input_min_neg[i]);
    }
    printf("\n");
    // CHECK-EXEC: Min-Negate Gradients: 0.000 0.000 0.000 -1.000 0.000 

    return 0;
}
