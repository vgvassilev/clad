// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oThrustReduce.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: ./ThrustReduce.out | %filecheck_exec %s
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


double sum_array(const thrust::device_vector<double>& vec) {
    return thrust::reduce(vec.begin(), vec.end(), 0.0, thrust::plus<double>());
}
// CHECK: void sum_array_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> *_d_vec) {
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     double _r2 = 0.;
// CHECK-NEXT:     thrust::plus<double> _r3 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::reduce_pullback(std::begin(vec), std::end(vec), 0., thrust::plus<double>(), 1, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT: }
// CHECK-NEXT: }

double max_array(const thrust::device_vector<double>& vec) {
    return thrust::reduce(vec.begin(), vec.end(), 0.0, thrust::maximum<double>());
}
// CHECK: void max_array_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> *_d_vec) {
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     double _r2 = 0.;
// CHECK-NEXT:     thrust::maximum<double> _r3 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::reduce_pullback(std::begin(vec), std::end(vec), 0., thrust::maximum<double>(), 1, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT: }
// CHECK-NEXT: }

double min_array(const thrust::device_vector<double>& vec) {
    return thrust::reduce(vec.begin(), vec.end(), 100.0, thrust::minimum<double>());
}
// CHECK: void min_array_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> *_d_vec) {
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     double _r2 = 0.;
// CHECK-NEXT:     thrust::minimum<double> _r3 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::reduce_pullback(std::begin(vec), std::end(vec), 100., thrust::minimum<double>(), 1, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT: }
// CHECK-NEXT: }

double product_array(const thrust::device_vector<double>& vec) {
    return thrust::reduce(vec.begin(), vec.end(), 1.0, thrust::multiplies<double>());
}
// CHECK: void product_array_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> *_d_vec) {
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     double _r2 = 0.;
// CHECK-NEXT:     thrust::multiplies<double> _r3 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::reduce_pullback(std::begin(vec), std::end(vec), 1., thrust::multiplies<double>(), 1, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT: }
// CHECK-NEXT: }

double product_array_init_zero(const thrust::device_vector<double>& vec) {
    return thrust::reduce(vec.begin(), vec.end(), 0.0, thrust::multiplies<double>());
}
// CHECK: void product_array_init_zero_grad(const thrust::device_vector<double> &vec, thrust::device_vector<double> *_d_vec) {
// CHECK-NEXT: {
// CHECK-NEXT:     const_iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     const_iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     double _r2 = 0.;
// CHECK-NEXT:     thrust::multiplies<double> _r3 = {};
// CHECK-NEXT:     clad::custom_derivatives::thrust::reduce_pullback(std::begin(vec), std::end(vec), 0., thrust::multiplies<double>(), 1, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT: }
// CHECK-NEXT: }


int main() {
    // --- Standard Tests ---
    std::vector<double> host_input = {10.0, 5.0, 2.0, 20.0};
    thrust::device_vector<double> device_input = host_input;

    // Test Summation
    INIT_GRADIENT(sum_array);
    thrust::device_vector<double> sum_gradients(host_input.size());
    sum_array_grad.execute(device_input, &sum_gradients);
    thrust::host_vector<double> host_sum_gradients = sum_gradients;
    printf("Sum Gradients: %.3f %.3f %.3f %.3f\n", host_sum_gradients[0], host_sum_gradients[1], host_sum_gradients[2], host_sum_gradients[3]);
    // CHECK-EXEC: Sum Gradients: 1.000 1.000 1.000 1.000

    // Test Maximum
    INIT_GRADIENT(max_array);
    thrust::device_vector<double> max_gradients(host_input.size());
    max_array_grad.execute(device_input, &max_gradients);
    thrust::host_vector<double> host_max_gradients = max_gradients;
    printf("Max Gradients: %.3f %.3f %.3f %.3f\n", host_max_gradients[0], host_max_gradients[1], host_max_gradients[2], host_max_gradients[3]);
    // CHECK-EXEC: Max Gradients: 0.000 0.000 0.000 1.000

    // Test Minimum
    INIT_GRADIENT(min_array);
    thrust::device_vector<double> min_gradients(host_input.size());
    min_array_grad.execute(device_input, &min_gradients);
    thrust::host_vector<double> host_min_gradients = min_gradients;
    printf("Min Gradients: %.3f %.3f %.3f %.3f\n", host_min_gradients[0], host_min_gradients[1], host_min_gradients[2], host_min_gradients[3]);
    // CHECK-EXEC: Min Gradients: 0.000 0.000 1.000 0.000

    // Test Product (No Zeros)
    INIT_GRADIENT(product_array);
    thrust::device_vector<double> product_gradients(host_input.size());
    product_array_grad.execute(device_input, &product_gradients);
    thrust::host_vector<double> host_product_gradients = product_gradients;
    printf("Product Gradients (No Zeros): %.3f %.3f %.3f %.3f\n", host_product_gradients[0], host_product_gradients[1], host_product_gradients[2], host_product_gradients[3]);
    // CHECK-EXEC: Product Gradients (No Zeros): 200.000 400.000 1000.000 100.000

    // --- Robust Product Tests ---
    
    // Test case 1: One zero in the input vector
    std::vector<double> host_input_one_zero = {10.0, 5.0, 0.0, 20.0};
    thrust::device_vector<double> device_input_one_zero = host_input_one_zero;
    thrust::device_vector<double> p_grads_one_zero(host_input_one_zero.size());
    product_array_grad.execute(device_input_one_zero, &p_grads_one_zero);
    thrust::host_vector<double> h_p_grads_one_zero = p_grads_one_zero;
    printf("Product Gradients (One Zero): %.3f %.3f %.3f %.3f\n", h_p_grads_one_zero[0], h_p_grads_one_zero[1], h_p_grads_one_zero[2], h_p_grads_one_zero[3]);
    // CHECK-EXEC: Product Gradients (One Zero): 0.000 0.000 1000.000 0.000

    // Test case 2: Multiple zeros in the input vector
    std::vector<double> host_input_multi_zero = {10.0, 0.0, 2.0, 0.0};
    thrust::device_vector<double> device_input_multi_zero = host_input_multi_zero;
    thrust::device_vector<double> p_grads_multi_zero(host_input_multi_zero.size());
    product_array_grad.execute(device_input_multi_zero, &p_grads_multi_zero);
    thrust::host_vector<double> h_p_grads_multi_zero = p_grads_multi_zero;
    printf("Product Gradients (Multiple Zeros): %.3f %.3f %.3f %.3f\n", h_p_grads_multi_zero[0], h_p_grads_multi_zero[1], h_p_grads_multi_zero[2], h_p_grads_multi_zero[3]);
    // CHECK-EXEC: Product Gradients (Multiple Zeros): 0.000 0.000 0.000 0.000

    // Test case 3: Initial value is zero
    INIT_GRADIENT(product_array_init_zero);
    std::vector<double> host_input_for_init_zero = {10.0, 5.0, 2.0, 20.0};
    thrust::device_vector<double> device_input_for_init_zero = host_input_for_init_zero;
    thrust::device_vector<double> p_grads_init_zero(host_input_for_init_zero.size());
    product_array_init_zero_grad.execute(device_input_for_init_zero, &p_grads_init_zero);
    thrust::host_vector<double> h_p_grads_init_zero = p_grads_init_zero;
    printf("Product Gradients (Init is Zero): %.3f %.3f %.3f %.3f\n", h_p_grads_init_zero[0], h_p_grads_init_zero[1], h_p_grads_init_zero[2], h_p_grads_init_zero[3]);
    // CHECK-EXEC: Product Gradients (Init is Zero): 0.000 0.000 0.000 0.000

    return 0;
} 