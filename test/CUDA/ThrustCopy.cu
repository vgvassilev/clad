// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oThrustCopy.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: ./ThrustCopy.out | %filecheck_exec %s
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

void copy(const thrust::device_vector<double>& src,
                 thrust::device_vector<double>& dst) {
    thrust::copy(src.begin(), src.end(), dst.begin());
}
// CHECK: void copy_grad(const thrust::device_vector<double> &src, thrust::device_vector<double> &dst, thrust::device_vector<double> *_d_src, thrust::device_vector<double> *_d_dst) {
// CHECK-NEXT:     {{.*}}thrust::copy_reverse_forw(std::begin(src), std::end(src), std::begin(dst), std::begin((*_d_src)), std::end((*_d_src)), std::begin((*_d_dst)));
// CHECK-NEXT:     {
// CHECK-NEXT:         const_iterator _r0 = std::begin((*_d_src));
// CHECK-NEXT:         const_iterator _r1 = std::end((*_d_src));
// CHECK-NEXT:         iterator _r2 = std::begin((*_d_dst));
// CHECK-NEXT:         clad::custom_derivatives::thrust::copy_pullback(std::begin(src), std::end(src), std::begin(dst), {}, &_r0, &_r1, &_r2);
// CHECK-NEXT:     }
// CHECK-NEXT: }

void copy_with_return_value_stored(const thrust::device_vector<double>& src,
                                   thrust::device_vector<double>& dst) {
    auto it = thrust::copy(src.begin(), src.end(), dst.begin());
    (void)it;
}
// CHECK: void copy_with_return_value_stored_grad(const thrust::device_vector<double> &src, thrust::device_vector<double> &dst, thrust::device_vector<double> *_d_src, thrust::device_vector<double> *_d_dst) {
// CHECK-NEXT:     clad::ValueAndAdjoint<{{.*}}> _t0 = {{.*}}thrust::copy_reverse_forw(std::begin(src), std::end(src), std::begin(dst), std::begin((*_d_src)), std::end((*_d_src)), std::begin((*_d_dst)));
// CHECK-NEXT:     thrust::detail::normal_iterator<device_ptr<double> > it = _t0.value;
// CHECK-NEXT:     thrust::detail::normal_iterator<device_ptr<double> > _d_it = _t0.adjoint;
// CHECK-NEXT:     {
// CHECK-NEXT:         const_iterator _r0 = std::begin((*_d_src));
// CHECK-NEXT:         const_iterator _r1 = std::end((*_d_src));
// CHECK-NEXT:         iterator _r2 = std::begin((*_d_dst));
// CHECK-NEXT:         clad::custom_derivatives::thrust::copy_pullback(std::begin(src), std::end(src), std::begin(dst), _d_it, &_r0, &_r1, &_r2);
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
    std::vector<double> h_src = {10.0, 5.0, 2.0, 20.0};
    thrust::device_vector<double> d_src_input = h_src;
    thrust::device_vector<double> d_dst_input(h_src.size(), 0.0);

    INIT_GRADIENT(copy);

    thrust::device_vector<double> d_src_grad(h_src.size(), 0.0);
    std::vector<double> h_dst_grad_init(h_src.size(), 1.0);
    thrust::device_vector<double> d_dst_grad = h_dst_grad_init;

    copy_grad.execute(d_src_input, d_dst_input, &d_src_grad, &d_dst_grad);

    thrust::host_vector<double> h_src_grad = d_src_grad;
    thrust::host_vector<double> h_dst_grad = d_dst_grad;

    printf("Copy Grad d_src: %.3f %.3f %.3f %.3f\n", h_src_grad[0], h_src_grad[1], h_src_grad[2], h_src_grad[3]);
    // CHECK-EXEC: Copy Grad d_src: 1.000 1.000 1.000 1.000
    printf("Copy Grad d_dst: %.3f %.3f %.3f %.3f\n", h_dst_grad[0], h_dst_grad[1], h_dst_grad[2], h_dst_grad[3]);
    // CHECK-EXEC: Copy Grad d_dst: 0.000 0.000 0.000 0.000


    thrust::device_vector<double> d_src_input2 = h_src;
    thrust::device_vector<double> d_dst_input2(h_src.size(), 0.0);
    INIT_GRADIENT(copy_with_return_value_stored);

    thrust::device_vector<double> d_src_grad2(h_src.size(), 0.0);
    thrust::device_vector<double> d_dst_grad2 = h_dst_grad_init;

    copy_with_return_value_stored_grad.execute(d_src_input2, d_dst_input2, &d_src_grad2, &d_dst_grad2);

    thrust::host_vector<double> h_src_grad2 = d_src_grad2;
    thrust::host_vector<double> h_dst_grad2 = d_dst_grad2;

    printf("Copy with return value stored Grad d_src: %.3f %.3f %.3f %.3f\n", h_src_grad2[0], h_src_grad2[1], h_src_grad2[2], h_src_grad2[3]);
    // CHECK-EXEC: Copy with return value stored Grad d_src: 1.000 1.000 1.000 1.000
    printf("Copy with return value stored Grad d_dst: %.3f %.3f %.3f %.3f\n", h_dst_grad2[0], h_dst_grad2[1], h_dst_grad2[2], h_dst_grad2[3]);
    // CHECK-EXEC: Copy with return value stored Grad d_dst: 0.000 0.000 0.000 0.000

    return 0;
}


