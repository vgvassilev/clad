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
#include <cmath>

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

CUDA_HOST_DEVICE inline void add_host_device(double* dst, double v) {
    #if defined(__CUDA_ARCH__)
        ::atomicAdd(dst, v);
    #else
        *dst += v;
    #endif
}
    
struct Affine {
    double a; double b;
    CUDA_HOST_DEVICE double operator()(double x) const { return a * x + b; }
    CUDA_HOST_DEVICE void operator_call_pullback(double x, double d_y, Affine* d_this, double* d_x) const {
        *d_x += d_y * a;
        if (d_this) {
            add_host_device(&d_this->a, d_y * x);
            add_host_device(&d_this->b, d_y);
        }
    }
};

void apply_affine(const thrust::device_vector<double>& v,
                    thrust::device_vector<double>& out,
                    Affine op) {
    thrust::transform(v.begin(), v.end(), out.begin(), op);
}
// CHECK: void apply_affine_grad(const thrust::device_vector<double> &v, thrust::device_vector<double> &out, Affine op, thrust::device_vector<double> *_d_v, thrust::device_vector<double> *_d_out, Affine *_d_op) {
// CHECK-NEXT: {{.*}}thrust::transform_reverse_forw(std::begin(v), std::end(v), std::begin(out), op, std::begin((*_d_v)), std::end((*_d_v)), std::begin((*_d_out)), (*_d_op));
// CHECK-NEXT: {
// CHECK-NEXT:     {{.*}}const_iterator _r0 = std::begin((*_d_v));
// CHECK-NEXT:     {{.*}}const_iterator _r1 = std::end((*_d_v));
// CHECK-NEXT:     {{.*}}iterator _r2 = std::begin((*_d_out));
// CHECK-NEXT:     Affine _r3 = (*_d_op);
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_pullback(std::begin(v), std::end(v), std::begin(out), op, {}, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT:     Affine::constructor_pullback(op, &_r3, _d_op);
// CHECK-NEXT: }
// CHECK-NEXT: }

struct SquarePlusBias {
    double a; double b; // y = a*x*x + b
    CUDA_HOST_DEVICE double operator()(double x) const { return a * x * x + b; }
    CUDA_HOST_DEVICE void operator_call_pullback(double x, double d_y, SquarePlusBias* d_this, double* d_x) const {
        *d_x += d_y * (2.0 * a * x);
        if (d_this) {
            add_host_device(&d_this->a, d_y * x * x);
            add_host_device(&d_this->b, d_y);
        }
    }
};

void apply_spb(const thrust::device_vector<double>& v,
                thrust::device_vector<double>& out,
                SquarePlusBias op) {
    thrust::transform(v.begin(), v.end(), out.begin(), op);
}
// CHECK: void apply_spb_grad(const thrust::device_vector<double> &v, thrust::device_vector<double> &out, SquarePlusBias op, thrust::device_vector<double> *_d_v, thrust::device_vector<double> *_d_out, SquarePlusBias *_d_op) {
// CHECK-NEXT: {{.*}}thrust::transform_reverse_forw(std::begin(v), std::end(v), std::begin(out), op, std::begin((*_d_v)), std::end((*_d_v)), std::begin((*_d_out)), (*_d_op));
// CHECK-NEXT: {
// CHECK-NEXT:     {{.*}}const_iterator _r0 = std::begin((*_d_v));
// CHECK-NEXT:     {{.*}}const_iterator _r1 = std::end((*_d_v));
// CHECK-NEXT:     {{.*}}iterator _r2 = std::begin((*_d_out));
// CHECK-NEXT:     SquarePlusBias _r3 = (*_d_op);
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_pullback(std::begin(v), std::end(v), std::begin(out), op, {}, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT:     SquarePlusBias::constructor_pullback(op, &_r3, _d_op);
// CHECK-NEXT: }
// CHECK-NEXT: }

struct Bilinear {
    double a; double b; double c;
    CUDA_HOST_DEVICE double operator()(double x1, double x2) const { return a * x1 + b * x2 + c; }
    CUDA_HOST_DEVICE void operator_call_pullback(double x1, double x2, double d_y, Bilinear* d_this, double* d_x1, double* d_x2) const {
        *d_x1 += d_y * a;
        *d_x2 += d_y * b;
        if (d_this) {
            add_host_device(&d_this->a, d_y * x1);
            add_host_device(&d_this->b, d_y * x2);
            add_host_device(&d_this->c, d_y);
        }
    }
};

void apply_bilinear(const thrust::device_vector<double>& v1,
                    const thrust::device_vector<double>& v2,
                    thrust::device_vector<double>& out,
                    Bilinear op) {
    thrust::transform(v1.begin(), v1.end(), v2.begin(), out.begin(), op);
}
// CHECK: void apply_bilinear_grad(const thrust::device_vector<double> &v1, const thrust::device_vector<double> &v2, thrust::device_vector<double> &out, Bilinear op, thrust::device_vector<double> *_d_v1, thrust::device_vector<double> *_d_v2, thrust::device_vector<double> *_d_out, Bilinear *_d_op) {
// CHECK-NEXT: {{.*}}thrust::transform_reverse_forw(std::begin(v1), std::end(v1), std::begin(v2), std::begin(out), op, std::begin((*_d_v1)), std::end((*_d_v1)), std::begin((*_d_v2)), std::begin((*_d_out)), (*_d_op));
// CHECK-NEXT: {
// CHECK-NEXT:     {{.*}}const_iterator _r0 = std::begin((*_d_v1));
// CHECK-NEXT:     {{.*}}const_iterator _r1 = std::end((*_d_v1));
// CHECK-NEXT:     {{.*}}const_iterator _r2 = std::begin((*_d_v2));
// CHECK-NEXT:     {{.*}}iterator _r3 = std::begin((*_d_out));
// CHECK-NEXT:     Bilinear _r4 = (*_d_op);
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_pullback(std::begin(v1), std::end(v1), std::begin(v2), std::begin(out), op, {}, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT:     Bilinear::constructor_pullback(op, &_r4, _d_op);
// CHECK-NEXT: }
// CHECK-NEXT: }

struct MixOp {
    double a; double b; double c; // y = x1*exp(b*x2 + c) + a*sin(x1)
    CUDA_HOST_DEVICE double operator()(double x1, double x2) const {
        return x1 * ::exp(b * x2 + c) + a * ::sin(x1);
    }
    CUDA_HOST_DEVICE void operator_call_pullback(double x1, double x2, double d_y, MixOp* d_this, double* d_x1, double* d_x2) const {
        const double t = ::exp(b * x2 + c);
        *d_x1 += d_y * (t + a * ::cos(x1));
        *d_x2 += d_y * (x1 * t * b);
        if (d_this) {
            add_host_device(&d_this->a, d_y * ::sin(x1));
            add_host_device(&d_this->b, d_y * (x1 * x2 * t));
            add_host_device(&d_this->c, d_y * (x1 * t));
        }
    }
};

void apply_mix(const thrust::device_vector<double>& v1,
                const thrust::device_vector<double>& v2,
                thrust::device_vector<double>& out,
                MixOp op) {
    thrust::transform(v1.begin(), v1.end(), v2.begin(), out.begin(), op);
}
// CHECK: void apply_mix_grad(const thrust::device_vector<double> &v1, const thrust::device_vector<double> &v2, thrust::device_vector<double> &out, MixOp op, thrust::device_vector<double> *_d_v1, thrust::device_vector<double> *_d_v2, thrust::device_vector<double> *_d_out, MixOp *_d_op) {
// CHECK-NEXT: {{.*}}thrust::transform_reverse_forw(std::begin(v1), std::end(v1), std::begin(v2), std::begin(out), op, std::begin((*_d_v1)), std::end((*_d_v1)), std::begin((*_d_v2)), std::begin((*_d_out)), (*_d_op));
// CHECK-NEXT: {
// CHECK-NEXT:     {{.*}}const_iterator _r0 = std::begin((*_d_v1));
// CHECK-NEXT:     {{.*}}const_iterator _r1 = std::end((*_d_v1));
// CHECK-NEXT:     {{.*}}const_iterator _r2 = std::begin((*_d_v2));
// CHECK-NEXT:     {{.*}}iterator _r3 = std::begin((*_d_out));
// CHECK-NEXT:     MixOp _r4 = (*_d_op);
// CHECK-NEXT:     clad::custom_derivatives::thrust::transform_pullback(std::begin(v1), std::end(v1), std::begin(v2), std::begin(out), op, {}, &_r0, &_r1, &_r2, &_r3, &_r4);
// CHECK-NEXT:     MixOp::constructor_pullback(op, &_r4, _d_op);
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

    // Test custom functor with operator_call_pullback (unary)
    thrust::device_vector<double> device_out(host_input1.size());
    thrust::fill(d_output.begin(), d_output.end(), 1.0);

    INIT_GRADIENT(apply_affine);
    Affine op{2.0, 3.0};
    Affine d_op{0.0, 0.0};
    thrust::device_vector<double> d_in_aff(host_input1.size());
    apply_affine_grad.execute(device_input1, device_out, op, &d_in_aff, &d_output, &d_op);
    thrust::host_vector<double> h_d_in_aff = d_in_aff;
    printf("Affine Grad x: %.3f %.3f %.3f %.3f\n", h_d_in_aff[0], h_d_in_aff[1], h_d_in_aff[2], h_d_in_aff[3]);
    // CHECK-EXEC: Affine Grad x: 2.000 2.000 2.000 2.000
    printf("Affine Grad a,b: %.3f %.3f\n", d_op.a, d_op.b);
    // With d_y = 1 and inputs [10,5,2,20] ⇒ d_a = sum(x) = 37, d_b = 4
    // CHECK-EXEC: Affine Grad a,b: 37.000 4.000

    // Test custom binary functor with operator_call_pullback
    thrust::fill(device_output.begin(), device_output.end(), 0.0);
    thrust::fill(d_output.begin(), d_output.end(), 1.0);
    INIT_GRADIENT(apply_bilinear);
    Bilinear bop{3.0, -2.0, 5.0};
    Bilinear d_bop{0.0, 0.0, 0.0};
    thrust::device_vector<double> d_in1(host_input1.size());
    thrust::device_vector<double> d_in2(host_input2.size());
    apply_bilinear_grad.execute(device_input1, device_input2, device_output, bop, &d_in1, &d_in2, &d_output, &d_bop);
    thrust::host_vector<double> h_d_in1 = d_in1;
    thrust::host_vector<double> h_d_in2 = d_in2;
    printf("Bilinear Grad x1: %.3f %.3f %.3f %.3f\n", h_d_in1[0], h_d_in1[1], h_d_in1[2], h_d_in1[3]);
    // CHECK-EXEC: Bilinear Grad x1: 3.000 3.000 3.000 3.000
    printf("Bilinear Grad x2: %.3f %.3f %.3f %.3f\n", h_d_in2[0], h_d_in2[1], h_d_in2[2], h_d_in2[3]);
    // CHECK-EXEC: Bilinear Grad x2: -2.000 -2.000 -2.000 -2.000
    printf("Bilinear Grad a,b,c: %.3f %.3f %.3f\n", d_bop.a, d_bop.b, d_bop.c);
    // d_a = sum(x1) = 37, d_b = sum(x2) = 10, d_c = N = 4
    // CHECK-EXEC: Bilinear Grad a,b,c: 37.000 10.000 4.000

    // More complex unary functor: y = a*x*x + b
    thrust::fill(device_output.begin(), device_output.end(), 0.0);
    thrust::fill(d_output.begin(), d_output.end(), 1.0);
    INIT_GRADIENT(apply_spb);
    SquarePlusBias sop{1.0, -3.0};
    SquarePlusBias d_sop{0.0, 0.0};
    thrust::device_vector<double> d_in_spb(host_input1.size());
    apply_spb_grad.execute(device_input1, device_output, sop, &d_in_spb, &d_output, &d_sop);
    thrust::host_vector<double> h_d_in_spb = d_in_spb;
    printf("SquarePlusBias Grad x: %.3f %.3f %.3f %.3f\n", h_d_in_spb[0], h_d_in_spb[1], h_d_in_spb[2], h_d_in_spb[3]);
    // CHECK-EXEC: SquarePlusBias Grad x: 20.000 10.000 4.000 40.000
    printf("SquarePlusBias Grad a,b: %.3f %.3f\n", d_sop.a, d_sop.b);
    // CHECK-EXEC: SquarePlusBias Grad a,b: 529.000 4.000

    // More complex binary functor: y = x1*exp(b*x2 + c) + a*sin(x1)
    thrust::fill(device_output.begin(), device_output.end(), 0.0);
    thrust::fill(d_output.begin(), d_output.end(), 1.0);
    INIT_GRADIENT(apply_mix);
    MixOp mop{0.0, 0.1, 0.0};
    MixOp d_mop{0.0, 0.0, 0.0};
    thrust::device_vector<double> d_in_mix1(host_input1.size());
    thrust::device_vector<double> d_in_mix2(host_input2.size());
    apply_mix_grad.execute(device_input1, device_input2, device_output, mop, &d_in_mix1, &d_in_mix2, &d_output, &d_mop);
    thrust::host_vector<double> h_d_in_mix1 = d_in_mix1;
    thrust::host_vector<double> h_d_in_mix2 = d_in_mix2;
    printf("MixOp Grad x1: %.6f %.6f %.6f %.6f\n", h_d_in_mix1[0], h_d_in_mix1[1], h_d_in_mix1[2], h_d_in_mix1[3]);
    // CHECK-EXEC: MixOp Grad x1: 1.105171 1.221403 1.349859 1.491825
    printf("MixOp Grad x2: %.6f %.6f %.6f %.6f\n", h_d_in_mix2[0], h_d_in_mix2[1], h_d_in_mix2[2], h_d_in_mix2[3]);
    // CHECK-EXEC: MixOp Grad x2: 1.105171 0.610701 0.269972 2.983649
    printf("MixOp Grad a,b,c: %.6f %.6f %.6f\n", d_mop.a, d_mop.b, d_mop.c);
    // CHECK-EXEC: MixOp Grad a,b,c: 0.319297 150.710865 49.694935

    return 0;
}
