// RUN: %cladclang %s -I%S/../../include -std=c++14 -oconstexprTest.out 2>&1 | FileCheck %s
// RUN: ./constexprTest.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -std=c++14 -oconstexprTest.out
// RUN: ./constexprTest.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

  double result[3] = {0};
  double jacobianou[6] = {0};
  double result1[3] = {0};
  double jacobianou1[9] = {0};

constexpr void fn_mul(double i, double j, double *res) {
   res[0] = i*i;
   res[1] = j*j;
   res[2] = i*j;
}

//CHECK: constexpr void fn_mul_jac(double i, double j, double *res, double *jacobianMatrix) {
//CHECK-NEXT:    res[0] = i * i;
//CHECK-NEXT:    res[1] = j * j;
//CHECK-NEXT:    res[2] = i * j;
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r4 = 1 * j;
//CHECK-NEXT:        jacobianMatrix[4UL] += _r4;
//CHECK-NEXT:        double _r5 = i * 1;
//CHECK-NEXT:        jacobianMatrix[5UL] += _r5;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r2 = 1 * j;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r2;
//CHECK-NEXT:        double _r3 = j * 1;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r3;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = 1 * i;
//CHECK-NEXT:        jacobianMatrix[0UL] += _r0;
//CHECK-NEXT:        double _r1 = i * 1;
//CHECK-NEXT:        jacobianMatrix[0UL] += _r1;
//CHECK-NEXT:    }
//CHECK-NEXT:}

constexpr void f_1(double x, double y, double z, double output[]) {
  output[0] = x * x * x;
  output[1] = x * y * x + y * x * x;
  output[2] = z * x * 10 - y * z;
}

//CHECK: constexpr void f_1_jac(double x, double y, double z, double output[], double *jacobianMatrix) {
//CHECK-NEXT:    output[0] = x * x * x;
//CHECK-NEXT:    output[1] = x * y * x + y * x * x;
//CHECK-NEXT:    output[2] = z * x * 10 - y * z;
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r12 = 1 * 10;
//CHECK-NEXT:        double _r13 = _r12 * x;
//CHECK-NEXT:        jacobianMatrix[8UL] += _r13;
//CHECK-NEXT:        double _r14 = z * _r12;
//CHECK-NEXT:        jacobianMatrix[6UL] += _r14;
//CHECK-NEXT:        double _r15 = -1 * z;
//CHECK-NEXT:        jacobianMatrix[7UL] += _r15;
//CHECK-NEXT:        double _r16 = y * -1;
//CHECK-NEXT:        jacobianMatrix[8UL] += _r16;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r4 = 1 * x;
//CHECK-NEXT:        double _r5 = _r4 * y;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r5;
//CHECK-NEXT:        double _r6 = x * _r4;
//CHECK-NEXT:        jacobianMatrix[4UL] += _r6;
//CHECK-NEXT:        double _r7 = x * y * 1;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r7;
//CHECK-NEXT:        double _r8 = 1 * x;
//CHECK-NEXT:        double _r9 = _r8 * x;
//CHECK-NEXT:        jacobianMatrix[4UL] += _r9;
//CHECK-NEXT:        double _r10 = y * _r8;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r10;
//CHECK-NEXT:        double _r11 = y * x * 1;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r11;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = 1 * x;
//CHECK-NEXT:        double _r1 = _r0 * x;
//CHECK-NEXT:        jacobianMatrix[0UL] += _r1;
//CHECK-NEXT:        double _r2 = x * _r0;
//CHECK-NEXT:        jacobianMatrix[0UL] += _r2;
//CHECK-NEXT:        double _r3 = x * x * 1;
//CHECK-NEXT:        jacobianMatrix[0UL] += _r3;
//CHECK-NEXT:    }
//CHECK-NEXT:}

int main() {
    
    INIT_JACOBIAN(fn_mul);
    INIT_JACOBIAN(f_1);

    TEST_JACOBIAN(fn_mul, 2, 6, 3, 1, result, jacobianou); // CHECK-EXEC: {6.00, 0.00, 0.00, 2.00, 1.00, 3.00}
    TEST_JACOBIAN(f_1, 3, 9, 4, 5, 6, result1, jacobianou1); // CHECK-EXEC: {48.00, 0.00, 0.00, 80.00, 32.00, 0.00, 60.00, -6.00, 35.00}
}