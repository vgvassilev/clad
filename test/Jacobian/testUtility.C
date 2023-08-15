// RUN: %cladclang %s -I%S/../../include -otestUtility.out
// RUN: ./testUtility.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

  double output[3] = {0};
  double jacobian[6] = {0};
  double output1[3] = {0};
  double jacobian1[9] = {0};

void fn_mul(double i, double j, double *res) {
   res[0] = i*i;
   res[1] = j*j;
   res[2] = i*j;
}

//CHECK: void fn_mul_jac(double i, double j, double *res, double *jacobianMatrix) {
//CHECK-NEXT:    double _t0;
//CHECK-NEXT:    double _t1;
//CHECK-NEXT:    double _t2;
//CHECK-NEXT:    double _t3;
//CHECK-NEXT:    double _t4;
//CHECK-NEXT:    double _t5;
//CHECK-NEXT:    _t1 = i;
//CHECK-NEXT:    _t0 = i;
//CHECK-NEXT:    res[0] = i * i;
//CHECK-NEXT:    _t3 = j;
//CHECK-NEXT:    _t2 = j;
//CHECK-NEXT:    res[1] = j * j;
//CHECK-NEXT:    _t5 = i;
//CHECK-NEXT:    _t4 = j;
//CHECK-NEXT:    res[2] = i * j;
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r4 = 1 * _t4;
//CHECK-NEXT:        jacobianMatrix[4UL] += _r4;
//CHECK-NEXT:        double _r5 = _t5 * 1;
//CHECK-NEXT:        jacobianMatrix[5UL] += _r5;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r2 = 1 * _t2;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r2;
//CHECK-NEXT:        double _r3 = _t3 * 1;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r3;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = 1 * _t0;
//CHECK-NEXT:        jacobianMatrix[0UL] += _r0;
//CHECK-NEXT:        double _r1 = _t1 * 1;
//CHECK-NEXT:        jacobianMatrix[0UL] += _r1;
//CHECK-NEXT:    }
//CHECK-NEXT:}


void f_1(double x, double y, double z, double output[]) {
  output[0] = x * x * x;
  output[1] = x * y * x + y * x * x;
  output[2] = z * x * 10 - y * z;
}

//CHECK: void f_1_jac(double x, double y, double z, double output[], double *jacobianMatrix) {
//CHECK-NEXT:    double _t0;
//CHECK-NEXT:    double _t1;
//CHECK-NEXT:    double _t2;
//CHECK-NEXT:    double _t3;
//CHECK-NEXT:    double _t4;
//CHECK-NEXT:    double _t5;
//CHECK-NEXT:    double _t6;
//CHECK-NEXT:    double _t7;
//CHECK-NEXT:    double _t8;
//CHECK-NEXT:    double _t9;
//CHECK-NEXT:    double _t10;
//CHECK-NEXT:    double _t11;
//CHECK-NEXT:    double _t12;
//CHECK-NEXT:    double _t13;
//CHECK-NEXT:    double _t14;
//CHECK-NEXT:    double _t15;
//CHECK-NEXT:    _t2 = x;
//CHECK-NEXT:    _t1 = x;
//CHECK-NEXT:    _t3 = _t2 * _t1;
//CHECK-NEXT:    _t0 = x;
//CHECK-NEXT:    output[0] = x * x * x;
//CHECK-NEXT:    _t6 = x;
//CHECK-NEXT:    _t5 = y;
//CHECK-NEXT:    _t7 = _t6 * _t5;
//CHECK-NEXT:    _t4 = x;
//CHECK-NEXT:    _t10 = y;
//CHECK-NEXT:    _t9 = x;
//CHECK-NEXT:    _t11 = _t10 * _t9;
//CHECK-NEXT:    _t8 = x;
//CHECK-NEXT:    output[1] = x * y * x + y * x * x;
//CHECK-NEXT:    _t13 = z;
//CHECK-NEXT:    _t12 = x;
//CHECK-NEXT:    _t15 = y;
//CHECK-NEXT:    _t14 = z;
//CHECK-NEXT:    output[2] = z * x * 10 - y * z;
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r12 = 1 * 10;
//CHECK-NEXT:        double _r13 = _r12 * _t12;
//CHECK-NEXT:        jacobianMatrix[8UL] += _r13;
//CHECK-NEXT:        double _r14 = _t13 * _r12;
//CHECK-NEXT:        jacobianMatrix[6UL] += _r14;
//CHECK-NEXT:        double _r15 = -1 * _t14;
//CHECK-NEXT:        jacobianMatrix[7UL] += _r15;
//CHECK-NEXT:        double _r16 = _t15 * -1;
//CHECK-NEXT:        jacobianMatrix[8UL] += _r16;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r4 = 1 * _t4;
//CHECK-NEXT:        double _r5 = _r4 * _t5;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r5;
//CHECK-NEXT:        double _r6 = _t6 * _r4;
//CHECK-NEXT:        jacobianMatrix[4UL] += _r6;
//CHECK-NEXT:        double _r7 = _t7 * 1;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r7;
//CHECK-NEXT:        double _r8 = 1 * _t8;
//CHECK-NEXT:        double _r9 = _r8 * _t9;
//CHECK-NEXT:        jacobianMatrix[4UL] += _r9;
//CHECK-NEXT:        double _r10 = _t10 * _r8;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r10;
//CHECK-NEXT:        double _r11 = _t11 * 1;
//CHECK-NEXT:        jacobianMatrix[3UL] += _r11;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = 1 * _t0;
//CHECK-NEXT:        double _r1 = _r0 * _t1;
//CHECK-NEXT:        jacobianMatrix[0UL] += _r1;
//CHECK-NEXT:        double _r2 = _t2 * _r0;
//CHECK-NEXT:        jacobianMatrix[0UL] += _r2;
//CHECK-NEXT:        double _r3 = _t3 * 1;
//CHECK-NEXT:        jacobianMatrix[0UL] += _r3;
//CHECK-NEXT:    }
//CHECK-NEXT:}

int main(){
    INIT_JACOBIAN(fn_mul);
    INIT_JACOBIAN(f_1);

    TEST_JACOBIAN(fn_mul, 2, 6, 3, 1, output, jacobian); // CHECK-EXEC: {6.00, 0.00, 0.00, 2.00, 1.00, 3.00}
    TEST_JACOBIAN(f_1, 3, 9, 4, 5, 6, output1, jacobian1); // CHECK-EXEC: {48.00, 0.00, 0.00, 80.00, 32.00, 0.00, 60.00, -6.00, 35.00}

}
