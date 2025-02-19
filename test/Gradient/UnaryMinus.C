// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oUnaryMinus.out 2>&1 | %filecheck %s
// RUN: ./UnaryMinus.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oUnaryMinus.out
// RUN: ./UnaryMinus.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

double f1(double x)
{
    return -(-(-1))*-(-(-x));
}

//CHECK: void f1_grad(double x, double *_d_x) {
//CHECK-NEXT:    *_d_x += -(-1 * 1);
//CHECK-NEXT: }

double f2(double x, double y)
{
    return -2*-(-(-x))*-y - 1*(-y)*(-(-x));
}

//CHECK: void f2_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:    {
//CHECK-NEXT:        *_d_x += -(-2 * 1 * -y);
//CHECK-NEXT:        *_d_y += -(-2 * -x * 1);
//CHECK-NEXT:        *_d_y += -1 * -1 * x;
//CHECK-NEXT:        *_d_x += 1 * -y * -1;
//CHECK-NEXT:    }
//CHECK-NEXT: }

double dx;
double arr[2] = {};
int main(){

    INIT_GRADIENT(f1);
    INIT_GRADIENT(f2);

    TEST_GRADIENT(f1, 1, 5, &dx); // CHECK-EXEC: 1.00
    TEST_GRADIENT(f2, 2, 3, 4, &arr[0], &arr[1]) // CHECK-EXEC: {-4.00, -3.00}
}