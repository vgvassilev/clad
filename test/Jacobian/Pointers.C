// RUN: %cladclang %s -I%S/../../include -oPointers.out 
// RUN: ./Pointers.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

void nonMemFn(double i, double j, double* out) {
  out[0] = i;
  out[1] = j;
}

// CHECK: void nonMemFn_jac(double i, double j, double *out, double *jacobianMatrix) {
// CHECK-NEXT:     out[0] = i;
// CHECK-NEXT:     out[1] = j;
// CHECK-NEXT:     jacobianMatrix[3UL] += 1;
// CHECK-NEXT:     jacobianMatrix[0UL] += 1;
// CHECK-NEXT: }

#define NON_MEM_FN_TEST(var)\
res[0]=res[1]=res[2]=res[3]=0;\
var.execute(5, 7, out, res);\
printf("{%.2f %.2f %.2f %.2f}\n", res[0], res[1], res[2], res[3]);

int main() {
  auto nonMemFnPtr = &nonMemFn;
  auto nonMemFnPtrToPtr = &nonMemFnPtr;

  double res[4];
  double out[2];
  auto d_nonMemFn = clad::jacobian(nonMemFn);
  auto d_nonMemFnPar = clad::jacobian((nonMemFn));
  auto d_nonMemFnPtr = clad::jacobian(nonMemFnPtr);
  auto d_nonMemFnPtrToPtr = clad::jacobian(*nonMemFnPtrToPtr);
  auto d_nonMemFnPtrToPtrPar = clad::jacobian((*(nonMemFnPtrToPtr)));

  NON_MEM_FN_TEST(d_nonMemFn); // CHECK-EXEC: {1.00 0.00 0.00 1.00}

  NON_MEM_FN_TEST(d_nonMemFnPar); // CHECK-EXEC: {1.00 0.00 0.00 1.00}

  NON_MEM_FN_TEST(d_nonMemFnPtr); // CHECK-EXEC: {1.00 0.00 0.00 1.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr); // CHECK-EXEC: {1.00 0.00 0.00 1.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrPar); // CHECK-EXEC: {1.00 0.00 0.00 1.00}
}