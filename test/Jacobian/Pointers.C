// RUN: %cladclang %s -I%S/../../include -oPointers.out 2>&1 | %filecheck %s
// RUN: ./Pointers.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oPointers.out
// RUN: ./Pointers.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

void nonMemFn(double i, double j, double* out) {
  out[0] = i;
  out[1] = j;
}

// CHECK: void nonMemFn_jac(double i, double j, double *out, double *jacobianMatrix) {
// CHECK-NEXT:     double _t0 = out[0];
// CHECK-NEXT:     out[0] = i;
// CHECK-NEXT:     double _t1 = out[1];
// CHECK-NEXT:     out[1] = j;
// CHECK-NEXT:     {
// CHECK-NEXT:         jacobianMatrix[{{3U|3UL|3ULL}}] += 1;
// CHECK-NEXT:         out[1] = _t1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         jacobianMatrix[{{0U|0UL|0ULL}}] += 1;
// CHECK-NEXT:         out[0] = _t0;
// CHECK-NEXT:     }
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

