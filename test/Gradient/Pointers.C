// RUN: %cladclang %s -I%S/../../include -oPointers.out 2>&1 | FileCheck %s
// RUN: ./Pointers.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double nonMemFn(double i) {
  return i*i;
}

// CHECK: void nonMemFn_grad(double i, clad::array_ref<double> _d_i) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 1 * _t0;
// CHECK-NEXT:         * _d_i += _r0;
// CHECK-NEXT:         double _r1 = _t1 * 1;
// CHECK-NEXT:         * _d_i += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

#define NON_MEM_FN_TEST(var)\
res[0]=0;\
var.execute(5,res);\
printf("%.2f\n", res[0]);

int main() {
  auto nonMemFnPtr = &nonMemFn;
  auto nonMemFnPtrToPtr = &nonMemFnPtr;
  auto nonMemFnPtrToPtrToPtr = &nonMemFnPtrToPtr;
  auto nonMemFnIndirectPtr = nonMemFnPtr;
  auto nonMemFnIndirectIndirectPtr = nonMemFnIndirectPtr;

  double res[2];

  auto d_nonMemFn = clad::gradient(nonMemFn, "i");
  auto d_nonMemFnPar = clad::gradient((nonMemFn), "i");
  auto d_nonMemFnPtr = clad::gradient(nonMemFnPtr, "i");
  auto d_nonMemFnPtrToPtr = clad::gradient(*nonMemFnPtrToPtr, "i");
  auto d_nonMemFnPtrToPtrPar = clad::gradient((*(nonMemFnPtrToPtr)), "i");
  auto d_nonMemFnPtrToPtr_1 = clad::gradient(**&nonMemFnPtrToPtr, "i");
  auto d_nonMemFnPtrToPtr_1Par = clad::gradient(**(&nonMemFnPtrToPtr), "i");
  auto d_nonMemFnPtrToPtr_1ParPar = clad::gradient(*(*(&nonMemFnPtrToPtr)), "i");
  auto d_nonMemFnPtrToPtrToPtr = clad::gradient(**nonMemFnPtrToPtrToPtr, "i");
  auto d_nonMemFnPtrToPtrToPtr_1 = clad::gradient(***&nonMemFnPtrToPtrToPtr, "i");
  auto d_nonMemFnPtrToPtrToPtr_1Par = clad::gradient(***(&nonMemFnPtrToPtrToPtr), "i");
  auto d_nonMemFnPtrToPtrToPtr_1ParPar = clad::gradient(*(**(&nonMemFnPtrToPtrToPtr)), "i");
  auto d_nonMemFnPtrToPtrToPtr_1ParParPar = clad::gradient((*(**((&nonMemFnPtrToPtrToPtr)))), "i");
  auto d_nonMemFnIndirectPtr = clad::gradient(nonMemFnIndirectPtr, "i");
  auto d_nonMemFnIndirectIndirectPtr = clad::gradient(nonMemFnIndirectIndirectPtr, "i");
  auto d_nonMemFnStaticCast = clad::gradient(static_cast<decltype(&nonMemFn)>(nonMemFn), "i");
  auto d_nonMemFnReinterpretCast = clad::gradient(reinterpret_cast<decltype(&nonMemFn)>(nonMemFn), "i");
  auto d_nonMemFnCStyleCast = clad::gradient((decltype(&nonMemFn))(nonMemFn), "i");


  NON_MEM_FN_TEST(d_nonMemFn); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPar); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtr); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrPar); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1Par); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1ParPar); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1Par); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1ParPar); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1ParParPar); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnIndirectPtr); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnIndirectIndirectPtr); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnStaticCast); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnReinterpretCast); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnCStyleCast); // CHECK-EXEC: 10.00
}
