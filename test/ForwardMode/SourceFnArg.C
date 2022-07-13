// RUN: %cladclang %s -I%S/../../include -oSourceFnArg.out 
// RUN: ./SourceFnArg.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

class Expression {
public:
  Expression() {}
  double memFn(double i) {
    return i*i;
  }

  // CHECK: double memFn_darg0(double i) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     Expression _d_this_obj;
  // CHECK-NEXT:     Expression *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_i * i + i * _d_i;
  // CHECK-NEXT: }

};

double nonMemFn(double i) {
  return 5*i;
}

// CHECK: double nonMemFn_darg0(double i) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     return 0 * i + 5 * _d_i;
// CHECK-NEXT: }

#define MEM_FN_TEST(var)\
printf("%.2f\n",var.execute(expr,7));

#define NON_MEM_FN_TEST(var)\
printf("%.2f\n",var.execute(3));

int main() {
  Expression expr;

  auto memFnPtr = &Expression::memFn;
  auto memFnPtrToPtr = &memFnPtr;
  auto memFnPtrToPtrToPtr = &memFnPtrToPtr;
  auto memFnIndirectPtr = memFnPtr;
  auto memFnIndirectIndirectPtr = memFnIndirectPtr;

  auto d_memFnPtr = clad::differentiate(memFnPtr, "i");
  auto d_memFnPtrPar = clad::differentiate((memFnPtr), "i");
  auto d_memFnPtrToPtr = clad::differentiate(*memFnPtrToPtr, "i");
  auto d_memFnPtrToPtr_1 = clad::differentiate(**&memFnPtrToPtr, "i");
  auto d_memFnPtrToPtr_1Par = clad::differentiate(**(&memFnPtrToPtr), "i");
  auto d_memFnPtrToPtr_1ParPar = clad::differentiate(*(*(&memFnPtrToPtr)), "i");
  auto d_memFnPtrToPtrToPtr = clad::differentiate(**memFnPtrToPtrToPtr, "i");
  auto d_memFnPtrToPtrToPtrPar = clad::differentiate(*(*memFnPtrToPtrToPtr), "i");
  auto d_memFnPtrToPtrToPtrParPar = clad::differentiate(((*(*memFnPtrToPtrToPtr))), "i");
  auto d_memFnPtrToPtrToPtr_1 = clad::differentiate(***&memFnPtrToPtrToPtr, "i");
  auto d_memFnPtrToPtrToPtr_1Par = clad::differentiate(***(&memFnPtrToPtrToPtr), "i");
  auto d_memFnPtrToPtrToPtr_1ParPar = clad::differentiate(**(*(&memFnPtrToPtrToPtr)), "i");
  auto d_memFnPtrToPtrToPtr_1ParParPar = clad::differentiate(*(*(*(&memFnPtrToPtrToPtr))), "i");
  auto d_memFnIndirectPtr = clad::differentiate(memFnIndirectPtr, "i");
  auto d_memFnIndirectIndirectPtr = clad::differentiate(memFnIndirectIndirectPtr, "i");
  auto d_memFnStaticCast = clad::differentiate(static_cast<decltype(&Expression::memFn)>(&Expression::memFn), "i");
  auto d_memFnReinterpretCast = clad::differentiate(reinterpret_cast<decltype(&Expression::memFn)>(&Expression::memFn), "i");
  auto d_memFnCStyleCast = clad::differentiate((decltype(&Expression::memFn))(&Expression::memFn), "i");

  MEM_FN_TEST(d_memFnPtr); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrPar); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrToPtr); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrToPtr_1); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrToPtr_1Par); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrToPtr_1ParPar); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrToPtrToPtr); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrToPtrToPtrPar); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrToPtrToPtrParPar); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrToPtrToPtr_1); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrToPtrToPtr_1Par); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrToPtrToPtr_1ParPar); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnPtrToPtrToPtr_1ParParPar); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnIndirectPtr); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnIndirectIndirectPtr); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnStaticCast); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnReinterpretCast); // CHECK-EXEC: 14.00

  MEM_FN_TEST(d_memFnCStyleCast); // CHECK-EXEC: 14.00

  auto nonMemFnPtr = &nonMemFn;
  auto nonMemFnPtrToPtr = &nonMemFnPtr;
  auto nonMemFnPtrToPtrToPtr = &nonMemFnPtrToPtr;
  auto nonMemFnIndirectPtr = nonMemFnPtr;
  auto nonMemFnIndirectIndirectPtr = nonMemFnIndirectPtr;

  auto d_nonMemFn = clad::differentiate(nonMemFn, "i");
  auto d_nonMemFnPar = clad::differentiate((nonMemFn), "i");
  auto d_nonMemFnPtr = clad::differentiate(nonMemFnPtr, "i");
  auto d_nonMemFnPtrToPtr = clad::differentiate(*nonMemFnPtrToPtr, "i");
  auto d_nonMemFnPtrToPtrPar = clad::differentiate((*(nonMemFnPtrToPtr)), "i");
  auto d_nonMemFnPtrToPtr_1 = clad::differentiate(**&nonMemFnPtrToPtr, "i");
  auto d_nonMemFnPtrToPtr_1Par = clad::differentiate(**(&nonMemFnPtrToPtr), "i");
  auto d_nonMemFnPtrToPtr_1ParPar = clad::differentiate(*(*(&nonMemFnPtrToPtr)), "i");
  auto d_nonMemFnPtrToPtrToPtr = clad::differentiate(**nonMemFnPtrToPtrToPtr, "i");
  auto d_nonMemFnPtrToPtrToPtr_1 = clad::differentiate(***&nonMemFnPtrToPtrToPtr, "i");
  auto d_nonMemFnPtrToPtrToPtr_1Par = clad::differentiate(***(&nonMemFnPtrToPtrToPtr), "i");
  auto d_nonMemFnPtrToPtrToPtr_1ParPar = clad::differentiate(*(**(&nonMemFnPtrToPtrToPtr)), "i");
  auto d_nonMemFnPtrToPtrToPtr_1ParParPar = clad::differentiate((*(**((&nonMemFnPtrToPtrToPtr)))), "i");
  auto d_nonMemFnIndirectPtr = clad::differentiate(nonMemFnIndirectPtr, "i");
  auto d_nonMemFnIndirectIndirectPtr = clad::differentiate(nonMemFnIndirectIndirectPtr, "i");
  auto d_nonMemFnStaticCast = clad::differentiate(static_cast<decltype(&nonMemFn)>(nonMemFn), "i");
  auto d_nonMemFnReinterpretCast = clad::differentiate(reinterpret_cast<decltype(&nonMemFn)>(nonMemFn), "i");
  auto d_nonMemFnCStyleCast = clad::differentiate((decltype(&nonMemFn))(nonMemFn), "i");

  NON_MEM_FN_TEST(d_nonMemFn); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPar); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPtr); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrPar); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1Par); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1ParPar); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1Par); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1ParPar); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1ParParPar); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnIndirectPtr); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnIndirectIndirectPtr); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnStaticCast); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnReinterpretCast); // CHECK-EXEC: 5.00

  NON_MEM_FN_TEST(d_nonMemFnCStyleCast); // CHECK-EXEC: 5.00
}