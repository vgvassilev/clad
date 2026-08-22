// RUN: %cladclang %s -I%S/../../include -oPointers.out 2>&1 | %filecheck %s
// RUN: ./Pointers.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oPointers.out
// RUN: ./Pointers.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

double nonMemFn(double i, double j) {
  return i*j;
}

// CHECK: inline clad::ValueAndPushforward<double, double> nonMemFn_pushforward(double i, double j, double _d_i, double _d_j) {
// CHECK-NEXT:     return {i * j, _d_i * j + i * _d_j};
// CHECK-NEXT: }


// CHECK: inline void nonMemFn_pushforward_pullback(double i, double j, double _d_i, double _d_j, clad::ValueAndPushforward<double, double> _d_y, double *_d_i0, double *_d_j0);

// CHECK: inline void nonMemFn_hessian(double i, double j, double *hessianMatrix) {
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d_y{0., 0.};
// CHECK-NEXT:     _d_y.pushforward = 1.;
// CHECK-NEXT:     double _d_i(0.);
// CHECK-NEXT:     double _d_j(0.);
// CHECK-NEXT:     _d_i = 1.;
// CHECK-NEXT:     nonMemFn_pushforward_pullback(i, j, _d_i, _d_j, _d_y, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
// CHECK-NEXT:     _d_i = 0.;
// CHECK-NEXT:     _d_j = 1.;
// CHECK-NEXT:     nonMemFn_pushforward_pullback(i, j, _d_i, _d_j, _d_y, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
// CHECK-NEXT:     _d_j = 0.;
// CHECK-NEXT: }

// CHECK: inline void nonMemFn_pushforward_pullback(double i, double j, double _d_i, double _d_j, clad::ValueAndPushforward<double, double> _d_y, double *_d_i0, double *_d_j0) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i0 += _d_y.value * j;
// CHECK-NEXT:         *_d_j0 += i * _d_y.value;
// CHECK-NEXT:         *_d_j0 += _d_i * _d_y.pushforward;
// CHECK-NEXT:         *_d_i0 += _d_y.pushforward * _d_j;
// CHECK-NEXT:     }
// CHECK-NEXT: }


#define NON_MEM_FN_TEST(var)\
res[0]=res[1]=res[2]=res[3]=0;\
var.execute(3, 4, res);\
printf("{%.2f %.2f %.2f %.2f}\n", res[0], res[1], res[2], res[3]);

int main() {
  auto nonMemFnPtr = &nonMemFn;
  auto nonMemFnPtrToPtr = &nonMemFnPtr;
  auto nonMemFnPtrToPtrToPtr = &nonMemFnPtrToPtr;
  auto nonMemFnIndirectPtr = nonMemFnPtr;
  auto nonMemFnIndirectIndirectPtr = nonMemFnIndirectPtr;

  double res[4];

  auto d_nonMemFn = clad::hessian(nonMemFn);
  auto d_nonMemFnPar = clad::hessian((nonMemFn));
  auto d_nonMemFnPtr = clad::hessian(nonMemFnPtr);
  auto d_nonMemFnPtrToPtr = clad::hessian(*nonMemFnPtrToPtr);
  auto d_nonMemFnPtrToPtrPar = clad::hessian((*(nonMemFnPtrToPtr)));
  auto d_nonMemFnPtrToPtr_1 = clad::hessian(**&nonMemFnPtrToPtr);
  auto d_nonMemFnPtrToPtr_1Par = clad::hessian(**(&nonMemFnPtrToPtr));
  auto d_nonMemFnPtrToPtr_1ParPar = clad::hessian(*(*(&nonMemFnPtrToPtr)));
  auto d_nonMemFnPtrToPtrToPtr = clad::hessian(**nonMemFnPtrToPtrToPtr);
  auto d_nonMemFnPtrToPtrToPtr_1 = clad::hessian(***&nonMemFnPtrToPtrToPtr);
  auto d_nonMemFnPtrToPtrToPtr_1Par = clad::hessian(***(&nonMemFnPtrToPtrToPtr));
  auto d_nonMemFnPtrToPtrToPtr_1ParPar = clad::hessian(*(**(&nonMemFnPtrToPtrToPtr)));
  auto d_nonMemFnPtrToPtrToPtr_1ParParPar = clad::hessian((*(**((&nonMemFnPtrToPtrToPtr)))));
  auto d_nonMemFnIndirectPtr = clad::hessian(nonMemFnIndirectPtr);
  auto d_nonMemFnIndirectIndirectPtr = clad::hessian(nonMemFnIndirectIndirectPtr);
  auto d_nonMemFnStaticCast = clad::hessian(static_cast<decltype(&nonMemFn)>(nonMemFn));
  auto d_nonMemFnReinterpretCast = clad::hessian(reinterpret_cast<decltype(&nonMemFn)>(nonMemFn));
  auto d_nonMemFnCStyleCast = clad::hessian((decltype(&nonMemFn))(nonMemFn));

  NON_MEM_FN_TEST(d_nonMemFn); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPar); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPtr); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrPar); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1Par); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1ParPar); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1Par); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1ParPar); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1ParParPar); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnIndirectPtr); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnIndirectIndirectPtr); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnStaticCast); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnReinterpretCast); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

  NON_MEM_FN_TEST(d_nonMemFnCStyleCast); // CHECK-EXEC: {0.00 1.00 1.00 0.00}

}
