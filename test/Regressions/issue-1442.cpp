// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

// Test for segmentation fault fix when asserts and PredefinedExpr are used
// This addresses issue #1442

#include "clad/Differentiator/Differentiator.h"
#include <cassert>

void calcViscFluxSide(int x, bool flag) {
    assert(x >= 0);
    // expected-warning@10 {{attempted to differentiate unsupported statement, no changes applied}}
}

void testPredefinedExpr(double x) {
    const char* fname = __func__;
    // expected-warning@15 {{attempted to differentiate unsupported statement, no changes applied}}
    const char* fname2 = __FUNCTION__;
    // expected-warning@17 {{attempted to differentiate unsupported statement, no changes applied}}
    const char* fname3 = __PRETTY_FUNCTION__;
    // expected-warning@19 {{attempted to differentiate unsupported statement, no changes applied}}
}

void testFunction(bool c) {
    calcViscFluxSide(5, c);
}

void testFunctionWithPredefined(double x) {
    testPredefinedExpr(x);
}

int main() {
    auto grad = clad::gradient(testFunction);
    auto grad2 = clad::gradient(testFunctionWithPredefined);
    return 0;
}

// CHECK: void testFunction_grad(bool c, bool *_d_c) {
// CHECK-NEXT: calcViscFluxSide(5, c);
// CHECK-NEXT: }
