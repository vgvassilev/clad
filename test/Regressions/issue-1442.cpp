// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

// Test for segmentation fault fix when asserts are used
// This addresses issue #1442

#include "clad/Differentiator/Differentiator.h"
#include <cassert>

void calcViscFluxSide(int x, bool flag) {
    assert(x >= 0);
    // expected-warning@10 {{attempted to differentiate unsupported statement, no changes applied}}
    // expected-warning@10 {{attempt to differentiate unsupported operator, ignored}}
}

void testFunction(bool c) {
    calcViscFluxSide(5, c);
}

int main() {
    // Test that this compiles without segfault - the main achievement of this fix
    auto grad = clad::gradient(testFunction);
    return 0;
}

// CHECK: void testFunction_grad(bool c, bool *_d_c) {
// CHECK-NEXT: calcViscFluxSide(5, c);
// CHECK-NEXT: }
