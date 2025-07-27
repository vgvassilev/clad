// RUN: %cladclang %s -I%S/../../include -oAssertFix.out 2>&1 | %filecheck %s
// RUN: ./AssertFix.out | %filecheck_exec %s

// Test for segmentation fault fix when asserts are used
// This addresses issue #1442

#include "clad/Differentiator/Differentiator.h"
#include <cassert>

void calcViscFluxSide(int x, bool flag) { 
    assert(x >= 0); // This should not cause a segfault during differentiation
}

void testFunction(bool c) { 
    calcViscFluxSide(5, c); 
}

int main() {
    // This should not segfault
    auto grad = clad::gradient(testFunction);
    
    bool dc = 0;
    grad.execute(true, &dc);
    
    printf("Test passed - no segfault with assert statements\n"); // CHECK-EXEC: Test passed - no segfault with assert statements
    
    return 0;
}

// CHECK: attempted to differentiate unsupported statement, no changes applied
// CHECK: assert