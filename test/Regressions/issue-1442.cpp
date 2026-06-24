// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

// Test for segmentation fault fix when asserts and PredefinedExpr are used.
// This addresses issue #1442.
//
// Clad would crash in ReferencesUpdater::updateType() and StmtClone::CloneType()
// when processing SourceLocExpr (__builtin_FILE, etc.) and PredefinedExpr
// (__func__, etc.) nodes.
//
// We use a custom function instead of __assert_fail to avoid conflicting with
// glibc's declaration (which Differentiator.h pulls in via <cassert>).

#include "clad/Differentiator/Differentiator.h"

// Custom assert handler — NOT named __assert_fail to avoid conflicts with
// the glibc declaration brought in transitively by Differentiator.h.
void test_assert_handler(const char* expr, const char* file,
                         unsigned int line,
                         const char* func) __attribute__((noreturn));

// Version 1: Using __FILE__, __LINE__, __func__ (string/int literals + PredefinedExpr)
#define TEST_ASSERT_V1(expr) \
    ((expr) ? (void)0 : test_assert_handler(#expr, __FILE__, __LINE__, __func__))

// Version 2: Using __builtin_* (SourceLocExpr nodes, as used in newer glibc)
#define TEST_ASSERT_V2(expr) \
    ((expr) ? (void)0 : test_assert_handler(#expr, __builtin_FILE(), __builtin_LINE(), __builtin_FUNCTION()))

void calcViscFluxSide(int x, bool flag) {
    TEST_ASSERT_V1(x >= 0);
    // expected-warning@-1 {{attempted to differentiate unsupported statement, no changes applied}}
}

void calcViscFluxSide2(int x, bool flag) {
    TEST_ASSERT_V2(x >= 0);
    // expected-warning@-1 {{attempted to differentiate unsupported statement, no changes applied}}
}

void testPredefinedExpr(double x) {
    const char* fname = __func__;
    // expected-warning@-1 {{attempted to differentiate unsupported statement, no changes applied}}
    const char* fname2 = __FUNCTION__;
    // expected-warning@-1 {{attempted to differentiate unsupported statement, no changes applied}}
    const char* fname3 = __PRETTY_FUNCTION__;
    // expected-warning@-1 {{attempted to differentiate unsupported statement, no changes applied}}
}

void testFunction(bool c) {
    calcViscFluxSide(5, c);
    calcViscFluxSide2(5, c);
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
// CHECK-NEXT: calcViscFluxSide2(5, c);
// CHECK-NEXT: }
