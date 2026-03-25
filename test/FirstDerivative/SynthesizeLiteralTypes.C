// RUN: %cladclang %s -I%S/../../include -oSynthesizeLiteralTypes.out 2>&1 | %filecheck %s
// RUN: ./SynthesizeLiteralTypes.out | %filecheck_exec %s
// XFAIL: valgrind

// Tests for improved type handling in ConstantFolder::synthesizeLiteral
// (issue #1073). Verifies that typedef, struct, and array-containing struct
// types are handled correctly during differentiation.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// Test 1: typedef types
// Typedefs are resolved via getCanonicalType() in synthesizeLiteral.
typedef double Real;

Real fn_typedef(Real x, Real y) {
    return x * y;
}

// CHECK: Real fn_typedef_darg0(Real x, Real y) {
// CHECK-NEXT:     Real _d_x = 1;
// CHECK-NEXT:     Real _d_y = 0;
// CHECK-NEXT:     return _d_x * y + x * _d_y;
// CHECK-NEXT: }

// Test 2: nested typedef
typedef Real MyReal;

MyReal fn_nested_typedef(MyReal x) {
    return x * x;
}

// CHECK: MyReal fn_nested_typedef_darg0(MyReal x) {
// CHECK-NEXT:     MyReal _d_x = 1;
// CHECK-NEXT:     return _d_x * x + x * _d_x;
// CHECK-NEXT: }

// Test 3: global struct object with scalar fields
struct Vec2 {
    double x, y;
};

Vec2 globalVec = {3.0, 5.0};

double fn_struct_global(double t) {
    return globalVec.x * t + globalVec.y;
}

// CHECK: double fn_struct_global_darg0(double t) {
// CHECK-NEXT:     double _d_t = 1;
// CHECK-NEXT:     double &_t0 = globalVec.x;
// CHECK-NEXT:     return 0. * t + _t0 * _d_t + 0.;
// CHECK-NEXT: }

// Test 4: struct with array field
struct ArrayWrapper {
    double data[3];
};

ArrayWrapper globalArrWrap = {{1.0, 2.0, 3.0}};

double fn_struct_array_field(double t) {
    return globalArrWrap.data[0] * t + globalArrWrap.data[2];
}

// CHECK: double fn_struct_array_field_darg0(double t) {
// CHECK-NEXT:     double _d_t = 1;
// CHECK-NEXT:     double &_t0 = globalArrWrap.data[0];
// CHECK-NEXT:     return 0. * t + _t0 * _d_t + 0.;
// CHECK-NEXT: }

// Test 5: typedef of struct type
typedef Vec2 Position;

Position globalPos = {4.0, 6.0};

double fn_typedef_struct(double t) {
    return globalPos.x * t;
}

// CHECK: double fn_typedef_struct_darg0(double t) {
// CHECK-NEXT:     double _d_t = 1;
// CHECK-NEXT:     double &_t0 = globalPos.x;
// CHECK-NEXT:     return 0. * t + _t0 * _d_t;
// CHECK-NEXT: }

int main() {
    auto d_typedef = clad::differentiate(fn_typedef, "x");
    printf("typedef: %.2f\n", d_typedef.execute(3.0, 5.0));
    // CHECK-EXEC: typedef: 5.00

    auto d_nested = clad::differentiate(fn_nested_typedef, "x");
    printf("nested_typedef: %.2f\n", d_nested.execute(4.0));
    // CHECK-EXEC: nested_typedef: 8.00

    auto d_struct = clad::differentiate(fn_struct_global, "t");
    printf("struct_global: %.2f\n", d_struct.execute(2.0));
    // CHECK-EXEC: struct_global: 3.00

    auto d_arr = clad::differentiate(fn_struct_array_field, "t");
    printf("struct_array: %.2f\n", d_arr.execute(2.0));
    // CHECK-EXEC: struct_array: 1.00

    auto d_tds = clad::differentiate(fn_typedef_struct, "t");
    printf("typedef_struct: %.2f\n", d_tds.execute(2.0));
    // CHECK-EXEC: typedef_struct: 4.00

    return 0;
}
