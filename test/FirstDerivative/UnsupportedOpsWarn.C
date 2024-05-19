// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

//CHECK-NOT: {{.*error|warning|note:.*}}

float binOpWarn_0(float x){
    int n = 3;
    return n << 1;  // expected-warning {{attempt to differentiate unsupported operator,  derivative                          set to 0}}
}

// CHECK: float binOpWarn_0_darg0(float x) {
// CHECK-NEXT:    float _d_x = 1;
// CHECK-NEXT:    int n = 3;
// CHECK-NEXT:    return 0;
// CHECK-NEXT: }


float binOpWarn_1(int x, double y){
    return x ^ 1;   // expected-warning {{attempt to differentiate unsupported operator, ignored.}}
}

// CHECK: void binOpWarn_1_grad_1(int x, double y, double *_d_y) {
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     ;
// CHECK-NEXT: }

float unOpWarn_0(float x){
    int n = 3;
    return ~n;  // expected-warning {{attempt to differentiate unsupported operator,  derivative                          set to 0}}
}

// CHECK: float unOpWarn_0_darg0(float x) {
// CHECK-NEXT:   float _d_x = 1;
// CHECK-NEXT:   int n = 3;
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }

int main(){
    clad::differentiate(binOpWarn_0, 0);
    clad::gradient(binOpWarn_1);
    clad::differentiate(unOpWarn_0, 0);
}
