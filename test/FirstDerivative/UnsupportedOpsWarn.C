// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

//CHECK-NOT: {{.*error|warning|note:.*}}

int binOpWarn_0(int x){
    return x << 1;  // expected-warning {{attempt to differentiate unsupported operator,  derivative                          set to 0}}
}

// CHECK: int binOpWarn_0_darg0(int x) {
// CHECK-NEXT:    int _d_x = 1;
// CHECK-NEXT:    return 0;
// CHECK-NEXT: }


int binOpWarn_1(int x){
    return x ^ 1;   // expected-warning {{attempt to differentiate unsupported operator, ignored.}}
}

// CHECK: void binOpWarn_1_grad(int x, int *_d_x) {
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     ;
// CHECK-NEXT: }

int unOpWarn_0(int x){
    return ~x;  // expected-warning {{attempt to differentiate unsupported operator,  derivative                          set to 0}}
}

// CHECK: int unOpWarn_0_darg0(int x) {
// CHECK-NEXT:   int _d_x = 1;
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }

int main(){

    clad::differentiate(binOpWarn_0, 0);
    clad::gradient(binOpWarn_1);
    clad::differentiate(unOpWarn_0, 0);
}
