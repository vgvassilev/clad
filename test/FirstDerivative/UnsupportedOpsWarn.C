// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | %filecheck %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

//CHECK-NOT: {{.*error|warning|note:.*}}

int binOpWarn_0(int x){
    return x << 1;  // expected-warning {{attempt to differentiate unsupported operator, ignored.}}                        set to 0}}
}

// CHECK: void binOpWarn_0_grad(int x, int *_d_x) {
// CHECK-NEXT: }


int binOpWarn_1(int x){
    return x ^ 1;   // expected-warning {{attempt to differentiate unsupported operator, ignored.}}
}

// CHECK: void binOpWarn_1_grad(int x, int *_d_x) {
// CHECK-NEXT: }

int unOpWarn_0(int x){
    return ~x;  // expected-warning {{attempt to differentiate unsupported operator, ignored.}}                        set to 0}}
}

// CHECK: void unOpWarn_0_grad(int x, int *_d_x) {
// CHECK-NEXT: }

int main(){
    clad::gradient(binOpWarn_0);
    clad::gradient(binOpWarn_1);
    clad::gradient(unOpWarn_0);
}
