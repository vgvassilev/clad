// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | %filecheck %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

double struct_in_function(double x, double y) {
   struct A { // expected-warning {{declaration kind 'CXXRecord' is not supported}} // expected-warning {{declaration kind 'CXXRecord' is not supported}}
      double x;
   };
   return 0;
}

int main(){
    clad::gradient(struct_in_function);
    clad::differentiate(struct_in_function, "x");
}
