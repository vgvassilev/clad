
// RUN: %clang++ -fsyntax-only -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang %clad_plugin_path %s 2>&1 | FileCheck %s

// CHECK-NOT: error: cannot initialize a variable of type

#include "clad/Differentiator/Differentiator.h"

struct B { double v; };
struct D : B {};

double f(double x) {
    D d;
    d.v = x * x;
    B* b = &d;
    D* p = static_cast<D*>(b);
    return p->v;
}

int main() {
    clad::gradient(f);
}
