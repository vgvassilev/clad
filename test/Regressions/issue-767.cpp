// RUN: %cladclang -I%S/../../include %s

#include "clad/Differentiator/Differentiator.h"

#include <iostream>

#define show(x) std::cout << #x << ": " << x << "\n";

double fn(double u, double v) {
    double res = 0;
    for (int i = 0; i < 2; ++i) {
        if (i & 1) {
            double &ref = u;
            res += ref;
        }
    }
    return res;
}

int main() {
    auto fn_grad = clad::gradient(fn);
    double u = 3, v = 5;
    double du = 0, dv = 0;
    fn_grad.execute(u, v, &du, &dv);
    show(du);
    show(dv);
}
