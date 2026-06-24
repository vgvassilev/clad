// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t -lpthread
// RUN: %t | %filecheck_exec %s
// XFAIL: valgrind

#include <iostream>
#include <thread>
#include "clad/Differentiator/Differentiator.h"

void add_square(double x, double& out) {
    out = x * x;
}

double f(double x, double y) {
    double rx = 0.0, ry = 0.0;
    std::thread t1(add_square, x, std::ref(rx));
    std::thread t2(add_square, y, std::ref(ry));
    t1.join();
    t2.join();
    return rx + ry;
}

int main() {
    auto df = clad::differentiate(f, "x");
    std::cout << "f'(3,4) = " << df.execute(3.0, 4.0) << "\n";
    // CHECK-EXEC: f'(3,4) = 6
    return 0;
}