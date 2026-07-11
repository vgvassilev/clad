// RUN: %cladclang -std=c++17 -O0 -I%S/../../include/ %s -o %t
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"
#include <vector>
#include <memory>
#include <iostream>

double f(double x) {
    std::vector<double> v(5); 
    return x * x; 
}

double f1(double x) {
    std::allocator<double> alloc;
    std::vector<double> v(5, alloc);
    return x * x * x;
}

int main() {

    auto df = clad::gradient(f);
    auto df1 = clad::gradient(f1);

    double dx1 = 0.0;
    df.execute(3.0, &dx1); 
    std::cout << "Diff result: " << dx1 << "\n";
    //CHECK-EXEC: Diff result: 6
    
    double dx2 = 0.0;
    df1.execute(2.0, &dx2); 
    std::cout << "Diff result: " << dx2 << "\n";
    //CHECK-EXEC: Diff result: 12

    return 0;
}