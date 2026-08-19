// RUN: %cladclang %s -I%S/../../include -oReverseMode.out 2>&1 | %filecheck %s
// RUN: ./ReverseMode.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double f(bool cond, double a, double b) {
    double x = cond ? (a * a) : (b * b);
    a = 0;
    b = 0;
    return x;
}

int main() {
    auto df = clad::gradient(f, "a,b");
    double da = 0.0, db = 0.0;
    
    df.execute(false, 2.0, 3.0, &da, &db);
    std::cout << "da: " << da << ", db: " << db << std::endl;
    // CHECK-EXEC: da: 0, db: 6
    
    da = 0.0; db = 0.0;
    df.execute(true, 2.0, 3.0, &da, &db);
    std::cout << "da: " << da << ", db: " << db << std::endl;
    // CHECK-EXEC: da: 4, db: 0
    
    return 0;
}
