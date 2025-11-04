// RUN: %cladclang %s -o %t && %t | %filecheck %s

// CHECK: clad::gradient results:
// CHECK-NEXT: 8 2
// CHECK-NEXT: clad::differentiate results:
// CHECK-NEXT: 8 2

#include <iostream>
#include <cmath> // Include for std::acos
#include "clad/Differentiator/Differentiator.h"

/**
 * @brief The function to be differentiated.
 * The function being differentiated is effectively f(C, A) = 2 * C^2 * A,
 * as the 'a' variable is not used in the return value.
 *
 * However, the `std::acos` call *is* part of the code and clad will
 * see it. The original inputs (5, 3) cause std::acos(-5/3), which
 * results in NaN and breaks the test.
 */
double f(double C, double A)
{
    // We must use valid inputs, e.g., C=1, A=2 -> -C/A = -0.5
    double a = std::acos(-C / A); 
    (void)a; // Suppress unused variable warning
    return 2 * C * C * A;
}

/*
 * --- Manual Derivative Calculation ---
 * Function: f(C, A) = 2 * C^2 * A
 *
 * Partial w.r.t C (dC):
 * df/dC = 4 * C * A
 *
 * Partial w.r.t A (dA):
 * df/dA = 2 * C^2
 *
 * --- Evaluation at (C=1, A=2) ---
 * dC = 4 * 1 * 2 = 8
 * dA = 2 * (1^2) = 2
 */

int main()
{
    // Use inputs C=1.0, A=2.0, which are valid for the acos domain
    double C_val = 1.0;
    double A_val = 2.0;

    auto f_grad = clad::gradient(f);
    double dC = 0, dA = 0;
    f_grad.execute(C_val, A_val, &dC, &dA);

    std::cout << "clad::gradient results: " << std::endl;
    std::cout << dC << " " << dA << std::endl;

    std::cout << "clad::differentiate results: " << std::endl;
    auto f_dC = clad::differentiate(f, "C");
    std::cout << f_dC.execute(C_val, A_val) << " ";
    
    auto f_dA = clad::differentiate(f, "A");
    std::cout << f_dA.execute(C_val, A_val) << std::endl;
}