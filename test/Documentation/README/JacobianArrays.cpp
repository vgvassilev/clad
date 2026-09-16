// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
// docs-readme: jacobian-arrays

// docs-begin-jacobian-arrays
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

void h(double a, double b, double _clad_out_arr[], double* _clad_out_ptr) {
    _clad_out_arr[0] = a * a * a;
    _clad_out_ptr[0] = _clad_out_arr[0] + b * b * b;
    _clad_out_arr[1] = 2 * (a + b);
}

int main() {
    auto h_jac = clad::jacobian(h);

    // One matrix per output parameter, each with a row per element of that
    // parameter and a column per independent variable, here a and b.
    clad::matrix<double> d_arr(2, 2);
    double arr[2] = {0};

    clad::matrix<double> d_ptr(1, 2);
    double ptr[1] = {0};

    h_jac.execute(/*a=*/3, /*b=*/4, arr, ptr, &d_arr, &d_ptr);

    std::cout << d_arr[0][0] << " " << d_arr[0][1] << std::endl
              << d_arr[1][0] << " " << d_arr[1][1] << std::endl;
    // prints: 27 0
    // prints: 2 2

    std::cout << d_ptr[0][0] << " " << d_ptr[0][1] << std::endl;
    // prints: 27 48
}
// docs-end-jacobian-arrays
