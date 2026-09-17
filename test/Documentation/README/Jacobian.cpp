// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
// docs-readme: jacobian

// docs-begin-jacobian
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

void h(double a, double b, double _clad_out_output[]) {
    _clad_out_output[0] = a * a * a;
    _clad_out_output[1] = a * a * a + b * b * b;
    _clad_out_output[2] = 2 * (a + b);
}

int main() {
    auto h_jac = clad::jacobian(h);

    // The jacobian matrix has one row per element of the output and one column
    // per independent variable. The _clad_out_ prefix marks output as an
    // output rather than an input, so the independent variables are a and b.
    clad::matrix<double> d_output(3, 2);
    double output[3] = {0};
    h_jac.execute(/*a=*/3, /*b=*/4, output, &d_output);

    // d_output[i][j] is the derivative of the i-th element of output w.r.t.
    // the j-th input.
    std::cout << d_output[0][0] << " " << d_output[0][1] << std::endl
              << d_output[1][0] << " " << d_output[1][1] << std::endl
              << d_output[2][0] << " " << d_output[2][1] << std::endl;
    // prints: 27 0
    // prints: 27 48
    // prints: 2 2
}
// docs-end-jacobian
