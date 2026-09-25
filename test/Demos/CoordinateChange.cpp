// The jacobian of the spherical-to-cartesian change of coordinates. Its
// determinant is known to be r*r*sin(theta), so the test asserts clad's
// number and the textbook one are the same rather than pinning a number.
//
// RUN: %cladclang %S/../../demos/CoordinateChange.cpp -I%S/../../include \
// RUN:     -o%t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// CHECK: void spherical_to_cartesian_jac(double r, double theta, double phi, double p[], clad::matrix<double> *_d_vector_p) {

// CHECK-EXEC: det J = [[DET:[0-9.]+]]   exact [[DET]]
