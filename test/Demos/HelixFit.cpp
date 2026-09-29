// A helix fitted to noisy points by Levenberg-Marquardt, with clad supplying
// the derivatives of both residuals with respect to a struct, which comes
// back as a struct of the same type. The test asks whether the fit recovered
// the helix it was generated from: the five parameters have to land on their
// true values, which they do not if a row of the Jacobian is wrong.
//
// RUN: %cladclang %S/../../demos/HelixFit.cpp -I%S/../../include -o%t 2>&1 \
// RUN:     | %filecheck %s
// RUN: %t | %filecheck_exec %s

// CHECK: void radial_residual_grad_0(Helix h, double x, double y, double z, Helix *_d_h) {
// CHECK: void z_residual_grad_0(Helix h, double x, double y, double z, Helix *_d_h) {

// CHECK-EXEC: converged after {{[0-9]+}} steps, cost 0.0{{[0-9]+}}
// CHECK-EXEC: cx  = +0.40{{[0-9]+}}   true +0.4000
// CHECK-EXEC: cy  = -0.29{{[0-9]+}}   true -0.3000
// CHECK-EXEC: r   = +2.49{{[0-9]+}}   true +2.5000
// CHECK-EXEC: z0  = +0.10{{[0-9]+}}   true +0.1000
// CHECK-EXEC: lam = +0.79{{[0-9]+}}   true +0.8000
