// Newton's method on the Rosenbrock function, with clad supplying both the
// gradient and the hessian. Where it ends up is the test: a wrong second
// derivative still walks downhill, just not to (1, 1) in six steps.
//
// RUN: %cladclang %S/../../demos/NewtonsMethod.cpp -I%S/../../include -o%t \
// RUN:     2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// CHECK: void rosenbrock_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: void rosenbrock_hessian(double x, double y, double *hessianMatrix) {

// CHECK-EXEC: step 1: x = -1.175281, y =  1.380674
// CHECK-EXEC: step 6: x =  1.000000, y =  1.000000, f = {{.*}}
