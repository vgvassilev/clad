// Clad differentiates a loop that runs as many times as the arguments make it
// run and stops on a value computed inside it. The demo prints its answer
// beside the one the implicit function theorem gives, so the test asserts the
// two are the same number rather than pinning a number of its own.
//
// RUN: %cladclang %S/../../demos/KeplerEquation.cpp -I%S/../../include -o%t \
// RUN:     2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// CHECK: void eccentric_anomaly_grad(double M, double e, double *_d_M, double *_d_e) {

// CHECK-EXEC: dE/dM = [[DM:[0-9.]+]]   exact [[DM]]
// CHECK-EXEC: dE/de = [[DE:[0-9.]+]]   exact [[DE]]
