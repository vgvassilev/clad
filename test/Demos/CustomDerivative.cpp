// A pullback supplied by hand for a function whose body is bit manipulation.
// Clad has to use it rather than differentiate the code, which is what the
// two exact derivatives show: they cannot come from the approximation, whose
// own value is several percent out.
//
// RUN: %cladclang %S/../../demos/CustomDerivative.cpp -I%S/../../include \
// RUN:     -o%t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// CHECK: void model_grad(float base, float exponent, float *_d_base, float *_d_exponent) {
// CHECK: fast_pow_pullback(base, exponent, 1, &_r0, &_r1);

// CHECK-EXEC: d/dbase     = [[DB:[0-9.]+]]   exact [[DB]]
// CHECK-EXEC: d/dexponent = [[DE:[0-9.]+]]   exact [[DE]]
