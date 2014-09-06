// RUN: %cladclang %S/../../demos/BasicUsage.cpp -I%S/../../include -fsyntax-only 2>&1
// RUN: %cladclang %S/../../demos/ControlFlow.cpp -I%S/../../include -fsyntax-only 2>&1
// RUN: %cladclang %S/../../demos/DebuggingClad.cpp -I%S/../../include -fsyntax-only 2>&1
// RUN: %cladclang %S/../../demos/RosenbrockFunction.cpp -I%S/../../include -fsyntax-only 2>&1


//-----------------------------------------------------------------------------/
//  Demo: Gradient.cpp
//-----------------------------------------------------------------------------/

// RUN: %cladclang %S/../../demos/Gradient.cpp -I%S/../../include -oGradient.out 2>&1 | FileCheck -check-prefix CHECK_GRADIENT %s 
// CHECK_GRADIENT-NOT:{{.*error|warning|note:.*}}
// CHECK_GRADIENT:float sphere_implicit_func_darg0(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK_GRADIENT:    return ((1.F - (0.F)) * (x - xc) + (x - xc) * (1.F - (0.F))) + (((0.F - (0.F)) * (y - yc) + (y - yc) * (0.F - (0.F)))) + (((0.F - (0.F)) * (z - zc) + (z - zc) * (0.F - (0.F)))) - ((0.F * r + r * 0.F));
// CHECK_GRADIENT:}

// CHECK_GRADIENT:float sphere_implicit_func_darg1(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK_GRADIENT:    return ((0.F - (0.F)) * (x - xc) + (x - xc) * (0.F - (0.F))) + (((1.F - (0.F)) * (y - yc) + (y - yc) * (1.F - (0.F)))) + (((0.F - (0.F)) * (z - zc) + (z - zc) * (0.F - (0.F)))) - ((0.F * r + r * 0.F));
// CHECK_GRADIENT:}

// CHECK_GRADIENT:float sphere_implicit_func_darg2(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK_GRADIENT:    return ((0.F - (0.F)) * (x - xc) + (x - xc) * (0.F - (0.F))) + (((0.F - (0.F)) * (y - yc) + (y - yc) * (0.F - (0.F)))) + (((1.F - (0.F)) * (z - zc) + (z - zc) * (1.F - (0.F)))) - ((0.F * r + r * 0.F));
// CHECK_GRADIENT:}

// RUN: ./Gradient.out | FileCheck -check-prefix CHECK_GRADIENT_EXEC %s
// CHECK_GRADIENT_EXEC: Result is N=(10.000000,0.000000,0.000000)

//-----------------------------------------------------------------------------/
// Demo: Rosenbrock Function
//-----------------------------------------------------------------------------/
// RUN: %cladclang %S/../../demos/RosenbrockFunction.cpp -I%S/../../include -oRosenbrockFunction.out 2>&1 | FileCheck -check-prefix CHECK_ROSENBROCK %s
// CHECK_ROSENBROCK-NOT:{{.*error|warning|note:.*}}
// CHECK_ROSENBROCK:double rosenbrock_func_darg0(double x, double y) {
// CHECK_ROSENBROCK: return ((1. - (0)) * (x - 1) + (x - 1) * (1. - (0))) + (((0 * (y - x * x) + 100 * (0. - ((1. * x + x * 1.)))) * (y - x * x) + 100 * (y - x * x) * (0. - ((1. * x + x * 1.)))));
// CHECK_ROSENBROCK:}
// CHECK_ROSENBROCK:double rosenbrock_func_darg1(double x, double y) {
// CHECK_ROSENBROCK: return ((0. - (0)) * (x - 1) + (x - 1) * (0. - (0))) + (((0 * (y - x * x) + 100 * (1. - ((0. * x + x * 0.)))) * (y - x * x) + 100 * (y - x * x) * (1. - ((0. * x + x * 0.)))));
// CHECK_ROSENBROCK:}
// RUN: ./RosenbrockFunction.out | FileCheck -check-prefix CHECK_ROSENBROCK_EXEC %s
// CHECK_ROSENBROCK_EXEC: The result is -899.000000.
