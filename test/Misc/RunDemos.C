// RUN: %cladclang %S/../../demos/BasicUsage.cpp -I%S/../../include -fsyntax-only 2>&1
// RUN: %cladclang %S/../../demos/ControlFlow.cpp -I%S/../../include -fsyntax-only 2>&1
// RUN: %cladclang %S/../../demos/DebuggingClad.cpp -I%S/../../include -fsyntax-only 2>&1


//-----------------------------------------------------------------------------/
//  Demo: Gradient.cpp
//-----------------------------------------------------------------------------/

// RUN: %cladclang %S/../../demos/Gradient.cpp -I%S/../../include -oGradient.out 2>&1 | FileCheck -check-prefix CHECK_GRADIENT %s 
// CHECK_GRADIENT-NOT:{{.*error|warning|note:.*}}
// CHECK_GRADIENT:float sphere_implicit_func_derived_x(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK_GRADIENT:    return ((1.F - (0.F)) * (x - xc) + (x - xc) * (1.F - (0.F))) + (((0.F - (0.F)) * (y - yc) + (y - yc) * (0.F - (0.F)))) + (((0.F - (0.F)) * (z - zc) + (z - zc) * (0.F - (0.F)))) - ((0.F * r + r * 0.F));
// CHECK_GRADIENT:}

// CHECK_GRADIENT:float sphere_implicit_func_derived_y(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK_GRADIENT:    return ((0.F - (0.F)) * (x - xc) + (x - xc) * (0.F - (0.F))) + (((1.F - (0.F)) * (y - yc) + (y - yc) * (1.F - (0.F)))) + (((0.F - (0.F)) * (z - zc) + (z - zc) * (0.F - (0.F)))) - ((0.F * r + r * 0.F));
// CHECK_GRADIENT:}

// CHECK_GRADIENT:float sphere_implicit_func_derived_z(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK_GRADIENT:    return ((0.F - (0.F)) * (x - xc) + (x - xc) * (0.F - (0.F))) + (((0.F - (0.F)) * (y - yc) + (y - yc) * (0.F - (0.F)))) + (((1.F - (0.F)) * (z - zc) + (z - zc) * (1.F - (0.F)))) - ((0.F * r + r * 0.F));
// CHECK_GRADIENT:}

// RUN: ./Gradient.out | FileCheck -check-prefix CHECK_GRADIENT_EXEC %s
// CHECK_GRADIENT_EXEC: Result is N=(10.000000,0.000000,0.000000)

