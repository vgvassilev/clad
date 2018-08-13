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
// CHECK_GRADIENT: float _d_x = 1;
// CHECK_GRADIENT: float _d_y = 0;
// CHECK_GRADIENT: float _d_z = 0;
// CHECK_GRADIENT: float _d_xc = 0;
// CHECK_GRADIENT: float _d_yc = 0;
// CHECK_GRADIENT: float _d_zc = 0;
// CHECK_GRADIENT: float _d_r = 0;
// CHECK_GRADIENT: float _t0 = (x - xc);
// CHECK_GRADIENT: float _t1 = (x - xc);
// CHECK_GRADIENT: float _t2 = (y - yc);
// CHECK_GRADIENT: float _t3 = (y - yc);
// CHECK_GRADIENT: float _t4 = (z - zc);
// CHECK_GRADIENT: float _t5 = (z - zc);
// CHECK_GRADIENT: return (_d_x - _d_xc) * _t1 + _t0 * (_d_x - _d_xc) + (_d_y - _d_yc) * _t3 + _t2 * (_d_y - _d_yc) + (_d_z - _d_zc) * _t5 + _t4 * (_d_z - _d_zc) - (_d_r * r + r * _d_r);
// CHECK_GRADIENT:}

// CHECK_GRADIENT:float sphere_implicit_func_darg1(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK_GRADIENT: float _d_x = 0;
// CHECK_GRADIENT: float _d_y = 1;
// CHECK_GRADIENT: float _d_z = 0;
// CHECK_GRADIENT: float _d_xc = 0;
// CHECK_GRADIENT: float _d_yc = 0;
// CHECK_GRADIENT: float _d_zc = 0;
// CHECK_GRADIENT: float _d_r = 0;
// CHECK_GRADIENT: float _t0 = (x - xc);
// CHECK_GRADIENT: float _t1 = (x - xc);
// CHECK_GRADIENT: float _t2 = (y - yc);
// CHECK_GRADIENT: float _t3 = (y - yc);
// CHECK_GRADIENT: float _t4 = (z - zc);
// CHECK_GRADIENT: float _t5 = (z - zc);
// CHECK_GRADIENT: return (_d_x - _d_xc) * _t1 + _t0 * (_d_x - _d_xc) + (_d_y - _d_yc) * _t3 + _t2 * (_d_y - _d_yc) + (_d_z - _d_zc) * _t5 + _t4 * (_d_z - _d_zc) - (_d_r * r + r * _d_r);
// CHECK_GRADIENT:}

// CHECK_GRADIENT:float sphere_implicit_func_darg2(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK_GRADIENT: float _d_x = 0;
// CHECK_GRADIENT: float _d_y = 0;
// CHECK_GRADIENT: float _d_z = 1;
// CHECK_GRADIENT: float _d_xc = 0;
// CHECK_GRADIENT: float _d_yc = 0;
// CHECK_GRADIENT: float _d_zc = 0;
// CHECK_GRADIENT: float _d_r = 0;
// CHECK_GRADIENT: float _t0 = (x - xc);
// CHECK_GRADIENT: float _t1 = (x - xc);
// CHECK_GRADIENT: float _t2 = (y - yc);
// CHECK_GRADIENT: float _t3 = (y - yc);
// CHECK_GRADIENT: float _t4 = (z - zc);
// CHECK_GRADIENT: float _t5 = (z - zc);
// CHECK_GRADIENT: return (_d_x - _d_xc) * _t1 + _t0 * (_d_x - _d_xc) + (_d_y - _d_yc) * _t3 + _t2 * (_d_y - _d_yc) + (_d_z - _d_zc) * _t5 + _t4 * (_d_z - _d_zc) - (_d_r * r + r * _d_r);
// CHECK_GRADIENT:}

// RUN: ./Gradient.out | FileCheck -check-prefix CHECK_GRADIENT_EXEC %s
// CHECK_GRADIENT_EXEC: Result is N=(10.000000,0.000000,0.000000)

//-----------------------------------------------------------------------------/
// Demo: Rosenbrock Function
//-----------------------------------------------------------------------------/
// RUN: %cladclang %S/../../demos/RosenbrockFunction.cpp -I%S/../../include -oRosenbrockFunction.out 2>&1 | FileCheck -check-prefix CHECK_ROSENBROCK %s
// CHECK_ROSENBROCK-NOT:{{.*error|warning|note:.*}}
// CHECK_ROSENBROCK:double rosenbrock_func_darg0(double x, double y) {
// CHECK_ROSENBROCK: double _d_x = 1;
// CHECK_ROSENBROCK: double _d_y = 0;
// CHECK_ROSENBROCK: double _t0 = (x - 1);
// CHECK_ROSENBROCK: double _t1 = (x - 1);
// CHECK_ROSENBROCK: double _t2 = (y - x * x);
// CHECK_ROSENBROCK: double _t3 = 100 * _t2;
// CHECK_ROSENBROCK: double _t4 = (y - x * x);
// CHECK_ROSENBROCK: return (_d_x - 0) * _t1 + _t0 * (_d_x - 0) + (0 * _t2 + 100 * (_d_y - (_d_x * x + x * _d_x))) * _t4 + _t3 * (_d_y - (_d_x * x + x * _d_x));
// CHECK_ROSENBROCK:}
// CHECK_ROSENBROCK:double rosenbrock_func_darg1(double x, double y) {
// CHECK_ROSENBROCK: double _d_x = 0;
// CHECK_ROSENBROCK: double _d_y = 1;
// CHECK_ROSENBROCK: double _t0 = (x - 1);
// CHECK_ROSENBROCK: double _t1 = (x - 1);
// CHECK_ROSENBROCK: double _t2 = (y - x * x);
// CHECK_ROSENBROCK: double _t3 = 100 * _t2;
// CHECK_ROSENBROCK: double _t4 = (y - x * x);
// CHECK_ROSENBROCK: return (_d_x - 0) * _t1 + _t0 * (_d_x - 0) + (0 * _t2 + 100 * (_d_y - (_d_x * x + x * _d_x))) * _t4 + _t3 * (_d_y - (_d_x * x + x * _d_x));
// CHECK_ROSENBROCK:}
// RUN: ./RosenbrockFunction.out | FileCheck -check-prefix CHECK_ROSENBROCK_EXEC %s
// CHECK_ROSENBROCK_EXEC: The result is -899.000000.
