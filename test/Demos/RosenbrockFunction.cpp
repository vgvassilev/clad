// The Rosenbrock function differentiated in forward mode, one partial per
// call, summed into the value the demo prints.
//
// RUN: %cladclang %S/../../demos/RosenbrockFunction.cpp -I%S/../../include -o%t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// CHECK:double rosenbrock_func_darg0(double x, double y) {
// CHECK: double _d_x = 1;
// CHECK: double _d_y = 0;
// CHECK: double _t0 = (x - 1);
// CHECK: double _t1 = (x - 1);
// CHECK: double _t2 = 100 * (y - x * x);
// CHECK: double _t3 = (y - x * x);
// CHECK: return _d_x * _t1 + _t0 * _d_x + (100 * (_d_y - (_d_x * x + x * _d_x))) * _t3 + _t2 * (_d_y - (_d_x * x + x * _d_x));
// CHECK:}
// CHECK:double rosenbrock_func_darg1(double x, double y) {
// CHECK: double _d_x = 0;
// CHECK: double _d_y = 1;
// CHECK: double _t0 = (x - 1);
// CHECK: double _t1 = (x - 1);
// CHECK: double _t2 = 100 * (y - x * x);
// CHECK: double _t3 = (y - x * x);
// CHECK: return _d_x * _t1 + _t0 * _d_x + (100 * (_d_y - (_d_x * x + x * _d_x))) * _t3 + _t2 * (_d_y - (_d_x * x + x * _d_x));
// CHECK:}

// CHECK-EXEC: The result is -899.000000.
