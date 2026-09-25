// An overload of clad::getErrorVal replaces clad's own error estimate.
// The demo dumps the code clad wrote when it runs, so the check is
// against the program's output rather than the compiler's.
//
// RUN: %cladclang %S/../../demos/ErrorEstimation/CustomModel/test.cpp -I%S/../../include -o%t
// RUN: %t | %filecheck_exec %s

// CHECK-EXEC: The code is:
// CHECK-EXEC-NEXT: void func_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
// CHECK-EXEC-NEXT:    float _d_z = 0.F;
// CHECK-EXEC-NEXT:    float z;
// CHECK-EXEC-NEXT:    float _t0 = z;
// CHECK-EXEC-NEXT:    z = x + y;
// CHECK-EXEC-NEXT:    _d_z += 1;
// CHECK-EXEC-NEXT:    {
// CHECK-EXEC-NEXT:        _final_error += clad::getErrorVal(_d_z, z, "z");
// CHECK-EXEC-NEXT:        z = _t0;
// CHECK-EXEC-NEXT:        *_d_x += _d_z;
// CHECK-EXEC-NEXT:        *_d_y += _d_z;
// CHECK-EXEC-NEXT:        _d_z = 0.F;
// CHECK-EXEC-NEXT:    }
// CHECK-EXEC-NEXT:    _final_error += clad::getErrorVal(*_d_x, x, "x");
// CHECK-EXEC-NEXT:    _final_error += clad::getErrorVal(*_d_y, y, "y");
// CHECK-EXEC-NEXT: }
