// A model that estimates nothing and reports every place clad would have
// accounted for error. Both what it dumps and what it reports are the
// program's own output.
//
// RUN: %cladclang %S/../../demos/ErrorEstimation/PrintModel/test.cpp -I%S/../../include -o%t
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
// CHECK-EXEC: Error in z : {{.+}}
// CHECK-EXEC-NEXT: Error in x : {{.+}}
// CHECK-EXEC-NEXT: Error in y : {{.+}}
