// A sphere's surface normal is the gradient of its implicit function, so
// the demo reads it off three forward-mode derivatives. A wrong partial
// shows up in the normal it prints as well as in the code clad wrote.
//
// RUN: %cladclang %S/../../demos/Gradient.cpp -I%S/../../include -o%t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// CHECK:float sphere_implicit_func_darg0(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK: float _d_x = 1;
// CHECK: float _d_y = 0;
// CHECK: float _d_z = 0;
// CHECK: float _d_xc = 0;
// CHECK: float _d_yc = 0;
// CHECK: float _d_zc = 0;
// CHECK: float _d_r = 0;
// CHECK: float _t0 = (x - xc);
// CHECK: float _t1 = (x - xc);
// CHECK: float _t2 = (y - yc);
// CHECK: float _t3 = (y - yc);
// CHECK: float _t4 = (z - zc);
// CHECK: float _t5 = (z - zc);
// CHECK: return (_d_x - _d_xc) * _t1 + _t0 * (_d_x - _d_xc) + (_d_y - _d_yc) * _t3 + _t2 * (_d_y - _d_yc) + (_d_z - _d_zc) * _t5 + _t4 * (_d_z - _d_zc) - (_d_r * r + r * _d_r);
// CHECK:}
// CHECK:float sphere_implicit_func_darg1(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK: float _d_x = 0;
// CHECK: float _d_y = 1;
// CHECK: float _d_z = 0;
// CHECK: float _d_xc = 0;
// CHECK: float _d_yc = 0;
// CHECK: float _d_zc = 0;
// CHECK: float _d_r = 0;
// CHECK: float _t0 = (x - xc);
// CHECK: float _t1 = (x - xc);
// CHECK: float _t2 = (y - yc);
// CHECK: float _t3 = (y - yc);
// CHECK: float _t4 = (z - zc);
// CHECK: float _t5 = (z - zc);
// CHECK: return (_d_x - _d_xc) * _t1 + _t0 * (_d_x - _d_xc) + (_d_y - _d_yc) * _t3 + _t2 * (_d_y - _d_yc) + (_d_z - _d_zc) * _t5 + _t4 * (_d_z - _d_zc) - (_d_r * r + r * _d_r);
// CHECK:}
// CHECK:float sphere_implicit_func_darg2(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK: float _d_x = 0;
// CHECK: float _d_y = 0;
// CHECK: float _d_z = 1;
// CHECK: float _d_xc = 0;
// CHECK: float _d_yc = 0;
// CHECK: float _d_zc = 0;
// CHECK: float _d_r = 0;
// CHECK: float _t0 = (x - xc);
// CHECK: float _t1 = (x - xc);
// CHECK: float _t2 = (y - yc);
// CHECK: float _t3 = (y - yc);
// CHECK: float _t4 = (z - zc);
// CHECK: float _t5 = (z - zc);
// CHECK: return (_d_x - _d_xc) * _t1 + _t0 * (_d_x - _d_xc) + (_d_y - _d_yc) * _t3 + _t2 * (_d_y - _d_yc) + (_d_z - _d_zc) * _t5 + _t4 * (_d_z - _d_zc) - (_d_r * r + r * _d_r);
// CHECK:}

// CHECK-EXEC: Result is N=(10.000000,0.000000,0.000000)
