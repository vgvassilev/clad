// Reverse mode over a least-squares cost, with a pullback for the
// hypothesis function it calls. The demo runs ten thousand steps and
// writes plot data, so this reads the code clad wrote and does not run
// it.
//
// RUN: %cladclang %S/../../demos/GradientDescent.cpp -I%S/../../include -o%t | %filecheck %s

// CHECK: void f_pullback(double theta_0, double theta_1, double x, double _d_y, double *_d_theta_0, double *_d_theta_1, double *_d_x) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_theta_0 += _d_y;
// CHECK-NEXT:         *_d_theta_1 += _d_y * x;
// CHECK-NEXT:         *_d_x += theta_1 * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }
// CHECK-NEXT: void cost_grad(double theta_0, double theta_1, double x, double y, double *_d_theta_0, double *_d_theta_1, double *_d_x, double *_d_y) {
// CHECK-NEXT:     double _d_f_x = 0.;
// CHECK-NEXT:     double f_x = f(theta_0, theta_1, x);
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_f_x += 1 * (f_x - y);
// CHECK-NEXT:         *_d_y += -1 * (f_x - y);
// CHECK-NEXT:         _d_f_x += (f_x - y) * 1;
// CHECK-NEXT:         *_d_y += -(f_x - y) * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         double _r2 = 0.;
// CHECK-NEXT:         f_pullback(theta_0, theta_1, x, _d_f_x, &_r0, &_r1, &_r2);
// CHECK-NEXT:         *_d_theta_0 += _r0;
// CHECK-NEXT:         *_d_theta_1 += _r1;
// CHECK-NEXT:         *_d_x += _r2;
// CHECK-NEXT:     }
// CHECK-NEXT: }
