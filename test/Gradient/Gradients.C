// RUN: %cladnumdiffclang %s  -I%S/../../include -oGradients.out 2>&1 | FileCheck %s
// RUN: ./Gradients.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s  -I%S/../../include -oGradients.out
// RUN: ./Gradients.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

#include "../TestUtils.h"

__attribute__((always_inline)) double f_add1(double x, double y);

__attribute__((always_inline)) double f_add1(double x, double y) {
  return x + y;
}

//CHECK: {{[__attribute__((always_inline))]*}}void f_add1_grad(double x, double y, double *_d_x, double *_d_y){{[ __attribute__((always_inline))]*}} {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 1;
//CHECK-NEXT:           *_d_y += 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_add1_grad(double x, double y, double *_d_x, double *_d_y);

double f_add2(double x, double y) {
  return 3*x + 4*y;
}

//CHECK:   void f_add2_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 3 * 1;
//CHECK-NEXT:           *_d_y += 4 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_add2_grad(double x, double y, double *_d_x, double *_d_y);

double f_add3(double x, double y) {
  return 3*x + 4*y*4;
}

//CHECK:   void f_add3_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 3 * 1;
//CHECK-NEXT:           *_d_y += 4 * 1 * 4;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_add3_grad(double x, double y, double *_d_x, double *_d_y);

double f_sub1(double x, double y) {
  return x - y;
}

//CHECK:   void f_sub1_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 1;
//CHECK-NEXT:           *_d_y += -1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }
void f_sub1_grad(double x, double y, double *_d_x, double *_d_y);

double f_sub2(double x, double y) {
  return 3*x - 4*y;
}

//CHECK:   void f_sub2_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 3 * 1;
//CHECK-NEXT:           *_d_y += 4 * -1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_sub2_grad(double x, double y, double *_d_x, double *_d_y);

double f_mult1(double x, double y) {
  return x*y;
}

//CHECK:   void f_mult1_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 1 * y;
//CHECK-NEXT:           *_d_y += x * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_mult1_grad(double x, double y, double *_d_x, double *_d_y);

double f_mult2(double x, double y) {
   return 3*x*4*y;
}

//CHECK:   void f_mult2_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 3 * 1 * y * 4;
//CHECK-NEXT:           *_d_y += 3 * x * 4 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_mult2_grad(double x, double y, double *_d_x, double *_d_y);

double f_div1(double x, double y) {
  return x/y;
}

//CHECK:   void f_div1_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 1 / y;
//CHECK-NEXT:           double _r0 = 1 * -x / (y * y);
//CHECK-NEXT:           *_d_y += _r0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_div1_grad(double x, double y, double *_d_x, double *_d_y);

double f_div2(double x, double y) {
  return 3*x/(4*y);
}

//CHECK:   void f_div2_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       _t0 = (4 * y);
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 3 * 1 / _t0;
//CHECK-NEXT:           double _r0 = 1 * -3 * x / (_t0 * _t0);
//CHECK-NEXT:           *_d_y += 4 * _r0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_div2_grad(double x, double y, double *_d_x, double *_d_y);

double f_div3(double x, double y) {
    return (x = y) / (y * y);
}

//CHECK: void f_div3_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     double _t1;
//CHECK-NEXT:     double _t2;
//CHECK-NEXT:     _t1 = x;
//CHECK-NEXT:     _t2 = (x = y);
//CHECK-NEXT:     _t0 = (y * y);
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += 1 / _t0;
//CHECK-NEXT:         x = _t1;
//CHECK-NEXT:         double _r_d0 = *_d_x;
//CHECK-NEXT:         *_d_x -= _r_d0;
//CHECK-NEXT:         *_d_y += _r_d0;
//CHECK-NEXT:         double _r0 = 1 * -_t2 / (_t0 * _t0);
//CHECK-NEXT:         *_d_y += _r0 * y;
//CHECK-NEXT:         *_d_y += y * _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

void f_div3_grad(double x, double y, double *_d_x, double *_d_y);

double f_c(double x, double y) {
  return -x*y + (x + y)*(x/y) - x*x;
}

//CHECK:   void f_c_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += -1 * y;
//CHECK-NEXT:           *_d_y += -x * 1;
//CHECK-NEXT:           *_d_x += 1 * (x / y);
//CHECK-NEXT:           *_d_y += 1 * (x / y);
//CHECK-NEXT:           *_d_x += (x + y) * 1 / y;
//CHECK-NEXT:           double _r0 = (x + y) * 1 * -x / (y * y);
//CHECK-NEXT:           *_d_y += _r0;
//CHECK-NEXT:           *_d_x += -1 * x;
//CHECK-NEXT:           *_d_x += x * -1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_c_grad(double x, double y, double *_d_x, double *_d_y);

double f_rosenbrock(double x, double y) {
  return (x - 1) * (x - 1) + 100 * (y - x * x) * (y - x * x);
}

//CHECK:   void f_rosenbrock_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 1 * (x - 1);
//CHECK-NEXT:           *_d_x += (x - 1) * 1;
//CHECK-NEXT:           *_d_y += 100 * 1 * (y - x * x);
//CHECK-NEXT:           *_d_x += -100 * 1 * (y - x * x) * x;
//CHECK-NEXT:           *_d_x += x * -100 * 1 * (y - x * x);
//CHECK-NEXT:           *_d_y += 100 * (y - x * x) * 1;
//CHECK-NEXT:           *_d_x += -100 * (y - x * x) * 1 * x;
//CHECK-NEXT:           *_d_x += x * -100 * (y - x * x) * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_rosenbrock_grad(double x, double y, double *_d_x, double *_d_y);

double f_cond1(double x, double y) {
  return (x > y ? x : y);
}

//CHECK:   void f_cond1_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       _cond0 = x > y;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       if (_cond0)
//CHECK-NEXT:           *_d_x += 1;
//CHECK-NEXT:       else
//CHECK-NEXT:           *_d_y += 1;
//CHECK-NEXT:   }

void f_cond1_grad(double x, double y, double *_d_x, double *_d_y);

double f_cond2(double x, double y) {
  return (x > y ? x : (y > 0 ? y : -y));
}

//CHECK:   void f_cond2_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       bool _cond1;
//CHECK-NEXT:       _cond0 = x > y;
//CHECK-NEXT:       if (_cond0)
//CHECK-NEXT:           ;
//CHECK-NEXT:       else
//CHECK-NEXT:           _cond1 = y > 0;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       if (_cond0)
//CHECK-NEXT:           *_d_x += 1;
//CHECK-NEXT:       else if (_cond1)
//CHECK-NEXT:           *_d_y += 1;
//CHECK-NEXT:       else
//CHECK-NEXT:           *_d_y += -1;
//CHECK-NEXT:   }

void f_cond2_grad(double x, double y, double *_d_x, double *_d_y);

double f_cond3(double x, double c) {
  return (c > 0 ? x + c : x - c);
}

//CHECK:   void f_cond3_grad(double x, double c, double *_d_x, double *_d_c) {
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       _cond0 = c > 0;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:           *_d_x += 1;
//CHECK-NEXT:           *_d_c += 1;
//CHECK-NEXT:       } else {
//CHECK-NEXT:           *_d_x += 1;
//CHECK-NEXT:           *_d_c += -1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_cond3_grad(double x, double c, double *_d_x, double *_d_y);

double f_cond4(double x, double y) {
    int i = 0;
    double arr[] = {x, y};
    if (x > 0) {
        y = arr[i] * x;
    }
    return y;
}

//CHECK:   void f_cond4_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       int _d_i = 0;
//CHECK-NEXT:       clad::array<double> _d_arr({{2U|2UL}});
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       int i = 0;
//CHECK-NEXT:       double arr[2] = {x, y};
//CHECK-NEXT:       _cond0 = x > 0;
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:           _t0 = y;
//CHECK-NEXT:           y = arr[i] * x;
//CHECK-NEXT:       }
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       *_d_y += 1;
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:           {
//CHECK-NEXT:               y = _t0;
//CHECK-NEXT:               double _r_d0 = *_d_y;
//CHECK-NEXT:               *_d_y -= _r_d0;
//CHECK-NEXT:               _d_arr[i] += _r_d0 * x;
//CHECK-NEXT:               *_d_x += arr[i] * _r_d0;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += _d_arr[0];
//CHECK-NEXT:           *_d_y += _d_arr[1];
//CHECK-NEXT:       }
//CHECK-NEXT:   }

void f_cond4_grad(double x, double c, double *_d_x, double *_d_y);

double f_if1(double x, double y) {
  if (x > y)
    return x;
  else
    return y;
}

//CHECK:   void f_if1_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       _cond0 = x > y;
//CHECK-NEXT:       if (_cond0)
//CHECK-NEXT:           goto _label0;
//CHECK-NEXT:       else
//CHECK-NEXT:           goto _label1;
//CHECK-NEXT:       if (_cond0)
//CHECK-NEXT:         _label0:
//CHECK-NEXT:           *_d_x += 1;
//CHECK-NEXT:       else
//CHECK-NEXT:         _label1:
//CHECK-NEXT:           *_d_y += 1;
//CHECK-NEXT:   }

void f_if1_grad(double x, double y, double *_d_x, double *_d_y);

double f_if2(double x, double y) {
  if (x > y)
    return x;
  else if (y > 0)
    return y;
  else
    return -y;
}

//CHECK:   void f_if2_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       bool _cond1;
//CHECK-NEXT:       _cond0 = x > y;
//CHECK-NEXT:       if (_cond0)
//CHECK-NEXT:           goto _label0;
//CHECK-NEXT:       else {
//CHECK-NEXT:           _cond1 = y > 0;
//CHECK-NEXT:           if (_cond1)
//CHECK-NEXT:               goto _label1;
//CHECK-NEXT:           else
//CHECK-NEXT:               goto _label2;
//CHECK-NEXT:       }
//CHECK-NEXT:       if (_cond0)
//CHECK-NEXT:         _label0:
//CHECK-NEXT:           *_d_x += 1;
//CHECK-NEXT:       else if (_cond1)
//CHECK-NEXT:         _label1:
//CHECK-NEXT:           *_d_y += 1;
//CHECK-NEXT:       else
//CHECK-NEXT:         _label2:
//CHECK-NEXT:           *_d_y += -1;
//CHECK-NEXT:   }

void f_if2_grad(double x, double y, double *_d_x, double *_d_y);

struct S {
  double c1;
  double c2;
  double f(double x, double y) {
    return c1 * x + c2 * y;
  }

  //CHECK:   void f_grad(double x, double y, S *_d_this, double *_d_x, double *_d_y) {
  //CHECK-NEXT:       goto _label0;
  //CHECK-NEXT:     _label0:
  //CHECK-NEXT:       {
  //CHECK-NEXT:           (*_d_this).c1 += 1 * x;
  //CHECK-NEXT:           *_d_x += this->c1 * 1;
  //CHECK-NEXT:           (*_d_this).c2 += 1 * y;
  //CHECK-NEXT:           *_d_y += this->c2 * 1;
  //CHECK-NEXT:       }
  //CHECK-NEXT:   }

  void f_grad(double x, double y, double *_d_x, double *_d_y);
};

double sum_of_powers(double x, double y, double z, double p) {
  return std::pow(x, p) + std::pow(y, p) + std::pow(z, p);
}

namespace clad {
namespace custom_derivatives {
  void sum_of_powers_pullback(double x,
                          double y,
                          double z,
                          double p,
                          double pullback,
                          double* _d_x,
                          double* _d_y,
                          double* _d_z,
                          double* _d_p) {
    *_d_x += clad::custom_derivatives::std::pow_pushforward(x, p, 1.0, 0.0).pushforward * pullback;
    *_d_p += clad::custom_derivatives::std::pow_pushforward(x, p, 0.0, 1.0).pushforward * pullback;
    *_d_y += clad::custom_derivatives::std::pow_pushforward(y, p, 1.0, 0.0).pushforward * pullback;
    *_d_p += clad::custom_derivatives::std::pow_pushforward(y, p, 0.0, 1.0).pushforward * pullback;
    *_d_z += clad::custom_derivatives::std::pow_pushforward(z, p, 1.0, 0.0).pushforward * pullback;
    *_d_p += clad::custom_derivatives::std::pow_pushforward(z, p, 0.0, 1.0).pushforward * pullback;
  }
}
}

double f_norm(double x, double y, double z, double d) {
  return std::pow(sum_of_powers(x, y, z, d), 1/d);
}

void f_norm_grad(double x,
                 double y,
                 double z,
                 double d,
                 double* _d_x,
                 double* _d_y,
                 double* _d_z,
                 double* _d_d);
//CHECK:   void f_norm_grad(double x, double y, double z, double d, double *_d_x, double *_d_y, double *_d_z, double *_d_d) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 0;
//CHECK-NEXT:           double _r5 = 0;
//CHECK-NEXT:           clad::custom_derivatives::pow_pullback(sum_of_powers(x, y, z, d), 1 / d, 1, &_r0, &_r5);
//CHECK-NEXT:           double _r1 = 0;
//CHECK-NEXT:           double _r2 = 0;
//CHECK-NEXT:           double _r3 = 0;
//CHECK-NEXT:           double _r4 = 0;
//CHECK-NEXT:           clad::custom_derivatives::sum_of_powers_pullback(x, y, z, d, _r0, &_r1, &_r2, &_r3, &_r4);
//CHECK-NEXT:           *_d_x += _r1;
//CHECK-NEXT:           *_d_y += _r2;
//CHECK-NEXT:           *_d_z += _r3;
//CHECK-NEXT:           *_d_d += _r4;
//CHECK-NEXT:           double _r6 = _r5 * -1 / (d * d);
//CHECK-NEXT:           *_d_d += _r6;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f_sin(double x, double y) {
  return (std::sin(x) + std::sin(y))*(x + y);
}

void f_sin_grad(double x, double y, double *_d_x, double *_d_y);
//CHECK:   void f_sin_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       _t0 = (std::sin(x) + std::sin(y));
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 0;
//CHECK-NEXT:           _r0 += 1 * (x + y) * clad::custom_derivatives::sin_pushforward(x, 1.).pushforward;
//CHECK-NEXT:           *_d_x += _r0;
//CHECK-NEXT:           double _r1 = 0;
//CHECK-NEXT:           _r1 += 1 * (x + y) * clad::custom_derivatives::sin_pushforward(y, 1.).pushforward;
//CHECK-NEXT:           *_d_y += _r1;
//CHECK-NEXT:           *_d_x += _t0 * 1;
//CHECK-NEXT:           *_d_y += _t0 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

unsigned f_types(int x, float y, double z) {
  return x + y + z;
}

void f_types_grad(int x,
                  float y,
                  double z,
                  int *_d_x,
                  float *_d_y,
                  double *_d_z);
//CHECK:   void f_types_grad(int x, float y, double z, int *_d_x, float *_d_y, double *_d_z) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 1;
//CHECK-NEXT:           *_d_y += 1;
//CHECK-NEXT:           *_d_z += 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f_decls1(double x, double y) {
  double a = 3 * x;
  double b = 5 * y;
  double c = a + b;
  return 2 * c;
}

void f_decls1_grad(double x, double y, double *_d_x, double *_d_y);
//CHECK:   void f_decls1_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       double _d_a = 0;
//CHECK-NEXT:       double _d_b = 0;
//CHECK-NEXT:       double _d_c = 0;
//CHECK-NEXT:       double a = 3 * x;
//CHECK-NEXT:       double b = 5 * y;
//CHECK-NEXT:       double c = a + b;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_c += 2 * 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_a += _d_c;
//CHECK-NEXT:           _d_b += _d_c;
//CHECK-NEXT:       }
//CHECK-NEXT:       *_d_y += 5 * _d_b;
//CHECK-NEXT:       *_d_x += 3 * _d_a;
//CHECK-NEXT:   }

double f_decls2(double x, double y) {
  double a = x * x;
  double b = x * y;
  double c = y * y;
  return a + 2 * b + c;
}

void f_decls2_grad(double x, double y, double *_d_x, double *_d_y);
//CHECK:   void f_decls2_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       double _d_a = 0;
//CHECK-NEXT:       double _d_b = 0;
//CHECK-NEXT:       double _d_c = 0;
//CHECK-NEXT:       double a = x * x;
//CHECK-NEXT:       double b = x * y;
//CHECK-NEXT:       double c = y * y;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_a += 1;
//CHECK-NEXT:           _d_b += 2 * 1;
//CHECK-NEXT:           _d_c += 1;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_y += _d_c * y;
//CHECK-NEXT:           *_d_y += y * _d_c;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += _d_b * y;
//CHECK-NEXT:           *_d_y += x * _d_b;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += _d_a * x;
//CHECK-NEXT:           *_d_x += x * _d_a;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f_decls3(double x, double y) {
  double a = 3 * x;
  double c = 333 * y;
  if (x > 1)
    return 2 * a;
  else if (x < -1)
    return -2 * a;
  double b = a * a;
  return b;
}

void f_decls3_grad(double x, double y, double *_d_x, double *_d_y);
//CHECK:   void f_decls3_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       double _d_a = 0;
//CHECK-NEXT:       double _d_c = 0;
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       bool _cond1;
//CHECK-NEXT:       double _d_b = 0;
//CHECK-NEXT:       double a = 3 * x;
//CHECK-NEXT:       double c = 333 * y;
//CHECK-NEXT:       _cond0 = x > 1;
//CHECK-NEXT:       if (_cond0)
//CHECK-NEXT:           goto _label0;
//CHECK-NEXT:       else {
//CHECK-NEXT:           _cond1 = x < -1;
//CHECK-NEXT:           if (_cond1)
//CHECK-NEXT:               goto _label1;
//CHECK-NEXT:       }
//CHECK-NEXT:       double b = a * a;
//CHECK-NEXT:       goto _label2;
//CHECK-NEXT:     _label2:
//CHECK-NEXT:       _d_b += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_a += _d_b * a;
//CHECK-NEXT:           _d_a += a * _d_b;
//CHECK-NEXT:       }
//CHECK-NEXT:       if (_cond0)
//CHECK-NEXT:         _label0:
//CHECK-NEXT:           _d_a += 2 * 1;
//CHECK-NEXT:       else if (_cond1)
//CHECK-NEXT:         _label1:
//CHECK-NEXT:           _d_a += -2 * 1;
//CHECK-NEXT:       *_d_y += 333 * _d_c;
//CHECK-NEXT:       *_d_x += 3 * _d_a;
//CHECK-NEXT:   }

double f_issue138(double x, double y) {
    double _t1 = 1; // expect it not to collide with _t*
    return x*x*x*x + y*y*y*y;
}

void f_issue138_grad(double x, double y, double *_d_x, double *_d_y);
//CHECK:   void f_issue138_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:       double _d__t1 = 0;
//CHECK-NEXT:       double _t10 = 1;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 1 * x * x * x;
//CHECK-NEXT:           *_d_x += x * 1 * x * x;
//CHECK-NEXT:           *_d_x += x * x * 1 * x;
//CHECK-NEXT:           *_d_x += x * x * x * 1;
//CHECK-NEXT:           *_d_y += 1 * y * y * y;
//CHECK-NEXT:           *_d_y += y * 1 * y * y;
//CHECK-NEXT:           *_d_y += y * y * 1 * y;
//CHECK-NEXT:           *_d_y += y * y * y * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f_const(const double a, const double b) {
  return a * b;
}

void f_const_grad(const double a, const double b, double *_d_a, double *_d_b);
//CHECK:   void f_const_grad(const double a, const double b, double *_d_a, double *_d_b) {
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_a += 1 * b;
//CHECK-NEXT:           *_d_b += a * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f_const_reference(double i, double j) {
  double a = i;
  const double& ar = a;
  double res = 2*ar;
  return res;
}
void f_const_reference_grad(double i, double j, double *_d_i, double *_d_j);
//CHECK: void f_const_reference_grad(double i, double j, double *_d_i, double *_d_j) {
//CHECK-NEXT:    double _d_a = 0;
//CHECK-NEXT:    double *_d_ar = 0;
//CHECK-NEXT:    double _d_res = 0;
//CHECK-NEXT:    double a = i;
//CHECK-NEXT:    _d_ar = &_d_a;
//CHECK-NEXT:    const double &ar = a;
//CHECK-NEXT:    double res = 2 * ar;
//CHECK-NEXT:    goto _label0;
//CHECK-NEXT:  _label0:
//CHECK-NEXT:    _d_res += 1;
//CHECK-NEXT:    *_d_ar += 2 * _d_res;
//CHECK-NEXT:    *_d_i += _d_a;
//CHECK-NEXT:}
double f_const02(double i, double j) {
  const double a = i;
  double res = a;
  return res;
}
void f_const02_grad(double i, double j, double *_d_i, double *_d_j);
//CHECK:  void f_const02_grad(double i, double j, double *_d_i, double *_d_j) {
//CHECK-NEXT:       double _d_a = 0;
//CHECK-NEXT:       double _d_res = 0;
//CHECK-NEXT:       const double a = i;
//CHECK-NEXT:       double res = a;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_res += 1;
//CHECK-NEXT:       _d_a += _d_res;
//CHECK-NEXT:       *_d_i += _d_a;
//CHECK-NEXT: }

float running_sum(float* p, int n) {
  for (int i = 1; i < n; i++) {
    p[i] += p[i - 1];
  }
  return p[n - 1];
}

// CHECK: void running_sum_grad(float *p, int n, float *_d_p, int *_d_n) {
// CHECK-NEXT:     unsigned {{int|long}} _t0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<float> _t1 = {};
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (i = 1; i < n; i++) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, p[i]);
// CHECK-NEXT:         p[i] += p[i - 1];
// CHECK-NEXT:     }
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_p[n - 1] += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         i--;
// CHECK-NEXT:         {
// CHECK-NEXT:             p[i] = clad::pop(_t1);
// CHECK-NEXT:             float _r_d0 = _d_p[i];
// CHECK-NEXT:             _d_p[i - 1] += _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

double global = 7;

double fn_global_var_use(double i, double j) {
  double& ref = global;
  return ref * i;
}

// CHECK: void fn_global_var_use_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _d_ref = 0;
// CHECK-NEXT:     double &ref = global;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_ref += 1 * i;
// CHECK-NEXT:         *_d_i += ref * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn_increment_in_return(double i, double j) {
  double temp = i;
  return (++i) * temp; // (i+1)*i
}
void fn_increment_in_return_grad(double i, double j, double *_d_i, double *_d_j);

// CHECK: void fn_increment_in_return_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _d_temp = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double temp = i;
// CHECK-NEXT:     _t0 = ++i;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1 * temp;
// CHECK-NEXT:         --i;
// CHECK-NEXT:         _d_temp += _t0 * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_i += _d_temp;
// CHECK-NEXT: }

template<size_t N>
double fn_template_non_type(double x) {
  const size_t maxN = 53;
  const size_t m = maxN < N ? maxN : N;
  return x*m;
}

// CHECK: void fn_template_non_type_grad(double x, double *_d_x) {
// CHECK-NEXT:     size_t _d_maxN = 0;
// CHECK-NEXT:     bool _cond0;
// CHECK-NEXT:     size_t _d_m = 0;
// CHECK-NEXT:     const size_t maxN = 53;
// CHECK-NEXT:     _cond0 = maxN < {{15U|15UL}};
// CHECK-NEXT:     const size_t m = _cond0 ? maxN : {{15U|15UL}};
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     *_d_x += 1 * m;
// CHECK-NEXT:     if (_cond0)
// CHECK-NEXT:         _d_maxN += _d_m;
// CHECK-NEXT: }

#define TEST(F, x, y)                                                          \
  {                                                                            \
    result[0] = 0;                                                             \
    result[1] = 0;                                                             \
    clad::gradient(F);                                                         \
    F##_grad(x, y, &result[0], &result[1]);                                    \
    printf("Result is = {%.2f, %.2f}\n", result[0], result[1]);                \
  }

int main() {
  double result[2];

  TEST(f_add1, 1, 1); // CHECK-EXEC: Result is = {1.00, 1.00}
  TEST(f_add2, 1, 1); // CHECK-EXEC: Result is = {3.00, 4.00}
  TEST(f_add3, 1, 1); // CHECK-EXEC: Result is = {3.00, 16.00}
  TEST(f_sub1, 1, 1); // CHECK-EXEC: Result is = {1.00, -1.00}
  TEST(f_sub2, 1, 1); // CHECK-EXEC: Result is = {3.00, -4.00}
  TEST(f_mult1, 1, 1); // CHECK-EXEC: Result is = {1.00, 1.00}
  TEST(f_mult2, 1, 1); // CHECK-EXEC: Result is = {12.00, 12.00}
  TEST(f_div1, 1, 1); // CHECK-EXEC: Result is = {1.00, -1.00}
  TEST(f_div2, 1, 1); // CHECK-EXEC: Result is = {0.75, -0.75}
  TEST(f_div3, 1, 1); // CHECK-EXEC: Result is = {0.00, -1.00}
  TEST(f_c, 1, 1); // CHECK-EXEC: Result is = {0.00, -2.00}
  TEST(f_rosenbrock, 1, 1); // CHECK-EXEC: Result is = {0.00, 0.00}
  TEST(f_cond1, 3, 2); // CHECK-EXEC: Result is = {1.00, 0.00}
  TEST(f_cond2, 3, -1); // CHECK-EXEC: Result is = {1.00, 0.00}
  TEST(f_cond3, 3, -1); // CHECK-EXEC: Result is = {1.00, -1.00}
  TEST(f_cond4, 3, -1); // CHECK-EXEC: Result is = {6.00, 0.00}
  TEST(f_if1, 3, 2); // CHECK-EXEC: Result is = {1.00, 0.00}
  TEST(f_if2, -5, -4); // CHECK-EXEC: Result is = {0.00, -1.00}
  clad::gradient(&S::f);
  clad::gradient(f_norm);
  clad::gradient(f_sin);
  clad::gradient(f_types);
  TEST(f_decls1, 3, 3); // CHECK-EXEC: Result is = {6.00, 10.00}
  TEST(f_decls2, 2, 2); // CHECK-EXEC: Result is = {8.00, 8.00}
  TEST(f_decls3, 3, 0); // CHECK-EXEC: Result is = {6.00, 0.00}
  TEST(f_decls3, -3, 0); // CHECK-EXEC: Result is = {-6.00, 0.00}
  TEST(f_decls3, 0.5, 0); // CHECK-EXEC: Result is = {9.00, 0.00}
  TEST(f_decls3, 0, 100); // CHECK-EXEC: Result is = {0.00, 0.00}
  TEST(f_issue138, 1, 2); // CHECK-EXEC: Result is = {4.00, 32.00}
  TEST(f_const, 2, 3); // CHECK-EXEC: Result is = {3.00, 2.00}
  TEST(f_const_reference, 5, 2); // CHECK-EXEC: Result is = {2.00, 0.00}
  TEST(f_const02, 8, 5); // CHECK-EXEC: Result is = {1.00, 0.00}
  clad::gradient(running_sum);

  INIT_GRADIENT(fn_global_var_use);
  double d_i, d_j;
  TEST_GRADIENT(fn_global_var_use, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);  // CHECK-EXEC: {7.00, 0.00}

  TEST(fn_increment_in_return, 3, 2); // CHECK-EXEC: Result is = {7.00, 0.00}

  auto fn_template_non_type_dx = clad::gradient(fn_template_non_type<15>);
  double x = 5, dx = 0;
  fn_template_non_type_dx.execute(x, &dx);
  printf("Result is = %.2f\n", dx); // CHECK-EXEC: Result is = 15.00
}
