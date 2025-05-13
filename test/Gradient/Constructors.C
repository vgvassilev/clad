// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oConstructors.out 2>&1 | %filecheck %s
// RUN: ./Constructors.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oConstructors.out
// RUN: ./Constructors.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <utility>
#include <complex>

#include "../TestUtils.h"
#include "../PrintOverloads.h"


struct argByVal {
    double x, y;
    argByVal(double val) {
        x = val;
        val *= val;
        y = val;
    }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
void constructor_pullback(double val, argByVal* d_this, double *_d_val) {
    double x, y;
    x = val;
    double _t0 = val;
    val *= val;
    y = val;
    {
        double _r_d2 = d_this->y;
        d_this->y = 0.;
        *_d_val += _r_d2;
    }
    {
        val = _t0;
        double _r_d1 = *_d_val;
        *_d_val = 0.;
        *_d_val += _r_d1 * val;
        *_d_val += val * _r_d1;
    }
    {
        double _r_d0 = d_this->x;
        d_this->x = 0.;
        *_d_val += _r_d0;
    }
}
}}}

double fn1(double x, double y) {
    argByVal g(x);
    y = x;
    return y + g.y;
}

// CHECK:  void fn1_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      argByVal g(x);
// CHECK-NEXT:      argByVal _d_g(g);
// CHECK-NEXT:      clad::zero_init(_d_g);
// CHECK-NEXT:      double _t0 = y; 
// CHECK-NEXT:      y = x;
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_y += 1;
// CHECK-NEXT:          _d_g.y += 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          y = _t0;
// CHECK-NEXT:          double _r_d0 = *_d_y;
// CHECK-NEXT:          *_d_y = 0.;
// CHECK-NEXT:          *_d_x += _r_d0;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          clad::custom_derivatives::class_functions::constructor_pullback(x, &_d_g, &_r0);
// CHECK-NEXT:          *_d_x += _r0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

int main() {
    double d_i, d_j;

    INIT_GRADIENT(fn1);
    TEST_GRADIENT(fn1, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);    // CHECK-EXEC: {7.00, 0.00}
}