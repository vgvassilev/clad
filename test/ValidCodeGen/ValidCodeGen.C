// RUN: %cladclang -std=c++14 -Xclang -verify -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oValidCodeGen.out 2>&1 | %filecheck %s
// RUN: ./ValidCodeGen.out | %filecheck_exec %s
// RUN: %cladclang -std=c++14 -Xclang -verify %s -I%S/../../include -oValidCodeGenWithTBR.out
// RUN: ./ValidCodeGenWithTBR.out | %filecheck_exec %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"
#include "../TestUtils.h"
#include "../PrintOverloads.h"

namespace TN {
    int coefficient = 3; // expected-warning {{The gradient utilizes a global variable 'coefficient'. Please make sure to properly reset 'coefficient' before re-running the gradient.}}
    template <typename T>
    struct Test2 {
        T operator[](T x) {
            return 4*x;
        }
    };
}

namespace clad {
namespace custom_derivatives {
namespace class_functions {
    template <typename T>
    void operator_subscript_pullback(::TN::Test2<T>* obj, T x, T d_u, ::TN::Test2<T>* d_obj, T* d_x) {
        (*d_x) += 4*d_u;
    }
}}}

double fn(double x) {
    // fwd and rvs mode test
    return x*TN::coefficient; // in this test, it's important that this nested name is copied into the generated code properly in both modes
}

double fn2(double x, double y) {
    // rvs mode test
    TN::Test2<double> t; // this type needs to be copied into the derived code properly
    auto q = t[x]; // in this test, it's important that this operator call is copied into the generated code properly and that the pullback function is called with all the needed namespace prefixes
    return q;
}

int main() {
    double dx, dy;
    INIT_DIFFERENTIATE(fn, "x");
    INIT_GRADIENT(fn);
    INIT_GRADIENT(fn2);

    TEST_GRADIENT(fn, /*numOfDerivativeArgs=*/1, 3, &dx);  // CHECK-EXEC: {3.00}
    TEST_GRADIENT(fn2, /*numOfDerivativeArgs=*/2, 3, 4, &dx, &dy);  // CHECK-EXEC: {4.00, 0.00}
    TEST_DIFFERENTIATE(fn, 3) // CHECK-EXEC: {3.00}
}

//CHECK:     double fn_darg0(double x) {
//CHECK-NEXT:         double _d_x = 1;
//CHECK-NEXT:         return _d_x * TN::coefficient + x * 0;
//CHECK-NEXT:     }

//CHECK:  int _d_coefficient = 0;
//CHECK-NEXT:  void fn_grad(double x, double *_d_x) {
//CHECK-NEXT:      _d_coefficient = 0;
//CHECK-NEXT:      {
//CHECK-NEXT:          *_d_x += 1 * TN::coefficient;
//CHECK-NEXT:          _d_coefficient += x * 1;
//CHECK-NEXT:      }
//CHECK-NEXT:  }

//CHECK:     void fn2_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:         TN::Test2<double> _d_t = {};
//CHECK-NEXT:         TN::Test2<double> t;
//CHECK-NEXT:         TN::Test2<double> _t0 = t;
//CHECK-NEXT:         double _d_q = 0.;
//CHECK-NEXT:         double q = t[x];
//CHECK-NEXT:         _d_q += 1;
//CHECK-NEXT:         {
//CHECK-NEXT:             double _r0 = 0.;
//CHECK-NEXT:             t = _t0;
//CHECK-NEXT:             clad::custom_derivatives::class_functions::operator_subscript_pullback(&t, x, _d_q, &_d_t, &_r0);
//CHECK-NEXT:             *_d_x += _r0;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
