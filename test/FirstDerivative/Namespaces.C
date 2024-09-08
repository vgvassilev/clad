// RUN: %cladclang %s -I%S/../../include -oNamespaces.out 2>&1 | %filecheck %s
// RUN: ./Namespaces.out
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

namespace A {
  double f(double x) { return 1 * x; }

  double f_darg0(double x);
  //CHECK:   double f_darg0(double x) {
  //CHECK-NEXT:       double _d_x = 1;
  //CHECK-NEXT:       return 0 * x + 1 * _d_x;
  //CHECK-NEXT:   }
}

namespace B {
  double f(double x) { return 2 * x; }

  double f_darg0(double x);
  //CHECK:   double f_darg0(double x) {
  //CHECK-NEXT:       double _d_x = 1;
  //CHECK-NEXT:       return 0 * x + 2 * _d_x;
  //CHECK-NEXT:   }
}

namespace C {
  namespace D {
    double f(double x) { return 3 * x; }

    double f_darg0(double x);
    //CHECK:   double f_darg0(double x) {
    //CHECK-NEXT:       double _d_x = 1;
    //CHECK-NEXT:       return 0 * x + 3 * _d_x;
    //CHECK-NEXT:   }
  }
  inline namespace E {
    double f(double x) { return 4 * x; }

    double f_darg0(double x);
    //CHECK:   double f_darg0(double x) {
    //CHECK-NEXT:       double _d_x = 1;
    //CHECK-NEXT:       return 0 * x + 4 * _d_x;
    //CHECK-NEXT:   }
  }
}

namespace {
  double f(double x) { return 5 * x; }

  double f_darg0(double x);
  //CHECK:   double f_darg0(double x) {
  //CHECK-NEXT:       double _d_x = 1;
  //CHECK-NEXT:       return 0 * x + 5 * _d_x;
  //CHECK-NEXT:   }
}

int main() {
  clad::differentiate(A::f, 0);
  printf("Result is = %.2f\n", A::f_darg0(1)); // CHECK-EXEC: Result is = 1.00

  clad::differentiate(B::f, 0);
  printf("Result is = %.2f\n", B::f_darg0(1)); // CHECK-EXEC: Result is = 2.00

  clad::differentiate(C::D::f, 0);
  printf("Result is = %.2f\n", C::D::f_darg0(1)); // CHECK-EXEC: Result is = 3.00
  
  clad::differentiate(C::E::f, 0);
  printf("Result is = %.2f\n", C::E::f_darg0(1)); // CHECK-EXEC: Result is = 4.00

  clad::differentiate(C::f, 0);
  printf("Result is = %.2f\n", C::f_darg0(1)); // CHECK-EXEC: Result is = 4.00

  clad::differentiate(f, 0);
  printf("Result is = %.2f\n", f_darg0(1)); // CHECK-EXEC: Result is = 5.00
}


