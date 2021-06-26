// RUN: %cladclang %s -I%S/../../include -oFunctors.out 2>&1 | FileCheck %s
// RUN: ./Functors.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

struct Experiment {
  double x, y;
  Experiment(double p_x, double p_y) : x(p_x), y(p_y) {}

  double operator()(double i, double j) {
    return x*i + y*j;
  }

  // CHECK: double operator_call_darg0(double i, double j) {
  // CHECK-NEXT:   double _d_i = 1;
  // CHECK-NEXT:   double _d_j = 0;
  // CHECK-NEXT:   double &_t0 = this->x;
  // CHECK-NEXT:   double &_t1 = this->y;
  // CHECK-NEXT:   return 0. * i + _t0 * _d_i + 0. * j + _t1 * _d_j;
  // CHECK-NEXT: }
};

struct ExperimentConstCallOperator {
  double x, y;
  ExperimentConstCallOperator(double p_x, double p_y) : x(p_x), y(p_y) {}

  double operator()(double i, double j) const {
    return x*i + y*j;
  }

  // CHECK: double operator_call_darg0(double i, double j) {
  // CHECK-NEXT:   double _d_i = 1;
  // CHECK-NEXT:   double _d_j = 0;
  // CHECK-NEXT:   double &_t0 = this->x;
  // CHECK-NEXT:   double &_t1 = this->y;
  // CHECK-NEXT:   return 0. * i + _t0 * _d_i + 0. * j + _t1 * _d_j;
  // CHECK-NEXT: }
};

namespace Outer {
  namespace Inner {
    struct Experiment {
      double x, y;
      Experiment(double p_x, double p_y) : x(p_x), y(p_y) {}

      double operator()(double i, double j) { return x * i + y * j; }

      // CHECK: double operator_call_darg0(double i, double j) {
      // CHECK-NEXT:   double _d_i = 1;
      // CHECK-NEXT:   double _d_j = 0;
      // CHECK-NEXT:   double &_t0 = this->x;
      // CHECK-NEXT:   double &_t1 = this->y;
      // CHECK-NEXT:   return 0. * i + _t0 * _d_i + 0. * j + _t1 * _d_j;
      // CHECK-NEXT: }
    };
  }
}

class Widget {
  mutable double x;
  double y;
public:
  Widget(double p_x, double p_y) : x(p_x), y(p_y) {}
  double operator()(double i, double j, double k) {
    return x*(i+j) + y*k;
  }
  void setX(double p_x) {
    x = p_x;
  }

  // CHECK: double operator_call_darg0(double i, double j, double k) {
  // CHECK-NEXT:  double _d_i = 1;
  // CHECK-NEXT:  double _d_j = 0;
  // CHECK-NEXT:  double _d_k = 0;
  // CHECK-NEXT:  double &_t0 = this->x;
  // CHECK-NEXT:  double _t1 = (i + j);
  // CHECK-NEXT:  double &_t2 = this->y;
  // CHECK-NEXT:  return 0. * _t1 + _t0 * (_d_i + _d_j) + 0. * k + _t2 * _d_k;
  // CHECK-NEXT:}

};

class WidgetConstCallOperator {
  double y;
  mutable double x;
public:
  WidgetConstCallOperator(double p_x, double p_y) : x(p_x), y(p_y) {}
  double operator()(double i, double j, double k) const {
    return x*(i+j) + y*k;
  }
  void setX(double p_x) const {
    x = p_x;
  }

  // CHECK: double operator_call_darg0(double i, double j, double k) {
  // CHECK-NEXT:  double _d_i = 1;
  // CHECK-NEXT:  double _d_j = 0;
  // CHECK-NEXT:  double _d_k = 0;
  // CHECK-NEXT:  double &_t0 = this->x;
  // CHECK-NEXT:  double _t1 = (i + j);
  // CHECK-NEXT:  double &_t2 = this->y;
  // CHECK-NEXT:  return 0. * _t1 + _t0 * (_d_i + _d_j) + 0. * k + _t2 * _d_k;
  // CHECK-NEXT:}

};

namespace Outer {
  namespace Inner {
    class Widget {
      mutable double x;
      double y;

    public:
      Widget(double p_x, double p_y) : x(p_x), y(p_y) {}
      double operator()(double i, double j, double k) {
        return x * (i + j) + y * k;
      }
      void setX(double p_x) { x = p_x; }

      // CHECK: double operator_call_darg0(double i, double j, double k) {
      // CHECK-NEXT:  double _d_i = 1;
      // CHECK-NEXT:  double _d_j = 0;
      // CHECK-NEXT:  double _d_k = 0;
      // CHECK-NEXT:  double &_t0 = this->x;
      // CHECK-NEXT:  double _t1 = (i + j);
      // CHECK-NEXT:  double &_t2 = this->y;
      // CHECK-NEXT:  return 0. * _t1 + _t0 * (_d_i + _d_j) + 0. * k + _t2 *
      // _d_k; CHECK-NEXT:}
    };
  }
}

#define EXECUTE(CF, ...)\
  printf("%.2f\n", CF.execute(__VA_ARGS__));

int main() {
  Experiment E(11, 33);
  Widget W(3, 7);

  const ExperimentConstCallOperator constE(11, 33);
  const WidgetConstCallOperator constW(3, 7);

  Outer::Inner::Experiment NSE(33, 35);
  Outer::Inner::Widget NSW(49, 121);

  auto lambda = [](double i, double j) {
    return i*i*j;
  };

  // CHECK: inline double operator_call_darg0(double i, double j) const {
  // CHECK-NEXT:   double _d_i = 1;
  // CHECK-NEXT:   double _d_j = 0;
  // CHECK-NEXT:   double _t0 = i * i;
  // CHECK-NEXT:   return (_d_i * i + i * _d_i) * j + _t0 * _d_j;
  // CHECK-NEXT: }

  auto d_E = clad::differentiate(&E, "i");
  auto d_ERef = clad::differentiate(E, "i");
  auto d_constE = clad::differentiate(&constE, "i");
  auto d_constERef = clad::differentiate(constE, "i");
  auto d_NSE = clad::differentiate(&NSE, "i");
  auto d_NSERef = clad::differentiate(NSE, "i");

  auto d_W = clad::differentiate(&W, "i");
  auto d_WRef = clad::differentiate(W, "i");
  auto d_constW = clad::differentiate(&constW, "i");
  auto d_constWRef = clad::differentiate(constW, "i");
  auto d_NSW = clad::differentiate(&NSW, "i");
  auto d_NSWRef = clad::differentiate(NSW, "i");

  auto d_lambda = clad::differentiate(&lambda, "i");
  auto d_lambdaRef = clad::differentiate(lambda, "i");

  EXECUTE(d_E, 1, 3);         // CHECK-EXEC: 11.00
  EXECUTE(d_ERef, 1, 3);      // CHECK-EXEC: 11.00
  EXECUTE(d_constE, 1, 3);    // CHECK-EXEC: 11.00
  EXECUTE(d_constERef, 1, 3); // CHECK-EXEC: 11.00
  EXECUTE(d_NSE, 1, 3);       // CHECK-EXEC: 33.00
  EXECUTE(d_NSERef, 1, 3);    // CHECK-EXEC: 33.00

  EXECUTE(d_W, 1, 3, 5);          // CHECK-EXEC: 3.00
  EXECUTE(d_WRef, 1, 3, 5);       // CHECK-EXEC: 3.00
  EXECUTE(d_constW, 1, 3, 5);     // CHECK-EXEC: 3.00
  EXECUTE(d_constWRef, 1, 3, 5);  // CHECK-EXEC: 3.00
  EXECUTE(d_NSW, 1, 3, 5);        // CHECK-EXEC: 49.00
  EXECUTE(d_NSWRef, 1, 3, 5);     // CHECK-EXEC: 49.00

  EXECUTE(d_lambda, 3, 5);    // CHECK-EXEC: 30.00
  EXECUTE(d_lambdaRef, 3, 5); // CHECK-EXEC: 30.00

  E.x = 121;
  NSE.x = 289;

  W.setX(9);
  constW.setX(9);
  NSW.setX(121);

  EXECUTE(d_E, 1, 3);     // CHECK-EXEC: 121.00
  EXECUTE(d_ERef, 1, 3);  // CHECK-EXEC: 121.00
  EXECUTE(d_NSE, 1, 3);   // CHECK-EXEC: 289.00
  EXECUTE(d_NSERef, 1, 3);// CHECK-EXEC: 289.00

  EXECUTE(d_W, 1, 3, 5);          // CHECK-EXEC: 9.00
  EXECUTE(d_WRef, 1, 3, 5);       // CHECK-EXEC: 9.00
  EXECUTE(d_constW, 1, 3, 5);     // CHECK-EXEC: 9.00
  EXECUTE(d_constWRef, 1, 3, 5);  // CHECK-EXEC: 9.00
  EXECUTE(d_NSW, 1, 3, 5);        // CHECK-EXEC: 121.00
  EXECUTE(d_NSWRef, 1, 3, 5);     // CHECK-EXEC: 121.00
}