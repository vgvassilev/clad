// RUN: %cladclang %s -I%S/../../include -oPointerAllocation.out
// RUN: ./PointerAllocation.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double allocThenAssign(double x) {
  double* p = nullptr;
  p = new double[2];
  double* owner = p;
  p[0] = x * x;
  double r = p[0];
  delete[] owner;
  return r;
}

// CHECK: void allocThenAssign_grad(double x, double *_d_x) {
// CHECK-NEXT:  double *_d_p = nullptr;
// CHECK-NEXT:  double *p = nullptr;
// CHECK-NEXT:  double *_t0 = _d_p;
// CHECK-NEXT:  _d_p = new double [2](/*implicit*/(double[2])0);
// CHECK-NEXT:  p = new double [2];
// CHECK-NEXT:  double *_d_owner = _d_p;
// CHECK-NEXT:  double *owner = p;
// CHECK-NEXT:  p[0] = x * x;
// CHECK-NEXT:  double _d_r = 0.;
// CHECK-NEXT:  double r = p[0];
// CHECK-NEXT:  _d_r += 1;
// CHECK-NEXT:  _d_p[0] += _d_r;
// CHECK-NEXT:  {
// CHECK-NEXT:    double _r_d0 = _d_p[0];
// CHECK-NEXT:    _d_p[0] = 0.;
// CHECK-NEXT:    *_d_x += _r_d0 * x;
// CHECK-NEXT:    *_d_x += x * _r_d0;
// CHECK-NEXT:  }
// CHECK-NEXT:  _d_p = _t0;
// CHECK-NEXT:  delete [] owner;
// CHECK-NEXT:  delete [] _d_owner;
// CHECK-NEXT:  }

struct Vec {
  double* data_ = nullptr;
  double* tmp = nullptr;
  void resize() {
    delete[] data_;
    data_ = tmp;
  }
  void resizeViaAlias() {
    double* old = data_;
    data_ = tmp;
    delete[] old;
  }
  double& at(int i) { return data_[i]; }
};

double reallocMember(Vec& v, double x) {
  v.at(0) = x * x;
  v.resize();
  v.at(1) = 3 * x;
  return v.at(1);
}

double reallocAlias(Vec& v, double x) {
  v.at(0) = x * x;
  v.resizeViaAlias();
  v.at(1) = 3 * x;
  return v.at(1);
}

int main() {
  double dx = 0;
  auto grad=clad::gradient(allocThenAssign);
  grad.execute(3, &dx);
  printf("{%.2f}\n", dx); // CHECK-EXEC: {6.00}

  double* buf[4] = {new double[2](), new double[3](), new double[2](),
                    new double[3]()};
  Vec v, dv;
  v.data_ = buf[0]; v.tmp = buf[1];
  dv.data_ = buf[2]; dv.tmp = buf[3];
  double dvx = 0;
  auto reallocGrad = clad::gradient(reallocMember, "v, x");
  reallocGrad.execute(v, 2, &dv, &dvx);
  printf("{%.2f}\n", dvx); // CHECK-EXEC: {3.00}

  v.data_ = buf[0]; v.tmp = buf[1];
  dv.data_ = buf[2]; dv.tmp = buf[3];
  dvx = 0;
  auto aliasGrad = clad::gradient(reallocAlias, "v, x");
  aliasGrad.execute(v, 2, &dv, &dvx);
  printf("{%.2f}\n", dvx); // CHECK-EXEC: {3.00}
  for (double* b : buf)
    delete[] b;
}
