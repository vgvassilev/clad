// RUN: %cladclang %s -I%S/../../include -oUserDefinedTypes.out 2>&1 | FileCheck %s
// RUN: ./UserDefinedTypes.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <utility>
#include <complex>

#include "../TestUtils.h"
#include "../PrintOverloads.h"

using pairdd = std::pair<double, double>;

double fn1(pairdd p, double i) {
    double res = p.first + 2*p.second + 3*i;
    return res;
}

// CHECK: void fn1_grad(pairdd p, double i, clad::array_ref<pairdd> _d_p, clad::array_ref<double> _d_i) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     _t0 = p.second;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     double res = p.first + 2 * _t0 + 3 * _t1;
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         (* _d_p).first += _d_res;
// CHECK-NEXT:         double _r0 = _d_res * _t0;
// CHECK-NEXT:         double _r1 = 2 * _d_res;
// CHECK-NEXT:         (* _d_p).second += _r1;
// CHECK-NEXT:         double _r2 = _d_res * _t1;
// CHECK-NEXT:         double _r3 = 3 * _d_res;
// CHECK-NEXT:         * _d_i += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

struct Tangent {
  Tangent() {}
  double data[5] = {};
  void updateTo(double d) {
    for (int i = 0; i < 5; ++i)
      data[i] = d;
  }

  double someMemFn(double i, double j) {
    return data[0] * i + data[1] * j + 3 * data[2] + data[3] * data[4];
  }
  double someMemFn2(double i, double j) const {
      return data[0]*i + data[1]*i*j;
  }
};

double sum(Tangent& t) {
    double res=0;
    for (int i=0; i<5; ++i)
        res += t.data[i];
    return res;
}

// CHECK: void sum_pullback(Tangent &t, double _d_y, clad::array_ref<Tangent> _d_t) {
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     clad::tape<int> _t1 = {};
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int i = 0; i < 5; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         res += t.data[clad::push(_t1, i)];
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += _d_y;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         _d_res += _r_d0;
// CHECK-NEXT:         int _t2 = clad::pop(_t1);
// CHECK-NEXT:         (* _d_t).data[_t2] += _r_d0;
// CHECK-NEXT:         _d_res -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double sum(double *data) {
    double res = 0;
    for (int i=0; i<5; ++i)
        res += data[i];
    return res;
}

// CHECK: void sum_pullback(double *data, double _d_y, clad::array_ref<double> _d_data) {
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     clad::tape<int> _t1 = {};
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int i = 0; i < 5; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         res += data[clad::push(_t1, i)];
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += _d_y;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         _d_res += _r_d0;
// CHECK-NEXT:         int _t2 = clad::pop(_t1);
// CHECK-NEXT:         _d_data[_t2] += _r_d0;
// CHECK-NEXT:         _d_res -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn2(Tangent t, double i) {
    double res = sum(t);
    res += sum(t.data) + i + 2*t.data[0];
    return res;
}

// CHECK: void fn2_grad(Tangent t, double i, clad::array_ref<Tangent> _d_t, clad::array_ref<double> _d_i) {
// CHECK-NEXT:     Tangent _t0;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     clad::array<double> _t1(5UL);
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     _t0 = t;
// CHECK-NEXT:     double res = sum(t);
// CHECK-NEXT:     _t1 = t.data;
// CHECK-NEXT:     _t3 = t.data[0];
// CHECK-NEXT:     res += sum(t.data) + i + 2 * _t3;
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         _d_res += _r_d0;
// CHECK-NEXT:         clad::array_ref<double> _t2 = {(* _d_t).data, 5UL};
// CHECK-NEXT:         sum_pullback(_t1, _r_d0, _t2);
// CHECK-NEXT:         clad::array<double> _r1({(* _d_t).data, 5UL});
// CHECK-NEXT:         * _d_i += _r_d0;
// CHECK-NEXT:         double _r2 = _r_d0 * _t3;
// CHECK-NEXT:         double _r3 = 2 * _r_d0;
// CHECK-NEXT:         (* _d_t).data[0] += _r3;
// CHECK-NEXT:         _d_res -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         sum_pullback(_t0, _d_res, &(* _d_t));
// CHECK-NEXT:         Tangent _r0 = (* _d_t);
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn3(double i, double j) {
    Tangent t;
    t.data[0] = 2*i;
    t.data[1] = 5*i + 3*j;
    return sum(t);
}

// CHECK: void fn3_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     Tangent _d_t({});
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     Tangent _t3;
// CHECK-NEXT:     Tangent t;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     t.data[0] = 2 * _t0;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     _t2 = j;
// CHECK-NEXT:     t.data[1] = 5 * _t1 + 3 * _t2;
// CHECK-NEXT:     _t3 = t;
// CHECK-NEXT:     {
// CHECK-NEXT:         sum_pullback(_t3, 1, &_d_t);
// CHECK-NEXT:         Tangent _r6 = _d_t;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = _d_t.data[1];
// CHECK-NEXT:         double _r2 = _r_d1 * _t1;
// CHECK-NEXT:         double _r3 = 5 * _r_d1;
// CHECK-NEXT:         * _d_i += _r3;
// CHECK-NEXT:         double _r4 = _r_d1 * _t2;
// CHECK-NEXT:         double _r5 = 3 * _r_d1;
// CHECK-NEXT:         * _d_j += _r5;
// CHECK-NEXT:         _d_t.data[1] -= _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_t.data[0];
// CHECK-NEXT:         double _r0 = _r_d0 * _t0;
// CHECK-NEXT:         double _r1 = 2 * _r_d0;
// CHECK-NEXT:         * _d_i += _r1;
// CHECK-NEXT:         _d_t.data[0] -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn4(double i, double j) {
    pairdd p(1, 3);
    pairdd q({7, 5});
    return p.first*i + p.second*j + q.first*i + q.second*j;
}

// CHECK: void fn4_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     pairdd _d_p({});
// CHECK-NEXT:     pairdd _d_q({});
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     double _t5;
// CHECK-NEXT:     double _t6;
// CHECK-NEXT:     double _t7;
// CHECK-NEXT:     pairdd p(1, 3);
// CHECK-NEXT:     pairdd q({7, 5});
// CHECK-NEXT:     _t1 = p.first;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     _t3 = p.second;
// CHECK-NEXT:     _t2 = j;
// CHECK-NEXT:     _t5 = q.first;
// CHECK-NEXT:     _t4 = i;
// CHECK-NEXT:     _t7 = q.second;
// CHECK-NEXT:     _t6 = j;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 1 * _t0;
// CHECK-NEXT:         _d_p.first += _r0;
// CHECK-NEXT:         double _r1 = _t1 * 1;
// CHECK-NEXT:         * _d_i += _r1;
// CHECK-NEXT:         double _r2 = 1 * _t2;
// CHECK-NEXT:         _d_p.second += _r2;
// CHECK-NEXT:         double _r3 = _t3 * 1;
// CHECK-NEXT:         * _d_j += _r3;
// CHECK-NEXT:         double _r4 = 1 * _t4;
// CHECK-NEXT:         _d_q.first += _r4;
// CHECK-NEXT:         double _r5 = _t5 * 1;
// CHECK-NEXT:         * _d_i += _r5;
// CHECK-NEXT:         double _r6 = 1 * _t6;
// CHECK-NEXT:         _d_q.second += _r6;
// CHECK-NEXT:         double _r7 = _t7 * 1;
// CHECK-NEXT:         * _d_j += _r7;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void someMemFn_grad(double i, double j, clad::array_ref<Tangent> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     double _t5;
// CHECK-NEXT:     double _t6;
// CHECK-NEXT:     _t1 = this->data[0];
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     _t3 = this->data[1];
// CHECK-NEXT:     _t2 = j;
// CHECK-NEXT:     _t4 = this->data[2];
// CHECK-NEXT:     _t6 = this->data[3];
// CHECK-NEXT:     _t5 = this->data[4];
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 1 * _t0;
// CHECK-NEXT:         (* _d_this).data[0] += _r0;
// CHECK-NEXT:         double _r1 = _t1 * 1;
// CHECK-NEXT:         * _d_i += _r1;
// CHECK-NEXT:         double _r2 = 1 * _t2;
// CHECK-NEXT:         (* _d_this).data[1] += _r2;
// CHECK-NEXT:         double _r3 = _t3 * 1;
// CHECK-NEXT:         * _d_j += _r3;
// CHECK-NEXT:         double _r4 = 1 * _t4;
// CHECK-NEXT:         double _r5 = 3 * 1;
// CHECK-NEXT:         (* _d_this).data[2] += _r5;
// CHECK-NEXT:         double _r6 = 1 * _t5;
// CHECK-NEXT:         (* _d_this).data[3] += _r6;
// CHECK-NEXT:         double _r7 = _t6 * 1;
// CHECK-NEXT:         (* _d_this).data[4] += _r7;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn5(const Tangent& t, double i) {
    return t.someMemFn2(i, i);
}

// CHECK: void someMemFn2_pullback(double i, double j, double _d_y, clad::array_ref<Tangent> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     double _t5;
// CHECK-NEXT:     _t1 = this->data[0];
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     _t4 = this->data[1];
// CHECK-NEXT:     _t3 = i;
// CHECK-NEXT:     _t5 = _t4 * _t3;
// CHECK-NEXT:     _t2 = j;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = _d_y * _t0;
// CHECK-NEXT:         (* _d_this).data[0] += _r0;
// CHECK-NEXT:         double _r1 = _t1 * _d_y;
// CHECK-NEXT:         * _d_i += _r1;
// CHECK-NEXT:         double _r2 = _d_y * _t2;
// CHECK-NEXT:         double _r3 = _r2 * _t3;
// CHECK-NEXT:         (* _d_this).data[1] += _r3;
// CHECK-NEXT:         double _r4 = _t4 * _r2;
// CHECK-NEXT:         * _d_i += _r4;
// CHECK-NEXT:         double _r5 = _t5 * _d_y;
// CHECK-NEXT:         * _d_j += _r5;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn5_grad(const Tangent &t, double i, clad::array_ref<Tangent> _d_t, clad::array_ref<double> _d_i) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     Tangent _t2;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     _t2 = t;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _grad0 = 0.;
// CHECK-NEXT:         double _grad1 = 0.;
// CHECK-NEXT:         _t2.someMemFn2_pullback(_t0, _t1, 1, &(* _d_t), &_grad0, &_grad1);
// CHECK-NEXT:         double _r0 = _grad0;
// CHECK-NEXT:         * _d_i += _r0;
// CHECK-NEXT:         double _r1 = _grad1;
// CHECK-NEXT:         * _d_i += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

using dcomplex = std::complex<double>;

double fn6(dcomplex c, double i) {
    c.real(5*i);
    double res = c.real() + 3*c.imag() + 6*i;
    res += 4*c.real();
    return res;
}
// CHECK: void real_pullback({{.*}} [[__val:.*]], clad::array_ref<complex<double> > _d_this, clad::array_ref<{{.*}}> [[_d___val:[a-zA-Z_]*]]){{.*}} {
// CHECK-NEXT:     {{(__real)?}} this->[[_M_value:.*]] = [[__val]];
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 ={{( __real)?}} (* _d_this).[[_M_value]];
// CHECK-NEXT:         * [[_d___val]] += _r_d0;
// CHECK-NEXT:         {{(__real)?}} (* _d_this).[[_M_value]] -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: constexpr void real_pullback(double _d_y, clad::array_ref<complex<double> > _d_this){{.*}} {
// CHECK-NEXT:     {{(__real)?}} (* _d_this).{{.*}} += _d_y;
// CHECK-NEXT: }

// CHECK: constexpr void imag_pullback(double _d_y, clad::array_ref<complex<double> > _d_this){{.*}} {
// CHECK-NEXT:     {{(__imag)?}} (* _d_this).{{.*}} += _d_y;
// CHECK-NEXT: }

// CHECK: void fn6_grad(dcomplex c, double i, clad::array_ref<dcomplex> _d_c, clad::array_ref<double> _d_i) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     dcomplex _t2;
// CHECK-NEXT:     dcomplex _t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     dcomplex _t5;
// CHECK-NEXT:     double _t6;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double _t7;
// CHECK-NEXT:     dcomplex _t8;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     _t1 = 5 * _t0;
// CHECK-NEXT:     _t2 = c;
// CHECK-NEXT:     c.real(_t1);
// CHECK-NEXT:     _t3 = c;
// CHECK-NEXT:     _t5 = c;
// CHECK-NEXT:     _t4 = c.imag();
// CHECK-NEXT:     _t6 = i;
// CHECK-NEXT:     double res = c.real() + 3 * _t4 + 6 * _t6;
// CHECK-NEXT:     _t8 = c;
// CHECK-NEXT:     _t7 = c.real();
// CHECK-NEXT:     res += 4 * _t7;
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         _d_res += _r_d0;
// CHECK-NEXT:         double _r7 = _r_d0 * _t7;
// CHECK-NEXT:         double _r8 = 4 * _r_d0;
// CHECK-NEXT:         _t8.real_pullback(_r8, &(* _d_c));
// CHECK-NEXT:         _d_res -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         _t3.real_pullback(_d_res, &(* _d_c));
// CHECK-NEXT:         double _r3 = _d_res * _t4;
// CHECK-NEXT:         double _r4 = 3 * _d_res;
// CHECK-NEXT:         _t5.imag_pullback(_r4, &(* _d_c));
// CHECK-NEXT:         double _r5 = _d_res * _t6;
// CHECK-NEXT:         double _r6 = 6 * _d_res;
// CHECK-NEXT:         * _d_i += _r6;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _grad0 = 0.;
// CHECK-NEXT:         _t2.real_pullback(_t1, &(* _d_c), &_grad0);
// CHECK-NEXT:         double _r0 = _grad0;
// CHECK-NEXT:         double _r1 = _r0 * _t0;
// CHECK-NEXT:         double _r2 = 5 * _r0;
// CHECK-NEXT:         * _d_i += _r2;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn7(dcomplex c1, dcomplex c2) {
    c1.real(c2.imag() + 5*c2.real());
    return c1.real() + 3*c1.imag();
}

// CHECK: void fn7_grad(dcomplex c1, dcomplex c2, clad::array_ref<dcomplex> _d_c1, clad::array_ref<dcomplex> _d_c2) {
// CHECK-NEXT:     dcomplex _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     dcomplex _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     dcomplex _t4;
// CHECK-NEXT:     dcomplex _t5;
// CHECK-NEXT:     double _t6;
// CHECK-NEXT:     dcomplex _t7;
// CHECK-NEXT:     _t0 = c2;
// CHECK-NEXT:     _t2 = c2;
// CHECK-NEXT:     _t1 = c2.real();
// CHECK-NEXT:     _t3 = c2.imag() + 5 * _t1;
// CHECK-NEXT:     _t4 = c1;
// CHECK-NEXT:     c1.real(_t3);
// CHECK-NEXT:     _t5 = c1;
// CHECK-NEXT:     _t7 = c1;
// CHECK-NEXT:     _t6 = c1.imag();
// CHECK-NEXT:     {
// CHECK-NEXT:         _t5.real_pullback(1, &(* _d_c1));
// CHECK-NEXT:         double _r3 = 1 * _t6;
// CHECK-NEXT:         double _r4 = 3 * 1;
// CHECK-NEXT:         _t7.imag_pullback(_r4, &(* _d_c1));
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _grad0 = 0.;
// CHECK-NEXT:         _t4.real_pullback(_t3, &(* _d_c1), &_grad0);
// CHECK-NEXT:         double _r0 = _grad0;
// CHECK-NEXT:         _t0.imag_pullback(_r0, &(* _d_c2));
// CHECK-NEXT:         double _r1 = _r0 * _t1;
// CHECK-NEXT:         double _r2 = 5 * _r0;
// CHECK-NEXT:         _t2.real_pullback(_r2, &(* _d_c2));
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn8(Tangent t, dcomplex c) {
  t.updateTo(c.real());
  return sum(t);
}

// CHECK: void updateTo_pullback(double d, clad::array_ref<Tangent> _d_this, clad::array_ref<double> _d_d) {
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     clad::tape<int> _t1 = {};
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int i = 0; i < 5; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         this->data[clad::push(_t1, i)] = d;
// CHECK-NEXT:     }
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         int _t2 = clad::pop(_t1);
// CHECK-NEXT:         double _r_d0 = (* _d_this).data[_t2];
// CHECK-NEXT:         * _d_d += _r_d0;
// CHECK-NEXT:         (* _d_this).data[_t2] -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn8_grad(Tangent t, dcomplex c, clad::array_ref<Tangent> _d_t, clad::array_ref<dcomplex> _d_c) {
// CHECK-NEXT:     dcomplex _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     Tangent _t2;
// CHECK-NEXT:     Tangent _t3;
// CHECK-NEXT:     _t0 = c;
// CHECK-NEXT:     _t1 = c.real();
// CHECK-NEXT:     _t2 = t;
// CHECK-NEXT:     t.updateTo(_t1);
// CHECK-NEXT:     _t3 = t;
// CHECK-NEXT:     {
// CHECK-NEXT:         sum_pullback(_t3, 1, &(* _d_t));
// CHECK-NEXT:         Tangent _r1 = (* _d_t);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _grad0 = 0.;
// CHECK-NEXT:         _t2.updateTo_pullback(_t1, &(* _d_t), &_grad0);
// CHECK-NEXT:         double _r0 = _grad0;
// CHECK-NEXT:         _t0.real_pullback(_r0, &(* _d_c));
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn9(Tangent t, dcomplex c) {
  double res = 0;
  for (int i=0; i<5; ++i) {
    res += c.real() + 2*c.imag();
  }
  res += sum(t);
  return res;
}

// CHECK: void fn9_grad(Tangent t, dcomplex c, clad::array_ref<Tangent> _d_t, clad::array_ref<dcomplex> _d_c) {
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     clad::tape<dcomplex> _t1 = {};
// CHECK-NEXT:     clad::tape<double> _t2 = {};
// CHECK-NEXT:     clad::tape<dcomplex> _t3 = {};
// CHECK-NEXT:     Tangent _t4;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int i = 0; i < 5; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, c);
// CHECK-NEXT:         clad::push(_t3, c);
// CHECK-NEXT:         res += c.real() + 2 * clad::push(_t2, c.imag());
// CHECK-NEXT:     }
// CHECK-NEXT:     _t4 = t;
// CHECK-NEXT:     res += sum(t);
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = _d_res;
// CHECK-NEXT:         _d_res += _r_d1;
// CHECK-NEXT:         sum_pullback(_t4, _r_d1, &(* _d_t));
// CHECK-NEXT:         Tangent _r4 = (* _d_t);
// CHECK-NEXT:         _d_res -= _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d0 = _d_res;
// CHECK-NEXT:             _d_res += _r_d0;
// CHECK-NEXT:             std{{(::__1)?}}::complex<double> _r0 = clad::pop(_t1);
// CHECK-NEXT:             _r0.real_pullback(_r_d0, &(* _d_c));
// CHECK-NEXT:             double _r1 = _r_d0 * clad::pop(_t2);
// CHECK-NEXT:             double _r2 = 2 * _r_d0;
// CHECK-NEXT:             std{{(::__1)?}}::complex<double> _r3 = clad::pop(_t3);
// CHECK-NEXT:             _r3.imag_pullback(_r2, &(* _d_c));
// CHECK-NEXT:             _d_res -= _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

void print(const Tangent& t) {
  for (int i = 0; i < 5; ++i) {
    printf("%.2f", t.data[i]);
    if (i != 4)
      printf(", ");
  }
}

int main() {
    pairdd p(3, 5), d_p;
    double i = 3, d_i, d_j;
    Tangent t, d_t;
    dcomplex c1, c2, d_c1, d_c2;
    auto memFn1 = &Tangent::someMemFn;

    INIT_GRADIENT(fn1);
    INIT_GRADIENT(fn2);
    INIT_GRADIENT(fn3);
    INIT_GRADIENT(fn4);
    INIT_GRADIENT(memFn1);
    INIT_GRADIENT(fn5);
    INIT_GRADIENT(fn6);
    INIT_GRADIENT(fn7);
    INIT_GRADIENT(fn8);
    INIT_GRADIENT(fn9);
    
    TEST_GRADIENT(fn1, /*numOfDerivativeArgs=*/2, p, i, &d_p, &d_i);    // CHECK-EXEC: {1.00, 2.00, 3.00}
    TEST_GRADIENT(fn2, /*numOfDerivativeArgs=*/2, t, i, &d_t, &d_i);    // CHECK-EXEC: {4.00, 2.00, 2.00, 2.00, 2.00, 1.00}
    TEST_GRADIENT(fn3, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {7.00, 3.00}
    TEST_GRADIENT(fn4, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {8.00, 8.00}
    t.updateTo(5);
    TEST_GRADIENT(memFn1, /*numOfDerivativeArgs=*/3, t, 3, 5, &d_t, &d_i, &d_j);   // CHECK-EXEC: {3.00, 5.00, 3.00, 5.00, 5.00, 5.00, 5.00}
    t.updateTo(5);
    TEST_GRADIENT(fn5, /*numOfDerivativeArgs=*/2, t, 3, &d_t, &d_i);    // CHECK-EXEC: {3.00, 9.00, 0.00, 0.00, 0.00, 35.00}
    TEST_GRADIENT(fn6, /*numOfDerivativeArgs=*/2, c1, 3, &d_c1, &d_i);  // CHECK-EXEC: {0.00, 3.00, 31.00}
    TEST_GRADIENT(fn7, /*numOfDerivativeArgs=*/2, c1, c2, &d_c1, &d_c2);// CHECK-EXEC: {0.00, 3.00, 5.00, 1.00}
    TEST_GRADIENT(fn8, /*numOfDerivativeArgs=*/2, t, c1, &d_t, &d_c1);  // CHECK-EXEC: {0.00, 0.00, 0.00, 0.00, 0.00, 5.00, 0.00}
    TEST_GRADIENT(fn9, /*numOfDerivativeArgs=*/2, t, c1, &d_t, &d_c1);  // CHECK-EXEC: {1.00, 1.00, 1.00, 1.00, 1.00, 5.00, 10.00}
}
