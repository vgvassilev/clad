// RUN: %cladclang %s -I%S/../../include -oUserDefinedTypes.out 2>&1 | %filecheck %s
// RUN: ./UserDefinedTypes.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oUserDefinedTypes.out
// RUN: ./UserDefinedTypes.out | %filecheck_exec %s
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

// CHECK: void fn1_grad(pairdd p, double i, pairdd *_d_p, double *_d_i) {
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = p.first + 2 * p.second + 3 * i;
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         (*_d_p).first += _d_res;
// CHECK-NEXT:         (*_d_p).second += 2 * _d_res;
// CHECK-NEXT:         *_d_i += 3 * _d_res;
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

// CHECK: void sum_pullback(Tangent &t, double _d_y, Tangent *_d_t);

double sum(double *data) {
    double res = 0;
    for (int i=0; i<5; ++i)
        res += data[i];
    return res;
}

// CHECK: void sum_pullback(double *data, double _d_y, double *_d_data);

double fn2(Tangent t, double i) {
    double res = sum(t);
    res += sum(t.data) + i + 2*t.data[0];
    return res;
}

// CHECK: void fn2_grad(Tangent t, double i, Tangent *_d_t, double *_d_i) {
// CHECK-NEXT:     Tangent _t0 = t;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = sum(t);
// CHECK-NEXT:     double _t1 = res;
// CHECK-NEXT:     res += sum(t.data) + i + 2 * t.data[0];
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         res = _t1;
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         sum_pullback(t.data, _r_d0, (*_d_t).data);
// CHECK-NEXT:         *_d_i += _r_d0;
// CHECK-NEXT:         (*_d_t).data[0] += 2 * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         t = _t0;
// CHECK-NEXT:         sum_pullback(_t0, _d_res, &(*_d_t));
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn3(double i, double j) {
    Tangent t;
    t.data[0] = 2*i;
    t.data[1] = 5*i + 3*j;
    return sum(t);
}

// CHECK: void fn3_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     Tangent _d_t({});
// CHECK-NEXT:     Tangent t;
// CHECK-NEXT:     double _t0 = t.data[0];
// CHECK-NEXT:     t.data[0] = 2 * i;
// CHECK-NEXT:     double _t1 = t.data[1];
// CHECK-NEXT:     t.data[1] = 5 * i + 3 * j;
// CHECK-NEXT:     Tangent _t2 = t;
// CHECK-NEXT:     {
// CHECK-NEXT:         t = _t2;
// CHECK-NEXT:         sum_pullback(_t2, 1, &_d_t);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         t.data[1] = _t1;
// CHECK-NEXT:         double _r_d1 = _d_t.data[1];
// CHECK-NEXT:         _d_t.data[1] = 0;
// CHECK-NEXT:         *_d_i += 5 * _r_d1;
// CHECK-NEXT:         *_d_j += 3 * _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         t.data[0] = _t0;
// CHECK-NEXT:         double _r_d0 = _d_t.data[0];
// CHECK-NEXT:         _d_t.data[0] = 0;
// CHECK-NEXT:         *_d_i += 2 * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn4(double i, double j) {
    pairdd p(1, 3);
    pairdd q({7, 5});
    return p.first*i + p.second*j + q.first*i + q.second*j;
}

// CHECK: void fn4_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     pairdd _d_p({});
// CHECK-NEXT:     pairdd p(1, 3);
// CHECK-NEXT:     pairdd _d_q({});
// CHECK-NEXT:     pairdd q({7, 5});
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_p.first += 1 * i;
// CHECK-NEXT:         *_d_i += p.first * 1;
// CHECK-NEXT:         _d_p.second += 1 * j;
// CHECK-NEXT:         *_d_j += p.second * 1;
// CHECK-NEXT:         _d_q.first += 1 * i;
// CHECK-NEXT:         *_d_i += q.first * 1;
// CHECK-NEXT:         _d_q.second += 1 * j;
// CHECK-NEXT:         *_d_j += q.second * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void someMemFn_grad(double i, double j, Tangent *_d_this, double *_d_i, double *_d_j) {
// CHECK-NEXT:     {
// CHECK-NEXT:         (*_d_this).data[0] += 1 * i;
// CHECK-NEXT:         *_d_i += this->data[0] * 1;
// CHECK-NEXT:         (*_d_this).data[1] += 1 * j;
// CHECK-NEXT:         *_d_j += this->data[1] * 1;
// CHECK-NEXT:         (*_d_this).data[2] += 3 * 1;
// CHECK-NEXT:         (*_d_this).data[3] += 1 * this->data[4];
// CHECK-NEXT:         (*_d_this).data[4] += this->data[3] * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn5(const Tangent& t, double i) {
    return t.someMemFn2(i, i);
}

// CHECK: void someMemFn2_pullback(double i, double j, double _d_y, Tangent *_d_this, double *_d_i, double *_d_j) const;

// CHECK: void fn5_grad(const Tangent &t, double i, Tangent *_d_t, double *_d_i) {
// CHECK-NEXT:     Tangent _t0 = t;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         double _r1 = 0;
// CHECK-NEXT:         _t0.someMemFn2_pullback(i, i, 1, &(*_d_t), &_r0, &_r1);
// CHECK-NEXT:         *_d_i += _r0;
// CHECK-NEXT:         *_d_i += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

using dcomplex = std::complex<double>;

double fn6(dcomplex c, double i) {
    c.real(5*i);
    double res = c.real() + 3*c.imag() + 6*i;
    res += 4*c.real();
    return res;
}
// CHECK: void real_pullback({{.*}} [[__val:.*]], std{{(::__1)?}}::complex<double> *_d_this, {{.*}} *[[_d___val:[a-zA-Z_]*]]){{.*}};

// CHECK: constexpr void real_pullback(double _d_y, std{{(::__1)?}}::complex<double> *_d_this){{.*}};

// CHECK: constexpr void imag_pullback(double _d_y, std{{(::__1)?}}::complex<double> *_d_this){{.*}};

// CHECK: void fn6_grad(dcomplex c, double i, dcomplex *_d_c, double *_d_i) {
// CHECK-NEXT:     dcomplex _t0 = c;
// CHECK-NEXT:     c.real(5 * i);
// CHECK-NEXT:     dcomplex _t1 = c;
// CHECK-NEXT:     dcomplex _t3 = c;
// CHECK-NEXT:     double _t2 = c.imag();
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = c.real() + 3 * _t2 + 6 * i;
// CHECK-NEXT:     double _t4 = res;
// CHECK-NEXT:     dcomplex _t6 = c;
// CHECK-NEXT:     double _t5 = c.real();
// CHECK-NEXT:     res += 4 * _t5;
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         res = _t4;
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         _t6.real_pullback(4 * _r_d0, &(*_d_c));
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         _t1.real_pullback(_d_res, &(*_d_c));
// CHECK-NEXT:         _t3.imag_pullback(3 * _d_res, &(*_d_c));
// CHECK-NEXT:         *_d_i += 6 * _d_res;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         _t0.real_pullback(5 * i, &(*_d_c), &_r0);
// CHECK-NEXT:         *_d_i += 5 * _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn7(dcomplex c1, dcomplex c2) {
    c1.real(c2.imag() + 5*c2.real());
    return c1.real() + 3*c1.imag();
}

// CHECK: void fn7_grad(dcomplex c1, dcomplex c2, dcomplex *_d_c1, dcomplex *_d_c2) {
// CHECK-NEXT:     dcomplex _t0 = c2;
// CHECK-NEXT:     dcomplex _t2 = c2;
// CHECK-NEXT:     double _t1 = c2.real();
// CHECK-NEXT:     dcomplex _t3 = c1;
// CHECK-NEXT:     c1.real(c2.imag() + 5 * _t1);
// CHECK-NEXT:     dcomplex _t4 = c1;
// CHECK-NEXT:     dcomplex _t6 = c1;
// CHECK-NEXT:     double _t5 = c1.imag();
// CHECK-NEXT:     {
// CHECK-NEXT:         _t4.real_pullback(1, &(*_d_c1));
// CHECK-NEXT:         _t6.imag_pullback(3 * 1, &(*_d_c1));
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         _t3.real_pullback(c2.imag() + 5 * _t1, &(*_d_c1), &_r0);
// CHECK-NEXT:         _t0.imag_pullback(_r0, &(*_d_c2));
// CHECK-NEXT:         _t2.real_pullback(5 * _r0, &(*_d_c2));
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn8(Tangent t, dcomplex c) {
  t.updateTo(c.real());
  return sum(t);
}

// CHECK: void updateTo_pullback(double d, Tangent *_d_this, double *_d_d);

// CHECK: void fn8_grad(Tangent t, dcomplex c, Tangent *_d_t, dcomplex *_d_c) {
// CHECK-NEXT:     dcomplex _t0 = c;
// CHECK-NEXT:     Tangent _t1 = t;
// CHECK-NEXT:     t.updateTo(c.real());
// CHECK-NEXT:     Tangent _t2 = t;
// CHECK-NEXT:     {
// CHECK-NEXT:         t = _t2;
// CHECK-NEXT:         sum_pullback(_t2, 1, &(*_d_t));
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         _t1.updateTo_pullback(c.real(), &(*_d_t), &_r0);
// CHECK-NEXT:         _t0.real_pullback(_r0, &(*_d_c));
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

// CHECK: void fn9_grad(Tangent t, dcomplex c, Tangent *_d_t, dcomplex *_d_c) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     clad::tape<dcomplex> _t2 = {};
// CHECK-NEXT:     clad::tape<double> _t3 = {};
// CHECK-NEXT:     clad::tape<dcomplex> _t4 = {};
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
// CHECK-NEXT:     for (i = 0; ; ++i) {
// CHECK-NEXT:     {
// CHECK-NEXT:          if (!(i < 5))
// CHECK-NEXT:          break;
// CHECK-NEXT:     }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, res);
// CHECK-NEXT:         clad::push(_t2, c);
// CHECK-NEXT:         clad::push(_t4, c);
// CHECK-NEXT:         res += c.real() + 2 * clad::push(_t3, c.imag());
// CHECK-NEXT:     }
// CHECK-NEXT:     double _t5 = res;
// CHECK-NEXT:     Tangent _t6 = t;
// CHECK-NEXT:     res += sum(t);
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         res = _t5;
// CHECK-NEXT:         double _r_d1 = _d_res;
// CHECK-NEXT:         t = _t6;
// CHECK-NEXT:         sum_pullback(_t6, _r_d1, &(*_d_t));
// CHECK-NEXT:     }
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         --i;
// CHECK-NEXT:         {
// CHECK-NEXT:             res = clad::pop(_t1);
// CHECK-NEXT:             double _r_d0 = _d_res;
// CHECK-NEXT:             clad::back(_t2).real_pullback(_r_d0, &(*_d_c));
// CHECK-NEXT:             clad::pop(_t2);
// CHECK-NEXT:             clad::back(_t4).imag_pullback(2 * _r_d0, &(*_d_c));
// CHECK-NEXT:             clad::pop(_t4);
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

// CHECK: void sum_pullback(Tangent &t, double _d_y, Tangent *_d_t) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
// CHECK-NEXT:     for (i = 0; ; ++i) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 5))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, res);
// CHECK-NEXT:         res += t.data[i];
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += _d_y;
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         --i;
// CHECK-NEXT:         res = clad::pop(_t1);
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         (*_d_t).data[i] += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void sum_pullback(double *data, double _d_y, double *_d_data) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
// CHECK-NEXT:     for (i = 0; ; ++i) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 5))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, res);
// CHECK-NEXT:         res += data[i];
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += _d_y;
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         --i;
// CHECK-NEXT:         res = clad::pop(_t1);
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         _d_data[i] += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void someMemFn2_pullback(double i, double j, double _d_y, Tangent *_d_this, double *_d_i, double *_d_j) const {
// CHECK-NEXT:     {
// CHECK-NEXT:         (*_d_this).data[0] += _d_y * i;
// CHECK-NEXT:         *_d_i += this->data[0] * _d_y;
// CHECK-NEXT:         (*_d_this).data[1] += _d_y * j * i;
// CHECK-NEXT:         *_d_i += this->data[1] * _d_y * j;
// CHECK-NEXT:         *_d_j += this->data[1] * i * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void real_pullback({{.*}} [[__val:.*]], std{{(::__1)?}}::complex<double> *_d_this, {{.*}} *[[_d___val:[a-zA-Z_]*]]){{.*}} {
// CHECK-NEXT:     double _t0 ={{( __real)?}} this->[[_M_value:.*]];
// CHECK-NEXT:     {{(__real)?}} this->[[_M_value:.*]] = [[__val]];
// CHECK-NEXT:     {
// CHECK-NEXT:         {{(__real)?}} this->[[_M_value:.*]] = _t0;
// CHECK-NEXT:         double _r_d0 ={{( __real)?}} (*_d_this).[[_M_value]];
// CHECK-NEXT:         {{(__real)?}} (*_d_this).[[_M_value]] = 0;
// CHECK-NEXT:         *[[_d___val]] += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: constexpr void real_pullback(double _d_y, std{{(::__1)?}}::complex<double> *_d_this){{.*}} {
// CHECK-NEXT:     {{(__real)?}} (*_d_this).{{.*}} += _d_y;
// CHECK-NEXT: }

// CHECK: constexpr void imag_pullback(double _d_y, std{{(::__1)?}}::complex<double> *_d_this){{.*}} {
// CHECK-NEXT:     {{(__imag)?}} (*_d_this).{{.*}} += _d_y;
// CHECK-NEXT: }

// CHECK: void updateTo_pullback(double d, Tangent *_d_this, double *_d_d) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
// CHECK-NEXT:     for (i = 0; ; ++i) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 5))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, this->data[i]);
// CHECK-NEXT:         this->data[i] = d;
// CHECK-NEXT:     }
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         --i;
// CHECK-NEXT:         this->data[i] = clad::pop(_t1);
// CHECK-NEXT:         double _r_d0 = (*_d_this).data[i];
// CHECK-NEXT:         (*_d_this).data[i] = 0;
// CHECK-NEXT:         *_d_d += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }