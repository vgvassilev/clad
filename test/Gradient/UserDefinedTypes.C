// RUN: %cladclang %s -I%S/../../include -oUserDefinedTypes.out 2>&1 -lstdc++ -lm | FileCheck %s
// RUN: ./UserDefinedTypes.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"
#include <utility>

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
// CHECK-NEXT:     double fn1_return = res;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
        for (int i=0; i<5; ++i)
            data[i] = d;
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
// CHECK-NEXT:     double sum_return = res;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     double sum_return = res;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     double fn2_return = res;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     double fn3_return = sum(t);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     double fn4_return = _t1 * _t0 + _t3 * _t2 + _t5 * _t4 + _t7 * _t6;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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

namespace std {
void print(const pairdd& p) { printf("%.2f, %.2f", p.first, p.second); }
} // namespace std

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

    INIT_GRADIENT(fn1);
    INIT_GRADIENT(fn2);
    INIT_GRADIENT(fn3);
    INIT_GRADIENT(fn4);

    TEST_GRADIENT(fn1, /*numOfDerivativeArgs=*/2, p, i, &d_p, &d_i);    // CHECK-EXEC: {1.00, 2.00, 3.00}
    TEST_GRADIENT(fn2, /*numOfDerivativeArgs=*/2, t, i, &d_t, &d_i);    // CHECK-EXEC: {4.00, 2.00, 2.00, 2.00, 2.00, 1.00}
    TEST_GRADIENT(fn3, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {7.00, 3.00}
    TEST_GRADIENT(fn4, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {8.00, 8.00}
}