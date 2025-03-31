// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oStructsInLoops.out 2>&1 | %filecheck %s
// RUN: ./StructsInLoops.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oStructsInLoops.out
// RUN: ./StructsInLoops.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <stdio.h>
#include <vector>

typedef struct {
    double val;
} Struct;

typedef struct {
    std::vector<float> vector;
} Struct2;

double fn1(double a) {
    double result = 0;
    for (int i=0;i<1;i++){
        Struct s;
        s.val += 2;
        result += s.val * a;
    }
    return result;
}

// CHECK-NEXT: void fn1_grad(double a, double *_d_a) {
// CHECK-NEXT:    int _d_i = 0;
// CHECK-NEXT:    int i = 0;
// CHECK-NEXT:    clad::tape<Struct> _t1 = {};
// CHECK-NEXT:    Struct _d_s = {0.};
// CHECK-NEXT:    Struct s = {0.};
// CHECK-NEXT:    double _d_result = 0.;
// CHECK-NEXT:    double result = 0;
// CHECK-NEXT:    unsigned long _t0 = 0UL;
// CHECK-NEXT:    for (i = 0; ; i++) {
// CHECK-NEXT:        {
// CHECK-NEXT:            if (!(i < 1))
// CHECK-NEXT:                break;
// CHECK-NEXT:        }
// CHECK-NEXT:        _t0++;
// CHECK-NEXT:        clad::push(_t1, std::move(s)) , s = {};
// CHECK-NEXT:        s.val += 2;
// CHECK-NEXT:        result += s.val * a;
// CHECK-NEXT:    }
// CHECK-NEXT:    _d_result += 1;
// CHECK-NEXT:    for (;; _t0--) {
// CHECK-NEXT:        {
// CHECK-NEXT:            if (!_t0)
// CHECK-NEXT:                break;
// CHECK-NEXT:        }
// CHECK-NEXT:        {
// CHECK-NEXT:            double _r_d1 = _d_result;
// CHECK-NEXT:            _d_s.val += _r_d1 * a;
// CHECK-NEXT:            *_d_a += s.val * _r_d1;
// CHECK-NEXT:        }
// CHECK-NEXT:        double _r_d0 = _d_s.val;
// CHECK-NEXT:        {
// CHECK-NEXT:            _d_s = {0.};
// CHECK-NEXT:            s = clad::pop(_t1);
// CHECK-NEXT:        }
// CHECK-NEXT:    }
// CHECK-NEXT: }

double fn2 (double a) {
    double result = 0;
    int i = 1;
    while (i--) {
        Struct s1;
        Struct s2;
        s2.val = s1.val + 2;
        result += s2.val * a * a;
    }
    return result;
}
 
// CHECK: void fn2_grad(double a, double *_d_a) {
// CHECK-NEXT:     clad::tape<Struct> _t1 = {};
// CHECK-NEXT:     Struct _d_s1 = {0.};
// CHECK-NEXT:     Struct s1 = {0.};
// CHECK-NEXT:     clad::tape<Struct> _t2 = {};
// CHECK-NEXT:     Struct _d_s2 = {0.};
// CHECK-NEXT:     Struct s2 = {0.};
// CHECK-NEXT:     double _d_result = 0.;
// CHECK-NEXT:     double result = 0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 1;
// CHECK-NEXT:     unsigned long _t0 = 0UL;
// CHECK-NEXT:     while (i--)
// CHECK-NEXT:         {
// CHECK-NEXT:             _t0++;
// CHECK-NEXT:             clad::push(_t1, std::move(s1)) , s1 = {};
// CHECK-NEXT:             clad::push(_t2, std::move(s2)) , s2 = {};
// CHECK-NEXT:             s2.val = s1.val + 2;
// CHECK-NEXT:             result += s2.val * a * a;
// CHECK-NEXT:         }
// CHECK-NEXT:     _d_result += 1;
// CHECK-NEXT:     while (_t0)
// CHECK-NEXT:         {
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     double _r_d1 = _d_result;
// CHECK-NEXT:                     _d_s2.val += _r_d1 * a * a;
// CHECK-NEXT:                     *_d_a += s2.val * _r_d1 * a;
// CHECK-NEXT:                     *_d_a += s2.val * a * _r_d1;
// CHECK-NEXT:                 }
// CHECK-NEXT:                 {
// CHECK-NEXT:                     double _r_d0 = _d_s2.val;
// CHECK-NEXT:                     _d_s2.val = 0.;
// CHECK-NEXT:                     _d_s1.val += _r_d0;
// CHECK-NEXT:                 }
// CHECK-NEXT:                 {
// CHECK-NEXT:                     _d_s2 = {0.};
// CHECK-NEXT:                     s2 = clad::pop(_t2);
// CHECK-NEXT:                 }
// CHECK-NEXT:                 {
// CHECK-NEXT:                     _d_s1 = {0.};
// CHECK-NEXT:                     s1 = clad::pop(_t1);
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:             _t0--;
// CHECK-NEXT:         }
// CHECK-NEXT: }

float fn3(float a) {
    float result;
    for (int i = 0; i < 1; i++) {
        Struct2 s;
        float size;
        if (!s.vector.size()) 
            size = 0;
        result =  size + a;
    }
    return result;
}

// CHECK: void fn3_grad(float a, float *_d_a) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<Struct2> _t1 = {};
// CHECK-NEXT:     Struct2 _d_s = {{}};
// CHECK-NEXT:     Struct2 s = {{}};
// CHECK-NEXT:     float _d_size = 0.F;
// CHECK-NEXT:     float size;
// CHECK-NEXT:     clad::tape<bool> _cond0 = {};
// CHECK-NEXT:     float _d_result = 0.F;
// CHECK-NEXT:     float result;
// CHECK-NEXT:     unsigned long _t0 = 0UL;
// CHECK-NEXT:     for (i = 0; ; i++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 1))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, std::move(s)) , s = {};
// CHECK-NEXT:         {
// CHECK-NEXT:             clad::push(_cond0, !s.vector.size());
// CHECK-NEXT:             if (clad::back(_cond0))
// CHECK-NEXT:                 size = 0;
// CHECK-NEXT:         }
// CHECK-NEXT:         result = size + a;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_result += 1;
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             float _r_d1 = _d_result;
// CHECK-NEXT:             _d_result = 0.F;
// CHECK-NEXT:             _d_size += _r_d1;
// CHECK-NEXT:             *_d_a += _r_d1;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             if (clad::back(_cond0)) {
// CHECK-NEXT:                 float _r_d0 = _d_size;
// CHECK-NEXT:                 _d_size = 0.F;
// CHECK-NEXT:             }
// CHECK-NEXT:             clad::pop(_cond0);
// CHECK-NEXT:         }
// CHECK-NEXT:         _d_size = 0.F;
// CHECK-NEXT:         {
// CHECK-NEXT:             _d_s = {{}};
// CHECK-NEXT:             s = clad::pop(_t1);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
    auto grad = clad::gradient(fn1);
    auto grad2 = clad::gradient(fn2);
    auto grad3 = clad::gradient(fn3);

    double a = 2.0;
    double b = 3.0;
    float c = 6.0;

    double d_a = 0;
    grad.execute(a, &d_a);
    printf("fn1 derivative: %.1f\n", d_a); //CHECK-EXEC: fn1 derivative: 2.0
    
    double d_b = 0;
    grad2.execute(b, &d_b);
    printf("fn2 derivative: %.1f\n", d_b); //CHECK-EXEC: fn2 derivative: 12.0

    float d_c = 0;
    grad3.execute(c, &d_c);
    printf("fn3 derivative: %.1f\n", d_c); //CHECK-EXEC: fn3 derivative: 1.0
    
    return 0;
}
