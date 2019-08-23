#include <cstdio>

void f_1_grad(double a, double b, double c, double output[], double _result[]) {
    double _t0;
    double _t1;
    double _t2;
    double _t3;
    double _t4;
    double _t5;
    double _t6;
    double _t7;
    double _t8;
    double _t9;
    double _t10;
    double _t11;
    double _t12;
    double _t13;
    double _t14;
    double _t15;
    _t2 = a;
    _t1 = a;
    _t3 = _t2 * _t1;
    _t0 = a;
    output[0] = a * a * a;
    _t6 = a;
    _t5 = a;
    _t7 = _t6 * _t5;
    _t4 = a;
    _t10 = b;
    _t9 = b;
    _t11 = _t10 * _t9;
    _t8 = b;
    output[1] = a * a * a + b * b * b;
    _t13 = c;
    _t12 = c;
    _t15 = a;
    _t14 = a;
    output[2] = c * c * 10 - a * a;
    {
        double _r12 = 1 * 10;
        double _r13 = _r12 * _t12;
        _result[8UL] += _r13;
        double _r14 = _t13 * _r12;
        _result[8UL] += _r14;
        double _r15 = -1 * _t14;
        _result[6UL] += _r15;
        double _r16 = _t15 * -1;
        _result[6UL] += _r16;
    }
    {
        double _r4 = 1 * _t4;
        double _r5 = _r4 * _t5;
        // printf("_r5: %.2f\n", _r5);
        _result[3UL] += _r5;
        double _r6 = _t6 * _r4;
        _result[3UL] += _r6;
        double _r7 = _t7 * 1;
        _result[3UL] += _r7;
        double _r8 = 1 * _t8;
        double _r9 = _r8 * _t9;
        _result[4UL] += _r9;
        double _r10 = _t10 * _r8;
        _result[4UL] += _r10;
        double _r11 = _t11 * 1;
        _result[4UL] += _r11;
    }
    {
        double _r0 = 1 * _t0;
        double _r1 = _r0 * _t1;
        // printf("_r1: %.2f\n", _r1);
        _result[0UL] += _r1;
        double _r2 = _t2 * _r0;
        _result[0UL] += _r2;
        double _r3 = _t3 * 1;
        // printf("_r3: %.2f\n", _r3);
        _result[0UL] += _r3;
        // printf("_result[0]: %.2f\n", _result[0UL]);
    }
}

int main() {
  double result[9] = {0};
  double outputa[10];
  f_1_grad(10, 5, 4, outputa, result);
  printf("%.2f %.2f %.2f\n%.2f %.2f %.2f\n%.2f %.2f %.2f\n", result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8]);
}
