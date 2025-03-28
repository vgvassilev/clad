// RUN: %cladclang %s -I%S/../../include -oStructsInLoops..out 2>&1
// RUN: ./StructsInLoops..out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <stdio.h>

typedef struct {
    double val;
} Struct;

double fn1(double a) {
    double result = 0;
    for (int i=0;i<1;i++){
        Struct s;
        s.val += 2;
        result += s.val * a;
    }
    return result;
}

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

int main() {
    auto grad = clad::gradient(fn1);
    auto grad2 = clad::gradient(fn2);

    double a = 2.0;
    double b = 3.0;

    double d_a = 0;
    grad.execute(a, &d_a);
    printf("fn1 derivative: %.1f\n", d_a); //CHECK-EXEC: fn1 derivative: 2.0
    
    double d_b = 0;
    grad2.execute(b, &d_b);
    printf("fn2 derivative: %.1f\n", d_b); //CHECK-EXEC: fn2 derivative: 12.0
    
    return 0;
}
